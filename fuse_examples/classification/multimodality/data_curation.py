from MedicalAnalyticsCore.DatabaseUtils.selected_studies_queries import get_annotations_and_findings
from MedicalAnalyticsCore.DatabaseUtils.tableResolver import TableResolver
from MedicalAnalyticsCore.DatabaseUtils.connection import create_homer_engine, Connection
from MedicalAnalyticsCore.DatabaseUtils import tableNames
from MedicalAnalyticsCore.DatabaseUtils import db_utils as db

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle

#------------------Baseline
def apply_gluon_baseline(train_set,test_set,label,save_path):
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(label=label, path=save_path, eval_metric='roc_auc').fit(train_set)
    results = predictor.fit_summary(show_plot=True)

    # Inference time:
    y_test = test_set[label]
    test_data = test_set.drop(labels=[label],
                               axis=1)  # delete labels from test data since we wouldn't have them in practice
    print(test_data.head())

    predictor = TabularPredictor.load(
        save_path)
    y_pred = predictor.predict_proba(test_data)
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

def mg_data_curation(imaging_filename):
    REVISION_DATE = '20200915'
    TableResolver().set_revision(REVISION_DATE)
    # extand dataset by adding  'baptist', 'froedtert', 'miami', 'ucsd'
    revision = {'prefix': 'sentara', 'suffix': REVISION_DATE}
    engine = Connection().get_engine()

    df_with_findings = get_annotations_and_findings(engine, revision,
                                                    exam_types=['MG'], viewpoints=None,  # ['CC','MLO'], \
                                                    include_findings=True, remove_invalids=True,
                                                    remove_heldout=False, \
                                                    remove_excluded=False, remove_less_than_4views=False, \
                                                    load_from_file=False, save_to_file=False)

    my_providers = ['sentara']#extand dataset by adding  'baptist', 'froedtert', 'miami', 'ucsd'
    df = df_with_findings.loc[df_with_findings['provider'].isin(my_providers)]
    # fixing assymetry
    asymmetries = ['asymmetry', 'developing asymmetry', 'focal asymmetry', 'global asymmetry']
    df['is_asymmetry'] = df['pathology'].isin(asymmetries)
    df['is_Breast_Assymetry'] = df['type'].isin(['Breast Assymetry'])
    df.loc[df['is_asymmetry'], 'pathology'] = df[df['is_asymmetry']]['biopsy_outcome']
    df.loc[df['is_Breast_Assymetry'], 'pathology'] = df[df['is_Breast_Assymetry']]['biopsy_outcome']
    # remove duble xmls
    aa_unsorted = df
    aa_unsorted.sort_values('xml_url', ascending=False, inplace=True)
    xml_url_to_keep = aa_unsorted.groupby(['image_id'])['xml_url'].transform('first')
    df = aa_unsorted[aa_unsorted['xml_url'] == xml_url_to_keep]
    remove_from_pathology = ['undefined', 'not_applicable', 'Undefined', 'extracapsular rupture of breast implant',
                             'intracapsular rupture of breast implant']
    is_pathology = ~df.pathology.isnull() & ~df.pathology.isin(remove_from_pathology)
    is_digital = df.image_source == 'Digital'
    is_biopsy = df.finding_biopsy.isin(['negative', 'negative high risk', 'positive'])
    df = df[(is_digital) & (is_pathology) & (is_biopsy)]
    df.to_csv(imaging_filename)

def encode_one_hot(original_dataframe, features_to_encode):
    for features in features_to_encode:
        original_dataframe[features] = original_dataframe[features].astype('category', copy=False)
    dummies = pd.get_dummies(data=original_dataframe[features_to_encode])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop(features_to_encode, axis=1)
    return res


if __name__ == '__main__':
    clinical_data_path = '/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/my_research/virtual_biopsy/experiments/biopsy-mal_benign_plus_clinical/'
    table_data_files = ['demographics_and_breast_density.csv',
                        'dicom_tags_extracted.csv',
                        'max_prev_birad_class.csv'
                        ]
    is_table_data_files = ['distortions_20210317.csv',
                           'spiculations_20210317.csv',
                           'susp_calcifications_20210317.csv',
                           ]
    path = clinical_data_path

    REVISION_DATE = '20210317'
    TableResolver().set_revision(REVISION_DATE)
    revision = {'prefix': '', 'suffix': REVISION_DATE}
    engine = Connection().get_engine()

    df_with_findings = get_annotations_and_findings(engine, revision,
                                                    exam_types=None, viewpoints=None,  # ['CC','MLO'],
                                                    include_findings=True, remove_invalids=True,
                                                    remove_heldout=False, \
                                                    remove_excluded=False, remove_less_than_4views=False, \
                                                    load_from_file=False, save_to_file=False)

    # negative 5011 positive 3464,negative high risk 907 - finding biopsy
    # negative 4906 positive 3561,negative high risk 915 - biopsy
    cured_df = mg_data_curation(df_with_findings)

    # add dicom tags
    dicom_tags_extracted_table = pd.read_csv(clinical_data_path + table_data_files[1])
    cured_df_w_dicoms = pd.merge(cured_df, dicom_tags_extracted_table, how='inner',
                                 left_on=['xml_url'], right_on=['xml_url'], suffixes=('', '_'))

    # add demographics tags
    demographics_and_breast_density_table = pd.read_csv(clinical_data_path + table_data_files[0])
    demographics_and_breast_density_table_cured = \
        demographics_and_breast_density_table[
            demographics_and_breast_density_table['study_id'].isin(cured_df_w_dicoms['study_id'])]
    cured_df_w_dicoms_w_demographic = pd.merge(cured_df_w_dicoms, demographics_and_breast_density_table_cured,
                                               how='inner',
                                               left_on=['provider', 'patient_id', 'study_id', 'breast_density'],
                                               right_on=['provider', 'patient_id', 'study_id', 'breast_density'])

    # add max_prev_birad_class
    max_prev_birad_class_table = pd.read_csv(clinical_data_path + table_data_files[2])
    max_prev_birad_class_table_cured = \
        max_prev_birad_class_table[
            max_prev_birad_class_table['study_id'].isin(cured_df_w_dicoms['study_id'])]
    cured_df_w_dicoms_w_demographic_w_birad = pd.merge(cured_df_w_dicoms_w_demographic,
                                                       max_prev_birad_class_table_cured, how='inner',
                                                       left_on=['provider', 'patient_id', 'study_id'],
                                                       right_on=['provider', 'patient_id', 'study_id'])

    # add features from report
    is_distortions_table = pd.read_csv(clinical_data_path + is_table_data_files[0])
    cured_df_w_dicoms_w_demographic_w_birad['is_distortions'] = 0
    cured_df_w_dicoms_w_demographic_w_birad['is_distortions'][
        (cured_df_w_dicoms_w_demographic_w_birad['provider'].isin(is_distortions_table['provider'])) &
        (cured_df_w_dicoms_w_demographic_w_birad['patient_id'].isin(is_distortions_table['patient_id'])) &
        (cured_df_w_dicoms_w_demographic_w_birad['study_id'].isin(is_distortions_table['study_id']))] = 1

    is_spiculations_table = pd.read_csv(clinical_data_path + is_table_data_files[1])
    cured_df_w_dicoms_w_demographic_w_birad['is_spiculations'] = 0
    cured_df_w_dicoms_w_demographic_w_birad['is_spiculations'][
        (cured_df_w_dicoms_w_demographic_w_birad['provider'].isin(is_spiculations_table['provider'])) &
        (cured_df_w_dicoms_w_demographic_w_birad['patient_id'].isin(is_spiculations_table['patient_id'])) &
        (cured_df_w_dicoms_w_demographic_w_birad['study_id'].isin(is_spiculations_table['study_id']))] = 1

    is_susp_calcifications_table = pd.read_csv(clinical_data_path + is_table_data_files[2])
    cured_df_w_dicoms_w_demographic_w_birad['is_susp_calcifications'] = 0
    cured_df_w_dicoms_w_demographic_w_birad['is_susp_calcifications'][
        (cured_df_w_dicoms_w_demographic_w_birad['provider'].isin(is_susp_calcifications_table['provider'])) &
        (cured_df_w_dicoms_w_demographic_w_birad['patient_id'].isin(is_susp_calcifications_table['patient_id'])) &
        (cured_df_w_dicoms_w_demographic_w_birad['study_id'].isin(is_susp_calcifications_table['study_id']))] = 1

    cured_df_w_dicoms_w_demographic_w_birad.to_csv(path + 'curated_set_full_table_v2.csv')
    cured_df_w_dicoms_w_demographic_w_birad = cured_df_w_dicoms_w_demographic_w_birad.drop(
        cured_df_w_dicoms_w_demographic_w_birad[cured_df_w_dicoms_w_demographic_w_birad['contour'] == '{}'].index)
    scanned_images = [
        '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Sentara/382590/20100802_MM10071593/MG/8_bit/metadata/annotations/1.2.392.200036.9125.4.0.185594307.2225150464.3063965447_8bit.xml',
        '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Sentara/382590/20100802_MM10071593/MG/8_bit/metadata/annotations/1.2.392.200036.9125.4.0.185594307.2359368192.3063965447_8bit.xml',
        '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Sentara/40052219/20130327_MM13037621/MG/8_bit/metadata/annotations/1.2.392.200036.9125.4.0.269456599.1298734592.3360623261_8bit.xml',
        '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Sentara/40052219/20130327_MM13037621/MG/8_bit/metadata/annotations/1.2.392.200036.9125.4.0.269456599.1365843456.3360623261_8bit.xml',
        '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Sentara/40113784/20090420_MM09036951/MG/8_bit/metadata/annotations/1.2.392.200036.9125.4.0.152018509.307699200.3360627099_8bit.xml',
        '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Sentara/40113784/20090420_MM09036951/MG/8_bit/metadata/annotations/1.2.392.200036.9125.4.0.152018509.374808064.3360627099_8bit.xml',
        '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Sentara/40124614/20121210_MM12151081/MG/8_bit/metadata/annotations/1.2.392.200036.9125.4.0.269446726.400694784.3360627099_8bit.xml',
        '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Sentara/40180058/20130219_MM13021798/MG/8_bit/metadata/annotations/1.2.392.200036.9125.4.0.269428388.2811436544.3360623261_8bit.xml',
        '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Sentara/40180058/20130219_MM13021798/MG/8_bit/metadata/annotations/1.2.392.200036.9125.4.0.269428900.2903318016.3360623261_8bit.xml',
        '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Sentara/50306630/20140206_MG140206004350/MG/8_bit/metadata/annotations/1.2.392.200036.9125.4.0.303013245.132455936.3360627099_8bit.xml',
        '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Sentara/50306630/20140206_MG140206004350/MG/8_bit/metadata/annotations/1.2.392.200036.9125.4.0.303013245.4226096640.3360627099_8bit.xml',
        '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Sentara/59991270/20100514_MM10044896/MG/8_bit/metadata/annotations/1.2.392.200036.9125.4.0.185536601.641998336.3360627099_8bit.xml',
        '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Sentara/59991270/20100514_MM10044896/MG/8_bit/metadata/annotations/1.2.392.200036.9125.4.0.185536601.776216064.3360627099_8bit.xml',
        '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Sentara/9985667/20090701_MM09062371/MG/8_bit/metadata/annotations/1.2.392.200036.9125.4.0.151978415.1949178112.179941807_8bit.xml',
        '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Sentara/9985667/20090701_MM09062371/MG/8_bit/metadata/annotations/1.2.392.200036.9125.4.0.151978415.2150570240.179941807_8bit.xml']
    cured_df_w_dicoms_w_demographic_w_birad = cured_df_w_dicoms_w_demographic_w_birad[
        ~cured_df_w_dicoms_w_demographic_w_birad['xml_url'].isin(scanned_images)]
    cured_df_w_dicoms_w_demographic_w_birad.to_csv(path + 'curated_set_full_table_v2_filtered.csv')
    # Load csv file after manually fixing outlier row
    df = pd.read_csv(path + 'curated_set_full_table_v2_filtered_fixed.csv')
    # Fixing missing values
    df['body_part_thickness'][
        df.xml_url == '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Baptist/FFF452F4CAC130D7556371C70B3459A4/20110206_73/MG/8_bit/metadata/annotations/FO-8874352202443175339_8bit_mn.xml'] = 183
    df['body_part_thickness'][
        df.xml_url == '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Baptist/6D3511BCDC25A14CD5F37D8BAE2B3F47/20111218_117/MG/8_bit/metadata/annotations/FO-3161254379748920352_8bit_mn.xml'] = 181
    df['body_part_thickness'][
        df.xml_url == '/gpfs/haifa/projects/m/msieve/MedicalSieve/PatientData/Baptist/225742C5EAC7484DCF4E59BDFA7D3F97/20140812_840/MG/8_bit/metadata/annotations/FO-6768169063846331327_8bit_mn.xml'] = 180
    df['breast_density'][df.breast_density == 'undefined'] = df['breast_density'].mode()[0]
    df['birad'][(df.birad == 'not_applicable') | (df.birad == 'undefined')] = df['final_side_birad'][
        (df.birad == 'not_applicable') | (df.birad == 'undefined')]
    df['age'][df.age == 0] = df['age'].mode()[0]
    max_prev_birad_class_lst = [0, 1, 2, 3]
    df['max_prev_birad_class'][~df.max_prev_birad_class.isin(max_prev_birad_class_lst)] = df['max_prev_birad_class'].mode()[0]
    race_lst = ['african american', 'amer indian/alaskan', 'American Indian', 'Asian', 'Black', 'caucasian', 'hispanic', 'other', 'Pacific Islander', 'unknown', 'White']
    df['race'][~df.race.isin(race_lst)] = 'unknown'
    longitudinal_change_lst = ['longitudinal_change', 'increase', 'new appearance', 'not_applicable', 'stable']
    df['longitudinal_change'][~df.longitudinal_change.isin(longitudinal_change_lst)] = 'unknown'

    df[['findings_x_max', 'findings_y_max']].multiply(df['x_pixel_spacing'], axis="index")
    df['findings_size'].multiply(df['x_pixel_spacing'].pow(2), axis="index")

    id_lst = ['patient_id', 'xml_url']
    clinical_features_lst = ['breast_density', 'final_side_birad', 'side', 'birad', 'calcification', 'findings_size',
                             'findings_x_max',
                             'findings_y_max', 'longitudinal_change', 'type', 'DistanceSourceToPatient',
                             'DistanceSourceToDetector', 'x_pixel_spacing',
                             'XRayTubeCurrent', 'CompressionForce', 'exposure_time', 'KVP', 'body_part_thickness',
                             'RelativeXRayExposure', 'exposure_in_mas',
                             'age', 'race', 'max_prev_birad_class', 'is_distortions', 'is_spiculations',
                             'is_susp_calcifications', 'biopsy']
    dataset = df[id_lst + clinical_features_lst]
    dataset.to_csv(path + 'dataset.csv')

    # Convert biopsy into 0/1 label
    dataset['biopsy'][dataset.biopsy == 'positive'] = 1
    dataset['biopsy'][(dataset.biopsy == 'negative') | (dataset.biopsy == 'negative high risk')] = 0

    dataset = dataset.drop_duplicates(subset=['patient_id'])
    dataset.to_csv(path + 'dataset_unique.csv')

    #Convert categorical data into numerical
    dataset['final_side_birad'] = dataset['final_side_birad'].astype('category')
    dataset['side'] = dataset['side'].astype('category')
    dataset['birad'] = dataset['birad'].astype('category')
    dataset['calcification'] = dataset['calcification'].astype('category')
    dataset['longitudinal_change'] = dataset['longitudinal_change'].astype('category')
    dataset['type'] = dataset['type'].astype('category')
    dataset['race'] = dataset['race'].astype('category')
    cat_columns = dataset.select_dtypes(['category']).columns
    dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
    dataset.to_csv(path + 'dataset_numerical_unique.csv')
    dataset['max_prev_birad_class'] = dataset['max_prev_birad_class'].astype(int)
    features_to_encode = ['breast_density', 'final_side_birad', 'birad',
                          'calcification', 'longitudinal_change', 'type', 'race', 'max_prev_birad_class']
    dataset = encode_one_hot(dataset, features_to_encode)
    dataset = dataset.rename(columns={'patient_id': "sample_desc"})

    FOLDS_NUMBER = 6
    X = dataset['sample_desc'].values
    y = np.zeros(X.shape)
    y[dataset['biopsy'].values > 0] = 1
    kfold = StratifiedKFold(n_splits=FOLDS_NUMBER, shuffle=True, random_state=1)
    # enumerate the splits and summarize the distributions
    db = {}
    f = 0
    for train_ix, test_ix in kfold.split(X, y):
        # select rows
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        # summarize train and test composition
        train_0, train_1 = len(train_y[train_y == 0]), len(train_y[train_y == 1])
        test_0, test_1 = len(test_y[test_y == 0]), len(test_y[test_y == 1])
        print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
        print(test_X)
        tt = dataset[dataset['sample_desc'].isin(test_X)]
        db['data_fold' + str(f)] = tt
        f += 1

    temp = db['data_fold0']
    for i in range(1,4):
        temp = temp.append(db['data_fold'+str(i)], ignore_index=True)
    train = temp
    validation = db['data_fold4']
    heldout = db['data_fold5']

    features_to_normalize = ['findings_size', 'findings_x_max', 'findings_y_max', 'DistanceSourceToPatient',
                             'DistanceSourceToDetector', 'x_pixel_spacing', 'XRayTubeCurrent', 'CompressionForce',
                             'exposure_time', 'KVP', 'body_part_thickness', 'RelativeXRayExposure', 'exposure_in_mas',
                             'age']

    for feature in features_to_normalize:
        train[feature] = (train[feature] - train[feature].mean()) / train[feature].std()
        validation[feature] = (validation[feature] - validation[feature].mean()) / validation[feature].std()
        heldout[feature] = (heldout[feature] - heldout[feature].mean()) /heldout[feature].std()

    with open(path + 'dataset_MG_clinical_train' + '.pickle','wb') as handle:
        pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path + 'dataset_MG_clinical_train_xml.txt', 'w') as f:
        f.write(train['xml_url'].str.cat(sep='\n'))
    train.to_csv(path + 'dataset_MG_clinical_train.csv')
    with open(path + 'dataset_MG_clinical_validation' + '.pickle', 'wb') as handle:
        pickle.dump(validation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path + 'dataset_MG_clinical_validation_xml.txt', 'w') as f:
        f.write(validation['xml_url'].str.cat(sep='\n'))
    validation.to_csv(path + 'dataset_MG_clinical_validation.csv')
    with open(path + 'dataset_MG_clinical_heldout' + '.pickle', 'wb') as handle:
        pickle.dump(heldout, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path + 'dataset_MG_clinical_heldout_xml.txt', 'w') as f:
        f.write(heldout['xml_url'].str.cat(sep='\n'))
    heldout.to_csv(path + 'dataset_MG_clinical_heldout.csv')



