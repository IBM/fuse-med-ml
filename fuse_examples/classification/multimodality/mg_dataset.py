import pandas as pd
import os
from typing import Callable, Optional
from typing import Tuple

from MedicalAnalyticsCore.DatabaseUtils.selected_studies_queries import get_annotations_and_findings
from MedicalAnalyticsCore.DatabaseUtils.tableResolver import TableResolver
from MedicalAnalyticsCore.DatabaseUtils.connection import create_homer_engine, Connection
from MedicalAnalyticsCore.DatabaseUtils import tableNames
from MedicalAnalyticsCore.DatabaseUtils import db_utils as db


# from autogluon.tabular import TabularPredictor
from fuse_examples.classification.multimodality.dataset import imaging_tabular_dataset
from fuse.data.dataset.dataset_default import FuseDatasetDefault

from fuse_examples.classification.cmmd.input_processor import FuseMGInputProcessor
from fuse.data.processor.processor_dataframe import FuseProcessorDataFrame


from typing import Dict, List
import torch
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict

class PostProcessing:
    def __init__(self, continuous_tabular_features_lst: List, categorical_tabular_features_lst: List, label_lst: List,
                 imaging_features_lst: List, non_imaging_features_lst: List, use_imaging: bool, use_non_imaging: bool):
        self.continuous_tabular_features_lst = continuous_tabular_features_lst
        self.categorical_tabular_features_lst = categorical_tabular_features_lst
        self.label_lst = label_lst
        self.imaging_features_lst = imaging_features_lst
        self.non_imaging_features_lst = non_imaging_features_lst
        self.use_imaging = use_imaging
        self.use_non_imaging = use_non_imaging

    def __call__(self, batch_dict: Dict) -> Dict:
        if not self.use_imaging and not self.use_non_imaging:
            raise ValueError('No features are in use')
        mask_list = self.use_imaging * self.imaging_features_lst + self.use_non_imaging * self.non_imaging_features_lst
        mask_continuous = torch.zeros(len( self.continuous_tabular_features_lst))
        for i in range(len(mask_list)):
            try:
                mask_continuous[self.continuous_tabular_features_lst.index(mask_list[i])] = 1
            except:
                pass
        mask_categorical = torch.zeros(len( self.categorical_tabular_features_lst))
        for i in range(len(mask_list)):
            try:
                mask_categorical[self.categorical_tabular_features_lst.index(mask_list[i])] = 1
            except:
                pass
        categorical = [FuseUtilsHierarchicalDict.get(batch_dict, 'data.' + feature_name) for feature_name in self.categorical_tabular_features_lst]
        for i in range(len(categorical)):
            if categorical[i].dim() == 0:
                categorical[i] = torch.unsqueeze(categorical[i], 0)
        categorical_tensor = torch.cat(tuple(categorical), 0)
        categorical_tensor = categorical_tensor.float()
        categorical_tensor = torch.mul(categorical_tensor, mask_categorical)
        FuseUtilsHierarchicalDict.set(batch_dict, 'data.categorical', categorical_tensor.float())
        continuous = [FuseUtilsHierarchicalDict.get(batch_dict, 'data.' + feature_name) for feature_name in self.continuous_tabular_features_lst]
        for i in range(len(continuous)):
            if continuous[i].dim() == 0:
                continuous[i] = torch.unsqueeze(continuous[i], 0)
        continuous_tensor = torch.cat(tuple(continuous), 0)
        continuous_tensor = continuous_tensor.float()
        continuous_tensor = torch.mul(continuous_tensor, mask_continuous)
        FuseUtilsHierarchicalDict.set(batch_dict, 'data.continuous', continuous_tensor.float())
        label = FuseUtilsHierarchicalDict.get(batch_dict, 'data.' + self.label_lst[0])
        FuseUtilsHierarchicalDict.set(batch_dict, 'data.label', label.long())
        feature_lst = self.continuous_tabular_features_lst + self.categorical_tabular_features_lst + self.label_lst
        for feature in feature_lst:
            FuseUtilsHierarchicalDict.pop(batch_dict, 'data.' + feature)
        return batch_dict


# feature selection univarient analysis

def get_selected_features_clinical(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initialize logger
    logger = logging.getLogger("BigMed")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    handler = logging.FileHandler(join(output_path, "feature_selection_messages.log"))
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("\n============ Configuration ============\n")
    logger.info("{")
    for key, value in config.items():
        logger.info("%s: %s" % (key, value))
    logger.info("}\n")

    if 'column_names' in config:
        all_df = ClinicalData.read_file(config['filename'], columns_names=config['column_names']).clinical_df
    else:
        all_df = pd.read_csv(config['filename'])
    if 'selected_columns' in config:
        all_df = all_df[[sc.SUBJECT_ID] + config['selected_columns'] + [config['label']]]
    all_df = all_df.loc[~np.isnan(all_df[config['label']].values), :] # remove rows with missing label
    scaler = MinMaxScaler(feature_range=(0, 1))

    if 'folds_file' in config:
        folds_df = pd.read_csv(config['folds_file'], sep=',', header=0, index_col=0)
        unq = folds_df['fold'].unique()
        n_folds = len(unq)-1 if HELD_OUT_KEY in unq else len(unq)
        all_train_rows = folds_df[~folds_df['fold'].isin([HELD_OUT_KEY, -1])]
        all_train_subjects = all_train_rows['patient'].tolist()
        mask = all_df[sc.SUBJECT_ID].isin(all_train_subjects)
        all_train_df = all_df.loc[mask]
    else:
        n_folds = 1

    selected_features = {}
    for fold in range(n_folds):
        if n_folds > 1:
            train_rows = folds_df[~folds_df['fold'].isin([fold, HELD_OUT_KEY, -1])]
            train_subjects = train_rows['patient'].tolist()
        else:
            train_subjects = all_df[sc.SUBJECT_ID].tolist()
        mask = all_df[sc.SUBJECT_ID].isin(train_subjects)
        fold_train_df = all_df.loc[mask]
        fold_train_df = fold_train_df.drop([sc.SUBJECT_ID], axis=1)

        fold_train_df = fold_train_df.fillna(fold_train_df.mean())
        y = fold_train_df[config['label']].values
        fold_train_df = fold_train_df.drop([config['label']], axis=1)

        logger.info('============ Features selection by SelectFromModel for fold {} ============\n'.format(fold))
        for i in range(len(config['classifiers'])):
            cur_df = fold_train_df
            if config['apply_scaler'][i]:
                cur_df[cur_df.columns] = scaler.fit_transform(cur_df[cur_df.columns])
            X = cur_df.values
            cls_name = config['classifiers'][i]
            if cls_name == 'RandomForestClassifier':
                cls = RandomForestClassifier(**config['classifiers_params'][i])
            elif cls_name == 'LogisticRegression':
                cls = LogisticRegression(**config['classifiers_params'][i])
            elif cls_name == 'XGBClassifier':
                cls = XGBClassifier(**config['classifiers_params'][i])

            sfm = SelectFromModel(cls).fit(X, y).get_support()
            cur_features = cur_df.columns.values[sfm].tolist()
            if cls_name not in selected_features:
                selected_features[cls_name] = [cur_features]
            else:
                selected_features[cls_name] += [cur_features]
            logger.info('Out of {} features, found for fold {} the following {} features by SelectFromModel with {}:\n{}\n'.format(len(X[0]), fold, len(cur_features), cls_name, cur_features))

        logger.info('\n============ Features selection by UnivariateTest for fold {} ============\n'.format(fold))
        _, p = f_classif(fold_train_df.values, y)   # analyze variance for all features
        cat_features_filter = (fold_train_df.nunique() <= 5)   # find categorial features that have at most 5 values
        if cat_features_filter.any():
            _, p[cat_features_filter] = chi2(fold_train_df.values[:, cat_features_filter], y)  # analyze dependence for categorial features

        sig_columns = []
        for i in np.argsort(p):
            name = fold_train_df.columns[i]
            if p[i] < config['p_threshold']:
                sig_columns += [name]
        if 'UnivariateTest' not in selected_features:
            selected_features['UnivariateTest'] = [sig_columns]
        else:
            selected_features['UnivariateTest'] += [sig_columns]
        logger.info('Out of {} features, found for fold {} the following {} significant columns in descending order:\n{}\n\n'.format(len(X[0]), fold, len(sig_columns), sig_columns))

    logger.info('\n============ Features selection Intersection ============\n')
    for key in selected_features:
        intersect_features = set(selected_features[key][0]).intersection(*selected_features[key])
        logger.info('Out of {} features, found the following {} intersect features for {}:\n{}\n\n'.format(len(X[0]), len(intersect_features), key, list(intersect_features)))

    logger.info('\n============ Features selection by UnivariateTest on all Train data ============\n')
    all_train_df = all_train_df.fillna(all_train_df.mean())
    y = all_train_df[config['label']].values
    all_train_df = all_train_df.drop([sc.SUBJECT_ID, config['label']], axis=1)
    _, p = f_classif(all_train_df.values, y)   # analyze variance for all features
    cat_features_filter = (all_train_df.nunique() <= 5)   # find categorial features that have at most 5 values
    if cat_features_filter.any():
        _, p[cat_features_filter] = chi2(all_train_df.values[:, cat_features_filter], y)  # analyze dependence for categorial features
    sig_columns = []
    for i in np.argsort(p):
        name = all_train_df.columns[i]
        if p[i] < config['p_threshold']:
            sig_columns += [name]
    logger.info('Out of {} features, found the following {} significant columns in descending order:\n{}\n\n'.format(len(X[0]), len(sig_columns), sig_columns))

    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
#-------------------Tabular
def get_selected_features_mg(data,features_mode,key_columns):
    features_dict = tabular_feature_mg(data)
    columns_names = list(data.columns)
    if features_mode == 'full':
        selected_col = \
            features_dict['icd9_feat'] + \
            features_dict['labs_feat'] + \
            features_dict['hrt_feat'] + \
            features_dict['fam_feats'] + \
            features_dict['biopsy_feat'] + \
            features_dict['smoking_feat'] + \
            features_dict['demo_feat'] + \
            features_dict['sympt_feat'] + \
            features_dict['meds_feat'] + \
            features_dict['prev_finding_feat'] + \
            features_dict['gynec_feat'] + \
            features_dict['genetic_feat'] + \
            features_dict['dicom']
    elif features_mode == 'icd9_feat':
        selected_col = \
            features_dict['icd9_feat']
    elif features_mode == 'labs_feat':
        selected_col = \
            features_dict['labs_feat']
    selected_col = selected_col + key_columns
    selected_colIx = [columns_names.index(selected_col[i]) for i in range(len(selected_col))]
    return selected_col,selected_colIx


def tabular_feature_mg(data):
    features_dict = {}
    features_dict['icd9_feat'] = [x for x in data if x.startswith('dx_')]
    features_dict['labs_feat'] = [x for x in data if x.startswith('labs')]
    features_dict['hrt_feat'] = [x for x in data if x.startswith('HRT')]
    features_dict['outcome_feat'] = [x for x in data if x.startswith('outcome')]
    features_dict['fam_feats'] = [x for x in data if x.startswith('family')]
    features_dict['biopsy_feat'] = ['prev_biopsy_result_max', 'past_biopsy_proc_ind', 'past_biopsy_proc_cnt']
    features_dict['smoking_feat'] = [x for x in data if x.startswith('smoking')]
    features_dict['radio_feat'] = [x for x in data if 'birads' in x] + [x for x in data if 'breast_density' in x] +\
                    [x for x in data if 'breast_MRI' in x]
    features_dict['demo_feat'] = [ 'age', 'race', 'religion', 'bmi_max', 'bmi_last', 'weight_max', 'weight_last',
                 'osteoporosis_ind', 'bmi_current', 'diabetes_ind']#, 'calc_bmi_current', 'calc_likelihood_obesity']
    features_dict['sympt_feat'] = [  'pain_cnt', 'nipple_retraction_ind_past',
                  'lump_by_dr_ind_past', 'nipple_retraction_ind_current',
                  'infection_current_ind', 'lump_by_dr_cnt', 'nipple_retraction_cnt',
                   'lump_by_dr_ind_current',  'breast_disorder_ind',
                  'breast_disorder_current_ind', 'nipple_allocation_ind_current', 'nipple_allocation_cnt',
                 'nipple_allocation_ind_past',  'complaint_ind_current', 'complaint_ind_past',
                 'pain_ind_past', 'pain_ind_current', 'infection_current_ind_last']
    features_dict['meds_feat'] = [ 'oral_contraceptives_ind_current', 'progesterons_ind', 'oral_contraceptives_ind_past']
    features_dict['gynec_feat'] = ['has_breastfed_ind', 'children_ind', 'children_cnt', 'age_last_menstruation', 'menopause_ind',
                 'age_first_childbirth', 'pregnancies_cnt', 'pregnancies_ind', 'age_first_menstruation',
                  'menstruation_years', 'menopause_dx_ind', 'menarche_to_ftp_years']
    features_dict['prev_finding_feat'] = ['prev_high_risk_ind', 'cancer_hist_any_ind', 'prev_benign_cnt', 'prev_benign_ind',
                        'prev_high_risk_cnt']
    features_dict['genetic_feat'] = ['genetic_consult_ind']
    features_dict['images'] = ['LCC_micro','RCC_micro','LMLO_micro','RMLO_micro',
     'LCC_pred_classA','LCC_pred_classB', 'LCC_pred_classC', 'LCC_pred_classD', 'LCC_pred_classE',
     'RCC_pred_classA', 'RCC_pred_classB','RCC_pred_classC', 'RCC_pred_classD','RCC_pred_classE',
     'LMLO_pred_classA', 'LMLO_pred_classB', 'LMLO_pred_classC', 'LMLO_pred_classD', 'LMLO_pred_classE',
     'RMLO_pred_classA', 'RMLO_pred_classB', 'RMLO_pred_classC', 'RMLO_pred_classD', 'RMLO_pred_classE',
     'LCC_findings_size', 'RCC_findings_size', 'LMLO_findings_size', 'RMLO_findings_size',
     'LCC_findings_x_max', 'RCC_findings_x_max', 'LMLO_findings_x_max', 'RMLO_findings_x_max',
     'LCC_findings_y_max', 'RCC_findings_y_max', 'LMLO_findings_y_max', 'RMLO_findings_y_max',
     'Calcification', 'Breast Assymetry', 'Tumor', 'Architectural Distortion', 'Axillary lymphadenopathy',
             'spiculated_lesions_report', 'architectural_distortion_report', 'suspicious_calcifications_report']
    features_dict['dicom'] = ['DistanceSourceToPatient_AVG_CC', 'DistanceSourceToDetector_AVG_CC',
           'XRayTubeCurrent_AVG_CC', 'CompressionForce_AVG_CC',
           'ExposureTime_AVG_CC', 'KVP_AVG_CC', 'BodyPartThickness_AVG_CC',
           'RelativeXRayExposure_AVG_CC', 'ExposureInuAs_AVG_CC',
           'DistanceSourceToPatient_AVG_MLO', 'DistanceSourceToDetector_AVG_MLO',
           'XRayTubeCurrent_AVG_MLO', 'CompressionForce_AVG_MLO',
           'ExposureTime_AVG_MLO', 'KVP_AVG_MLO', 'BodyPartThickness_AVG_MLO',
           'RelativeXRayExposure_AVG_MLO', 'ExposureInuAs_AVG_MLO']

    return features_dict


def tabular_mg(tabular_filename,key_columns):
    data = pd.read_csv(tabular_filename)
    column_names,column_colIx = get_selected_features_mg(data, 'full',key_columns)
    df_tabular = data[column_names]
    return df_tabular


#------------------Imaging
def imaging_mg(imaging_filename,key_columns):
    label_column = 'finding_biopsy'
    img_sample_column = 'dcm_url'

    if os.path.exists(imaging_filename):
        df = pd.read_csv(imaging_filename)
    else:
        REVISION_DATE = '20200915'
        TableResolver().set_revision(REVISION_DATE)
        revision = {'prefix': 'sentara', 'suffix': REVISION_DATE}
        engine = Connection().get_engine()

        df_with_findings = get_annotations_and_findings(engine, revision,
                                                        exam_types=['MG'], viewpoints=None,  # ['CC','MLO'], \
                                                        include_findings=True, remove_invalids=True,
                                                        remove_heldout=False, \
                                                        remove_excluded=False, remove_less_than_4views=False, \
                                                        load_from_file=False, save_to_file=False)

        # dicom_table = db.get_table_as_dataframe(engine, tableNames.get_dicom_tags_table_name(revision))
        # study_statuses = db.get_table_as_dataframe(engine, tableNames.get_study_statuses_table_name(revision))
        my_providers = ['sentara']
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

    df1 = df.groupby(key_columns)[img_sample_column].apply(lambda x:  list(map(str, x))).reset_index()
    df2 = df.groupby(key_columns)[label_column].apply(lambda x:  list(map(str, x))).reset_index()


    return pd.merge(df1,df2,on=key_columns)


#------------------Imaging+Tabular
def merge_datasets(tabular_filename,imaging_filename,key_columns):
    tabular_data = tabular_mg(tabular_filename, key_columns)
    imaging_data = imaging_mg(imaging_filename, key_columns)
    tabular_columns = tabular_data.columns.values
    imaging_columns = imaging_data.columns.values
    dataset = pd.merge(tabular_data, imaging_data, on=key_columns, how='inner')
    return dataset,tabular_columns,imaging_columns

#------------------Baseline
def apply_gluon_baseline(train_set,test_set,label,save_path):

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

def MG_dataset(tabular_filename:str,
               imaging_filename:str,
               train_val_test_filenames:list,

               imaging_processor,
               tabular_processor,

               key_columns:list,
               label_key:str,
               img_key:str,
               sample_key: str,

               cache_dir: str = 'cache',
               reset_cache: bool = False,
               post_cache_processing_func: Optional[Callable] = None) -> Tuple[FuseDatasetDefault, FuseDatasetDefault]:


    dataset, tabular_columns, imaging_columns = merge_datasets(tabular_filename, imaging_filename, key_columns)

    dataset['finding_biopsy'] = [1 if 'positive' in sample else 0 for sample in list(dataset[label_key])]
    dataset = dataset.loc[:, ~dataset.columns.duplicated()]
    dataset.rename(columns={'study_id': sample_key}, inplace=True)

    train_set = dataset[dataset[sample_key].isin(pd.read_csv(train_val_test_filenames[0])['study_id'])]
    val_set = dataset[dataset[sample_key].isin(pd.read_csv(train_val_test_filenames[1])['study_id'])]
    test_set = dataset[dataset[sample_key].isin(pd.read_csv(train_val_test_filenames[2])['study_id'])]

    features_list = list(tabular_columns)
    [features_list.remove(x) for x in key_columns]
    train_dataset, validation_dataset, test_dataset = imaging_tabular_dataset(
                                                                        df=[train_set, val_set, test_set],
                                                                        imaging_processor=imaging_processor,
                                                                        tabular_processor=tabular_processor,
                                                                        label_key=label_key,
                                                                        img_key=img_key,
                                                                        tabular_features_lst=features_list + [label_key] + [sample_key],
                                                                        sample_key=sample_key,
                                                                        cache_dir=cache_dir,
                                                                        reset_cache=reset_cache,
                                                                        post_cache_processing_func=post_cache_processing_func
                                                                        )

    return train_dataset, validation_dataset, test_dataset



if __name__ == "__main__":
    data_path = '/projects/msieve_dev3/usr/Tal/my_research/multi-modality/mg_clinical_dicom_sentra/'
    tabular_filename = os.path.join(data_path, 'fx_sentara_cohort_processed.csv')
    imaging_filename = os.path.join(data_path, 'mg_sentara_cohort.csv')

    train_val_test_filenames = [os.path.join(data_path, 'sentara_train_pathologies.csv'),
                                os.path.join(data_path, 'sentara_val_pathologies.csv'),
                                os.path.join(data_path, 'sentara_test_pathologies.csv'), ]

    key_columns = ['patient_id', 'study_id']
    fuse_key_column = 'sample_desc'
    label_column = 'finding_biopsy'
    img_sample_column = 'dcm_url'
    train_dataset, validation_dataset, test_dataset = \
                                                    MG_dataset(tabular_filename=tabular_filename,
                                                               imaging_filename=imaging_filename,
                                                               train_val_test_filenames=train_val_test_filenames,
                                                               key_columns=key_columns,
                                                               sample_key=fuse_key_column,
                                                               label_key=label_column,
                                                               img_key=img_sample_column,
                                                               cache_dir='./lala/',
                                                               reset_cache=False,
                                                               imaging_processor=FuseMGInputProcessor,
                                                               tabular_processor=FuseProcessorDataFrame,
                                                               )


    # apply_gluon_baseline(train_set[tabular_columns+[label_column]],
    #                      test_set[tabular_columns+[label_column]],label_column,'./Results/MG+clinical/gluon_baseline/')