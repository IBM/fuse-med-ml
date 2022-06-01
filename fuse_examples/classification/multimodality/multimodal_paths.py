import os
import pathlib
import pandas as pd
from fuse.utils.rand.seed import Seed

def multimodal_paths(dataset_name,root_data,root, experiment,cache_path):
    if dataset_name=='mg_clinical':
        paths = {
            # paths
            'data_dir': root_data,
            'tabular_filename': os.path.join(root_data, 'mg_clinical_dicom_sentra/fx_sentara_cohort_processed.csv'),
            'imaging_filename': os.path.join(root_data, 'mg_clinical_dicom_sentra/mg_sentara_cohort.csv'),
            'train_val_test_filenames': [os.path.join(root_data, 'mg_clinical_dicom_sentra/sentara_train_pathologies.csv'),
                                         os.path.join(root_data, 'mg_clinical_dicom_sentra/sentara_val_pathologies.csv'),
                                         os.path.join(root_data, 'mg_clinical_dicom_sentra/sentara_test_pathologies.csv'), ],

            # keys to extract from dataframe
            'key_columns': ['patient_id', 'study_id'],
            'sample_key': 'sample_desc',
            'label_key': 'finding_biopsy',
            'img_key': 'dcm_url',

            'model_dir': os.path.join(root, experiment, 'model_mg_clinical_dicom_sentra'),
            'force_reset_model_dir': True,
            # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
            'cache_dir': os.path.join(cache_path, '/lala/'),
            'inference_dir': os.path.join(root, experiment, 'infer_mg_clinical_dicom_sentra')}
    if dataset_name == 'mg_radiologic':
        paths = {
    # paths
    'data_dir': root_data,
    'tabular_filename': os.path.join(root_data, 'mg_radiologist_usa/dataset_MG_clinical.csv'),
    'imaging_filename': os.path.join(root_data, 'mg_radiologist_usa/mg_usa_cohort.csv'),
    'train_val_test_filenames': [os.path.join(root_data, 'mg_radiologist_usa/dataset_MG_clinical_train.csv'),
                                 os.path.join(root_data, 'mg_radiologist_usa/dataset_MG_clinical_validation.csv'),
                                 os.path.join(root_data, 'mg_radiologist_usa/dataset_MG_clinical_heldout.csv'), ],

    # keys to extract from dataframe
    'key_columns': ['patient_id'],
    'sample_key': 'sample_desc',
    'label_key': 'finding_biopsy',
    'img_key': 'dcm_url',

    'model_dir': os.path.join(root_data,'model_mg_radiologist_usa/'+experiment),
    'force_reset_model_dir': True,
    # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
    'cache_dir': os.path.join(cache_path),
    'inference_dir': os.path.join(root_data,'model_mg_radiologist_usa/'+experiment)}
    if dataset_name == 'knight':
        # read train/val splits file. for convenience, we use the one
        # auto-generated by the nnU-Net framework for the KiTS21 data
        dir_path = '/projects/msieve_dev2/usr/Tal/git_repos_multimodality/fuse-med-ml/fuse_examples/classification/knight/baseline'
        splits = pd.read_pickle(os.path.join(dir_path, 'splits_final.pkl'))
        # For this example, we use split 0 out of the 5 available cross validation splits
        split = splits[0]

        # set constant seed for reproducibility.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"  # required for pytorch deterministic mode
        rand_gen = Seed.set_seed(1234, deterministic_mode=False)

        paths = {
            # paths
            'data_dir': '/projects/msieve/MedicalSieve/PatientData/KNIGHT/',
            'split': split,
            'seed':rand_gen,
            'force_reset_model_dir': False,
            # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
            'cache_dir': os.path.join(cache_path),
            'rand_gen':rand_gen,
            'model_dir': os.path.join(root_data, 'knight/' + experiment),
            'inference_dir': os.path.join(root_data, 'knight/' + experiment)}

    return paths