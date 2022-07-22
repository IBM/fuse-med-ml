import os
import numpy as np

import fuseimg.datasets.duke_label_type

import getpass
from fuseimg.datasets import duke
import logging
from fuse.utils.utils_logger import fuse_logger_start
from fuse.data.utils.split import dataset_balanced_division_to_folds

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def main():
    mode = "debug"  # 'default'  #  'default'  # Options: 'default', 'fast', 'debug', 'verbose', 'user'. See details in FuseDebug

    PATHS, TRAIN_COMMON_PARAMS, INFER_COMMON_PARAMS, EVAL_COMMON_PARAMS = get_setting(mode)
    print(PATHS)

    RUNNING_MODES = ["train", "infer", "eval"]  # Options: 'train', 'infer', 'eval'
    # train
    if "train" in RUNNING_MODES:
        print(TRAIN_COMMON_PARAMS)
        run_train(paths=PATHS, train_params=TRAIN_COMMON_PARAMS)

    # infer
    if 'infer' in RUNNING_MODES:
        print(INFER_COMMON_PARAMS)
        # run_infer(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS)

    # eval
    if 'eval' in RUNNING_MODES:
        print(EVAL_COMMON_PARAMS)
        # run_eval(paths=PATHS, eval_common_params=EVAL_COMMON_PARAMS)


def get_setting(
    mode, label_type=fuseimg.datasets.duke_label_type.DukeLabelType.STAGING_TUMOR_SIZE, n_folds=5, heldout_fold=4
):
    ###########################################################################################################
    # Fuse
    ###########################################################################################################
    ##########################################
    # Debug modes
    ##########################################

    # debug = FuseDebug(mode)

    ##########################################
    # Output Paths
    ##########################################
    assert (
        "DUKE_DATA_PATH" in os.environ
    ), "Expecting environment variable DUKE_DATA_PATH to be set. Follow the instruction in example README file to download and set the path to the data"
    ROOT = f"/projects/msieve_dev3/usr/{getpass.getuser()}/fuse_examples/duke_radiomics"
    model_dir = os.path.join(ROOT, "model_dir")

    if mode == "debug":
        num_workers = 16  # 0
        selected_sample_ids = duke.get_samples_for_debug(n_pos=10, n_neg=10, label_type=label_type)
        cache_dir = os.path.join(ROOT, "cache_dir_debug")
        data_split_file = os.path.join(ROOT, "DUKE_radiomics_folds_debug.pkl")

    else:
        num_workers = 16
        selected_sample_ids = None
        cache_dir = os.path.join(ROOT, "cache_dir")
        data_split_file = os.path.join(ROOT, "DUKE_radiomics_folds.pkl")

    cache_dir = os.path.join(ROOT, cache_dir)

    PATHS = {
        "model_dir": model_dir,
        "force_reset_model_dir": True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
        "cache_dir": cache_dir,
        "data_split_filename": os.path.join(ROOT, data_split_file),
        "data_dir": os.environ["DUKE_DATA_PATH"],
        "inference_dir": os.path.join(model_dir, "infer_dir"),
        "eval_dir": os.path.join(model_dir, "eval_dir"),
    }

    ##########################################
    # Train Common Params
    ##########################################
    TRAIN_COMMON_PARAMS = {}
    # ============
    # Model
    # ============

    # ============
    # Data
    # ============

    train_folds = [i % n_folds for i in range(heldout_fold + 1, heldout_fold + n_folds - 1)]
    validation_fold = (heldout_fold - 1) % n_folds
    TRAIN_COMMON_PARAMS["data.selected_sample_ids"] = selected_sample_ids
    TRAIN_COMMON_PARAMS["data.num_folds"] = n_folds
    TRAIN_COMMON_PARAMS["data.train_folds"] = train_folds
    TRAIN_COMMON_PARAMS["data.validation_folds"] = [validation_fold]
    TRAIN_COMMON_PARAMS["data.train_num_workers"] = num_workers

    def get_selected_series_index_radiomics(sample_id, seq_id):
        patient_id = sample_id[0]
        if patient_id in ["Breast_MRI_120", "Breast_MRI_596"]:
            map = {"DCE_mix": [1, 2], "MASK": [0]}
        else:
            map = {"DCE_mix": [0, 1], "MASK": [0]}
        return map[seq_id]

    TRAIN_COMMON_PARAMS["data.get_selectedseries_index_func"] = get_selected_series_index_radiomics

    TRAIN_COMMON_PARAMS["radiomics_extractor_setting"] = get_radiomics_extractor_setting2()

    # classification_task:
    # supported tasks are: 'Staging Tumor Size','Histology Type','is High Tumor Grade Total','PCR'
    TRAIN_COMMON_PARAMS["classification_task"] = label_type
    TRAIN_COMMON_PARAMS["models"] = [("lr", LogisticRegression()), ("rf", RandomForestClassifier())]

    ######################################
    # Inference Common Params
    ######################################
    INFER_COMMON_PARAMS = {}
    INFER_COMMON_PARAMS["infer_filename"] = "radiomics_validation_set_infer.gz"
    INFER_COMMON_PARAMS["data.infer_folds"] = [heldout_fold]  # infer validation set
    INFER_COMMON_PARAMS["classification_task"] = TRAIN_COMMON_PARAMS["classification_task"]

    ######################################
    # Analyze Common Params
    ######################################
    EVAL_COMMON_PARAMS = {}
    EVAL_COMMON_PARAMS["infer_filename"] = INFER_COMMON_PARAMS["infer_filename"]

    return PATHS, TRAIN_COMMON_PARAMS, INFER_COMMON_PARAMS, EVAL_COMMON_PARAMS


def run_train(paths: dict, train_params: dict, reset_cache=False):
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")
    lgr.info("Fuse Train", {"attrs": ["bold", "underline"]})

    lgr.info(f'model_dir={paths["model_dir"]}', {"color": "magenta"})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {"color": "magenta"})

    # ==============================================================================
    # Data
    # ==============================================================================
    # Train Data
    lgr.info("Train Data:", {"attrs": "bold"})

    # split to folds randomly - temp
    params = dict(
        label_type=train_params["classification_task"],
        data_dir=paths["data_dir"],
        cache_dir=paths["cache_dir"],
        reset_cache=False,
        sample_ids=train_params["data.selected_sample_ids"],
        num_workers=train_params["data.train_num_workers"],
        radiomics_extractor_setting=train_params["radiomics_extractor_setting"],
        select_series_func=train_params["data.get_selectedseries_index_func"],
        cache_kwargs=dict(audit_first_sample=False, audit_rate=None),  # None
    )
    dataset_all = duke.DukeRadiomics.dataset(**params)
    folds = dataset_balanced_division_to_folds(
        dataset=dataset_all,
        output_split_filename=paths["data_split_filename"],
        keys_to_balance=["data.ground_truth"],
        nfolds=train_params["data.num_folds"],
    )

    print("---------------")
    train_sample_ids = []
    for fold in train_params["data.train_folds"]:
        train_sample_ids += folds[fold]
    validation_sample_ids = []
    for fold in train_params["data.validation_folds"]:
        validation_sample_ids += folds[fold]

    params["sample_ids"] = train_sample_ids
    train_dataset = duke.DukeRadiomics.dataset(**params)

    params["sample_ids"] = validation_sample_ids
    # validation_dataset = duke.DukeRadiomics.dataset(**params)

    # ==============================================================================
    # Model
    # ==============================================================================

    # model1 = RandomForestClassifier()
    # model2 = LogisticRegression()
    X_train, xnames, y_train = get_X_y(train_dataset)
    print(X_train.shape, len(xnames), y_train.shape)
    print("#nans=", np.isnan(X_train).sum().sum())
    #
    # X_val, y_val = get_X_y(validation_dataset)

    lgr.info("Train: Done", {"attrs": "bold"})


def get_radiomics_extractor_setting(norm_method="default"):
    setting = {}

    setting["seq_vec"] = ["DCE"]
    setting["seq_inx_list"] = [0]
    setting["maskType"] = "full"  # alternatives: 'edge'
    setting["norm_method"] = norm_method  # alternative: 'tumor_area', 'breast_area'

    if norm_method == "default":
        setting["normalize"] = True
        setting["normalizeScale"] = 100
    else:
        setting["normalize"] = False

    setting["binWidth"] = 5
    setting["preCrop"] = True
    setting["applyLog"] = False
    setting["applyWavelet"] = False


def get_radiomics_extractor_setting2(norm_method="default"):
    setting = {}
    setting["seq_list"] = ["DCE0", "DCE1"]
    setting["seq_inx_list"] = [0, 1]
    setting["norm_method"] = norm_method
    setting["maskType"] = "full"

    if norm_method == "default":
        setting["normalize"] = True
        setting["normalizeScale"] = 100
    else:
        setting["normalize"] = False

    setting["binWidth"] = 5
    setting["preCrop"] = True
    setting["applyLog"] = False
    setting["applyWavelet"] = False
    ###

    return setting


def get_X_y(a_dataset):
    feature_keys = fnames = None
    y_list = []
    X_list = []
    for i, d in enumerate(a_dataset):
        if i == 0:
            feature_keys = sorted([s for s in d.flatten() if s.startswith("data.radiomics")])
            fnames = [s.replace("data.radiomics.", "") for s in feature_keys]
            is_ndarray = [isinstance(d[k], np.ndarray) and len(d[k].shape) > 0 for k in feature_keys]

        y = d["data.ground_truth"]

        X = np.asarray([d[k][0] if flag else float(d[k]) for k, flag in zip(feature_keys, is_ndarray)])
        y_list.append(y)
        X_list.append(X)

    y = np.asarray(y_list)
    X = np.asarray(X_list)
    print(y.shape, X.shape)
    return X, fnames, y


if __name__ == "__main__":
    main()
