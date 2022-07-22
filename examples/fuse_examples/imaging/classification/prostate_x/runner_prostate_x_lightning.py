"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""

import torch
import logging
import os
import copy
from typing import OrderedDict, Optional

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

import fuse.utils.gpu as GPU
from fuse_fuse_examples_utils import ask_user
from fuse_examples.imaging.utils.backbone_3d_multichannel import Fuse_model_3d_multichannel, ResNet
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.split import dataset_balanced_division_to_folds
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models.heads import Head1DClassifier
from fuse.eval.evaluator import EvaluatorDefault
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.utils.rand.seed import Seed
from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.file_io.file_io import create_dir, load_pickle, save_dataframe

from fuseimg.datasets import prostate_x

from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe
import pytorch_lightning as pl
from fuse_examples import fuse_examples_utils


def main():
    mode = "default"  # Options: 'default', 'fast', 'debug', 'verbose', 'user'. See details in FuseDebug

    # allocate gpus
    # To use cpu - set NUM_GPUS to 0
    if mode == "debug":
        NUM_GPUS = 1
    else:
        NUM_GPUS = 1
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    GPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    PATHS, TRAIN_COMMON_PARAMS, INFER_COMMON_PARAMS, EVAL_COMMON_PARAMS = get_setting(
        mode, num_devices=NUM_GPUS, n_folds=8, heldout_fold=4
    )
    print(PATHS)

    RUNNING_MODES = ["train", "infer", "eval"]  # Options: 'train', 'infer', 'eval'
    # train
    if "train" in RUNNING_MODES:
        print(TRAIN_COMMON_PARAMS)
        run_train(paths=PATHS, train_params=TRAIN_COMMON_PARAMS)

    # infer
    if "infer" in RUNNING_MODES:
        print(INFER_COMMON_PARAMS)
        run_infer(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS, audit_cache="train" not in RUNNING_MODES)

    # eval
    if "eval" in RUNNING_MODES:
        print(EVAL_COMMON_PARAMS)
        run_eval(paths=PATHS, eval_common_params=EVAL_COMMON_PARAMS)

    print(f"Done running with heldout={INFER_COMMON_PARAMS['data.infer_folds']}")


def get_setting(mode, num_devices, label_type=prostate_x.ProstateXLabelType.ClinSig, n_folds=8, heldout_fold=7):
    ###########################################################################################################
    # Fuse
    ###########################################################################################################
    ##########################################
    # Debug modes
    ##########################################

    input_channels_num = 5
    ##########################################
    # Output Paths
    ##########################################
    assert (
        "PROSTATEX_DATA_PATH" in os.environ
    ), "Expecting environment variable PROSTATEX_DATA_PATH to be set. Follow the instruction in example README file to download and set the path to the data"
    data_dir = os.environ["PROSTATEX_DATA_PATH"]
    ROOT = os.path.join(fuse_examples_utils.get_fuse_examples_user_dir(), "prostate_x")

    if mode == "debug":
        data_split_file = os.path.join(ROOT, f"prostate_x_{n_folds}folds_debug.pkl")
        selected_sample_ids = prostate_x.get_samples_for_debug(n_pos=10, n_neg=10, label_type=label_type)
        print(selected_sample_ids)
        cache_dir = os.path.join(ROOT, "cache_dir_debug")
        model_dir = os.path.join(ROOT, "model_dir_debug")
        num_workers = 0
        batch_size = 2
        num_epoch = 5
    else:
        data_split_file = os.path.join(ROOT, f"prostatex_{n_folds}folds.pkl")
        cache_dir = os.path.join(ROOT, "cache_dir_pl")
        model_dir = os.path.join(ROOT, f"model_dir_pl_{heldout_fold}")
        selected_sample_ids = None

        num_workers = 16
        batch_size = 50
        num_epoch = 50
    PATHS = {
        "model_dir": model_dir,
        "cache_dir": cache_dir,
        "data_split_filename": os.path.join(ROOT, data_split_file),
        "data_dir": data_dir,
        "inference_dir": os.path.join(model_dir, "infer_dir"),
        "eval_dir": os.path.join(model_dir, "eval_dir"),
    }

    ##########################################
    # Train Common Params
    ##########################################
    TRAIN_COMMON_PARAMS = {}

    # ============
    # Data
    # ============

    train_folds = [i % n_folds for i in range(heldout_fold + 1, heldout_fold + n_folds - 1)]
    validation_fold = (heldout_fold - 1) % n_folds
    TRAIN_COMMON_PARAMS["data.selected_sample_ids"] = selected_sample_ids
    TRAIN_COMMON_PARAMS["data.batch_size"] = batch_size
    TRAIN_COMMON_PARAMS["data.train_num_workers"] = num_workers
    TRAIN_COMMON_PARAMS["data.validation_num_workers"] = num_workers
    TRAIN_COMMON_PARAMS["data.num_folds"] = n_folds
    TRAIN_COMMON_PARAMS["data.train_folds"] = train_folds
    TRAIN_COMMON_PARAMS["data.validation_folds"] = [validation_fold]

    # ===============
    # PL Trainer
    # ===============
    TRAIN_COMMON_PARAMS["trainer.num_epochs"] = num_epoch
    TRAIN_COMMON_PARAMS["trainer.num_devices"] = num_devices
    TRAIN_COMMON_PARAMS["trainer.accelerator"] = "gpu"
    TRAIN_COMMON_PARAMS["trainer.ckpt_path"] = None  # path to the checkpoint you wish continue the training from

    # ===============
    # Optimizer
    # ===============
    TRAIN_COMMON_PARAMS["opt.lr"] = 1e-3
    TRAIN_COMMON_PARAMS["opt.weight_decay"] = 0.005

    # ===============
    # Manager - Train
    # ===============
    TRAIN_COMMON_PARAMS["manager.train_params"] = {
        "num_epochs": num_epoch,
        "virtual_batch_size": 1,  # number of batches in one virtual batch
        "start_saving_epochs": 10,  # first epoch to start saving checkpoints from
        "gap_between_saving_epochs": 5,  # number of epochs between saved checkpoint
    }
    TRAIN_COMMON_PARAMS["manager.best_epoch_source"] = {
        "source": "metrics.auc",  # can be any key from 'epoch_results'
        "optimization": "max",  # can be either min/max
        "on_equal_values": "better",
        # can be either better/worse - whether to consider best epoch when values are equal
    }
    TRAIN_COMMON_PARAMS["manager.learning_rate"] = 1e-5
    TRAIN_COMMON_PARAMS["manager.weight_decay"] = 1e-4
    TRAIN_COMMON_PARAMS["manager.dropout"] = 0.5
    TRAIN_COMMON_PARAMS["manager.momentum"] = 0.9
    TRAIN_COMMON_PARAMS["manager.resume_checkpoint_filename"] = None  # if not None, will try to load the checkpoint
    # TRAIN_COMMON_PARAMS['imaging_dropout'] = 0.25
    # # TRAIN_COMMON_PARAMS['fused_dropout'] = 0.0
    # # TRAIN_COMMON_PARAMS['clinical_dropout'] = 0.0

    TRAIN_COMMON_PARAMS["num_backbone_features_imaging"] = 512

    # in order to add relevant tabular feature uncomment:
    # num_backbone_features_clinical, post_concat_inputs,post_concat_model
    TRAIN_COMMON_PARAMS["num_backbone_features_clinical"] = None  # 256
    TRAIN_COMMON_PARAMS["post_concat_inputs"] = None  # [('data.clinical_features',9),]
    TRAIN_COMMON_PARAMS["post_concat_model"] = None  # (256,256)

    if TRAIN_COMMON_PARAMS["num_backbone_features_clinical"] is None:
        TRAIN_COMMON_PARAMS["num_backbone_features"] = TRAIN_COMMON_PARAMS["num_backbone_features_imaging"]
    else:
        TRAIN_COMMON_PARAMS["num_backbone_features"] = (
            TRAIN_COMMON_PARAMS["num_backbone_features_imaging"] + TRAIN_COMMON_PARAMS["num_backbone_features_clinical"]
        )

    # classification task:
    # supported tasks are: 'ClinSig'
    TRAIN_COMMON_PARAMS["label_type"] = label_type
    TRAIN_COMMON_PARAMS["class_num"] = label_type.get_num_classes()

    # backbone parameters
    TRAIN_COMMON_PARAMS["backbone_model_dict"] = {
        "input_channels_num": input_channels_num,
    }

    # ============
    # Model
    # ============

    TRAIN_COMMON_PARAMS["model"] = dict(
        imaging_dropout=0.25,
        # fused_dropout=0.0,
        # clinical_dropout=0.0,
        num_backbone_features=TRAIN_COMMON_PARAMS["num_backbone_features"],
        input_channels_num=5,
    )

    ######################################
    # Inference Common Params
    ######################################
    INFER_COMMON_PARAMS = {}
    INFER_COMMON_PARAMS["infer_filename"] = "infer_file.gz"
    INFER_COMMON_PARAMS["checkpoint"] = "best_epoch.ckpt"
    INFER_COMMON_PARAMS["data.infer_folds"] = [heldout_fold]  # infer validation set
    INFER_COMMON_PARAMS["data.batch_size"] = 4
    INFER_COMMON_PARAMS["data.num_workers"] = num_workers
    INFER_COMMON_PARAMS["label_type"] = TRAIN_COMMON_PARAMS["label_type"]
    INFER_COMMON_PARAMS["model"] = TRAIN_COMMON_PARAMS["model"]
    INFER_COMMON_PARAMS["trainer.num_devices"] = num_devices
    INFER_COMMON_PARAMS["trainer.accelerator"] = "gpu"

    ######################################
    # Analyze Common Params
    ######################################
    EVAL_COMMON_PARAMS = {}
    EVAL_COMMON_PARAMS["infer_filename"] = INFER_COMMON_PARAMS["infer_filename"]

    return PATHS, TRAIN_COMMON_PARAMS, INFER_COMMON_PARAMS, EVAL_COMMON_PARAMS


def create_model(imaging_dropout: float, num_backbone_features: int, input_channels_num: int) -> torch.nn.Module:
    """
    creates the model
    See Head3DClassifier for details about imaging_dropout, clinical_dropout, fused_dropout
    """
    conv_inputs = (("data.input.patch_volume", 1),)

    model = Fuse_model_3d_multichannel(
        conv_inputs=conv_inputs,  # previously 'data.input'. could be either 'data.input.patch_volume' or  'data.input.patch_volume_orig'
        backbone=ResNet(conv_inputs=conv_inputs, ch_num=input_channels_num),
        # since backbone resnet contains pooling and fc, the feature output is 1D,
        # hence we use Head1dClassifier as classification head
        heads=[
            Head1DClassifier(
                head_name="classification",
                conv_inputs=[("model.backbone_features", num_backbone_features)],
                post_concat_inputs=None,  # [('data.clinical_features',9),]
                post_concat_model=None,  # (256,256)
                dropout_rate=imaging_dropout,
                # append_dropout_rate=train_params['clinical_dropout'],
                # fused_dropout_rate=train_params['fused_dropout'],
                shared_classifier_head=None,
                layers_description=None,
                num_classes=2,
                # append_features=[("data.input.clinical", 8)],
                # append_layers_description=(256,128),
            ),
        ],
    )
    return model


#################################
# Train Template
#################################
def run_train(paths: dict, train_params: dict):
    Seed.set_seed(222, False)

    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")
    lgr.info("Fuse Train", {"attrs": ["bold", "underline"]})

    lgr.info(f'model_dir={paths["model_dir"]}', {"color": "magenta"})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {"color": "magenta"})
    lgr.info(f'train folds={train_params["data.train_folds"]}', {"color": "magenta"})
    lgr.info(f'validation folds={train_params["data.validation_folds"]}', {"color": "magenta"})

    # ==============================================================================
    # Data
    # ==============================================================================
    # Train Data
    lgr.info("Train Data:", {"attrs": "bold"})

    reset_cache = ask_user("Do you want to reset cache?")
    cache_kwargs = {"use_pipeline_hash": False}
    if not reset_cache:
        audit_cache = ask_user("Do you want to audit cache?")
        if not audit_cache:
            cache_kwargs2 = dict(audit_first_sample=False, audit_rate=None)
            cache_kwargs = {**cache_kwargs, **cache_kwargs2}

    # split to folds randomly
    params = dict(
        label_type=train_params["label_type"],
        data_dir=paths["data_dir"],
        cache_dir=paths["cache_dir"],
        reset_cache=reset_cache,
        sample_ids=train_params["data.selected_sample_ids"],
        num_workers=train_params["data.train_num_workers"],
        cache_kwargs=cache_kwargs,
        train=False,
        verbose=False,
    )

    dataset_all = prostate_x.ProstateX.dataset(**params)
    # ExportDataset.export_to_dir(dataset=dataset_all, output_dir=f'/tmp/ozery/prostatex_{my_version}')

    folds = dataset_balanced_division_to_folds(
        dataset=dataset_all,
        output_split_filename=paths["data_split_filename"],
        keys_to_balance=["data.ground_truth"],
        id="data.input.patient_id",
        workers=0,  # todo: stuck in Export to dataframe
        nfolds=train_params["data.num_folds"],
        verbose=True,
    )

    train_sample_ids = []
    for fold in train_params["data.train_folds"]:
        train_sample_ids += folds[fold]
    validation_sample_ids = []
    for fold in train_params["data.validation_folds"]:
        validation_sample_ids += folds[fold]

    params["sample_ids"] = train_sample_ids
    params["reset_cache"] = False
    params["train"] = True
    params["cache_kwargs"] = dict(use_pipeline_hash=False, audit_first_sample=False, audit_rate=None)
    train_dataset = prostate_x.ProstateX.dataset(**params)
    # for _ in train_dataset:
    #     pass
    params["sample_ids"] = validation_sample_ids
    params["train"] = False
    validation_dataset = prostate_x.ProstateX.dataset(**params)

    lgr.info("- Create sampler:")
    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name="data.ground_truth",
        num_balanced_classes=train_params["class_num"],
        batch_size=train_params["data.batch_size"],
        workers=0,  # train_params['data.train_num_workers'] #todo: stuck
    )
    lgr.info("- Create sampler: Done")

    # Create dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=train_params["data.train_num_workers"],
    )
    lgr.info("Train Data: Done", {"attrs": "bold"})

    # dataloader
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=train_params["data.batch_size"],
        collate_fn=CollateDefault(),
        num_workers=train_params["data.validation_num_workers"],
    )
    lgr.info("Validation Data: Done", {"attrs": "bold"})

    # ==============================================================================
    # Model
    # ==============================================================================
    lgr.info("Model:", {"attrs": "bold"})

    model = create_model(**train_params["model"])

    lgr.info("Model: Done", {"attrs": "bold"})

    # ====================================================================================
    #  Loss
    # ====================================================================================
    losses = {
        "cls_loss": LossDefault(
            pred="model.logits.classification", target="data.ground_truth", callable=F.cross_entropy, weight=1.0
        ),
    }

    # ====================================================================================
    # Metrics
    # ====================================================================================
    lgr.info("Metrics:", {"attrs": "bold"})
    train_metrics = OrderedDict([("auc", MetricAUCROC(pred="model.output.classification", target="data.ground_truth"))])
    validation_metrics = copy.deepcopy(train_metrics)  # use the same metrics in validation as well

    # either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric).
    best_epoch_source = dict(
        monitor="validation.metrics.auc",
        mode="max",
    )

    # create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_params["opt.lr"],
        weight_decay=train_params["opt.weight_decay"],
        momentum=0.9,
        nesterov=True,
    )

    # create learning scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")

    # optimizier and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    # =====================================================================================
    #  Train
    # =====================================================================================
    lgr.info("Train:", {"attrs": "bold"})

    # create instance of PL module - FuseMedML generic version
    pl_module = LightningModuleDefault(
        model_dir=paths["model_dir"],
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_schs,
    )

    # create lightining trainer.
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        max_epochs=train_params["trainer.num_epochs"],
        accelerator=train_params["trainer.accelerator"],
        devices=train_params["trainer.num_devices"],
        auto_select_gpus=True,
    )

    # train
    pl_trainer.fit(pl_module, train_dataloader, validation_dataloader, ckpt_path=train_params["trainer.ckpt_path"])

    lgr.info("Train: Done", {"attrs": "bold"})


######################################
# Inference Template
######################################
def run_infer(paths: dict, infer_common_params: dict, audit_cache: Optional[bool] = True):
    create_dir(paths["inference_dir"])
    infer_file = os.path.join(paths["inference_dir"], infer_common_params["infer_filename"])
    checkpoint_file = os.path.join(paths["model_dir"], infer_common_params["checkpoint"])

    #### Logger
    fuse_logger_start(output_path=paths["inference_dir"], console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")
    lgr.info("Fuse Inference", {"attrs": ["bold", "underline"]})
    lgr.info(f"infer_filename={infer_file}", {"color": "magenta"})
    lgr.info(f'infer folds={infer_common_params["data.infer_folds"]}', {"color": "magenta"})

    ## Data
    folds = load_pickle(paths["data_split_filename"])  # assume exists and created in train func

    infer_sample_ids = []
    for fold in infer_common_params["data.infer_folds"]:
        infer_sample_ids += folds[fold]

    params = dict(
        label_type=infer_common_params["label_type"],
        data_dir=paths["data_dir"],
        cache_dir=paths["cache_dir"],
        train=False,
        sample_ids=infer_sample_ids,
        verbose=False,
    )
    if not audit_cache:
        params["cache_kwargs"] = dict(use_pipeline_hash=False, audit_first_sample=False, audit_rate=None)
    else:
        params["cache_kwargs"] = dict(use_pipeline_hash=False)
    infer_dataset = prostate_x.ProstateX.dataset(**params)

    # dataloader
    infer_dataloader = DataLoader(
        dataset=infer_dataset,
        batch_size=infer_common_params["data.batch_size"],
        collate_fn=CollateDefault(),
        num_workers=infer_common_params["data.num_workers"],
    )

    # load python lightning module
    model = create_model(**infer_common_params["model"])
    pl_module = LightningModuleDefault.load_from_checkpoint(
        checkpoint_file, model_dir=paths["model_dir"], model=model, map_location="cpu", strict=True
    )
    # set the prediction keys to extract (the ones used be the evaluation function).
    pl_module.set_predictions_keys(
        ["model.output.classification", "data.ground_truth"]
    )  # which keys to extract and dump into file

    # create a trainer instance
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        accelerator=infer_common_params["trainer.accelerator"],
        devices=infer_common_params["trainer.num_devices"],
        auto_select_gpus=True,
    )
    predictions = pl_trainer.predict(pl_module, infer_dataloader, return_predictions=True)

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file)


######################################
# Analyze Template
######################################
def run_eval(paths: dict, eval_common_params: dict):
    infer_file = os.path.join(paths["inference_dir"], eval_common_params["infer_filename"])

    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")
    lgr.info("Fuse Analyze", {"attrs": ["bold", "underline"]})

    # metrics
    metrics = OrderedDict(
        [
            ("operation_point", MetricApplyThresholds(pred="model.output.classification")),  # will apply argmax
            ("accuracy", MetricAccuracy(pred="results:metrics.operation_point.cls_pred", target="data.ground_truth")),
            (
                "roc",
                MetricROCCurve(
                    pred="model.output.classification",
                    target="data.ground_truth",
                    output_filename=os.path.join(paths["inference_dir"], "roc_curve.png"),
                ),
            ),
            ("auc", MetricAUCROC(pred="model.output.classification", target="data.ground_truth")),
        ]
    )

    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(ids=None, data=infer_file, metrics=metrics, output_dir=paths["eval_dir"])

    return results


######################################
# Run
######################################
if __name__ == "__main__":
    main()
