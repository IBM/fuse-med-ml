from collections import OrderedDict
import pathlib
from pickle import FALSE
from fuse.utils.utils_logger import fuse_logger_start
import os
import sys

# add parent directory to path, so that 'baseline' folder is treated as a module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from baseline.dataset import knight_dataset
import pandas as pd
from fuse.dl.models import ModelMultiHead
from fuse.dl.models.backbones.backbone_resnet_3d import BackboneResnet3D
from fuse.dl.models.heads.head_3D_classifier import Head3DClassifier
from fuse.dl.losses.loss_default import LossDefault
import torch.nn.functional as F
import torch.nn as nn
from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC, MetricAccuracy, MetricConfusion
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
import torch.optim as optim
import fuse.utils.gpu as GPU
from fuse.utils.rand.seed import Seed
import logging
import time
import copy
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.lightning.pl_module import LightningModuleDefault
import pytorch_lightning as pl

## Parameters:
##############################################################################
# Data sources to use in model. Set {'imaging': True, 'clinical': False} for imaging only setting,
# and vice versa, or set both to True to use both.
# allocate gpus
# uncomment if you want to use specific gpus instead of automatically looking for free ones
experiment_num = 0
task_num = 1  # 1 or 2
num_gpus = 1
use_data = {"imaging": True, "clinical": True}  # specify whether to use imaging, clinical data or both
batch_size = 2
resize_to = (80, 256, 256)

if task_num == 1:
    num_epochs = 150
    num_classes = 2
    learning_rate = 1e-4 if use_data["clinical"] else 1e-5
    imaging_dropout = 0.5
    clinical_dropout = 0.0
    fused_dropout = 0.5
    target_name = "data.gt.gt_global.task_1_label"
    target_metric = "validation.metrics.auc"

elif task_num == 2:
    num_epochs = 150
    num_classes = 5
    learning_rate = 1e-4
    imaging_dropout = 0.7
    clinical_dropout = 0.0
    fused_dropout = 0.0
    target_name = "data.gt.gt_global.task_2_label"
    target_metric = "validation.metrics.auc.macro_avg"


def main():
    # read train/val splits file. for convenience, we use the one
    # auto-generated by the nnU-Net framework for the KiTS21 data
    dir_path = pathlib.Path(__file__).parent.resolve()
    splits = pd.read_pickle(os.path.join(dir_path, "splits_final.pkl"))
    # For this example, we use split 0 out of the 5 available cross validation splits
    split = splits[0]

    # read environment variables for data, cache and results locations
    data_path = os.environ["KNIGHT_DATA"]
    cache_path = os.path.join(os.environ["KNIGHT_CACHE"], str(experiment_num))
    results_path = os.environ["KNIGHT_RESULTS"]

    ## Basic settings:
    ##############################################################################
    # create model results dir:
    # we use a time stamp in model directory name, to prevent re-writing
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(results_path, timestr)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # start logger
    fuse_logger_start(output_path=model_dir, console_verbose_level=logging.INFO)
    print("Done")

    # set constant seed for reproducibility.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required for pytorch deterministic mode
    rand_gen = Seed.set_seed(1234, deterministic_mode=True)

    # select gpus
    GPU.choose_and_enable_multiple_gpus(num_gpus, force_gpus=None)

    ## FuseMedML dataset preparation
    ##############################################################################

    train_dl, valid_dl, _, _, _, _ = knight_dataset(
        data_dir=data_path,
        cache_dir=cache_path,
        split=split,
        reset_cache=False,
        rand_gen=rand_gen,
        batch_size=batch_size,
        resize_to=resize_to,
        task_num=task_num,
        target_name=target_name,
        num_classes=num_classes,
    )

    ## Model definition
    ##############################################################################

    if use_data["imaging"]:
        backbone = BackboneResnet3D(in_channels=1)
        conv_inputs = [("model.backbone_features", 512)]
    else:
        backbone = nn.Identity()
        conv_inputs = None
    if use_data["clinical"]:
        append_features = [("data.input.clinical.all", 11)]
    else:
        append_features = None

    model = ModelMultiHead(
        conv_inputs=(("data.input.img", 1),),
        backbone=backbone,
        heads=[
            Head3DClassifier(
                head_name="head_0",
                conv_inputs=conv_inputs,
                dropout_rate=imaging_dropout,
                num_classes=num_classes,
                append_features=append_features,
                append_layers_description=(256, 128),
                append_dropout_rate=clinical_dropout,
                fused_dropout_rate=fused_dropout,
            ),
        ],
    )

    # Loss definition:
    ##############################################################################
    losses = {
        "cls_loss": LossDefault(pred="model.logits.head_0", target=target_name, callable=F.cross_entropy, weight=1.0)
    }

    # Metrics definition:
    ##############################################################################
    train_metrics = OrderedDict(
        [
            ("op", MetricApplyThresholds(pred="model.output.head_0")),  # will apply argmax
            ("auc", MetricAUCROC(pred="model.output.head_0", target=target_name)),
            ("accuracy", MetricAccuracy(pred="results:metrics.op.cls_pred", target=target_name)),
            (
                "sensitivity",
                MetricConfusion(pred="results:metrics.op.cls_pred", target=target_name, metrics=("sensitivity",)),
            ),
        ]
    )
    val_metrics = copy.deepcopy(train_metrics)  # use the same metrics in validation as well

    best_epoch_source = dict(
        monitor=target_metric,  # can be any key from losses or metrics dictionaries
        mode="max",  # can be either min/max
    )

    # Optimizer definition:
    ##############################################################################
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    # Scheduler definition:
    ##############################################################################
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    ## Training
    ##############################################################################

    # create instance of PL module - FuseMedML generic version
    pl_module = LightningModuleDefault(
        model_dir=model_dir,
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=val_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_schs,
    )
    # create lightining trainer.
    pl_trainer = pl.Trainer(
        default_root_dir=model_dir,
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=num_gpus,
        strategy=None,
        auto_select_gpus=True,
        num_sanity_val_steps=-1,
    )

    # train
    pl_trainer.fit(pl_module, train_dl, valid_dl, ckpt_path=None)


if __name__ == "__main__":
    main()
