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
import os
import copy
import logging
from typing import OrderedDict

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from fuse.dl.models import ModelMultiHead
from fuse.dl.models.backbones.backbone_resnet import BackboneResnet
from fuse.dl.models.heads.head_global_pooling_classifier import HeadGlobalPoolingClassifier
from fuse.dl.models.backbones.backbone_inception_resnet_v2 import BackboneInceptionResnetV2

from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve

from fuse.utils.utils_debug import FuseDebug
from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.file_io.file_io import create_dir, save_dataframe, load_pickle
import fuse.utils.gpu as GPU

from fuse.eval.evaluator import EvaluatorDefault
from fuse.dl.losses.loss_default import LossDefault

from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.split import dataset_balanced_division_to_folds

import pytorch_lightning as pl
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe

from fuseimg.datasets.isic import ISIC
from fuse_examples.imaging.classification.isic.golden_members import FULL_GOLDEN_MEMBERS


###########################################################################################################
# Fuse
###########################################################################################################
##########################################
# Debug modes
##########################################
mode = 'default'  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)

##########################################
# Output Paths
##########################################

# TODO: Path to save model
ROOT = '_examples/isic/'

PATHS = {'data_dir': os.path.join(ROOT, 'data_dir'),
         'model_dir': os.path.join(ROOT, 'model_dir'),
         'cache_dir': os.path.join(ROOT, 'cache_dir'),
         'inference_dir': os.path.join(ROOT, 'infer_dir'),
         'eval_dir': os.path.join(ROOT, 'eval_dir'),
         'data_split_filename': os.path.join(ROOT, 'isic_split.pkl')}

##########################################
# Train Common Params
##########################################
TRAIN_COMMON_PARAMS = {}
# ============
# Data
# ============
TRAIN_COMMON_PARAMS['data.batch_size'] = 8
TRAIN_COMMON_PARAMS['data.train_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.validation_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.num_folds'] = 5
TRAIN_COMMON_PARAMS['data.train_folds'] = [0, 1, 2]
TRAIN_COMMON_PARAMS['data.validation_folds'] = [3]
TRAIN_COMMON_PARAMS['data.samples_ids'] = FULL_GOLDEN_MEMBERS # Change to None to use all members

# ===============
# PL Trainer
# ===============
TRAIN_COMMON_PARAMS['trainer.num_epochs'] = 20
TRAIN_COMMON_PARAMS['trainer.num_devices'] = 1
TRAIN_COMMON_PARAMS['trainer.accelerator'] = "gpu"
TRAIN_COMMON_PARAMS['trainer.ckpt_path'] = None

# ===============
# Optimizer
# ===============
TRAIN_COMMON_PARAMS['opt.lr'] = 1e-5
TRAIN_COMMON_PARAMS['opt.weight_decay'] = 1e-3

# ===============
# Model
# ===============
TRAIN_COMMON_PARAMS['model'] = dict(dropout_rate=0.5)

# ===============
# Manager - Train
# ===============
# TRAIN_COMMON_PARAMS['manager.train_params'] = {
#     'virtual_batch_size': 1,  # number of batches in one virtual batch
#     'start_saving_epochs': 20,  # first epoch to start saving checkpoints from
#     'gap_between_saving_epochs': 10,  # number of epochs between saved checkpoint
# }
# # best_epoch_source
# # if an epoch values are the best so far, the epoch is saved as a checkpoint.
# TRAIN_COMMON_PARAMS['manager.best_epoch_source'] = {
#     'source': 'metrics.auc.macro_avg',  # can be any key from 'epoch_results'
#     'optimization': 'max',  # can be either min/max
#     'on_equal_values': 'better',
#     # can be either better/worse - whether to consider best epoch when values are equal
# }
# TRAIN_COMMON_PARAMS['manager.resume_checkpoint_filename'] = None  # if not None, will try to load the checkpoint

def create_model(dropout_rate: float) -> torch.nn.Module:
    """ 
    creates the model 
    """
    model = ModelMultiHead(
        conv_inputs=(('data.input.img', 3),),
        backbone={'Resnet18': BackboneResnet(pretrained=True, in_channels=3, name='resnet18'),
                  'InceptionResnetV2': BackboneInceptionResnetV2(input_channels_num=3, logical_units_num=43)}['InceptionResnetV2'],
        heads=[
            HeadGlobalPoolingClassifier(head_name='head_0',
                                            dropout_rate=dropout_rate,
                                            conv_inputs=[('model.backbone_features', 1536)],
                                            num_classes=8,
                                            pooling="avg"),
        ]
    )
    return model

#################################
# Train Template
#################################
def run_train(paths: dict, train_common_params: dict):
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths['model_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Train', {'attrs': ['bold', 'underline']})

    lgr.info(f'model_dir={paths["model_dir"]}', {'color': 'magenta'})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {'color': 'magenta'})

    # ==============================================================================
    # Data
    # ==============================================================================
    # Train Data
    lgr.info(f'Train Data:', {'attrs': 'bold'})

    # split to folds randomly - temp
    all_dataset = ISIC.dataset(paths['data_dir'], paths['cache_dir'], reset_cache=False,
                               num_workers=train_common_params['data.train_num_workers'],
                               samples_ids=train_common_params['data.samples_ids'])

    folds = dataset_balanced_division_to_folds(dataset=all_dataset,
                                                output_split_filename=paths['data_split_filename'],
                                                keys_to_balance=['data.label'],
                                                nfolds=train_common_params['data.num_folds'])
    
    train_sample_ids = []
    for fold in train_common_params["data.train_folds"]:
        train_sample_ids += folds[fold]
    validation_sample_ids = []
    for fold in train_common_params["data.validation_folds"]:
        validation_sample_ids += folds[fold]

    train_dataset = ISIC.dataset(paths['data_dir'], paths['cache_dir'], samples_ids=train_sample_ids, train=True)

    lgr.info(f'- Create sampler:')
    sampler = BatchSamplerDefault(dataset=train_dataset,
                                       balanced_class_name='data.label',
                                       num_balanced_classes=8,
                                       batch_size=train_common_params['data.batch_size'])
    lgr.info(f'- Create sampler: Done')

    # Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_sampler=sampler,
                                  collate_fn=CollateDefault(),
                                  num_workers=train_common_params['data.train_num_workers'])

    lgr.info(f'Train Data: Done', {'attrs': 'bold'})

    ## Validation data
    lgr.info(f'Validation Data:', {'attrs': 'bold'})

    # dataset
    validation_dataset = ISIC.dataset(paths['data_dir'], paths['cache_dir'], samples_ids=validation_sample_ids, train=False)

    # dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       batch_size=train_common_params['data.batch_size'],
                                       collate_fn=CollateDefault(),
                                       num_workers=train_common_params['data.validation_num_workers'])
    lgr.info(f'Validation Data: Done', {'attrs': 'bold'})

    # ==============================================================================
    # Model
    # ==============================================================================
    lgr.info('Model:', {'attrs': 'bold'})

    model = create_model(**train_common_params["model"])

    lgr.info('Model: Done', {'attrs': 'bold'})

    # ====================================================================================
    #  Loss
    # ====================================================================================
    losses = {
        'cls_loss': LossDefault(pred='model.logits.head_0', target='data.label', callable=F.cross_entropy, weight=1.0),
    }

    # ====================================================================================
    # Metrics
    # ====================================================================================
    class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    train_metrics = OrderedDict([
        ('op', MetricApplyThresholds(pred='model.output.head_0')), # will apply argmax
        ('auc', MetricAUCROC(pred='model.output.head_0', target='data.label', class_names=class_names)),
        ('accuracy', MetricAccuracy(pred='results:metrics.op.cls_pred', target='data.label')),
    ])

    validation_metrics = copy.deepcopy(train_metrics) # use the same metrics in validation as well

    best_epoch_source = dict(
        monitor="validation.metrics.auc.macro_avg",
        mode="max"
    )

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=train_common_params['opt.lr'], weight_decay=train_common_params['opt.weight_decay'])

    # create learning scheduler
    lr_scheduler = {'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                 'CosineAnnealing': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)}['ReduceLROnPlateau']
    lr_sch_config = dict(scheduler=lr_scheduler,
                        monitor="validation.losses.total_loss")

    # optimizier and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    # =====================================================================================
    #  Train
    # =====================================================================================
    lgr.info('Train:', {'attrs': 'bold'})

    # create instance of PL module - FuseMedML generic version
    pl_module = LightningModuleDefault(model_dir=paths["model_dir"], 
                                       model=model,
                                       losses=losses,
                                       train_metrics=train_metrics,
                                       validation_metrics=validation_metrics,
                                       best_epoch_source=best_epoch_source,
                                       optimizers_and_lr_schs=optimizers_and_lr_schs)

    # create lightining trainer.
    pl_trainer = pl.Trainer(default_root_dir=paths['model_dir'],
                            max_epochs=train_common_params['trainer.num_epochs'],
                            accelerator=train_common_params["trainer.accelerator"],
                            devices=train_common_params["trainer.num_devices"],
                            auto_select_gpus=True)

    # train
    pl_trainer.fit(pl_module, train_dataloader, validation_dataloader, ckpt_path=train_common_params['trainer.ckpt_path'])

    lgr.info('Train: Done', {'attrs': 'bold'})

######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS['infer_filename'] = 'infer_file.gz'
INFER_COMMON_PARAMS['checkpoint'] = 'best_epoch.ckpt'
INFER_COMMON_PARAMS['data.num_workers'] = TRAIN_COMMON_PARAMS['data.train_num_workers']
INFER_COMMON_PARAMS['data.validation_num_workers'] =  TRAIN_COMMON_PARAMS['data.validation_num_workers']
INFER_COMMON_PARAMS['data.infer_folds'] = [4]  # infer validation set
INFER_COMMON_PARAMS['data.batch_size'] = 4

INFER_COMMON_PARAMS['model'] = TRAIN_COMMON_PARAMS['model']
INFER_COMMON_PARAMS['trainer.num_devices'] = 1
INFER_COMMON_PARAMS['trainer.accelerator'] = "gpu"

######################################
# Inference Template
######################################

def run_infer(paths: dict, infer_common_params: dict):
    create_dir(paths['inference_dir'])
    infer_file = os.path.join(paths['inference_dir'], infer_common_params['infer_filename'])
    checkpoint_file  = os.path.join(paths['model_dir'], infer_common_params['checkpoint'])

    ## Logger
    fuse_logger_start(output_path=paths['inference_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})
    lgr.info(f'infer_filename={infer_file}', {'color': 'magenta'})

    ## Data
    folds = load_pickle(paths["data_split_filename"]) # assume exists and created in train func

    infer_sample_ids = []                              
    for fold in infer_common_params["data.infer_folds"]:
        infer_sample_ids += folds[fold]

    # Create dataset
    infer_dataset = ISIC.dataset(paths['data_dir'], paths['cache_dir'], samples_ids=infer_sample_ids, train=False)

    # dataloader
    infer_dataloader = DataLoader(dataset=infer_dataset, collate_fn=CollateDefault(),
                                    batch_size=infer_common_params['data.batch_size'],
                                    num_workers=infer_common_params['data.num_workers'])
                            
    # load python lightning module
    model = create_model(**infer_common_params["model"])
    pl_module = LightningModuleDefault.load_from_checkpoint(checkpoint_file, model_dir=paths["model_dir"], model=model, map_location="cpu", strict=True)
    # set the prediction keys to extract (the ones used be the evaluation function).
    pl_module.set_predictions_keys(['model.output.head_0', 'data.label']) # which keys to extract and dump into file

    # create a trainer instance
    pl_trainer = pl.Trainer(default_root_dir=paths['model_dir'],
                            accelerator=infer_common_params["trainer.accelerator"],
                            devices=infer_common_params["trainer.num_devices"],
                            auto_select_gpus=True)
    predictions = pl_trainer.predict(pl_module, infer_dataloader, return_predictions=True)

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file)


######################################
# Eval Common Params
######################################
EVAL_COMMON_PARAMS = {}
EVAL_COMMON_PARAMS['infer_filename'] = INFER_COMMON_PARAMS['infer_filename']

######################################
# Eval Template
######################################
def run_eval(paths: dict, eval_common_params: dict):
    infer_file = os.path.join(paths['inference_dir'], eval_common_params['infer_filename'])

    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Eval', {'attrs': ['bold', 'underline']})

    # metrics
    metrics = OrderedDict([
        ('op', MetricApplyThresholds(pred='model.output.head_0')), # will apply argmax
        ('auc', MetricAUCROC(pred='model.output.head_0', target='data.label')),
        ('accuracy', MetricAccuracy(pred='results:metrics.op.cls_pred', target='data.label')),
        ('roc', MetricROCCurve(pred='model.output.head_0', target='data.label',
                                  output_filename=os.path.join(paths["inference_dir"], "roc_curve.png"))),
    ])
   
    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(ids=None,
                     data=infer_file,
                     metrics=metrics,
                     output_dir=paths['eval_dir'])

    return results


######################################
# Run
######################################
if __name__ == "__main__":
    # allocate gpus
    # To use cpu - set NUM_GPUS to 0
    NUM_GPUS = 1
    if NUM_GPUS == 0:
        TRAIN_COMMON_PARAMS['manager.train_params']['device'] = 'cpu' 
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    GPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    ISIC.download(data_path = PATHS['data_dir'])

    RUNNING_MODES = ['train', 'infer', 'eval']  # Options: 'train', 'infer', 'eval'

    # train
    if 'train' in RUNNING_MODES:
        run_train(paths=PATHS, train_common_params=TRAIN_COMMON_PARAMS)

    # infer
    if 'infer' in RUNNING_MODES:
        run_infer(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS)

    # eval
    if 'eval' in RUNNING_MODES:
        run_eval(paths=PATHS, eval_common_params=EVAL_COMMON_PARAMS)
