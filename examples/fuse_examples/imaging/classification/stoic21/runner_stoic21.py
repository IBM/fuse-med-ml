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

import logging
import os
from typing import OrderedDict
import copy
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe



import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from fuse.eval.evaluator import EvaluatorDefault 
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve

from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.split import dataset_balanced_division_to_folds
    
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models.backbones.backbone_resnet_3d import BackboneResnet3D
from fuse.dl.models import ModelMultiHead
from fuse.dl.models.heads.head_3D_classifier import Head3DClassifier
from fuse.dl.lightning.pl_module import LightningModuleDefault

from fuse.utils.utils_debug import FuseDebug
from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.file_io.file_io import create_dir, load_pickle, save_dataframe
import fuse.utils.gpu as GPU

from fuseimg.datasets.stoic21 import STOIC21

###########################################################################################################
# Fuse
###########################################################################################################
##########################################
# Debug modes
##########################################
mode = 'default'  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)

##########################################qQ
# Output Paths
##########################################
assert "STOIC21_DATA_PATH" in os.environ, "Expecting environment variable STOIC21_DATA_PATH to be set. Follow the instruction in example README file to download and set the path to the data"
ROOT = '_examples/stoic21' # TODO: fill path here
model_dir = os.path.join(ROOT, 'model_dir')
PATHS = {'model_dir': model_dir,
         'cache_dir': os.path.join(ROOT, 'cache_dir'),
         'data_split_filename': os.path.join(ROOT, 'stoic21_split.pkl'),
         'data_dir': os.environ["STOIC21_DATA_PATH"],
         'inference_dir': os.path.join(model_dir, 'infer_dir'),
         'eval_dir': os.path.join(model_dir, 'eval_dir')}

##########################################
# Train Common Params
##########################################
TRAIN_COMMON_PARAMS = {}
# ============
# Model
# ============
TRAIN_COMMON_PARAMS['model'] = dict(imaging_dropout=0.5,
                                     fused_dropout=0.0,
                                     clinical_dropout=0.0)

# ============
# Data
# ============
TRAIN_COMMON_PARAMS['data.batch_size'] = 4
TRAIN_COMMON_PARAMS['data.train_num_workers'] = 16
TRAIN_COMMON_PARAMS['data.validation_num_workers'] = 16
TRAIN_COMMON_PARAMS['data.num_folds'] = 5
TRAIN_COMMON_PARAMS['data.train_folds'] = [0, 1, 2, 3]
TRAIN_COMMON_PARAMS['data.validation_folds'] = [4]

# ===============
# PL Trainer
# ===============
TRAIN_COMMON_PARAMS['trainer.num_epochs'] = 50
NUM_GPUS = 1
TRAIN_COMMON_PARAMS['trainer.num_devices'] = NUM_GPUS
TRAIN_COMMON_PARAMS['trainer.accelerator'] = "gpu"
TRAIN_COMMON_PARAMS['trainer.ckpt_path'] = None  #  path to the checkpoint you wish continue the training from

# ===============
# Optimizer
# ===============
TRAIN_COMMON_PARAMS['opt.lr'] = 1e-3
TRAIN_COMMON_PARAMS['opt.weight_decay'] = 0.005

def create_model(imaging_dropout: float, clinical_dropout: float, fused_dropout: float) -> torch.nn.Module:
    """ 
    creates the model 
    See Head3DClassifier for details about imaging_dropout, clinical_dropout, fused_dropout
    """
    model = ModelMultiHead(
        conv_inputs=(('data.input.img', 1),),
        backbone=BackboneResnet3D(in_channels=1),
        heads=[
            Head3DClassifier(head_name='classification',
                                conv_inputs=[("model.backbone_features", 512)],
                                dropout_rate=imaging_dropout,
                                append_dropout_rate=clinical_dropout,
                                fused_dropout_rate=fused_dropout,
                                num_classes=2,
                                append_features=[("data.input.clinical", 8)],
                                append_layers_description=(256,128),
                                ),
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
    dataset_all = STOIC21.dataset(paths["data_dir"], paths["cache_dir"], reset_cache=False)
    folds = dataset_balanced_division_to_folds(dataset=dataset_all,
                                        output_split_filename=paths["data_split_filename"], 
                                        keys_to_balance=["data.gt.probSevere"], 
                                        nfolds=train_common_params["data.num_folds"])

    train_sample_ids = []
    for fold in train_common_params["data.train_folds"]:
        train_sample_ids += folds[fold]
    validation_sample_ids = []
    for fold in train_common_params["data.validation_folds"]:
        validation_sample_ids += folds[fold]

    train_dataset = STOIC21.dataset(paths["data_dir"], paths["cache_dir"], sample_ids=train_sample_ids, train=True)
    # for _ in train_dataset:
    #     pass
    validation_dataset = STOIC21.dataset(paths["data_dir"], paths["cache_dir"], sample_ids=validation_sample_ids, train=False)

    lgr.info(f'- Create sampler:')
    sampler = BatchSamplerDefault(dataset=train_dataset,
                                       balanced_class_name='data.gt.probSevere',
                                       num_balanced_classes=2,
                                       batch_size=train_common_params['data.batch_size'],
                                       balanced_class_weights=None)
    lgr.info(f'- Create sampler: Done')

    # Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=sampler, collate_fn=CollateDefault(), num_workers=train_common_params['data.train_num_workers'])
    lgr.info(f'Train Data: Done', {'attrs': 'bold'})


    # dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=train_common_params['data.batch_size'], collate_fn=CollateDefault(),
                                       num_workers=train_common_params['data.validation_num_workers'])
    lgr.info(f'Validation Data: Done', {'attrs': 'bold'})

    # ==============================================================================
    # Model
    # ==============================================================================
    model = create_model(**train_common_params["model"])

    # ====================================================================================
    #  Loss
    # ====================================================================================
    losses = {
        'cls_loss': LossDefault(pred='model.logits.classification', target='data.gt.probSevere', callable=F.cross_entropy, weight=1.0),
    }

    # ====================================================================================
    # Metrics
    # ====================================================================================
    train_metrics = OrderedDict([
        ('auc', MetricAUCROC(pred='model.output.classification', target="data.gt.probSevere")),
    ])

    validation_metrics = copy.deepcopy(train_metrics) # use the same metrics in validation as well
 
    # either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric). 
    best_epoch_source = dict(
        monitor="validation.metrics.auc",
        mode="max",
    )
    
    # create optimizer
    optimizer = optim.SGD(model.parameters(), lr=train_common_params['opt.lr'], weight_decay=train_common_params['opt.weight_decay'], momentum=0.9, nesterov=True)

    # create learning scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
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
INFER_COMMON_PARAMS['checkpoint'] = "best_epoch.ckpt"
INFER_COMMON_PARAMS['data.infer_folds'] = [4]  # infer validation set
INFER_COMMON_PARAMS['data.batch_size'] = 4
INFER_COMMON_PARAMS['data.num_workers'] = 16
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
    #### Logger
    fuse_logger_start(output_path=paths['inference_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})
    lgr.info(f'infer_filename={infer_file}', {'color': 'magenta'})

    ## Data
    folds = load_pickle(paths["data_split_filename"]) # assume exists and created in train func

    infer_sample_ids = []                              
    for fold in infer_common_params["data.infer_folds"]:
        infer_sample_ids += folds[fold]

    infer_dataset = STOIC21.dataset(paths["data_dir"], paths["cache_dir"], sample_ids=infer_sample_ids, train=False)
    infer_dataloader = DataLoader(dataset=infer_dataset, collate_fn=CollateDefault(), batch_size=2, num_workers=2)

    # load python lightning module
    model = create_model(**infer_common_params["model"])
    pl_module = LightningModuleDefault.load_from_checkpoint(checkpoint_file, model_dir=paths["model_dir"], model=model, map_location="cpu", strict=True)
    # set the prediction keys to extract (the ones used be the evaluation function).
    pl_module.set_predictions_keys(['model.output.classification', 'data.gt.probSevere']) # which keys to extract and dump into file

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
    lgr.info('Fuse Analyze', {'attrs': ['bold', 'underline']})

    # metrics
    metrics = OrderedDict([
        ('operation_point', MetricApplyThresholds(pred='model.output.classification')), # will apply argmax
        ('accuracy', MetricAccuracy(pred='results:metrics.operation_point.cls_pred', target='data.gt.probSevere')),
        ('roc', MetricROCCurve(pred='model.output.classification', target='data.gt.probSevere', output_filename=os.path.join(paths['inference_dir'], 'roc_curve.png'))),
        ('auc', MetricAUCROC(pred='model.output.classification', target='data.gt.probSevere')),
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
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None # [0]
    GPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

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

