
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

from collections import OrderedDict
import os
import sys
import copy
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
import pandas as pd
from fuse.utils.utils_debug import FuseDebug
from fuse.utils.gpu import choose_and_enable_multiple_gpus

import logging

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils import NDict
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.split import dataset_balanced_division_to_folds
from fuse.dl.models import ModelMultiHead
from fuse.utils.file_io.file_io import load_pickle
from fuse.dl.losses.loss_default import LossDefault

from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC, MetricAccuracy
from fuseimg.datasets.ukbb_neck_to_knee import UKBB
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe
from pytorch_lightning import Trainer
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.utils.file_io.file_io import create_dir, load_pickle, save_dataframe
from fuse.eval.evaluator import EvaluatorDefault
import torch
import hydra
from typing import Dict
from omegaconf import DictConfig, OmegaConf

from fuse.dl.models.backbones.backbone_resnet_3d import BackboneResnet3D
from fuse.dl.models import ModelMultiHead
from fuse.dl.models.heads.head_3D_classifier import Head3DClassifier

assert "UKBB_DATA_PATH" in os.environ, "Expecting environment variable UKBB_DATA_PATH to be set. Follow the instruction in example README file to download and set the path to the data"
##########################################
# Debug modes
##########################################
mode = 'default'  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)

def create_model(num_classes: int) -> torch.nn.Module:
    """ 
    creates the model 
    See HeadGlobalPoolingClassifier for details 
    """
    model = ModelMultiHead(
    conv_inputs=(('data.input.img', 1),),
    backbone=BackboneResnet3D(in_channels=1),
    heads=[
        Head3DClassifier(head_name='head_0',
                             conv_inputs=[("model.backbone_features", 512)],
                            #  dropout_rate=train_params['imaging_dropout'],
                            #  append_dropout_rate=train_params['clinical_dropout'],
                            #  fused_dropout_rate=train_params['fused_dropout'],
                             num_classes=num_classes,
                            #  append_features=[("data.input.clinical", 8)],
                            #  append_layers_description=(256,128),
                             ),
    ])
    return model
#################################
# Train Template
#################################
def run_train(paths : NDict , train: NDict ) -> torch.nn.Module:
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')

    # Download data
    # TBD

    lgr.info('\nFuse Train', {'attrs': ['bold', 'underline']})

    lgr.info(f'model_dir={paths["model_dir"]}', {'color': 'magenta'})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {'color': 'magenta'})

    #### Train Data
    if train['target'] == "classification" :
        num_classes = 2
        mode = "approx"
        gt_label = "data.gt.classification"
        skip_keys=['data.gt.subtype']
        class_names = ["Male", "Female"] 
    else:
        raise("unsuported target!!")
    # split to folds randomly - temp
    sample_ids = pd.read_csv(os.path.join(paths["data_misc_dir"],"samples.csv"))['file'].to_list()
    print(sample_ids)
    dataset_all = UKBB.dataset(paths["data_dir"], paths["data_misc_dir"], train['target'], paths["cache_dir"], reset_cache=False, num_workers=train["num_workers"], sample_ids=sample_ids,train=True)
    folds = dataset_balanced_division_to_folds(dataset=dataset_all,
                                        output_split_filename=os.path.join( paths["data_misc_dir"], paths["data_split_filename"]), 
                                        id = 'data.patientID',
                                        keys_to_balance=[gt_label], 
                                        nfolds=train["num_folds"],
                                        workers= train["num_workers"])

    train_sample_ids = []
    for fold in train["train_folds"]:
        train_sample_ids += folds[fold]
    validation_sample_ids = []
    for fold in train["validation_folds"]:
        validation_sample_ids += folds[fold]

    train_dataset = UKBB.dataset(paths["data_dir"], paths["data_misc_dir"], train['target'], paths["cache_dir"], reset_cache=False, num_workers=train["num_workers"], sample_ids=train_sample_ids, train=True)
    
    validation_dataset = UKBB.dataset(paths["data_dir"], paths["data_misc_dir"], train['target'], paths["cache_dir"],  reset_cache=False, num_workers=train["num_workers"], sample_ids=validation_sample_ids)

    ## Create sampler
    lgr.info(f'- Create sampler:')
    sampler = BatchSamplerDefault(dataset=train_dataset,
                                       balanced_class_name=gt_label,
                                       num_balanced_classes=num_classes,
                                       batch_size=train["batch_size"],
                                       mode = mode,
                                       workers=train["num_workers"],
                                       balanced_class_weights=None
                                       )

    lgr.info(f'- Create sampler: Done')

    ## Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  shuffle=False, drop_last=False,
                                  batch_sampler=sampler, collate_fn=CollateDefault(skip_keys=skip_keys),
                                  num_workers=train["num_workers"])
    lgr.info(f'Train Data: Done', {'attrs': 'bold'})

    #### Validation data
    lgr.info(f'Validation Data:', {'attrs': 'bold'})

    ## Create dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       shuffle=False,
                                       drop_last=False,
                                       batch_sampler=None,
                                       batch_size=train["batch_size"],
                                       num_workers=train["num_workers"],
                                       collate_fn=CollateDefault(skip_keys=skip_keys))
    lgr.info(f'Validation Data: Done', {'attrs': 'bold'})
    # ===================================================================
    # ==============================================================================
    # Model
    # ==============================================================================
    lgr.info('Model:', {'attrs': 'bold'})

    model = create_model(num_classes)
    lgr.info('Model: Done', {'attrs': 'bold'})


    # ====================================================================================
    #  Loss
    # ====================================================================================
    losses = {
        'cls_loss': LossDefault(pred='model.logits.head_0', target=gt_label,
                                    callable=F.cross_entropy, weight=1.0)
    }


    # ====================================================================================
    # Metrics
    # ====================================================================================
    train_metrics = OrderedDict([
        ('op', MetricApplyThresholds(pred='model.output.head_0')), # will apply argmax
        ('auc', MetricAUCROC(pred='model.output.head_0', target=gt_label, class_names = class_names)),
        ('accuracy', MetricAccuracy(pred='results:metrics.op.cls_pred', target=gt_label)),
    ])
    
    validation_metrics = copy.deepcopy(train_metrics) # use the same metrics in validation as well

    # either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric). 
    best_epoch_source = dict(
        monitor="validation.metrics.auc.macro_avg",
        mode="max",
    )

    # =====================================================================================
    #  Manager - Train
    #  Create a manager, training objects and run a training process.
    # =====================================================================================
    lgr.info('Train:', {'attrs': 'bold'})

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=train["learning_rate"],
                           weight_decay=train["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    lr_sch_config = dict(scheduler=scheduler,monitor="validation.losses.total_loss")
    
    # optimizier and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    # create instance of PL module - FuseMedML generic version
    
    pl_module = LightningModuleDefault(model_dir=paths["model_dir"], 
                                       model=model,
                                       losses=losses,
                                       train_metrics=train_metrics,
                                       validation_metrics=validation_metrics,
                                       best_epoch_source=best_epoch_source,
                                       optimizers_and_lr_schs=optimizers_and_lr_schs)

    # create lightining trainer.
    pl_trainer = Trainer(default_root_dir=paths['model_dir'],
                            max_epochs=train['trainer']['num_epochs'],
                            accelerator=train['trainer']['accelerator'],
                            devices=train['trainer']['devices'],
                            num_sanity_val_steps = -1,
                            auto_select_gpus=True)

    # train from scratch
    pl_trainer.fit(pl_module, train_dataloader, validation_dataloader, ckpt_path=train['trainer']['ckpt_path'])
    lgr.info('Train: Done', {'attrs': 'bold'})
    
    return model, pl_trainer

######################################
# Inference Template
######################################
def run_infer(model: torch.nn.Module, pl_trainer: Trainer, paths : NDict , infer: NDict):
    create_dir(paths['inference_dir'])
    #### Logger
    fuse_logger_start(output_path=paths["inference_dir"], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})
    infer_file = os.path.join(paths['inference_dir'], infer['infer_filename'])
    checkpoint_file = os.path.join(paths["model_dir"], infer["checkpoint"])
    lgr.info(f'infer_filename={checkpoint_file}', {'color': 'magenta'})


    ## Data
    folds = load_pickle(os.path.join( paths["data_misc_dir"], paths["data_split_filename"])) # assume exists and created in train func

    infer_sample_ids = []                              
    for fold in infer["infer_folds"]:
        infer_sample_ids += folds[fold]

    test_dataset = UKBB.dataset(paths["data_dir"], paths["data_misc_dir"], infer['target'], paths["cache_dir"], sample_ids=infer_sample_ids, train=False)

    ## Create dataloader
    infer_dataloader = DataLoader(dataset=test_dataset,
                                  shuffle=False, drop_last=False,
                                  collate_fn=CollateDefault(),
                                  num_workers=infer["num_workers"])
    # load python lightning module
    pl_module = LightningModuleDefault.load_from_checkpoint(checkpoint_file, model_dir=paths["model_dir"], model=model, map_location="cpu", strict=True)
    # set the prediction keys to extract (the ones used be the evaluation function).
    pl_module.set_predictions_keys(['model.output.head_0', 'data.gt.classification']) # which keys to extract and dump into file
    lgr.info(f'Test Data: Done', {'attrs': 'bold'})

    # create a trainer instance
    predictions = pl_trainer.predict(pl_module, infer_dataloader, return_predictions=True)

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file)
    


######################################
# Analyze Template
######################################
def run_eval(paths : NDict , infer: NDict):
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Eval', {'attrs': ['bold', 'underline']})

     # metrics
    metrics = OrderedDict([
        ('op', MetricApplyThresholds(pred='model.output.head_0')), # will apply argmax
        ('auc', MetricAUCROC(pred='model.output.head_0', target='data.gt.classification')),
        ('accuracy', MetricAccuracy(pred='results:metrics.op.cls_pred', target='data.gt.classification')),
    ])

    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(ids=None,
                     data=os.path.join(paths["inference_dir"], infer["infer_filename"]),
                     metrics=metrics,
                     output_dir=paths["eval_dir"])

    return results

@hydra.main(config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    cfg = NDict(OmegaConf.to_object(cfg))
    print(cfg)
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    choose_and_enable_multiple_gpus(cfg["train.trainer.devices"], force_gpus=force_gpus)


    # Path to the stored dataset location
    # dataset should be download from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508
    # download requires NBIA data retriever https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
    # put on the following in the main folder  - 
    # 1. CMMD_clinicaldata_revision.csv which is a converted version of CMMD_clinicaldata_revision.xlsx 
    # 2. folder named CMMD which is the downloaded data folder

    # train
    if 'train' in cfg["run.running_modes"]:
        model, trainer = run_train(cfg["paths"] ,cfg["train"])
    else:
        assert "Expecting train mode to be set."

    # infer
    if 'infer' in cfg["run.running_modes"]:
        run_infer(model, trainer, cfg["paths"] , cfg["infer"])
    #
    # analyze
    if 'eval' in cfg["run.running_modes"]:
        run_eval(cfg["paths"] ,cfg["infer"])
if __name__ == "__main__":
    sys.argv.append('hydra.run.dir=working_dir')
    main()
