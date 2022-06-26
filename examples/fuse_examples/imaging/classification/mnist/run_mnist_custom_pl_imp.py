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

"""
MNIST classfier implementation that demonstrate end to end training, inference and evaluation using FuseMedML
This example shows how to directly train a model using custom (your own) pytorch lightning module implementation 
"""


import copy
import logging
import os
from typing import Any, Dict, List, OrderedDict, Sequence
from fuse.utils.file_io.file_io import create_dir, save_dataframe
from fuse.utils.ndict import NDict

import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl 
from torch.utils.data.dataloader import DataLoader

from fuse.eval.evaluator import EvaluatorDefault 
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve

from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault

from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.lightning.pl_funcs import *

from fuse.utils.utils_debug import FuseDebug
from fuse.utils.utils_logger import fuse_logger_start

from fuseimg.datasets.mnist import MNIST

from fuse_examples.imaging.classification.mnist import lenet
###########################################################################################################
# Fuse
###########################################################################################################

##########################################
# Lightning Module
##########################################
class LightningModuleMnist(pl.LightningModule):
    """
    Implementation of pl.LightningModule
    Demonstrate how to use FuseMedML with with your own PyTorch lightining implementaiton.
    """
    def __init__(self, model_dir: str, opt_lr: float, opt_weight_decay: float, **kwargs):
        """
        :param model_dir: location for checkpoints and logs
        :param opt_lr: learning rate for Adam optimizer 
        :param opt_weight_decay: weight decay for Adam optimizer
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model_dir"])

        # store arguments
        self._model_dir = model_dir
        self._opt_lr = opt_lr
        self._opt_weight_decay = opt_weight_decay

        # model
        torch_model = lenet.LeNet()

        # wrap basic torch model to automatically read inputs from batch_dict and save its outputs to batch_dict
        self._model = ModelWrapSeqToDict(model=torch_model,
                            model_inputs=["data.image"],
                            post_forward_processing_function=perform_softmax,
                            model_outputs=['logits.classification', 'output.classification']
                            )

        # losses
        self._losses = {
            'cls_loss': LossDefault(pred='model.logits.classification', target='data.label', callable=F.cross_entropy, weight=1.0),
        }

        # metrics
        self._train_metrics = OrderedDict([
            ('operation_point', MetricApplyThresholds(pred='model.output.classification')), # will apply argmax
            ('accuracy', MetricAccuracy(pred='results:metrics.operation_point.cls_pred', target='data.label'))
        ])

        self._validation_metrics = copy.deepcopy(self._train_metrics) # use the same metrics in validation as well
        
        
    
    ## forward
    def forward(self, batch_dict: NDict) -> NDict:
        return self._model(batch_dict)
    
    ## Step
    def training_step(self, batch_dict: NDict, batch_idx: int) -> dict:
        # run forward function and store the outputs in batch_dict["model"]
        batch_dict["model"] = self.forward(batch_dict)
        # given the batch_dict and FuseMedML style losses - compute the losses, return the total loss and save losses values in batch_dict["losses"]
        total_loss = step_losses(self._losses, batch_dict)
        # given the batch_dict and FuseMedML style losses - collect the required values to compute the metrics on epoch_end
        step_metrics(self._train_metrics, batch_dict)

        # return the total_loss, the losses and drop everything else
        return {"loss": total_loss, "losses": batch_dict["losses"]} 
     
    def validation_step(self, batch_dict: NDict, batch_idx: int) -> dict:
        # run forward function and store the outputs in batch_dict["model"]
        batch_dict["model"] = self.forward(batch_dict)
        # given the batch_dict and FuseMedML style losses - compute the losses, return the total loss (ignored) and save losses values in batch_dict["losses"]
        _ = step_losses(self._losses, batch_dict)
        # given the batch_dict and FuseMedML style losses - collect the required values to compute the metrics on epoch_end
        step_metrics(self._validation_metrics, batch_dict)

        # return just the losses and drop everything else
        return {"losses": batch_dict["losses"]} 
     
    def predict_step(self, batch_dict: NDict, batch_idx: int) -> dict:
        # run forward function and store the outputs in batch_dict["model"]
        batch_dict["model"] = self.forward(batch_dict)
        # extract the requried keys - defined in self.set_predictions_keys()
        return step_extract_predictions(self._prediction_keys, batch_dict)
        
    ## Epoch end
    def training_epoch_end(self, step_outputs) -> None:
        # calc average epoch loss and log it
        epoch_end_compute_and_log_losses(self, "train", [e["losses"] for e in step_outputs])
        # evaluate  and log it
        epoch_end_compute_and_log_metrics(self, "train", self._train_metrics)
    
    def validation_epoch_end(self, step_outputs) -> None:
        # calc average epoch loss and log it
        epoch_end_compute_and_log_losses(self, "validation", [e["losses"] for e in step_outputs])
        # evaluate  and log it
        epoch_end_compute_and_log_metrics(self, "validation", self._validation_metrics)

    
    def configure_callbacks(self) -> Sequence[pl.Callback]:
        """ Create callbacks to monitor the metrics and print epoch summary """
        best_epoch_source = dict(
            monitor="validation.metrics.accuracy",
            mode="max",
        )
        return model_checkpoint_callbacks(self._model_dir, best_epoch_source)
    
    def configure_optimizers(self) -> Any:
        """ See pl.LightningModule.configure_optimizers return value for all options """
        # create optimizer
        optimizer = optim.Adam(self._model.parameters(), lr=self._opt_lr, weight_decay=self._opt_weight_decay)

        # create learning scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        lr_sch_config = dict(scheduler=lr_scheduler,
                            monitor="validation.losses.total_loss")
        return dict(optimizer=optimizer, lr_scheduler=lr_sch_config)
    
    def set_predictions_keys(self, keys: List[str]) -> None:
        """ Define which keys to extract from batch_dict  on prediction mode """
        self._prediction_keys = keys

##########################################
# Debug modes
##########################################
mode = 'default'  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)

##########################################
# Output Paths
##########################################
ROOT = '_examples' # TODO: fill path here
PATHS = {'model_dir': os.path.join(ROOT, 'mnist/model_dir'),
         'force_reset_model_dir': True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
         'cache_dir': os.path.join(ROOT, 'mnist/cache_dir'),
         'eval_dir': os.path.join(ROOT, 'mnist/eval_dir')}

##########################################
# Train Common Params
##########################################
TRAIN_COMMON_PARAMS = {}

# ============
# Data
# ============
TRAIN_COMMON_PARAMS['data.batch_size'] = 100
TRAIN_COMMON_PARAMS['data.train_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.validation_num_workers'] = 8

# ===============
# PL Trainer
# ===============
TRAIN_COMMON_PARAMS['trainer.num_epochs'] = 2
TRAIN_COMMON_PARAMS['trainer.num_devices'] = 1
TRAIN_COMMON_PARAMS['trainer.accelerator'] = "gpu"
TRAIN_COMMON_PARAMS['trainer.ckpt_path'] = None  #  path to the checkpoint you wish continue the training from

# ===============
# PL Module
# ===============
TRAIN_COMMON_PARAMS['pl_module.opt_lr'] = 1e-4
TRAIN_COMMON_PARAMS['pl_module.opt_weight_decay'] = 0.001


def perform_softmax(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cls_preds = F.softmax(logits, dim=1)
    return logits, cls_preds

#################################
# Train Template
#################################
def run_train(paths: dict, train_params: dict):
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths['model_dir'], console_verbose_level=logging.INFO)
    print('Fuse Train')

    # ==============================================================================
    # Data
    # ==============================================================================
    # Train Data
    print(f'Data - trainset:')
    train_dataset = MNIST.dataset(paths["cache_dir"], train=True)
    print(f'- Create sampler:')
    sampler = BatchSamplerDefault(dataset=train_dataset,
                                       balanced_class_name='data.label',
                                       num_balanced_classes=10,
                                       batch_size=train_params['data.batch_size'],
                                       balanced_class_weights=None)
    print(f'- Create sampler: Done')

    # Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=sampler, collate_fn=CollateDefault(), num_workers=train_params['data.train_num_workers'])
    print(f'Data - trainset: Done')

    ## Validation data
    print(f'Data - validation set:')
    # wrapping torch dataset
    validation_dataset = MNIST.dataset(paths["cache_dir"], train=False)
    
    # dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=train_params['data.batch_size'], collate_fn=CollateDefault(),
                                       num_workers=train_params['data.validation_num_workers'])
    print(f'Data - validation set: Done')



    print('Train:')
    # create instance of PL module 
    pl_module = LightningModuleMnist(model_dir=paths["model_dir"], 
                                     opt_lr=train_params["pl_module.opt_lr"], 
                                     opt_weight_decay=train_params["pl_module.opt_weight_decay"])
                

    # create lightining trainer.
    pl_trainer = pl.Trainer(default_root_dir=paths['model_dir'],
                            max_epochs=train_params['trainer.num_epochs'],
                            accelerator=train_params["trainer.accelerator"],
                            devices=train_params["trainer.num_devices"],
                            auto_select_gpus=True)

    # train    
    pl_trainer.fit(pl_module, train_dataloader, validation_dataloader, ckpt_path=train_params['trainer.ckpt_path'])
    print('Train: Done')


######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS['infer_filename'] = os.path.join(PATHS["model_dir"], 'infer.gz')
INFER_COMMON_PARAMS['checkpoint'] = os.path.join(PATHS["model_dir"], "best_epoch.ckpt")


######################################
# Inference Template
######################################
def run_infer(paths: dict, infer_common_params: dict):
    #### Logger
    fuse_logger_start(output_path=paths['model_dir'], console_verbose_level=logging.INFO)

    print('Fuse Inference')
    print(f'infer_filename={infer_common_params["infer_filename"]}')

    ## Data
    # Create dataset
    validation_dataset = MNIST.dataset(paths["cache_dir"], train=False)
    # dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset, collate_fn=CollateDefault(), batch_size=2, num_workers=2)

    # load pytorch lightning module
    pl_module = LightningModuleMnist.load_from_checkpoint(infer_common_params['checkpoint'], model_dir=paths["model_dir"], map_location="cpu", strict=True)
    # set the prediction keys to extract (the ones used be the evaluation function).
    pl_module.set_predictions_keys(['model.output.classification', 'data.label']) # which keys to extract and dump into file

    print('Model: Done')
    # create a trainer instance
    pl_trainer = pl.Trainer(default_root_dir=paths['model_dir'],
                            accelerator=TRAIN_COMMON_PARAMS["trainer.accelerator"],
                            devices=TRAIN_COMMON_PARAMS["trainer.num_devices"],
                            auto_select_gpus=True)
    predictions = pl_trainer.predict(pl_module, validation_dataloader, return_predictions=True)

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_common_params['infer_filename'])
    

######################################
# Analyze Common Params
######################################
EVAL_COMMON_PARAMS = {}
EVAL_COMMON_PARAMS['infer_filename'] = INFER_COMMON_PARAMS['infer_filename']


######################################
# Eval Template
######################################
def run_eval(paths: dict, eval_common_params: dict):
    create_dir(paths["eval_dir"])
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)

    print('Fuse Eval')

    # metrics
    class_names = [str(i) for i in range(10)]

    metrics = OrderedDict([
        ('operation_point', MetricApplyThresholds(pred='model.output.classification')), # will apply argmax
        ('accuracy', MetricAccuracy(pred='results:metrics.operation_point.cls_pred', target='data.label')),
        ('roc', MetricROCCurve(pred='model.output.classification', target='data.label', class_names=class_names, output_filename=os.path.join(paths['eval_dir'], 'roc_curve.png'))),
        ('auc', MetricAUCROC(pred='model.output.classification', target='data.label', class_names=class_names)),
    ])
   
    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(ids=None,
                     data=eval_common_params["infer_filename"],
                     metrics=metrics,
                     output_dir=paths['eval_dir'])

    return results


######################################
# Run
######################################
if __name__ == "__main__":
    RUNNING_MODES = ['train', 'infer', 'eval']  # Options: 'train', 'infer', 'eval'
    # train
    if 'train' in RUNNING_MODES:
        run_train(paths=PATHS, train_params=TRAIN_COMMON_PARAMS)

    # infer
    if 'infer' in RUNNING_MODES:
        run_infer(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS)

    # eval
    if 'eval' in RUNNING_MODES:
        run_eval(paths=PATHS, eval_common_params=EVAL_COMMON_PARAMS)
