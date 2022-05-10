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
import logging
from glob import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from pathlib import Path
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from fuse.data.augmentor.augmentor_toolbox import aug_op_affine_group, aug_op_affine, aug_op_color, aug_op_gaussian, aug_op_elastic_transform
# from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerUniform as Uniform
# from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandBool as RandBool
# from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandInt as RandInt
# from fuse.utils.utils_gpu import FuseUtilsGPU
import fuse.utils.gpu as FuseUtilsGPU
from fuse.utils.utils_logger import fuse_logger_start
# from fuse.data.augmentor.augmentor_default import FuseAugmentorDefault
# from fuse.data.visualizer.visualizer_default import FuseVisualizerDefault
# from fuse.data.dataset.dataset_default import FuseDatasetDefault
from fuse.dl.models.model_wrapper import ModelWrapper
from fuse.dl.losses.segmentation.loss_dice import DiceBCELoss
from fuse.dl.losses.segmentation.loss_dice import FuseDiceLoss
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.managers.manager_default import ManagerDefault
from fuse.dl.managers.callbacks.callback_tensorboard import TensorboardCallback
from fuse.dl.managers.callbacks.callback_metric_statistics import MetricStatisticsCallback
from fuse.dl.managers.callbacks.callback_time_statistics import TimeStatisticsCallback
# from fuse.dl.data.processor.processor_dataframe import FuseProcessorDataFrame
from fuse.eval.evaluator import EvaluatorDefault
from fuse.eval.metrics.segmentation.metrics_segmentation_common import MetricDice, MetricIouJaccard, MetricOverlap, Metric2DHausdorff, MetricPixelAccuracy
from fuse.utils.utils_debug import FuseDebug

from data_source_segmentation import get_data_sample_ids # FuseDataSourceSeg
from seg_input_processor import SegInputProcessor
from image_mask_loader import OpImageMaskLoader

from unet import UNet

# fuse2 imports
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.ops.op_base import OpBase
from fuse.data.ops.ops_aug_common import OpSample
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.ops.ops_common import OpLambda
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data import PipelineDefault, OpSampleAndRepeat, OpToTensor, OpRepeat
from fuse.utils.rand.param_sampler import RandBool, RandInt, Uniform
import torch
import numpy as np
from functools import partial
from tempfile import mkdtemp

import os
from fuse.data.ops.ops_cast import OpToTensor
from fuse.utils.ndict import NDict
from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.color import OpClip, OpToRange
from fuseimg.data.ops.aug.color import OpAugColor
from fuseimg.data.ops.aug.geometry import OpAugAffine2D

from fuseimg.datasets.kits21 import OpKits21SampleIDDecode, KITS21
##########################################
# Debug modes
##########################################
mode = 'default'  # Options: 'default', 'fast', 'debug', 'verbose', 'user'. See details in FuseUtilsDebug
debug = FuseDebug(mode)

##########################################
# Output and data Paths
##########################################

# # TODO: path to save model
ROOT = '../results/'

# TODO: path for siim data
# Download instructions can be found in README
DATA_ROOT = '../siim/'

# TODO: Name of the experiment
EXPERIMENT = 'unet_seg_results'
# TODO: Path to cache data
CACHE_PATH = '../results/'
# TODO: Name of the cached data folder
EXPERIMENT_CACHE = 'exp_cache'

PATHS = {#'data_dir': [TRAIN, MASKS, TEST],
         'train_rle_file': os.path.join(DATA_ROOT, 'train-rle.csv'),
         'train_folder': os.path.join(DATA_ROOT, 'dicom-images-train'),
         'test_folder': os.path.join(DATA_ROOT, 'dicom-images-test'),
         'model_dir': os.path.join(ROOT, EXPERIMENT, 'model_dir'),
         'force_reset_model_dir': True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
         'cache_dir': os.path.join(CACHE_PATH, EXPERIMENT_CACHE+'_cache_dir'),
         'inference_dir': os.path.join(ROOT, EXPERIMENT, 'infer_dir'),
         'eval_dir': os.path.join(ROOT, EXPERIMENT, 'eval_dir')}

##########################################
# Train Common Params
##########################################
# ============
# Data
# ============
TRAIN_COMMON_PARAMS = {}
TRAIN_COMMON_PARAMS['data.image_size'] = 512
TRAIN_COMMON_PARAMS['data.batch_size'] = 8
TRAIN_COMMON_PARAMS['data.train_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.validation_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.augmentation_pipeline'] = [
    [
        ('data.input.input_0','data.gt.gt_global'),
        aug_op_affine_group,
        {'rotate': Uniform(-20.0, 20.0),  
        'flip': (RandBool(0.0), RandBool(0.5)),  # only flip right-to-left
        'scale': Uniform(0.9, 1.1),
        'translate': (RandInt(-50, 50), RandInt(-50, 50))},
        {'apply': RandBool(0.9)}
    ],
    [
        ('data.input.input_0','data.gt.gt_global'),
        aug_op_elastic_transform,
        {'sigma': 7,
         'num_points': 3},
        {'apply': RandBool(0.7)}
    ],
    [
        ('data.input.input_0',),
        aug_op_color,
        {
         'add': Uniform(-0.06, 0.06), 
         'mul': Uniform(0.95, 1.05), 
         'gamma': Uniform(0.9, 1.1),
         'contrast': Uniform(0.85, 1.15)
        },
        {'apply': RandBool(0.7)}
    ],
    [
        ('data.input.input_0',),
        aug_op_gaussian,
        {'std': 0.05},
        {'apply': RandBool(0.7)}
    ],
]

# ===============
# Manager - Train1
# ===============
TRAIN_COMMON_PARAMS['manager.train_params'] = {
    'num_epochs': 50,
    'virtual_batch_size': 1,  # number of batches in one virtual batch
    'start_saving_epochs': 10,  # first epoch to start saving checkpoints from
    'gap_between_saving_epochs': 5,  # number of epochs between saved checkpoint
}
TRAIN_COMMON_PARAMS['manager.best_epoch_source'] = {
    'source': 'losses.total_loss',  # can be any key from 'epoch_results' (either metrics or losses result)
    'optimization': 'min',  # can be either min/max
}
TRAIN_COMMON_PARAMS['manager.learning_rate'] = 1e-2
TRAIN_COMMON_PARAMS['manager.weight_decay'] = 1e-4  
TRAIN_COMMON_PARAMS['manager.resume_checkpoint_filename'] = None  # if not None, will try to load the checkpoint
TRAIN_COMMON_PARAMS['partition_file'] = 'train_val_split.pickle'

#################################
# Train Template
#################################
def run_train(paths: dict, train_common_params: dict):
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths['model_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')

    lgr.info('\nFuse Train', {'attrs': ['bold', 'underline']})

    lgr.info(f'model_dir={paths["model_dir"]}', {'color': 'magenta'})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {'color': 'magenta'})

    #### Train Data
    lgr.info(f'Train Data:', {'attrs': 'bold'})

    train_sample_ids = get_data_sample_ids(phase='train',
                                          data_folder=paths['train_folder'],
                                          partition_file=train_common_params['partition_file'])

    static_pipeline = PipelineDefault("static", [
        (OpImageMaskLoader(size=train_common_params['data.image_size']), 
            dict(key_in="data.input.img_path", key_out="data.input.img")),
        (OpImageMaskLoader(size=train_common_params['data.image_size'], 
                           data_csv=paths['train_rle_file']), 
            dict(key_in="data.gt.seg_path", key_out="data.gt.seg")),
        ])


    # cache_dir = mkdtemp(prefix="kits_21")
    cacher = SamplesCacher('siim_cache', 
                           static_pipeline,
                           cache_dirs=[paths['cache_dir']], 
                           restart_cache=True)   

    my_dataset = DatasetDefault(sample_ids=train_sample_ids[:5],
                                static_pipeline=static_pipeline,
                                dynamic_pipeline=None,
                                cacher=cacher)            
    my_dataset.create()

    ## Create data processors:
    input_processors = {
        'input_0': SegInputProcessor(name='image',
                                     size=train_common_params['data.image_size'])
    }
    gt_processors = {
        'gt_global': SegInputProcessor(name='mask', 
                                       data_csv=paths['train_rle_file'],
                                       size=train_common_params['data.image_size'])
    }

    ## Create data augmentation (optional)
    # augmentor = FuseAugmentorDefault(augmentation_pipeline=train_common_params['data.augmentation_pipeline'])
    augmentor = []

    # Create visualizer (optional)
    # visualiser = FuseVisualizerDefault(image_name='data.input.input_0', 
    #                                    mask_name='data.gt.gt_global',
    #                                    pred_name='model.logits.segmentation')
    visualiser = []

    train_dataset = FuseDatasetDefault(cache_dest=paths['cache_dir'],
                                       data_source=train_data_source,
                                       input_processors=input_processors,
                                       gt_processors=gt_processors,
                                       augmentor=augmentor,
                                       visualizer=visualiser)

    lgr.info(f'- Load and cache data:')
    train_dataset.create()
    lgr.info(f'- Load and cache data: Done')

    ## Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  shuffle=True, 
                                  drop_last=False,
                                  batch_size=train_common_params['data.batch_size'],
                                  collate_fn=train_dataset.collate_fn,
                                  num_workers=train_common_params['data.train_num_workers'])
    lgr.info(f'Train Data: Done', {'attrs': 'bold'})
    # ==================================================================
    # Validation dataset
    lgr.info(f'Validation Data:', {'attrs': 'bold'})

    valid_data_source = FuseDataSourceSeg(phase='validation',
                                          data_folder=paths['train_folder'],
                                          partition_file=train_common_params['partition_file'])
    print(valid_data_source.summary())

    valid_dataset = FuseDatasetDefault(cache_dest=paths['cache_dir'],
                                       data_source=valid_data_source,
                                       input_processors=input_processors,
                                       gt_processors=gt_processors,
                                       visualizer=visualiser)

    lgr.info(f'- Load and cache data:')
    valid_dataset.create()
    lgr.info(f'- Load and cache data: Done')

    ## Create dataloader
    validation_dataloader = DataLoader(dataset=valid_dataset,
                                       shuffle=False, 
                                       drop_last=False,
                                       batch_size=train_common_params['data.batch_size'],
                                       collate_fn=valid_dataset.collate_fn,
                                       num_workers=train_common_params['data.validation_num_workers'])

    lgr.info(f'Validation Data: Done', {'attrs': 'bold'})
    # ==================================================================
    # # Training graph
    lgr.info('Model:', {'attrs': 'bold'})
    torch_model = UNet(n_channels=1, n_classes=1, bilinear=False)

    model = FuseModelWrapper(model=torch_model,
                            model_inputs=['data.input.input_0'],
                            model_outputs=['logits.segmentation']
                            )

    lgr.info('Model: Done', {'attrs': 'bold'})
    # ====================================================================================
    #  Loss
    # ====================================================================================
    losses = {
        'dice_loss': DiceBCELoss(pred_name='model.logits.segmentation', 
                                 target_name='data.gt.gt_global')
    }

    optimizer = optim.SGD(model.parameters(), 
                          lr=train_common_params['manager.learning_rate'],
                          momentum=0.9,
                          weight_decay=train_common_params['manager.weight_decay'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # train from scratch
    manager = FuseManagerDefault(output_model_dir=paths['model_dir'], 
                                force_reset=paths['force_reset_model_dir'])

    # =====================================================================================
    #  Callbacks
    # =====================================================================================
    callbacks = [
        # default callbacks
        # FuseTensorboardCallback(model_dir=paths['model_dir']),  # save statistics for tensorboard
        FuseMetricStatisticsCallback(output_path=paths['model_dir'] + "/metrics.csv"),  # save statistics in a csv file
        FuseTimeStatisticsCallback(num_epochs=train_common_params['manager.train_params']['num_epochs'], load_expected_part=0.1)  # time profiler
    ]

    # Providing the objects required for the training process.
    manager.set_objects(net=model,
                        optimizer=optimizer,
                        losses=losses,
                        lr_scheduler=scheduler,
                        callbacks=callbacks,
                        best_epoch_source=train_common_params['manager.best_epoch_source'],
                        train_params=train_common_params['manager.train_params'],
                        output_model_dir=paths['model_dir'])

    manager.train(train_dataloader=train_dataloader,
                  validation_dataloader=validation_dataloader)
    lgr.info('Train: Done', {'attrs': 'bold'})


######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS['infer_filename'] = os.path.join(PATHS['inference_dir'], 'validation_set_infer.gz')
INFER_COMMON_PARAMS['checkpoint'] = 'last'  # Fuse TIP: possible values are 'best', 'last' or epoch_index.
INFER_COMMON_PARAMS['data.train_num_workers'] = TRAIN_COMMON_PARAMS['data.train_num_workers']
INFER_COMMON_PARAMS['partition_file'] = TRAIN_COMMON_PARAMS['partition_file']
INFER_COMMON_PARAMS['data.image_size'] = TRAIN_COMMON_PARAMS['data.image_size']
INFER_COMMON_PARAMS['data.batch_size'] = TRAIN_COMMON_PARAMS['data.batch_size']

######################################
# Inference Template
######################################
def run_infer(paths: dict, infer_common_params: dict):
    #### Logger
    fuse_logger_start(output_path=paths['inference_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})
    lgr.info(f'infer_filename={os.path.join(paths["inference_dir"], infer_common_params["infer_filename"])}', {'color': 'magenta'})

    # ==================================================================
    # Validation dataset
    lgr.info(f'Test Data:', {'attrs': 'bold'})

    infer_data_source = FuseDataSourceSeg(phase='validation',
                                          data_folder=paths['train_folder'],
                                          partition_file=infer_common_params['partition_file'])
    print(infer_data_source.summary())

    ## Create data processors:
    input_processors = {
        'input_0': SegInputProcessor(name='image',
                                     size=infer_common_params['data.image_size'])
    }
    gt_processors = {
        'gt_global': SegInputProcessor(name='mask', 
                                       data_csv=paths['train_rle_file'],
                                       size=infer_common_params['data.image_size'])
    }

    # Create visualizer (optional)
    visualiser = FuseVisualizerDefault(image_name='data.input.input_0', 
                                       mask_name='data.gt.gt_global',
                                       pred_name='model.logits.segmentation')

    infer_dataset = FuseDatasetDefault(cache_dest=paths['cache_dir'],
                                       data_source=infer_data_source,
                                       input_processors=input_processors,
                                       gt_processors=gt_processors,
                                       visualizer=visualiser)

    lgr.info(f'- Load and cache data:')
    infer_dataset.create()
    lgr.info(f'- Load and cache data: Done')

    ## Create dataloader
    infer_dataloader = DataLoader(dataset=infer_dataset,
                                  shuffle=False, 
                                  drop_last=False,
                                  batch_size=infer_common_params['data.batch_size'],
                                  collate_fn=infer_dataset.collate_fn,
                                  num_workers=infer_common_params['data.train_num_workers'])

    lgr.info(f'Test Data: Done', {'attrs': 'bold'})

    #### Manager for inference
    manager = FuseManagerDefault()
    # extract just the global segmentation per sample and save to a file
    output_columns = ['model.logits.segmentation', 'data.gt.gt_global']
    manager.infer(data_loader=infer_dataloader,
                  input_model_dir=paths['model_dir'],
                  checkpoint=infer_common_params['checkpoint'],
                  output_columns=output_columns,
                  output_file_name=infer_common_params["infer_filename"])

    # visualize the predictions
    infer_processor = FuseProcessorDataFrame(data_pickle_filename=infer_common_params['infer_filename'])
    descriptors_list = infer_processor.get_samples_descriptors()
    out_name = 'model.logits.segmentation'
    gt_name = 'data.gt.gt_global' 
    for desc in descriptors_list[:10]:
        data = infer_processor(desc)
        pred = np.squeeze(data[out_name])
        gt = np.squeeze(data[gt_name])
        _, ax = plt.subplots(1,2)
        ax[0].imshow(pred)
        ax[0].set_title('prediction')
        ax[1].imshow(gt)
        ax[1].set_title('gt')
        fn = os.path.join(paths["inference_dir"], Path(desc[0]).name)
        plt.savefig(fn)

######################################
# Evaluation Common Params
######################################
EVAL_COMMON_PARAMS = {}
EVAL_COMMON_PARAMS['infer_filename'] = INFER_COMMON_PARAMS['infer_filename']
EVAL_COMMON_PARAMS['output_filename'] = os.path.join(PATHS['eval_dir'], 'all_metrics.txt')
EVAL_COMMON_PARAMS['num_workers'] = 4
EVAL_COMMON_PARAMS['batch_size'] = 8

######################################
# Analyze Template
######################################
def run_eval(paths: dict, eval_common_params: dict):
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse eval', {'attrs': ['bold', 'underline']})

    # define iterator
    def data_iter():
        data = pd.read_pickle(eval_common_params['infer_filename'])
        n_samples = data.shape[0]
        threshold = 1e-7 #0.5
        for inx in range(n_samples):
            row = data.loc[inx]
            sample_dict = {}
            sample_dict["id"] = row['id']
            sample_dict["pred.array"] = row['model.logits.segmentation'] > threshold
            sample_dict["label.array"] = row['data.gt.gt_global']
            yield sample_dict

    metrics = OrderedDict([
            ("dice", MetricDice(pred='pred.array', target='label.array')),
            ("IOU", MetricIouJaccard(pred='pred.array', target='label.array')),
            ("Overlap", MetricOverlap(pred='pred.array', target='label.array')),
            ("PixelAcc", MetricPixelAccuracy(pred='pred.array', target='label.array')),
    ])

    # create evaluator
    evaluator = EvaluatorDefault()

    results = evaluator.eval(ids=None, 
                             data=data_iter(),
                             batch_size=1,
                             metrics=metrics) 


######################################
# Run
######################################
if __name__ == "__main__":
    # allocate gpus
    NUM_GPUS = 0
    if NUM_GPUS == 0:
        TRAIN_COMMON_PARAMS['manager.train_params']['device'] = 'cpu'
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    FuseUtilsGPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

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
