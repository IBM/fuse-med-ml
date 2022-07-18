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
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn

import fuse.utils.gpu as FuseUtilsGPU
from fuse.utils.utils_debug import FuseUtilsDebug
from fuse.utils.utils_logger import fuse_logger_start
from fuse.managers.callbacks.callback_tensorboard import FuseTensorboardCallback
from fuse.managers.callbacks.callback_metric_statistics import FuseMetricStatisticsCallback
from fuse.managers.callbacks.callback_time_statistics import FuseTimeStatisticsCallback
from fuse.managers.manager_default import FuseManagerDefault
from fuse.models.backbones.backbone_inception_resnet_v2 import FuseBackboneInceptionResnetV2
from fuse.models.backbones.backbone_mlp import FuseMultilayerPerceptronBackbone
from fuse.data.sampler.sampler_balanced_batch import FuseSamplerBalancedBatch
from fuse.eval.evaluator import EvaluatorDefault
from fuse.data.processor.processor_dataframe import FuseProcessorDataFrame

from fuse_examples.classification.cmmd.input_processor import FuseMGInputProcessor
from fuse_examples.classification.multimodality.mg_dataset_clinical_and_annotations import PostProcessing,tabular_feature_mg
from fuse_examples.classification.multimodality.multimodel_parameters import multimodal_parameters
from fuse_examples.classification.multimodality.mg_dataset_clinical_and_annotations import mg_clinical_annotations_dataset
from fuse_examples.classification.multimodality.multimodal_paths import multimodal_paths
from fuse_examples.classification.multimodality.model_tabular_imaging import project_imaging, project_tabular




##########################################
# Debug modes
##########################################
mode = 'default'  # Options: 'default', 'fast', 'debug', 'verbose', 'user'. See details in FuseUtilsDebug
debug = FuseUtilsDebug(mode)

##########################################
# Train Common Params
##########################################
# ============
# Data
# ============
TRAIN_COMMON_PARAMS = {}

TRAIN_COMMON_PARAMS['data.train_num_workers'] = 8
TRAIN_COMMON_PARAMS['data.validation_num_workers'] = 8

##########################################
# Dataset
##########################################
dataset_name = 'mg_radiologic'
root = ''
root_data = '/projects/msieve_dev3/usr/Tal/my_research/multi-modality/mg/'  # TODO: add path to the data folder
assert root_data is not None, "Error: please set root_data, the path to the stored MM dataset location"
# Name of the experiment
experiment = 'late_fusion2'
# Path to cache data
cache_path = root_data+'/mg_radiologic/'

paths = multimodal_paths(dataset_name, root_data, root, experiment, cache_path)
TRAIN_COMMON_PARAMS['paths'] = paths
TRAIN_COMMON_PARAMS['fusion_type'] = 'late_fusion'
######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS['infer_filename'] = os.path.join(TRAIN_COMMON_PARAMS['paths']['inference_dir'],'validation_set_infer.gz')
INFER_COMMON_PARAMS['checkpoint'] = 'best'  # Fuse TIP: possible values are 'best', 'last' or epoch_index.
INFER_COMMON_PARAMS['data.train_num_workers'] = TRAIN_COMMON_PARAMS['data.train_num_workers']
INFER_COMMON_PARAMS['output_keys'] = ['model.output.multimodal','data.gt']
INFER_COMMON_PARAMS['model_dir'] = TRAIN_COMMON_PARAMS['paths']['model_dir']
# Analyze Common Params
######################################
ANALYZE_COMMON_PARAMS = {}
ANALYZE_COMMON_PARAMS['infer_filename'] = INFER_COMMON_PARAMS['infer_filename']
ANALYZE_COMMON_PARAMS['output_filename'] = os.path.join(TRAIN_COMMON_PARAMS['paths']['inference_dir'],'all_metrics.txt')
ANALYZE_COMMON_PARAMS['num_workers'] = 4
ANALYZE_COMMON_PARAMS['batch_size'] = 8

# ===============
# Manager - Train
# ===============
NUM_GPUS = 2
TRAIN_COMMON_PARAMS['data.batch_size'] = 2 * NUM_GPUS
TRAIN_COMMON_PARAMS['manager.train_params'] = {
    'num_gpus': NUM_GPUS,
    'num_epochs': 300,
    'virtual_batch_size': 1,  # number of batches in one virtual batch
    'start_saving_epochs': 10,  # first epoch to start saving checkpoints from
    'gap_between_saving_epochs': 5,  # number of epochs between saved checkpoint
}

# best_epoch_source
# if an epoch values are the best so far, the epoch is saved as a checkpoint.
TRAIN_COMMON_PARAMS['manager.best_epoch_source'] = {
    'source':'metrics.auc',#'losses.cls_loss',# 'metrics.auc.macro_avg',  # can be any key from losses or metrics dictionaries
    'optimization': 'max',  # can be either min/max
    'on_equal_values': 'better',
    # can be either better/worse - whether to consider best epoch when values are equal
}

TRAIN_COMMON_PARAMS['manager.resume_checkpoint_filename'] = None



#define postprocessing function
features_dic = tabular_feature_mg()
TRAIN_COMMON_PARAMS['post_processing'] = PostProcessing(features_dic['continuous_clinical_feat'],
                                                        features_dic['categorical_clinical_feat'],
                                                        ['gt'],
                                                        features_dic['annotated_feat'],
                                                        features_dic['non_annotated_feat'],
                                                        use_annotated=True, use_non_annotated=True)

#define processors
TRAIN_COMMON_PARAMS['imaging_processor'] = FuseMGInputProcessor
TRAIN_COMMON_PARAMS['tabular_processor'] = FuseProcessorDataFrame

#define encoders
TRAIN_COMMON_PARAMS['imaging_feature_size'] = 384
TRAIN_COMMON_PARAMS['tabular_feature_size'] = 256
TRAIN_COMMON_PARAMS['tabular_encoder_categorical'] = FuseMultilayerPerceptronBackbone(
                                                       layers=[128, len(features_dic['categorical_clinical_feat'])],
                                                       mlp_input_size=len(features_dic['categorical_clinical_feat']))
TRAIN_COMMON_PARAMS['tabular_encoder_continuous'] = None
TRAIN_COMMON_PARAMS['tabular_encoder_cat'] = FuseMultilayerPerceptronBackbone(
                                                       layers=[TRAIN_COMMON_PARAMS['tabular_feature_size']],
                                                       mlp_input_size=len(features_dic['categorical_clinical_feat'])+\
                                                       len(features_dic['continuous_clinical_feat']))

TRAIN_COMMON_PARAMS['imaging_encoder'] = FuseBackboneInceptionResnetV2(input_channels_num=1)
TRAIN_COMMON_PARAMS['imaging_projector'] = project_imaging(projection_imaging=nn.Conv2d(TRAIN_COMMON_PARAMS['imaging_feature_size'], TRAIN_COMMON_PARAMS['tabular_feature_size'], kernel_size=1, stride=1))
TRAIN_COMMON_PARAMS['tabular_projector'] = None

TRAIN_COMMON_PARAMS['dataset_func'] = mg_clinical_annotations_dataset(
                                                          tabular_filename=TRAIN_COMMON_PARAMS['paths']['tabular_filename'],
                                                          imaging_filename=TRAIN_COMMON_PARAMS['paths']['imaging_filename'],
                                                          train_val_test_filenames=TRAIN_COMMON_PARAMS['paths']['train_val_test_filenames'],
                                                          key_columns=TRAIN_COMMON_PARAMS['paths']['key_columns'],
                                                          label_key=TRAIN_COMMON_PARAMS['paths']['label_key'],
                                                          img_key=TRAIN_COMMON_PARAMS['paths']['img_key'],
                                                          sample_key=TRAIN_COMMON_PARAMS['paths']['sample_key'],

                                                          imaging_processor=TRAIN_COMMON_PARAMS['imaging_processor'],
                                                          tabular_processor=TRAIN_COMMON_PARAMS['tabular_processor'],

                                                          cache_dir=TRAIN_COMMON_PARAMS['paths']['cache_dir'],
                                                          reset_cache=False,
                                                          post_cache_processing_func=TRAIN_COMMON_PARAMS['post_processing'],
                                                          )

TRAIN_COMMON_PARAMS,INFER_COMMON_PARAMS,ANALYZE_COMMON_PARAMS = multimodal_parameters(TRAIN_COMMON_PARAMS,INFER_COMMON_PARAMS,ANALYZE_COMMON_PARAMS)

#################################
# Train Template
#################################
def run_train(paths: dict, train_common_params: dict, reset_cache: bool):
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths['model_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')

    # Download data
    # TBD

    lgr.info('\nFuse Train', {'attrs': ['bold', 'underline']})

    lgr.info(f'model_dir={paths["model_dir"]}', {'color': 'magenta'})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {'color': 'magenta'})

    #### Train Data
    lgr.info(f'Train Data:', {'attrs': 'bold'})
    #Mo: function that returns dataset datasetfun in train_params, kwargs
    train_dataset, validation_dataset, _ = train_common_params['dataset_func']


    ## Create sampler
    lgr.info(f'- Create sampler:')
    sampler = FuseSamplerBalancedBatch(dataset=train_dataset,
                                       balanced_class_name='data.gt',
                                       num_balanced_classes=2,
                                       batch_size=train_common_params['data.batch_size'],
                                       use_dataset_cache=True)

    lgr.info(f'- Create sampler: Done')

    ## Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  shuffle=False, drop_last=False,
                                  batch_sampler=sampler, collate_fn=train_dataset.collate_fn,
                                  num_workers=train_common_params['data.train_num_workers'])
    lgr.info(f'Train Data: Done', {'attrs': 'bold'})

    #### Validation data
    lgr.info(f'Validation Data:', {'attrs': 'bold'})

    ## Create dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       shuffle=False,
                                       drop_last=False,
                                       batch_sampler=None,
                                       batch_size=train_common_params['data.batch_size'],
                                       num_workers=train_common_params['data.validation_num_workers'],
                                       collate_fn=validation_dataset.collate_fn)
    lgr.info(f'Validation Data: Done', {'attrs': 'bold'})


    # ===================================================================
    # ==============================================================================
    # Model
    # ==============================================================================
    lgr.info('Model:', {'attrs': 'bold'})

    model = train_common_params['model']

    lgr.info('Model: Done', {'attrs': 'bold'})

    # ====================================================================================
    #  Loss
    # ====================================================================================

    losses = train_common_params['loss']

    # ====================================================================================
    # Metrics
    # ====================================================================================

    metrics=train_common_params['metrics']

    # =====================================================================================
    #  Callbacks
    # =====================================================================================
    callbacks = [
        # default callbacks
        FuseTensorboardCallback(model_dir=paths['model_dir']),  # save statistics for tensorboard
        FuseMetricStatisticsCallback(output_path=paths['model_dir'] + "/metrics.csv"),  # save statistics in a csv file
        FuseTimeStatisticsCallback(num_epochs=train_common_params['manager.train_params']['num_epochs'],
                                   load_expected_part=0.1)  # time profiler
    ]

    # =====================================================================================
    #  Manager - Train
    #  Create a manager, training objects and run a training process.
    # =====================================================================================
    lgr.info('Train:', {'attrs': 'bold'})

    # create optimizer
    optimizer = TRAIN_COMMON_PARAMS['optimizer']
    # create scheduler
    scheduler = TRAIN_COMMON_PARAMS['scheduler']

    # train from scratch
    manager = FuseManagerDefault(output_model_dir=paths['model_dir'], force_reset=paths['force_reset_model_dir'])
    # Providing the objects required for the training process.
    manager.set_objects(net=model,
                        optimizer=optimizer,
                        losses=losses,
                        metrics=metrics,
                        best_epoch_source=train_common_params['manager.best_epoch_source'],
                        lr_scheduler=scheduler,
                        callbacks=callbacks,
                        train_params=train_common_params['manager.train_params'],
                        output_model_dir=paths['model_dir'])

    # Continue training
    if train_common_params['manager.resume_checkpoint_filename'] is not None:
        # Loading the checkpoint including model weights, learning rate, and epoch_index.
        manager.load_checkpoint(checkpoint=train_common_params['manager.resume_checkpoint_filename'], mode='train',
                                values_to_resume=['net'])
    # # Start training
    manager.train(train_dataloader=train_dataloader,
                  validation_dataloader=validation_dataloader)

    lgr.info('Train: Done', {'attrs': 'bold'})


######################################
# Inference Template
######################################
def run_infer(paths: dict, train_common_params: dict,infer_common_params: dict):
    #### Logger
    fuse_logger_start(output_path=paths['inference_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Inference', {'attrs': ['bold', 'underline']})
    lgr.info(f'infer_filename={os.path.join(paths["inference_dir"], infer_common_params["infer_filename"])}',
             {'color': 'magenta'})

    # Create data source:
    _, val_dataset, test_dataset = train_common_params['dataset_func']

    ## Create dataloader
    infer_dataloader = DataLoader(dataset=val_dataset,
                                  shuffle=False, drop_last=False,
                                  collate_fn=test_dataset.collate_fn,
                                  num_workers=infer_common_params['data.train_num_workers'])
    lgr.info(f'Test Data: Done', {'attrs': 'bold'})

    #### Manager for inference
    manager = FuseManagerDefault()
    # extract just the global classification per sample and save to a file
    output_columns = infer_common_params['output_keys']
    manager.infer(data_loader=infer_dataloader,
                  input_model_dir=infer_common_params['model_dir'],
                  checkpoint=infer_common_params['checkpoint'],
                  output_columns=output_columns,
                  output_file_name=os.path.join(paths["inference_dir"], infer_common_params["infer_filename"]))

    ######################################



######################################
# Analyze Template
######################################
def run_analyze(paths: dict, analyze_common_params: dict):
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')
    lgr.info('Fuse Analyze', {'attrs': ['bold', 'underline']})

    # metrics
    metrics = analyze_common_params['metrics']


    # create analyzer
    analyzer = EvaluatorDefault()

    # run
    # FIXME: simplify analyze interface for this case
    results = analyzer.eval(ids=None,
                               data=os.path.join(paths["inference_dir"],
                                                                 analyze_common_params["infer_filename"]),
                               metrics=metrics,
                               output_dir=analyze_common_params['output_filename'])

    return results


######################################
# Run
######################################


if __name__ == "__main__":
    # allocate gpus
    if NUM_GPUS == 0:
        TRAIN_COMMON_PARAMS['manager.train_params']['device'] = 'cpu'
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    FuseUtilsGPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    RUNNING_MODES = ['train','infer','analyze']#['train', 'infer', 'analyze']  # Options: 'train', 'infer', 'analyze'


    paths = TRAIN_COMMON_PARAMS['paths']
    # train
    if 'train' in RUNNING_MODES:
        run_train(paths=paths, train_common_params=TRAIN_COMMON_PARAMS, reset_cache=False)

    # infer
    if 'infer' in RUNNING_MODES:
        run_infer(paths=paths, train_common_params=TRAIN_COMMON_PARAMS,infer_common_params=INFER_COMMON_PARAMS)
    #
    # analyze
    if 'analyze' in RUNNING_MODES:
        run_analyze(paths=paths, analyze_common_params=ANALYZE_COMMON_PARAMS)
