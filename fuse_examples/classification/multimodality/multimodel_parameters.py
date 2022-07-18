import torch.optim as optim
from fuse_examples.classification.multimodality.model_tabular_imaging import *
from fuse.losses.loss_default import FuseLossDefault
import torch.nn.functional as F
from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC
from fuse_examples.classification.multimodality.loss_multimodal_contrastive_learning import FuseLossMultimodalContrastiveLearning
from fuse.models.heads.head_global_pooling_classifier import FuseHeadGlobalPoolingClassifier
from fuse.models.heads.head_1d_classifier import FuseHead1dClassifier
from fuse.models.model_ensemble import FuseModelEnsemble
from fuse.models.heads.head_3D_classifier import FuseHead3dClassifier

def multimodal_parameters(train_common_params: dict,infer_common_params: dict,analyze_common_params: dict):

    num_classes = train_common_params['num_classes']
    target_metric = train_common_params['target_metric'].replace('metrics.','')
    ################################################
    # backbone_models
    model_tabular = FuseModelTabularContinuousCategorical(
        continuous_tabular_input=(('data.continuous', 1),),
        categorical_tabular_input=(('data.categorical', 1),),
        backbone_categorical_tabular=train_common_params['tabular_encoder_categorical'],
        backbone_continuous_tabular=train_common_params['tabular_encoder_continuous'],
        backbone_cat_tabular=train_common_params['tabular_encoder_cat'],
        heads=None,
        )

    model_imaging = train_common_params['imaging_encoder']
    model_multimodel_concat = TabularImagingConcat()
    model_projection_imaging = train_common_params['imaging_projector']
    model_projection_tabular = train_common_params['tabular_projector']

    model_interactive_3d =  FuseBackboneResnet3DInteractive(
                                                             conv_inputs=(('data.image', 1),),
                                                             fcn_inputs=(('data.input.clinical.all', 1),),
                                                             )

    heads_for_multimodal = {
                         'multimodal_head':
                             [
                                FuseHead1dClassifier(
                                    head_name='multimodal',
                                    conv_inputs=(('model.multimodal_features',
                                                  train_common_params['tabular_feature_size'] * 2),),
                                    num_classes=num_classes,
                                )
                            ],
                         'tabular_head':
                             [
                                 FuseHead1dClassifier(
                                     head_name='tabular',
                                     conv_inputs=(('model.tabular_features',
                                                   train_common_params['tabular_feature_size']),),
                                     num_classes=num_classes,
                                 )
                             ],
                         'imaging_head':
                             [
                                FuseHeadGlobalPoolingClassifier(
                                    head_name='imaging',
                                    dropout_rate=0.5,
                                    layers_description=(256,),
                                    conv_inputs=(('model.imaging_features',
                                                  train_common_params['imaging_feature_size']),),
                                    num_classes=num_classes,
                                    pooling="avg",
                                )
                            ],

                        'imaging_head_3d':
                            [
                                FuseHead3dClassifier(
                                        head_name='imaging',
                                        dropout_rate=0.5,
                                        layers_description=(256,),
                                        conv_inputs=(('model.imaging_features',
                                                      train_common_params['imaging_feature_size']),),
                                        num_classes=num_classes,
                                    )
                            ],

                        'interactive_head_3d':
                            [
                                FuseHead3dClassifier(
                                    head_name='interactive',
                                    dropout_rate=0.5,
                                    layers_description=(256,),
                                    conv_inputs=(('model.backbone_features',
                                                  train_common_params['imaging_feature_size']),),
                                    num_classes=num_classes,
                                )
                            ],


                        }
    loss_for_multimodal = {
                        'multimodal_loss':FuseLossDefault(pred_name='model.logits.multimodal', target_name='data.gt',
                                                                        callable=F.cross_entropy, weight=1.0),
                        'tabular_loss':FuseLossDefault(pred_name='model.logits.tabular', target_name='data.gt',
                                                                        callable=F.cross_entropy, weight=1.0),
                        'imaging_loss':FuseLossDefault(pred_name='model.logits.imaging', target_name='data.gt',
                                                                        callable=F.cross_entropy, weight=1.0),
                        'interactive_loss': FuseLossDefault(pred_name='model.logits.interactive', target_name='data.gt',
                                                        callable=F.cross_entropy, weight=1.0),
                        'ensemble_loss':FuseLossDefault(pred_name='model.output.tabular_ensemble_average', target_name='data.gt',
                                                                        callable=F.nll_loss, weight=1.0,reduction='sum'),

                        }
    metric_for_multimodal = {
                        'multimodal_auc': MetricAUCROC(pred='model.output.multimodal', target='data.gt'),
                        'tabular_auc': MetricAUCROC(pred='model.output.tabular', target='data.gt'),
                        'imaging_auc': MetricAUCROC(pred='model.output.imaging', target='data.gt'),
                        'interactive_auc': MetricAUCROC(pred='model.output.interactive', target='data.gt'),
                        'ensemble_auc':MetricAUCROC(pred='model.output.tabular_ensemble_average', target='data.gt'),
                        }
    ################################################


    if train_common_params['fusion_type'] == 'mono_tabular':

        train_common_params['model'] = FuseMultiModalityModel(
                                            tabular_inputs=(('data.continuous', 1), ('data.categorical', 1),),
                                            tabular_backbone=model_tabular,
                                            heads=heads_for_multimodal['tabular_head'],
                                        )
        train_common_params['loss'] = {
            'cls_loss': loss_for_multimodal['tabular_loss'],
        }
        train_common_params['metrics'] = {
            target_metric: metric_for_multimodal['tabular_auc'],
        }

        train_common_params['manager.learning_rate'] = 1e-4
        train_common_params['manager.weight_decay'] = 1e-4
        train_common_params['manager.momentum'] = 0.9
        train_common_params['manager.step_size'] = 150
        train_common_params['manager.gamma'] = 0.1
        train_common_params['optimizer'] = optim.SGD(train_common_params['model'].parameters(),
                  lr=train_common_params['manager.learning_rate'],
                  momentum=train_common_params['manager.momentum'],
                  weight_decay=train_common_params['manager.weight_decay'])
        train_common_params['scheduler'] = optim.lr_scheduler.StepLR(train_common_params['optimizer'], step_size=train_common_params['manager.step_size'],
                                  gamma=train_common_params['manager.gamma'])
        analyze_common_params['metrics'] =  train_common_params['metrics']
        infer_common_params['output_keys'] = ['model.output.tabular', 'data.gt']

    if train_common_params['fusion_type'] == 'mono_imaging':

        train_common_params['model'] = FuseMultiModalityModel(
                                            imaging_inputs=(('data.image', 1),),
                                            imaging_backbone=model_imaging,
                                            heads=heads_for_multimodal['imaging_head_3d'],
                                        )
        train_common_params['loss'] = {
            'cls_loss': loss_for_multimodal['imaging_loss'],
        }
        train_common_params['metrics'] = {
            target_metric: metric_for_multimodal['imaging_auc'],
        }
        train_common_params['manager.learning_rate'] = 1e-5
        train_common_params['manager.weight_decay'] = 0.001

        train_common_params['optimizer'] = optim.Adam(train_common_params['model'].parameters(), lr=train_common_params['manager.learning_rate'],
                                                         weight_decay=train_common_params['manager.weight_decay'])
        train_common_params['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(train_common_params['optimizer'])
        analyze_common_params['metrics'] =  train_common_params['metrics']
        infer_common_params['output_keys'] = ['model.output.imaging', 'data.gt']

    if train_common_params['fusion_type'] == 'late_fusion':

        train_common_params['model'] = FuseMultiModalityModel(
                                            tabular_inputs=(('data.continuous', 1), ('data.categorical', 1),),
                                            imaging_inputs=(('data.image', 1),),
                                            tabular_backbone=model_tabular,
                                            imaging_backbone=model_imaging,
                                            multimodal_backbone=model_multimodel_concat,
                                            imaging_projection= model_projection_imaging,
                                            tabular_projection=model_projection_tabular,
                                            heads=[heads_for_multimodal['multimodal_head'][0]],
                                        )

        train_common_params['loss'] = {
            'cls_loss': loss_for_multimodal['multimodal_loss'],
        }
        train_common_params['metrics'] = {
            target_metric: metric_for_multimodal['multimodal_auc'],
        }
        train_common_params['manager.learning_rate'] = 1e-4
        train_common_params['manager.weight_decay'] = 1e-4
        train_common_params['manager.momentum'] = 0.9
        train_common_params['manager.step_size'] = 150
        train_common_params['manager.gamma'] = 0.1
        train_common_params['optimizer'] = optim.SGD(train_common_params['model'].parameters(),
                  lr=train_common_params['manager.learning_rate'],
                  momentum=train_common_params['manager.momentum'],
                  weight_decay=train_common_params['manager.weight_decay'])
        train_common_params['scheduler'] = optim.lr_scheduler.StepLR(train_common_params['optimizer'], step_size=train_common_params['manager.step_size'],
                                  gamma=train_common_params['manager.gamma'])

        analyze_common_params['metrics'] =  train_common_params['metrics'] =  {
            target_metric: metric_for_multimodal['multimodal_auc'],
        }
        infer_common_params['output_keys'] = ['model.output.multimodal', 'data.gt']

    if train_common_params['fusion_type'] == 'ensemble':

        train_common_params['tabular_dir'] = '/projects/msieve_dev3/usr/Tal/my_research/multi-modality/knight/mono_tabular/'
        train_common_params['imaging_dir'] = '/projects/msieve_dev3/usr/Tal/my_research/multi-modality/knight/mono_imaging/'
        train_common_params['model'] = FuseModelEnsemble(input_model_dirs=[train_common_params['tabular_dir'],
                                                                           train_common_params['imaging_dir']])
        infer_common_params['model_dir'] = [train_common_params['tabular_dir'],
                                                                           train_common_params['imaging_dir']]

        infer_common_params['output_keys'] = ['data.gt',
                                              'model.output.ensemble_output_0.tabular',
                                              'model.output.ensemble_output_1.tabular',
                                              'model.output.ensemble_output_1.imaging',
                                              'model.output.tabular_ensemble_average',
                                              'model.output.tabular_ensemble_majority_vote']

        analyze_common_params['metrics'] =  train_common_params['metrics'] = {
                                                    target_metric: metric_for_multimodal['ensemble_auc'],
                                                }

    if train_common_params['fusion_type'] == 'contrastive':
        train_common_params['model'] = FuseMultiModalityModel(
                                            tabular_inputs=(('data.continuous', 1), ('data.categorical', 1),),
                                            imaging_inputs=(('data.image', 1),),
                                            tabular_backbone=model_tabular,
                                            imaging_backbone=model_imaging,
                                            multimodal_backbone=None,
                                            imaging_projection= model_projection_imaging,
                                            tabular_projection=model_projection_tabular,
                                        )

        train_common_params['loss'] = {'cls_loss': FuseLossMultimodalContrastiveLearning(
            imaging_representations='model.imaging_features',
            tabular_representations='model.tabular_features',
            label='data.gt',
            temperature=0.1,
            alpha=0.5)
        }
        train_common_params['metrics'] = None
        train_common_params['manager.learning_rate'] = 1e-4
        train_common_params['manager.weight_decay'] = 1e-4
        train_common_params['manager.momentum'] = 0.9
        train_common_params['manager.step_size'] = 150
        train_common_params['manager.gamma'] = 0.1
        train_common_params['optimizer'] = optim.SGD(train_common_params['model'].parameters(),
                  lr=train_common_params['manager.learning_rate'],
                  momentum=train_common_params['manager.momentum'],
                  weight_decay=train_common_params['manager.weight_decay'])
        train_common_params['scheduler'] = optim.lr_scheduler.StepLR(train_common_params['optimizer'], step_size=train_common_params['manager.step_size'],
                                  gamma=train_common_params['manager.gamma'])



    if train_common_params['fusion_type'] == 'interactive':

        train_common_params['model'] = FuseModelDefaultInteractive(backbone=model_interactive_3d,
                                                                    heads=heads_for_multimodal['interactive_head_3d'],
                                                                    )

        train_common_params['loss'] = {
            'cls_loss': loss_for_multimodal['interactive_loss'],
        }
        train_common_params['metrics'] = {
            target_metric: metric_for_multimodal['interactive_auc'],
        }
        train_common_params['manager.learning_rate'] = 1e-5
        train_common_params['manager.weight_decay'] = 0.001

        train_common_params['optimizer'] = optim.Adam(train_common_params['model'].parameters(), lr=train_common_params['manager.learning_rate'],
                                                         weight_decay=train_common_params['manager.weight_decay'])
        train_common_params['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(train_common_params['optimizer'])
        analyze_common_params['metrics'] =  train_common_params['metrics']
        infer_common_params['output_keys'] = ['model.output.interactive', 'data.gt']




    return train_common_params,infer_common_params,analyze_common_params