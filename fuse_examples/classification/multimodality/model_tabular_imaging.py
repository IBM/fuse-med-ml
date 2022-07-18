import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torchvision.models.video.resnet import VideoResNet, BasicBlock, Conv3DSimple, BasicStem, model_urls
from typing import Dict, Tuple, Any, List, Sequence, Callable
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
from fuse.models.model_default import FuseModelDefault
from fuse.models.heads.head_3D_classifier import FuseHead3dClassifier

#-----------------------------------------------------------------
#Encoders fusion models
#-----------------------------------------------------------------

class project_imaging(nn.Module):

    def __init__(self, pooling='max', dim='2d', projection_imaging: nn.Module = None):
        super().__init__()
        assert pooling in ('max', 'avg')
        assert dim in ('2d', '3d')
        self.pooling = pooling
        self.dim = dim
        self.projection_imaging = projection_imaging

    def forward(self, imaging_features):
        if self.pooling == 'max':
            if self.dim == '2d':
                imaging_features = F.max_pool2d(imaging_features, kernel_size=imaging_features.shape[2:])
            else:
                imaging_features = F.max_pool3d(imaging_features, kernel_size=imaging_features.shape[2:])
                imaging_features = torch.squeeze(imaging_features,len(imaging_features.shape)-1)

        elif self.pooling == 'avg':
            if self.dim == '2d':
                imaging_features = F.avg_pool2d(imaging_features, kernel_size=imaging_features.shape[2:])
            else:
                imaging_features = F.max_pool3d(imaging_features, kernel_size=imaging_features.shape[2:])
                imaging_features = torch.squeeze(imaging_features,len(imaging_features.shape)-1)

        if self.projection_imaging is not None:
            imaging_features = self.projection_imaging.forward(imaging_features)
            imaging_features = torch.squeeze(torch.squeeze(imaging_features,dim=3),dim=2)

        return imaging_features

class project_tabular(nn.Module):

    def __init__(self, projection_tabular: nn.Module = None):
        super().__init__()
        self.projection_tabular = projection_tabular

    def forward(self, tabular_features):
        if self.projection_imaging is not None:
            tabular_features = self.projection_tabular.forward(tabular_features)
            tabular_features = torch.squeeze(torch.squeeze(tabular_features, dim=3), dim=2)

        return tabular_features

# concat model
class TabularImagingConcat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_dict):
        tabular_features = FuseUtilsHierarchicalDict.get(batch_dict, 'model.tabular_features')
        imaging_features = FuseUtilsHierarchicalDict.get(batch_dict, 'model.imaging_features')
        res = torch.cat([tabular_features, imaging_features], dim=1)
        return res

#Tabular model
class FuseModelTabularContinuousCategorical(torch.nn.Module):
    def __init__(self,
                 continuous_tabular_input: Tuple[Tuple[str, int], ...],
                 categorical_tabular_input: Tuple[Tuple[str, int], ...],
                 backbone_categorical_tabular: torch.nn.Module,
                 backbone_continuous_tabular: torch.nn.Module,
                 backbone_cat_tabular: torch.nn.Module,
                 heads: Sequence[torch.nn.Module],
                 ) -> None:
        super().__init__()
        self.continuous_tabular_input = continuous_tabular_input
        self.categorical_tabular_input = categorical_tabular_input
        self.backbone_categorical_tabular = backbone_categorical_tabular
        self.backbone_continuous_tabular = backbone_continuous_tabular
        self.backbone_cat_tabular = backbone_cat_tabular
        self.heads = torch.nn.ModuleList(heads)
        self.add_module('heads', self.heads)

    def forward(self, batch_dict: Dict) -> Dict:
        if self.backbone_categorical_tabular:
            categorical_input = FuseUtilsHierarchicalDict.get(batch_dict, self.categorical_tabular_input[0][0])
            categorical_embeddings = self.backbone_categorical_tabular(categorical_input)
        else:
            categorical_embeddings = FuseUtilsHierarchicalDict.get(batch_dict, self.categorical_tabular_input[0][0])

        if self.backbone_continuous_tabular:
            continuous_input = FuseUtilsHierarchicalDict.get(batch_dict, self.continuous_tabular_input[0][0])
            continuous_embeddings = self.backbone_categorical_tabular(continuous_input)
        else:
            continuous_embeddings = FuseUtilsHierarchicalDict.get(batch_dict, self.continuous_tabular_input[0][0])

        input_cat = torch.cat((categorical_embeddings, continuous_embeddings), 1)
        tabular_features = self.backbone_cat_tabular(input_cat)
        FuseUtilsHierarchicalDict.set(batch_dict, 'model.tabular_features', tabular_features)

        for head in self.heads:
            batch_dict = head.forward(batch_dict)
        return batch_dict['model']

#Tabular Imaging model
class FuseMultiModalityModel(torch.nn.Module):
    def __init__(self,
                 tabular_inputs: Tuple[Tuple[str, int], ...]=None,
                 imaging_inputs: Tuple[Tuple[str, int], ...]=None,
                 tabular_backbone: torch.nn.Module=None,
                 imaging_backbone: torch.nn.Module=None,
                 tabular_projection: torch.nn.Module=None,
                 imaging_projection: torch.nn.Module = None,
                 multimodal_backbone: torch.nn.Module=None,
                 heads: Sequence[torch.nn.Module]=None,
                 ) -> None:
        super().__init__()

        self.tabular_inputs = tabular_inputs
        self.tabular_backbone = tabular_backbone
        if self.tabular_backbone:
            self.add_module('tabular_backbone', self.tabular_backbone)

        self.imaging_inputs = imaging_inputs
        self.imaging_backbone = imaging_backbone
        if self.imaging_backbone:
            self.add_module('imaging_backbone', self.imaging_backbone)

        self.multimodal_backbone = multimodal_backbone
        if self.multimodal_backbone:
            self.add_module('multimodal_backbone', multimodal_backbone)

        self.tabular_projection = tabular_projection
        if self.tabular_projection:
            self.add_module('tabular_projection', tabular_projection)

        self.imaging_projection = imaging_projection
        if self.imaging_projection:
            self.add_module('imaging_projection', imaging_projection)

        self.heads = torch.nn.ModuleList(heads)
        if self.heads:
            self.add_module('heads', self.heads)


    def forward(self, batch_dict: Dict) -> Dict:

        if self.tabular_backbone:
            tabular_features = self.tabular_backbone.forward(batch_dict)

        if self.imaging_backbone:
            imaging_input = FuseUtilsHierarchicalDict.get(batch_dict, self.imaging_inputs[0][0])
            imaging_features = self.imaging_backbone.forward(imaging_input)
            FuseUtilsHierarchicalDict.set(batch_dict, 'model.imaging_features', imaging_features)

        if self.tabular_projection:
            tabular_features = FuseUtilsHierarchicalDict.get(batch_dict, 'model.tabular_features')
            tabular_features = self.tabular_projection.forward(tabular_features)
            FuseUtilsHierarchicalDict.set(batch_dict, 'model.tabular_features', tabular_features)

        if self.imaging_projection:
            imaging_features = FuseUtilsHierarchicalDict.get(batch_dict, 'model.imaging_features')
            imaging_features = self.imaging_projection.forward(imaging_features)
            FuseUtilsHierarchicalDict.set(batch_dict, 'model.imaging_features', imaging_features)

        if self.multimodal_backbone:
            multimodal_features = self.multimodal_backbone.forward(batch_dict)
            FuseUtilsHierarchicalDict.set(batch_dict, 'model.multimodal_features', multimodal_features)


        # run through heads
        if self.heads:
            for head in self.heads:
                batch_dict = head.forward(batch_dict)

        return batch_dict['model']


#-----------------------------------------------------------------
#Interactive models
#-----------------------------------------------------------------

def channel_multiplication(vector, matrix):
    return matrix * vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

class FuseBackboneResnet3DInteractive(VideoResNet):
    """
    3D model classifier (ResNet architecture"
    """

    def __init__(self,
                 conv_inputs: Tuple[Tuple[str, int], ...] = (('data.image', 1),),
                 fcn_inputs: Tuple[Tuple[str, int], ...] = (('data.input.clinical.all', 1),),
                 fcn_layers: List[int] = [64, 64, 128, 256, 512], #VideoResNet layers
                 fcn_input_size: int = 11,
                 interact_function: Callable = channel_multiplication,
                 cnn_interact_function: Callable = None,
                 fcn_cnn_layers_interactions: List[int] = None,
                 fcn_cnn_layers_parallels: List[int] = None,
                 use_relu_in_fcn: bool = True,
                 use_batcn_norm_in_fcn: bool = False,
                 pretrained: bool = False, in_channels: int = 1,
                 name: str = "r3d_18") -> None:
        """
        Create 3D ResNet model
        :param pretrained: Use pretrained weights
        :param in_channels: number of input channels
        :param name: model name. currently only 'r3d_18' is supported
        """
        # init parameters per required backbone
        init_parameters = {
            'r3d_18': {'block': BasicBlock,
                       'conv_makers': [Conv3DSimple] * 4,
                       'layers': [2, 2, 2, 2],
                       'stem': BasicStem},
        }[name]

        # init original model
        super().__init__(**init_parameters)

        # load pretrained parameters if required
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[name])
            self.load_state_dict(state_dict)

        # =================================
        self.use_relu_in_fcn = use_relu_in_fcn
        self.use_batcn_norm_in_fcn = use_batcn_norm_in_fcn
        self.cnn_interact_function = cnn_interact_function

        fcn_data = [nn.Linear(fcn_input_size, fcn_layers[0])]
        if self.use_relu_in_fcn:
            fcn_data.append(nn.ReLU(inplace=False))
        if self.use_batcn_norm_in_fcn:
            fcn_data.append(nn.BatchNorm1d(fcn_layers[0], eps=0.001, momentum=0.01, affine=True))

        for layer_idx in range(len(fcn_layers) - 1):
            fcn_data.append(nn.Linear(fcn_layers[layer_idx], fcn_layers[layer_idx + 1]))
            fcn_data.append(nn.ReLU(inplace=False)) if self.use_relu_in_fcn else None
            fcn_data.append(nn.BatchNorm1d(fcn_layers[layer_idx + 1], eps=0.001, momentum=0.01,
                                           affine=True)) if self.use_batcn_norm_in_fcn else None

        self.interactive_fcn = nn.ModuleList(fcn_data)
        if interact_function is None and fcn_cnn_layers_interactions is not None:
            assert "fcn_cnn_layers_interactions are defined but no interactive function is provided"
        self.fcn_interact_function = interact_function
        self.fcn_cnn_layers_parallels = fcn_cnn_layers_parallels or range(
            len(self.interactive_fcn))  # either specified or all cnn layers
        self.fcn_cnn_layers_interactions = fcn_cnn_layers_interactions or self.fcn_cnn_layers_parallels  # either specified or all parallel layers

        #=================================
        # save input parameters
        self.pretrained = pretrained
        self.in_channels = in_channels
        # override the first convolution layer to support any number of input channels
        self.stem = nn.Sequential(
            nn.Conv3d(self.in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_inputs = conv_inputs
        self.fcn_inputs = fcn_inputs

    def features(self, batch_dict: Dict) -> Any:
        """
        Extract spatial features - given a 3D tensor
        :param x: Input tensor - shape: [batch_size, channels, z, y, x]
        :return: spatial features - shape [batch_size, n_features, z', y', x']
        """

        conv_input = torch.cat([FuseUtilsHierarchicalDict.get(batch_dict, conv_input[0]) for conv_input in self.conv_inputs], 1)
        fcn_input = torch.cat([FuseUtilsHierarchicalDict.get(batch_dict, fcn_input[0]) for fcn_input in self.fcn_inputs], 1)


        interactive_idx=0
        x = self.stem(conv_input)
        out, y, interactive_idx = self.apply_interactive_fcn(interactive_idx, x, fcn_input, 0)
        out = self.layer1(out)
        out, y, interactive_idx = self.apply_interactive_fcn(interactive_idx, out, y, 1)
        out = self.layer2(out)
        out, y, interactive_idx = self.apply_interactive_fcn(interactive_idx, out, y, 2)
        out = self.layer3(out)
        out, y, interactive_idx = self.apply_interactive_fcn(interactive_idx, out, y, 3)
        out = self.layer4(out)
        out, y, interactive_idx = self.apply_interactive_fcn(interactive_idx, out, y, 4)

        return out

    def apply_interactive_fcn(self, interactive_idx, x, y, cnn_layer_idx):
        if cnn_layer_idx in self.fcn_cnn_layers_parallels:  # only if this resnet layer should be interacted with fcn
            if interactive_idx < len(self.interactive_fcn):
                y = self.interactive_fcn[interactive_idx](y)  # fully connected layer
                interactive_idx += 1
                if self.use_relu_in_fcn:
                    y = self.interactive_fcn[interactive_idx](y)  # ReLU
                    interactive_idx += 1
                if self.use_batcn_norm_in_fcn:
                    y = self.interactive_fcn[interactive_idx](y)  # BatchNorm
                    interactive_idx += 1
            if self.fcn_interact_function is not None:
                # only apply interact function if the layer is in the resnet layer we want to interact with
                if cnn_layer_idx in self.fcn_cnn_layers_interactions:
                    x = self.fcn_interact_function(y, x)
            if self.cnn_interact_function is not None:
                # only apply interact function if the layer is in the resnet layer we want to interact with
                if cnn_layer_idx in self.fcn_cnn_layers_interactions:
                    y = self.cnn_interact_function(y, x)

        return x, y, interactive_idx

    def forward(self, x: Tensor) -> Tuple[Tensor, None, None, None]:  # type: ignore
        """
        Forward pass. 3D global classification given a volume
        :param x: Input volume. shape: [batch_size, channels, z, y, x]
        :return: logits for global classification. shape: [batch_size, n_classes].
        """
        x = self.features(x)
        return x

class FuseModelDefaultInteractive(FuseModelDefault):
    def __init__(self,
                 conv_inputs: Tuple[Tuple[str, int], ...] = (('data.input', 1),),
                 cnn_inputs: Tuple[Tuple[str, int], ...] = (('data.clinical', 1),),
                 backbone: torch.nn.Module = None,
                 heads: Sequence[torch.nn.Module] = None,
                 freeze_backbone=False,
                 ) -> None:
        """
        Default Fuse model - convolutional neural network with multiple heads
        :param conv_inputs:     batch_dict name for model input and its number of input channels, imaging
        :param cnn_inputs:     batch_dict name for model input and its number of input channels, tabular feature

        :param backbone:        PyTorch backbone module - a convolutional neural network
        :param heads:           Sequence of head modules
        """
        FuseModelDefault.__init__(self,conv_inputs=conv_inputs,heads=heads,backbone=backbone)

        self.cnn_inputs = cnn_inputs
        self.freeze_backbone = freeze_backbone

    def train(self, model: bool=True):
        if self.freeze_backbone:
            self.backbone.eval()
            self.backbone.interactive_fcn.train()
            self.heads.train()
        else:
            self.backbone.train()
            self.heads.train()

    def forward(self,
                batch_dict: Dict) -> Dict:
        """
        Forward function of the model
        :param input: Tensor [BATCH_SIZE, 1, H, W]
        :return: classification scores - [BATCH_SIZE, num_classes]
        """

        features = self.backbone(batch_dict)
        FuseUtilsHierarchicalDict.set(batch_dict, 'model.backbone_features', features)

        for head in self.heads:
            batch_dict = head.forward(batch_dict)

        return batch_dict['model']

