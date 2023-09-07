import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Output


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, lrelu_alpha, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.act = nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, lrelu_alpha, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.act = nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class ResNet_LReLU(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_features,
                 lrelu_alpha,
                 dropout_p,
                 linear_units,
                 clinical_variables_position,
                 clinical_variables_linear_units,
                 clinical_variables_dropout_p,
                 use_bias=False,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 num_classes=2):
        super().__init__()
        self.n_features = n_features
        self.clinical_variables_position = clinical_variables_position
        self.clinical_variables_linear_units = clinical_variables_linear_units

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.act = nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type, lrelu_alpha)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       lrelu_alpha,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       lrelu_alpha,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       lrelu_alpha,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        output_channel = block_inplanes[-1] * block.expansion

        # Initialize MLP layers for clinical variables
        if (self.n_features > 0) and (clinical_variables_linear_units is not None):
            self.clinical_variables_layers = torch.nn.ModuleList()

            clinical_variables_linear_units = [n_features] + clinical_variables_linear_units

            for i in range(len(clinical_variables_linear_units) - 1):
                self.clinical_variables_layers.add_module(
                    'dropout%s' % i,
                    torch.nn.Dropout(clinical_variables_dropout_p[i])
                )
                self.clinical_variables_layers.add_module(
                    'linear%s' % i,
                    torch.nn.Linear(in_features=clinical_variables_linear_units[i],
                                    out_features=clinical_variables_linear_units[i + 1],
                                    bias=use_bias)
                )
                self.clinical_variables_layers.add_module('leaky_relu%s' % i, self.act)
        else:
            clinical_variables_linear_units = [n_features]

        # Initialize linear layers
        self.linear_layers = torch.nn.ModuleList()

        if (linear_units is None) or (len(linear_units) == 0):
            self.out_layer = Output(in_features=output_channel + clinical_variables_linear_units[-1],
                                    out_features=num_classes,
                                    bias=use_bias)
        else:
            linear_units = [output_channel] + linear_units
            for i in range(len(linear_units) - 1):
                self.linear_layers.add_module('dropout%s' % i, torch.nn.Dropout(dropout_p[i]))

                if self.clinical_variables_position + 1 == i:
                    self.linear_layers.add_module('linear%s' % i,
                                                  torch.nn.Linear(
                                                      in_features=linear_units[i] + clinical_variables_linear_units[-1],
                                                      out_features=linear_units[i + 1], bias=use_bias))
                else:
                    self.linear_layers.add_module('linear%s' % i,
                                                  torch.nn.Linear(in_features=linear_units[i],
                                                                  out_features=linear_units[i + 1],
                                                                  bias=use_bias))

                self.linear_layers.add_module('leaky_relu%s' % i, self.act)
            self.n_sublayers_per_linear_layer = len(self.linear_layers) / (len(linear_units) - 1)

            # Initialize output layer
            if (self.n_features > 0) and (clinical_variables_linear_units is not None):
                # Add additional input units to the output layer if MLP output is concatenated at the last linear layer
                if self.clinical_variables_position + 1 == len(linear_units) - 1:
                    self.out_layer = Output(in_features=linear_units[-1] + clinical_variables_linear_units[-1],
                                            out_features=num_classes, bias=use_bias)
                else:
                    self.out_layer = Output(in_features=linear_units[-1], out_features=num_classes, bias=use_bias)
            else:
                self.out_layer = Output(in_features=linear_units[-1] + self.n_features, out_features=num_classes,
                                        bias=use_bias)
            # self.out_layer.__class__.__name__ = 'Output'

        # self.fc = Output(block_inplanes[3] * block.expansion + self.n_features, num_classes)
        # self.fc.__class__.__name__ = 'Output'

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, lrelu_alpha, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  lrelu_alpha=lrelu_alpha,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, features):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        # MLP clinical variables
        if (self.n_features > 0) and (self.clinical_variables_linear_units is not None):
            for layer in self.clinical_variables_layers:
                features = layer(features)

        if (self.n_features > 0) and (len(self.linear_layers) == 0):
            # Add features to flattened layer
            x = torch.cat([x, features], dim=1)
        else:
            # Linear layers
            for i, layer in enumerate(self.linear_layers):

                # Add features to flattened layer
                if (self.n_features > 0) and (i == 0) and (self.clinical_variables_position == -1):
                    # Add features to flattened layer
                    x = torch.cat([x, features], dim=1)

                x = layer(x)

                # Add features to a linear layer
                if (self.n_features > 0) and (
                        (i + 1) / self.n_sublayers_per_linear_layer == self.clinical_variables_position + 1) and not (
                        self.clinical_variables_position == -1
                ):
                    # Add features to a linear layer
                    x = torch.cat([x, features], dim=1)

        x = self.out_layer(x)

        return x


def get_resnet_lrelu(model_depth, channels, n_features, filters, lrelu_alpha,
                     dropout_p, linear_units, clinical_variables_position,
                     clinical_variables_linear_units, clinical_variables_dropout_p, **kwargs):
    """
    ResNet

    Args:
        model_depth:
        channels:
        n_features:
        filters:
        lrelu_alpha:
        dropout_p:
        linear_units:
        clinical_variables_position:
        clinical_variables_linear_units:
        clinical_variables_dropout_p:
        **kwargs:

    Returns:

    """
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    assert len(filters) == 4

    if model_depth == 10:
        model = ResNet_LReLU(BasicBlock, [1, 1, 1, 1], block_inplanes=filters, n_input_channels=channels, n_features=n_features, lrelu_alpha=lrelu_alpha,
                             dropout_p=dropout_p, linear_units=linear_units,
                             clinical_variables_position=clinical_variables_position,
                             clinical_variables_linear_units=clinical_variables_linear_units,
                             clinical_variables_dropout_p=clinical_variables_dropout_p,
                             **kwargs)
    elif model_depth == 18:
        model = ResNet_LReLU(BasicBlock, [2, 2, 2, 2], block_inplanes=filters, n_input_channels=channels, n_features=n_features, lrelu_alpha=lrelu_alpha,
                             dropout_p=dropout_p, linear_units=linear_units,
                             clinical_variables_position=clinical_variables_position,
                             clinical_variables_linear_units=clinical_variables_linear_units,
                             clinical_variables_dropout_p=clinical_variables_dropout_p,
                             **kwargs)
    elif model_depth == 34:
        model = ResNet_LReLU(BasicBlock, [3, 4, 6, 3], block_inplanes=filters, n_input_channels=channels, n_features=n_features, lrelu_alpha=lrelu_alpha,
                             dropout_p=dropout_p, linear_units=linear_units,
                             clinical_variables_position=clinical_variables_position,
                             clinical_variables_linear_units=clinical_variables_linear_units,
                             clinical_variables_dropout_p=clinical_variables_dropout_p,
                             **kwargs)
    elif model_depth == 50:
        model = ResNet_LReLU(Bottleneck, [3, 4, 6, 3], block_inplanes=filters, n_input_channels=channels, n_features=n_features, lrelu_alpha=lrelu_alpha,
                             dropout_p=dropout_p, linear_units=linear_units,
                             clinical_variables_position=clinical_variables_position,
                             clinical_variables_linear_units=clinical_variables_linear_units,
                             clinical_variables_dropout_p=clinical_variables_dropout_p,
                             **kwargs)
    elif model_depth == 101:
        model = ResNet_LReLU(Bottleneck, [3, 4, 23, 3], block_inplanes=filters,  n_input_channels=channels, n_features=n_features, lrelu_alpha=lrelu_alpha,
                             dropout_p=dropout_p, linear_units=linear_units,
                             clinical_variables_position=clinical_variables_position,
                             clinical_variables_linear_units=clinical_variables_linear_units,
                             clinical_variables_dropout_p=clinical_variables_dropout_p,
                             **kwargs)
    elif model_depth == 152:
        model = ResNet_LReLU(Bottleneck, [3, 8, 36, 3], block_inplanes=filters, n_input_channels=channels, n_features=n_features, lrelu_alpha=lrelu_alpha,
                             dropout_p=dropout_p, linear_units=linear_units,
                             clinical_variables_position=clinical_variables_position,
                             clinical_variables_linear_units=clinical_variables_linear_units,
                             clinical_variables_dropout_p=clinical_variables_dropout_p,
                             **kwargs)
    elif model_depth == 200:
        model = ResNet_LReLU(Bottleneck, [3, 24, 36, 3], block_inplanes=filters, n_input_channels=channels, n_features=n_features, lrelu_alpha=lrelu_alpha,
                             dropout_p=dropout_p, linear_units=linear_units,
                             clinical_variables_position=clinical_variables_position,
                             clinical_variables_linear_units=clinical_variables_linear_units,
                             clinical_variables_dropout_p=clinical_variables_dropout_p,
                             **kwargs)

    return model


# def get_model():
#     """
#     Initialize model.
#
#     Returns:
#
#     """
#     model = DenseNet121(
#         spatial_dims=config.spatial_dims,
#         in_channels=config.n_input_channels,
#         out_channels=config.num_classes,
#     )
#     return model.to(device=config.device)


