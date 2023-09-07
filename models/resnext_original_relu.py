# https://github.com/miraclewkf/ResNeXt-PyTorch/blob/master/resnext.py
import torch
import torch.nn as nn
import math
from .layers import Output

__all__ = ['ResNeXt', 'resnext18', 'resnext34', 'resnext50', 'resnext101',
           'resnext152']


def conv3x3(in_planes, out_planes, groups=1, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)


class BasicBlock(nn.Module):
    # expansion = 2
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes=inplanes, out_planes=planes, stride=stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes, groups=num_group)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes * 2)
        self.conv2 = nn.Conv3d(planes * 2, planes * 2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm3d(planes * 2)
        self.conv3 = nn.Conv3d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, n_features,
                 dropout_p,
                 linear_units,
                 clinical_variables_position,
                 clinical_variables_linear_units,
                 clinical_variables_dropout_p,
                 use_bias,
                 input_channels=3, num_classes=2, num_group=32):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.n_features = n_features
        self.clinical_variables_position = clinical_variables_position
        self.clinical_variables_linear_units = clinical_variables_linear_units
        self.use_bias = use_bias

        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_group)
        self.layer2 = self._make_layer(block, 128, layers[1], num_group, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], num_group, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        output_channel = 512 * block.expansion
        # self.out_layer = Output(512 * block.expansion + self.n_features, num_classes)
        # self.out_layer.__class__.__name__ = 'Output'

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
                self.clinical_variables_layers.add_module('relu%s' % i, self.relu)
        else:
            clinical_variables_linear_units = [n_features]
        #####

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

                self.linear_layers.add_module('relu%s' % i, self.relu)
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

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return nn.Sequential(*layers)

    def forward(self, x, features):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # flatten

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
                x = layer(x)
                if (self.n_features > 0) and (
                        (i + 1) / self.n_sublayers_per_linear_layer == self.clinical_variables_position + 1):
                    # Add features to a linear layer
                    x = torch.cat([x, features], dim=1)

        x = self.out_layer(x)

        return x


def get_resnext_original_relu(model_depth, channels, n_features, num_classes, dropout_p, linear_units,
                              clinical_variables_position, clinical_variables_linear_units,
                              clinical_variables_dropout_p, use_bias):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNeXt(BasicBlock, [1, 1, 1, 1], input_channels=channels, n_features=n_features,
                        num_classes=num_classes,
                        dropout_p=dropout_p, linear_units=linear_units,
                        clinical_variables_position=clinical_variables_position,
                        clinical_variables_linear_units=clinical_variables_linear_units,
                        clinical_variables_dropout_p=clinical_variables_dropout_p,
                        use_bias=use_bias,
                        )
    elif model_depth == 18:
        model = ResNeXt(BasicBlock, [2, 2, 2, 2], input_channels=channels, n_features=n_features,
                        num_classes=num_classes,
                        dropout_p=dropout_p, linear_units=linear_units,
                        clinical_variables_position=clinical_variables_position,
                        clinical_variables_linear_units=clinical_variables_linear_units,
                        clinical_variables_dropout_p=clinical_variables_dropout_p,
                        use_bias=use_bias,
                        )

    return model


def resnext10(**kwargs):
    """Constructs a ResNeXt-10 model.
    """
    model = ResNeXt(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnext18(**kwargs):
    """Constructs a ResNeXt-18 model.
    """
    model = ResNeXt(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnext34(**kwargs):
    """Constructs a ResNeXt-34 model.
    """
    model = ResNeXt(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnext50(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model = ResNeXt(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNeXt-152 model.
    """
    model = ResNeXt(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
