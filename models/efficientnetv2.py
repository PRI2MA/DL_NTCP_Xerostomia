# https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021).
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
import from https://github.com/d-li14/mobilenetv2.pytorch
"""

import torch
import torch.nn as nn
import math
from .layers import Output

__all__ = ['efficientnetv2_xs', 'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l', 'efficientnetv2_xl']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm3d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv3d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    """
    t: expand ratio in (Fused-)MBConv block
    c: output channels
    n: number of layers
    s: stride
    use_se: use Fused-MBConv block if True, else use MBConv block

    """
    def __init__(self, cfgs, n_features,
                 dropout_p,
                 linear_units,
                 clinical_variables_position,
                 clinical_variables_linear_units,
                 clinical_variables_dropout_p,
                 use_bias,
                 lrelu_alpha,
                 channels=3, num_classes=2, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs
        self.n_features = n_features
        self.clinical_variables_position = clinical_variables_position
        self.clinical_variables_linear_units = clinical_variables_linear_units
        self.use_bias = use_bias
        self.lrelu_alpha = lrelu_alpha

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(channels, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        # output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self._initialize_weights()

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
                self.clinical_variables_layers.add_module('lrelu%s' % i, torch.nn.LeakyReLU(negative_slope=lrelu_alpha))
        else:
            clinical_variables_linear_units = [n_features]

        # Initialize linear layers
        self.linear_layers = torch.nn.ModuleList()
        if (linear_units is None) or (len(linear_units) == 0):
            self.out_layer = Output(in_features=output_channel + clinical_variables_linear_units[-1], out_features=num_classes,
                                    bias=use_bias)
        else:
            linear_units = [output_channel] + linear_units
            for i in range(len(linear_units) - 1):
                self.linear_layers.add_module('dropout%s' % i, torch.nn.Dropout(dropout_p[i]))

                if self.clinical_variables_position + 1 == i:
                    self.linear_layers.add_module('linear%s' % i,
                                                  torch.nn.Linear(in_features=linear_units[i] + clinical_variables_linear_units[-1],
                                                                  out_features=linear_units[i + 1], bias=use_bias))
                else:
                    self.linear_layers.add_module('linear%s' % i,
                                                  torch.nn.Linear(in_features=linear_units[i], out_features=linear_units[i + 1],
                                                                  bias=use_bias))

                self.linear_layers.add_module('lrelu%s' % i, torch.nn.LeakyReLU(negative_slope=lrelu_alpha))
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


    def forward(self, x, features):
        x = self.features(x)
        x = self.conv(x)
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
                x = layer(x)
                if (self.n_features > 0) and (
                        (i + 1) / self.n_sublayers_per_linear_layer == self.clinical_variables_position + 1):
                    # Add features to a linear layer
                    x = torch.cat([x, features], dim=1)

        x = self.out_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                receptive_field_size = 1
                for s in m.kernel_size:
                    receptive_field_size *= s

                n = m.out_channels * receptive_field_size
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def efficientnetv2_xs(**kwargs):
    """
    Constructs a EfficientNetV2-XS model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 12, 2, 1, 0],
        [4, 24, 4, 2, 0],
        [4, 32, 4, 2, 0],
        [4, 64, 6, 2, 1],
        [6, 80, 9, 1, 1],
        [6, 128, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def efficientnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def efficientnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 3, 1, 0],
        [4, 48, 5, 2, 0],
        [4, 80, 5, 2, 0],
        [4, 160, 7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512, 5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def efficientnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 7, 2, 0],
        [4, 96, 7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640, 7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def efficientnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 8, 2, 0],
        [4, 96, 8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640, 8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def get_efficientnetv2(model_name, channels, n_features, num_classes,
                       dropout_p, linear_units, clinical_variables_position, clinical_variables_linear_units, clinical_variables_dropout_p, use_bias, lrelu_alpha):

    if model_name == 'efficientnetv2_xs':
        model = efficientnetv2_xs(channels=channels, n_features=n_features, num_classes=num_classes, dropout_p=dropout_p, linear_units=linear_units,
                                  clinical_variables_position=clinical_variables_position, clinical_variables_linear_units=clinical_variables_linear_units,
                                  clinical_variables_dropout_p=clinical_variables_dropout_p,
                                  use_bias=use_bias, lrelu_alpha=lrelu_alpha)
    elif model_name == 'efficientnetv2_s':
        model = efficientnetv2_s(channels=channels, n_features=n_features, num_classes=num_classes, dropout_p=dropout_p, linear_units=linear_units,
                                 clinical_variables_position=clinical_variables_position, clinical_variables_linear_units=clinical_variables_linear_units,
                                 clinical_variables_dropout_p=clinical_variables_dropout_p,
                                 use_bias=use_bias, lrelu_alpha=lrelu_alpha)
    elif model_name == 'efficientnetv2_m':
        model = efficientnetv2_m(channels=channels, n_features=n_features, num_classes=num_classes, dropout_p=dropout_p, linear_units=linear_units,
                                 clinical_variables_position=clinical_variables_position, clinical_variables_linear_units=clinical_variables_linear_units,
                                 clinical_variables_dropout_p=clinical_variables_dropout_p,
                                  use_bias=use_bias,lrelu_alpha=lrelu_alpha)
    elif model_name == 'efficientnetv2_l':
        model = efficientnetv2_l(channels=channels, n_features=n_features, num_classes=num_classes, dropout_p=dropout_p, linear_units=linear_units,
                                 clinical_variables_position=clinical_variables_position, clinical_variables_linear_units=clinical_variables_linear_units,
                                 clinical_variables_dropout_p=clinical_variables_dropout_p,
                                  use_bias=use_bias,lrelu_alpha=lrelu_alpha)
    elif model_name == 'efficientnetv2_xl':
        model = efficientnetv2_xl(channels=channels, n_features=n_features, num_classes=num_classes, dropout_p=dropout_p, linear_units=linear_units,
                                  clinical_variables_position=clinical_variables_position, clinical_variables_linear_units=clinical_variables_linear_units,
                                  clinical_variables_dropout_p=clinical_variables_dropout_p,
                                  use_bias=use_bias,lrelu_alpha=lrelu_alpha)
    else:
        raise ValueError('Model_name = {} is not valid!'.format(model_name))

    return model

