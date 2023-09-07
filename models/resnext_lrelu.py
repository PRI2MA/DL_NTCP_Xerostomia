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

    def __init__(self, inplanes, planes, lrelu_alpha, stride=1, downsample=None, num_group=32):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes=inplanes, out_planes=planes, stride=stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.act = nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes, groups=num_group)
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

    def __init__(self, inplanes, planes, lrelu_alpha, stride=1, downsample=None, num_group=32):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes*2)
        self.conv2 = nn.Conv3d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm3d(planes*2)
        self.conv3 = nn.Conv3d(planes*2, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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


class ResNeXt_LReLU(nn.Module):

    def __init__(self, block, layers, n_features, lrelu_alpha, input_channels=3, filters=[64, 128, 256, 512],
                 num_group=[32, 32, 32, 32], num_classes=2):
        self.inplanes = filters[0]
        super(ResNeXt_LReLU, self).__init__()
        self.n_features = n_features
        self.conv1 = nn.Conv3d(input_channels, filters[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(filters[0])
        self.act = nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, filters[0], layers[0], num_group[0], lrelu_alpha)
        self.layer2 = self._make_layer(block, filters[1], layers[1], num_group[1], lrelu_alpha, stride=2)
        self.layer3 = self._make_layer(block, filters[2], layers[2], num_group[2], lrelu_alpha, stride=2)
        self.layer4 = self._make_layer(block, filters[3], layers[3], num_group[3], lrelu_alpha, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = Output(filters[3] * block.expansion + self.n_features, num_classes)
        # self.fc.__class__.__name__ = 'Output'

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, lrelu_alpha, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, lrelu_alpha, stride, downsample, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, lrelu_alpha, num_group=num_group))

        return nn.Sequential(*layers)

    def forward(self, x, features):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Add features
        if self.n_features > 0:
            x = torch.cat([x, features], dim=1)

        x = self.fc(x)

        return x


def get_resnext_lrelu(model_depth, channels, n_features, filters, num_group, lrelu_alpha, num_classes):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    assert len(filters) == 4

    if model_depth == 10:
        model = ResNeXt_LReLU(BasicBlock, [1, 1, 1, 1], input_channels=channels, n_features=n_features, filters=filters,
                              lrelu_alpha=lrelu_alpha, num_group=num_group, num_classes=num_classes)
    elif model_depth == 18:
        model = ResNeXt_LReLU(BasicBlock, [2, 2, 2, 2], input_channels=channels, n_features=n_features, filters=filters,
                              lrelu_alpha=lrelu_alpha, num_group=num_group, num_classes=num_classes)

    return model


def resnext10(**kwargs):
    """Constructs a ResNeXt-10 model.
    """
    model = ResNeXt_LReLU(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnext18(**kwargs):
    """Constructs a ResNeXt-18 model.
    """
    model = ResNeXt_LReLU(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnext34(**kwargs):
    """Constructs a ResNeXt-34 model.
    """
    model = ResNeXt_LReLU(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnext50(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt_LReLU(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model = ResNeXt_LReLU(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNeXt-152 model.
    """
    model = ResNeXt_LReLU(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


