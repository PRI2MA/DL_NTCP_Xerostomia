# https://github.com/Tencent/MedicalNet
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import math
from functools import partial
from .layers import Output

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_C,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 n_features,
                 num_classes,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.n_features = n_features
        self.conv1 = nn.Conv3d(
            sample_input_C,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = Output(512 * block.expansion + self.n_features, num_classes)
        # self.fc.__class__.__name__ = 'Output'

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

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
        x = x.view(x.size(0), -1)

        # Add features
        if self.n_features > 0:
            x = torch.cat([x, features], dim=1)

        x = self.fc(x)

        return x


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


def get_resnet_original_relu(model_depth, channels, depth, height, width, n_features, resnet_shortcut, num_classes):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = resnet10(
            sample_input_C=channels,
            sample_input_D=depth,
            sample_input_H=height,
            sample_input_W=width,
            n_features=n_features,
            shortcut_type=resnet_shortcut,
            num_classes=num_classes)
    elif model_depth == 18:
        model = resnet18(
            sample_input_C=channels,
            sample_input_D=depth,
            sample_input_H=height,
            sample_input_W=width,
            n_features=n_features,
            shortcut_type=resnet_shortcut,
            num_classes=num_classes)
    #
    # if not opt['no_cuda']:
    #     # if len(opt['gpu_id']) > 1:
    #     # Decide which device we want to run on
    #     gpu_condition = torch.cuda.is_available()
    #     device = torch.device('cuda') if gpu_condition else torch.device('cpu')
    #
    #     model = model.to(device)
    #     # model = nn.DataParallel(model).to(device)
    #     model_dict = model.state_dict()
    # else:
    #     model_dict = model.state_dict()
    #
    # # load pretrain
    # if opt['phase'] != 'test' and opt['pretrain_path']:
    #     print('Loading pretrained model {}'.format(opt['pretrain_path']))
    #     checkpoint = torch.load(opt['pretrain_path'])
    #
    #     # Pretrained model assumes 1 input channel, so input shape = (B, 1, D, H, W), however
    #     # we use input shape (B, 3, D, H, W). Therefore we repeat the filter weights for the
    #     # channels.
    #     module_conv1_weight = checkpoint['state_dict']['module.conv1.weight']
    #     checkpoint['state_dict']['module.conv1.weight'] = module_conv1_weight.repeat(1, opt['input_C'], 1, 1, 1)
    #
    #     # Remove 'module.' in front of the layer names
    #     #         for k, v in checkpoint['state_dict'].items():
    #     #             k_new = k.replace('module.', '')
    #     #             checkpoint['state_dict'][k_new] = checkpoint['state_dict'][k]
    #
    #     # Option A
    #     print('checkpoint["state_dict"].keys():', checkpoint['state_dict'].keys())
    #     print('model_dict.keys():', model_dict.keys())
    #     pretrain_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()
    #                      if k.replace('module.', '') in model_dict.keys()}
    #     if len(pretrain_dict) == 0:
    #         assert False
    #     print('pretrain_dict.keys():', pretrain_dict.keys())
    #     model_dict.update(pretrain_dict)
    #     model.load_state_dict(model_dict)
    #
    #     new_parameters = []
    #     for pname, p in model.named_parameters():
    #         for layer_name in opt['new_layer_names']:
    #             if pname.find(layer_name) >= 0:
    #                 new_parameters.append(p)
    #                 break
    #
    #     new_parameters_id = list(map(id, new_parameters))
    #     base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
    #     parameters = {'base_parameters': base_parameters,
    #                   'new_parameters': new_parameters}
    #
    #     return model, parameters

    return model