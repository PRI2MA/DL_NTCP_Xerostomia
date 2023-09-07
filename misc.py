"""
Miscellaneous functions.
"""
import os
import math
import torch
import optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.nn.functional import one_hot

import config
from lr_finder import LearningRateFinder
from models.cnn_lrelu import CNN_LReLU
from models.convnext_original import ConvNeXt
from models.dcnn_lrelu import DCNN_LReLU
from models.dcnn_dws_lrelu import DCNN_DWS_LReLU
from models.dcnn_lrelu_ln import DCNN_LReLU_LN
from models.dcnn_lrelu_gn import DCNN_LReLU_GN
from models.dcnn_selu import DCNN_SELU
from models.efficientnet import EfficientNet3D
from models.efficientnetv2 import get_efficientnetv2
from models.efficientnetv2_selu import get_efficientnetv2_selu
from models.mlp import MLP
from models.resnet_lrelu import get_resnet_lrelu
from models.resnet_original_relu import get_resnet_original_relu
from models.resnext_lrelu import get_resnext_lrelu
from models.resnext_original_relu import get_resnext_original_relu
from models.resnet_dcnn_lrelu import ResNet_DCNN_LReLU
from models.resnet_dcnn_dws_lrelu_v2 import ResNet_DCNN_DWS_LReLU_V2
from models.resnet_dcnn_lrelu_v2 import ResNet_DCNN_LReLU_V2
from models.resnet_dcnn_lrelu_gn import ResNet_DCNN_LReLU_GN
from models.resnet_dcnn_selu import ResNet_DCNN_SELU
from models.resnet_mp_lrelu import ResNet_MP_LReLU


def proba(pred):
    """
    Map probability to One-Hot Encoding form. This is required for monai.metrics.ROCAUCMetric.

    Example: pred = tensor([0.3, 0.9]), then return [tensor([0.7, 0.3]), tensor([0.1, 0.9])].

    Args:
        pred

    Returns:

    """
    return [torch.as_tensor([1 - x, x], device=pred.device) for x in pred]


def compute_auc(y_pred_list, y_true_list, auc_metric, nr_of_decimals=config.nr_of_decimals):
    """
    Compute AUC.

    Args:
        y_pred (list): list with tensors of shape (num_ohe_classes), i.e. One-Hot Encoded
        y_true (list): list with tensors of shape (num_ohe_classes), i.e. One-Hot Encoded
        auc_metric: monai.metrics.ROCAUCMetric class

    Returns:

    """
    auc_metric(y_pred=y_pred_list, y=y_true_list)
    auc_value = auc_metric.aggregate()
    auc_metric.reset()

    return round(auc_value, nr_of_decimals)


def compute_mse(y_pred_list, y_true_list, mse_metric, nr_of_decimals=config.nr_of_decimals):
    mse_value = mse_metric(y_pred_list, y_true_list).mean().item()
    return round(mse_value, nr_of_decimals)


def create_results(**kwargs):
    """
    Create results.csv.

    Returns:

    """
    df = pd.DataFrame([x for x in kwargs.values()],
                      index=[x for x in kwargs.keys()],
                      columns=[config.exp_name])
    df.to_csv(os.path.join(config.exp_dir, config.filename_results_csv), sep=';')


class DiceLoss(torch.nn.Module):
    """
    https://neptune.ai/blog/pytorch-loss-functions
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.softmax_act = torch.nn.Softmax(dim=-1)

    def forward(self, y_pred, y_true, smooth=1):
        # Convert logits to probabilities
        y_pred = self.softmax_act(y_pred)
        y_pred = y_pred[:, 1]

        # Map [0, 1] to [-1, 1]
        # y_pred = (y_pred - 0.5) * 2
        # y_true = (y_true - 0.5) * 2

        # y_pred = y_pred.view(-1)
        # y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

        return 1 - dice


class F1_Loss(torch.nn.Module):
    '''
    Calculate F1 score. Can work with gpu tensors

    The original implementation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = torch.nn.functional.one_hot(y_true, 2).to(torch.float32)
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


class L1Loss(torch.nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.l1 = torch.nn.SmoothL1Loss(reduction=reduction, beta=0.05)
        self.softmax_act = torch.nn.Softmax(dim=-1)

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(
            self.l1(self.softmax_act(y_pred), one_hot(y_true, num_classes=config.num_ohe_classes).float()))

        return loss


class RankingLoss(torch.nn.Module):
    """
    In self.rank(): if y = 1 then it assumed the first input should be ranked higher (have a larger value) than the
    second input, and vice-versa for y = -1.

    y_pred.shape: torch.Size([7, 2])
    y_pred: tensor([[0.0916, 0.4190],
            [0.5505, 0.2531],
            [0.4484, 0.1766],
            [0.4597, 0.2745],
            [0.6642, 0.3163],
            [0.6221, 0.5239],
            [0.5956, 0.3042]], device='cuda:0', grad_fn=<AddmmBackward0>)
    y_true.shape: torch.Size([7])
    y_true: tensor([0, 0, 0, 1, 1, 0, 0], device='cuda:0')
    y: tensor([ 1.,  1.,  1., -1., -1.,  1.,  1.], device='cuda:0')

    """

    def __init__(self, reduction):
        super().__init__()
        self.softmax_act = torch.nn.Softmax(dim=-1)
        self.rank = torch.nn.MarginRankingLoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        # Convert logits to probabilities
        y_pred = self.softmax_act(y_pred)

        x1 = y_pred[:, 0]
        x2 = y_pred[:, 1]
        y = - (y_true - 0.5) * 2

        loss = self.rank(x1, x2, y)

        return loss


class SoftAUCLoss(torch.nn.Module):
    """
    Soft version of AUC that uses Wilcoxon-Mann-Whitney U statistic.

    Approximates the Area Under Curve score, using approximation based on the Wilcoxon-Mann-Whitney U statistic.

    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.

    Measures overall performance for a full range of threshold levels.

    Source: https://github.com/mysterefrank/pytorch_differentiable_auc/blob/master/auc_loss.py
    Source: https://github.com/tflearn/tflearn/blob/c0baee9d34f41b84dbc43ea28e37baa5dbd465e4/tflearn/objectives.py#L179
    Source: https://discuss.pytorch.org/t/loss-backward-found-long-expected-float/121743/4
    """

    def __init__(self):
        super().__init__()
        self.softmax_act = torch.nn.Softmax(dim=-1)
        self.sigmoid_act = torch.nn.Sigmoid()
        self.gamma = 0.2  # 0.7
        self.p = 3

    def forward(self, y_pred, y_true):
        # Convert logits to probabilities
        y_pred = self.softmax_act(y_pred)[:, 1]
        # y_pred = self.sigmoid_act(y_pred)
        # Get the predictions of all the positive and negative examples
        pos = y_pred[y_true.bool()].view(1, -1)
        neg = y_pred[~y_true.bool()].view(-1, 1)
        # Compute Wilcoxon-Mann-Whitney U statistic
        difference = torch.zeros_like(pos * neg) + pos - neg - self.gamma
        masked = difference[difference < 0.0]
        loss = torch.sum(torch.pow(-masked, self.p))

        return loss


class CustomLoss(torch.nn.Module):
    def __init__(self, weight, reduction, label_smoothing, loss_weights, device):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction, label_smoothing=label_smoothing)
        self.dice = DiceLoss()
        self.f1 = F1_Loss().to(device)
        self.l1 = L1Loss(reduction=reduction)
        self.ranking = RankingLoss(reduction=reduction)
        self.soft_auc = SoftAUCLoss()
        self.loss_weights = loss_weights

    def forward(self, y_pred, y_true):
        fcns = [self.ce, self.dice, self.f1, self.l1, self.ranking, self.soft_auc]

        loss = 0
        for i, (w, fcn) in enumerate(zip(self.loss_weights, fcns)):
            if w != 0:
                loss = loss + w * fcn(y_pred, y_true)

        return loss


def get_loss_function(loss_function_name, label_weights, label_smoothing, loss_weights, device):
    """
    CrossEntropyLoss: softmax is computed as part of the loss. In other words, the model outputs should be logits,
        i.e. the model output should NOT be softmax'd.
    BCEWithLogitsLoss: sigmoid is computed as part of the loss. In other words, the model outputs should be logits,
        i.e. the model output should NOT be sigmoid'd.

    Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#CrossEntropyLoss
    Source: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html

    Args:
        loss_function_name:
        label_weights:
        label_smoothing:
        loss_weights:
        device:

    Returns:

    """
    # Initialize variables
    label_weights = torch.as_tensor(label_weights, dtype=torch.float32, device=device)
    reduction = config.loss_reduction

    if loss_function_name == 'cross_entropy':
        loss_function = torch.nn.CrossEntropyLoss(weight=label_weights, reduction=reduction,
                                                  label_smoothing=label_smoothing)
    elif loss_function_name == 'bce':
        loss_function = torch.nn.BCEWithLogitsLoss(weight=label_weights, reduction=reduction)
    elif loss_function_name == 'dice':
        loss_function = DiceLoss()
    elif loss_function_name == 'f1':
        loss_function = F1_Loss().to(device)
    elif loss_function_name == 'ranking':
        loss_function = RankingLoss(reduction=reduction)
    elif loss_function_name == 'soft_auc':
        loss_function = SoftAUCLoss()
    elif loss_function_name == 'custom':
        loss_function = CustomLoss(weight=label_weights, reduction=reduction, label_smoothing=label_smoothing,
                                   loss_weights=loss_weights, device=device)
    else:
        raise ValueError('Invalid loss_function_name: {}.'.format(loss_function_name))

    return loss_function


def get_model(model_name, num_ohe_classes, channels, depth, height, width, n_features, resnet_shortcut, num_classes,
              filters, kernel_sizes, strides, pad_value, lrelu_alpha, dropout_p, pooling_conv_filters,
              perform_pooling, linear_units, clinical_variables_position, clinical_variables_linear_units,
              clinical_variables_dropout_p, use_bias, pretrained_path, logger):
    """
    Initialize model.
    """
    # Determine number of downsampling blocks
    n_down_blocks = [sum([x[0] == 2 for x in strides]), sum([x[1] == 2 for x in strides]),
                     sum([x[2] == 2 for x in strides])]

    if model_name == 'cnn_lrelu':
        model = CNN_LReLU(n_input_channels=channels, depth=depth, height=height, width=width, n_features=n_features,
                          num_classes=num_classes, filters=filters, kernel_sizes=kernel_sizes, strides=strides,
                          pad_value=pad_value, n_down_blocks=n_down_blocks, lrelu_alpha=lrelu_alpha,
                          dropout_p=dropout_p, pooling_conv_filters=pooling_conv_filters,
                          perform_pooling=perform_pooling, linear_units=linear_units, use_bias=use_bias)
    elif model_name == 'convnext_tiny':
        # https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
        model = ConvNeXt(in_chans=channels, n_features=n_features, num_classes=num_classes, depths=[3, 3, 9, 3],
                         dims=[96, 192, 384, 768])
    elif model_name == 'convnext_small':
        model = ConvNeXt(in_chans=channels, n_features=n_features, num_classes=num_classes, depths=[3, 3, 27, 3],
                         dims=[96, 192, 384, 768])
    elif model_name == 'convnext_base':
        model = ConvNeXt(in_chans=channels, n_features=n_features, num_classes=num_classes, depths=[3, 3, 27, 3],
                         dims=[128, 256, 512, 1024])
    elif model_name == 'dcnn_lrelu':
        model = DCNN_LReLU(n_input_channels=channels, depth=depth, height=height, width=width, n_features=n_features,
                           num_classes=num_classes, filters=filters, kernel_sizes=kernel_sizes, strides=strides,
                           pad_value=pad_value, n_down_blocks=n_down_blocks, lrelu_alpha=lrelu_alpha,
                           dropout_p=dropout_p, pooling_conv_filters=pooling_conv_filters,
                           perform_pooling=perform_pooling, linear_units=linear_units,
                           clinical_variables_position=clinical_variables_position,
                           clinical_variables_linear_units=clinical_variables_linear_units,
                           clinical_variables_dropout_p=clinical_variables_dropout_p, use_bias=use_bias)
    elif model_name == 'dcnn_dws_lrelu':
        model = DCNN_DWS_LReLU(n_input_channels=channels, depth=depth, height=height, width=width,
                               n_features=n_features, num_classes=num_classes, filters=filters,
                               kernel_sizes=kernel_sizes, strides=strides, pad_value=pad_value,
                               n_down_blocks=n_down_blocks, lrelu_alpha=lrelu_alpha, dropout_p=dropout_p,
                               pooling_conv_filters=pooling_conv_filters, perform_pooling=perform_pooling,
                               linear_units=linear_units, use_bias=use_bias)
    elif model_name == 'dcnn_lrelu_ln':
        model = DCNN_LReLU_LN(n_input_channels=channels, depth=depth, height=height, width=width, n_features=n_features,
                              num_classes=num_classes, filters=filters, kernel_sizes=kernel_sizes, strides=strides,
                              pad_value=pad_value, n_down_blocks=n_down_blocks, lrelu_alpha=lrelu_alpha,
                              dropout_p=dropout_p, pooling_conv_filters=pooling_conv_filters,
                              perform_pooling=perform_pooling, linear_units=linear_units, use_bias=use_bias)
    elif model_name == 'dcnn_lrelu_gn':
        model = DCNN_LReLU_GN(n_input_channels=channels, depth=depth, height=height, width=width, n_features=n_features,
                              num_classes=num_classes, filters=filters, kernel_sizes=kernel_sizes, strides=strides,
                              pad_value=pad_value, n_down_blocks=n_down_blocks, lrelu_alpha=lrelu_alpha,
                              dropout_p=dropout_p, pooling_conv_filters=pooling_conv_filters,
                              perform_pooling=perform_pooling, linear_units=linear_units, use_bias=use_bias)
    elif model_name == 'dcnn_selu':
        model = DCNN_SELU(n_input_channels=channels, depth=depth, height=height, width=width, n_features=n_features,
                          num_classes=num_classes, filters=filters, kernel_sizes=kernel_sizes, strides=strides,
                          pad_value=pad_value, n_down_blocks=n_down_blocks, dropout_p=dropout_p,
                          pooling_conv_filters=pooling_conv_filters, perform_pooling=perform_pooling,
                          linear_units=linear_units, use_bias=use_bias)
    elif model_name in ['efficientnet-b{}'.format(i) for i in range(9)]:
        # https://github.com/shijianjian/EfficientNet-PyTorch-3D
        model = EfficientNet3D.from_name(model_name=model_name, n_features=n_features,
                                         override_params={'num_classes': num_classes}, in_channels=channels)
    elif model_name in ['efficientnetv2_{}'.format(i) for i in ['xs', 's', 'm', 'l', 'xl']]:
        # https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
        model = get_efficientnetv2(model_name=model_name, channels=channels, n_features=n_features,
                                   num_classes=num_classes, dropout_p=dropout_p, linear_units=linear_units,
                                   clinical_variables_position=clinical_variables_position,
                                   clinical_variables_linear_units=clinical_variables_linear_units,
                                   clinical_variables_dropout_p=clinical_variables_dropout_p, use_bias=use_bias,
                                   lrelu_alpha=lrelu_alpha)
    elif model_name in ['efficientnetv2_{}_selu'.format(i) for i in ['s', 'm', 'l', 'xl']]:
        # https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
        model = get_efficientnetv2_selu(model_name=model_name, channels=channels, n_features=n_features,
                                        num_classes=num_classes)
    elif model_name == 'mlp':
        assert n_features > 0
        model = MLP(n_features=n_features, num_classes=num_classes, lrelu_alpha=lrelu_alpha, dropout_p=dropout_p,
                    linear_units=linear_units, use_bias=use_bias)
    elif model_name == 'resnet_lrelu':
        model = get_resnet_lrelu(model_depth=10, channels=channels, n_features=n_features, lrelu_alpha=lrelu_alpha,
                                 num_classes=num_classes, filters=filters,
                                 dropout_p=dropout_p, linear_units=linear_units,
                                 clinical_variables_position=clinical_variables_position,
                                 clinical_variables_linear_units=clinical_variables_linear_units,
                                 clinical_variables_dropout_p=clinical_variables_dropout_p)
    elif model_name == 'resnet_dcnn_lrelu':
        model = ResNet_DCNN_LReLU(n_input_channels=channels, depth=depth, height=height, width=width,
                                  n_features=n_features, num_classes=num_classes, filters=filters,
                                  kernel_sizes=kernel_sizes, strides=strides, pad_value=pad_value,
                                  n_down_blocks=n_down_blocks, lrelu_alpha=lrelu_alpha, dropout_p=dropout_p,
                                  pooling_conv_filters=pooling_conv_filters, perform_pooling=perform_pooling,
                                  linear_units=linear_units, clinical_variables_position=clinical_variables_position,
                                  clinical_variables_linear_units=clinical_variables_linear_units,
                                  clinical_variables_dropout_p=clinical_variables_dropout_p, use_bias=use_bias)
    elif model_name == 'resnet_dcnn_dws_lrelu_v2':
        model = ResNet_DCNN_DWS_LReLU_V2(n_input_channels=channels, depth=depth, height=height, width=width,
                                         n_features=n_features, num_classes=num_classes, filters=filters,
                                         kernel_sizes=kernel_sizes, strides=strides, pad_value=pad_value,
                                         n_down_blocks=n_down_blocks, lrelu_alpha=lrelu_alpha, dropout_p=dropout_p,
                                         pooling_conv_filters=pooling_conv_filters, perform_pooling=perform_pooling,
                                         linear_units=linear_units, use_bias=use_bias)
    elif model_name == 'resnet_dcnn_lrelu_v2':
        model = ResNet_DCNN_LReLU_V2(n_input_channels=channels, depth=depth, height=height, width=width,
                                     n_features=n_features, num_classes=num_classes, filters=filters,
                                     kernel_sizes=kernel_sizes, strides=strides, pad_value=pad_value,
                                     n_down_blocks=n_down_blocks, lrelu_alpha=lrelu_alpha, dropout_p=dropout_p,
                                     pooling_conv_filters=pooling_conv_filters, perform_pooling=perform_pooling,
                                     linear_units=linear_units, use_bias=use_bias)
    elif model_name == 'resnet_dcnn_lrelu_gn':
        model = ResNet_DCNN_LReLU_GN(n_input_channels=channels, depth=depth, height=height, width=width,
                                     n_features=n_features, num_classes=num_classes, filters=filters,
                                     kernel_sizes=kernel_sizes, strides=strides, pad_value=pad_value,
                                     n_down_blocks=n_down_blocks, lrelu_alpha=lrelu_alpha, dropout_p=dropout_p,
                                     pooling_conv_filters=pooling_conv_filters, perform_pooling=perform_pooling,
                                     linear_units=linear_units, use_bias=use_bias)
    elif model_name == 'resnet_dcnn_selu':
        model = ResNet_DCNN_SELU(n_input_channels=channels, depth=depth, height=height, width=width,
                                 n_features=n_features, num_classes=num_classes, filters=filters,
                                 kernel_sizes=kernel_sizes, strides=strides, pad_value=pad_value,
                                 n_down_blocks=n_down_blocks, dropout_p=dropout_p,
                                 pooling_conv_filters=pooling_conv_filters, perform_pooling=perform_pooling,
                                 linear_units=linear_units, use_bias=use_bias)
    elif model_name == 'resnet_mp_lrelu':
        model = ResNet_MP_LReLU(n_input_channels=channels, depth=depth, height=height, width=width,
                                n_features=n_features, num_classes=num_classes, filters=filters,
                                kernel_sizes=kernel_sizes, strides=strides, pad_value=pad_value,
                                n_down_blocks=n_down_blocks, lrelu_alpha=lrelu_alpha, dropout_p=dropout_p,
                                pooling_conv_filters=pooling_conv_filters, perform_pooling=perform_pooling,
                                linear_units=linear_units, use_bias=use_bias)
    elif model_name == 'resnet_original_relu':
        # https://github.com/Tencent/MedicalNet
        model = get_resnet_original_relu(model_depth=10, channels=channels, depth=depth, height=height, width=width,
                                         n_features=n_features, resnet_shortcut=resnet_shortcut,
                                         num_classes=num_classes)
    elif model_name == 'resnext_lrelu':
        model = get_resnext_lrelu(model_depth=10, channels=channels, n_features=n_features, filters=filters,
                                  num_group=filters, num_classes=num_classes, lrelu_alpha=lrelu_alpha)
    elif model_name == 'resnext_original_relu':
        # https://github.com/miraclewkf/ResNeXt-PyTorch/blob/master/resnext.py
        model = get_resnext_original_relu(model_depth=10, channels=channels, n_features=n_features,
                                          num_classes=num_classes, dropout_p=dropout_p, linear_units=linear_units,
                                          clinical_variables_position=clinical_variables_position,
                                          clinical_variables_linear_units=clinical_variables_linear_units,
                                          clinical_variables_dropout_p=clinical_variables_dropout_p, use_bias=use_bias)

    else:
        raise ValueError('Invalid model_name: {}.'.format(model_name))

    # Load pretrained model weights
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path)

        if model_name == 'resnet_original':
            # The pretrained model assumes 1 input channel, i.e. input_shape = (B, 1, D, H, W). However
            # we use input_shape = (B, 3, D, H, W). Therefore we repeat the filter weights across the different channels.
            # This is only required for the very first layer.
            module_conv1_weight = checkpoint['state_dict']['module.conv1.weight']
            checkpoint['state_dict']['module.conv1.weight'] = module_conv1_weight.repeat(1, channels, 1, 1, 1)

            # Only consider layers that are present in the newly initialized model and in the pretrained model
            model_dict = model.state_dict()
            # Somehow the pretrained model has 'module.' preceding the layer's name
            pretrain_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items() if
                             k.replace('module.', '') in model_dict.keys()}
            if len(pretrain_dict) == 0:
                logger.my_print('checkpoint["state_dict"].keys():', checkpoint['state_dict'].keys())
                logger.my_print('model_dict.keys():', model_dict.keys())
                raise ValueError('Invalid pretrain_dict. No weights gets transferred.\nPretrain_dict: {}.'.
                                 format(pretrain_dict))

            # update(): updates key-value pairs in model_dict with key-value pairs in pretrain_dict
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)

            # Check
            # TODO: quick check (remove later)
            pname, p = next(model.named_parameters())
            # pname: parameter name
            # p: parameter weights
            logger.my_print('pname: {}.'.format(pname))
            logger.my_print(p.shape)
            # Check for elementwise-equality
            check_equality = torch.eq(p.data.cpu(), checkpoint['state_dict']['module.conv1.weight'].cpu())
            if False in check_equality:
                raise ValueError('Check of equality failed.')
                assert False
        else:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                model.load_state_dict(checkpoint)

    return model


def get_model_summary(model, input_size, device, logger):
    """
    Get model summary and number of trainable parameters.

    Args:
        model:
        input_size:
        device:
        logger:

    Returns:

    """
    # Get and save summary
    txt = str(summary(model=model, input_size=input_size, device=device))
    file = open(os.path.join(config.exp_dir, config.filename_model_txt), 'a+', encoding='utf-8')
    file.write(txt)
    file.close()

    # Determine number of trainable parameters
    # Source: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    # total_params = sum(p.numel() for p in model.parameters())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params


def get_optimizer(optimizer_name, model, lr, momentum, weight_decay, hessian_power, num_batches_per_epoch,
                  use_lookahead, lookahead_k, lookahead_alpha, logger):
    """
    https://github.com/jettify/pytorch-optimizer

    Args:
        optimizer_name:
        model:
        lr:
        momentum:
        momentum:
        weight_decay:
        hessian_power:
        num_batches_per_epoch:
        use_lookahead:
        lookahead_k:
        lookahead_alpha:
        logger:

    Returns:

    """
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam_w':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'sgd_w':
        optimizer = optimizers.SGDW(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'ada_hessian':
        optimizer = optimizers.Adahessian(model.parameters(), lr=lr, weight_decay=weight_decay,
                                          hessian_power=hessian_power)
    elif optimizer_name == 'acc_sgd':
        optimizer = optimizers.AccSGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'ada_belief':
        optimizer = optimizers.AdaBelief(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'ada_bound':
        optimizer = optimizers.AdaBound(model.parameters(), lr=lr, final_lr=lr * 100, weight_decay=weight_decay)
    elif optimizer_name == 'ada_bound_w':
        optimizer = optimizers.AdaBoundW(model.parameters(), lr=lr, final_lr=lr * 100, weight_decay=weight_decay)
    elif optimizer_name == 'ada_mod':
        optimizer = optimizers.AdaMod(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'apollo':
        optimizer = optimizers.Apollo(model.parameters(), lr=lr, init_lr=lr / 100, warmup=500,
                                      weight_decay=weight_decay)
    elif optimizer_name == 'diff_grad':
        optimizer = optimizers.DiffGrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'madgrad':
        optimizer = optimizers.MADGRAD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'novo_grad':
        optimizer = optimizers.NovoGrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'pid':
        optimizer = optimizers.PID(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'qh_adam':
        optimizer = optimizers.QHAdam(model.parameters(), lr=lr, nus=(0.7, 1.0), betas=(0.995, 0.999),
                                      weight_decay=weight_decay)
    elif optimizer_name == 'qhm':
        optimizer = optimizers.QHM(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'r_adam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'ranger_qh':
        optimizer = optimizers.RangerQH(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'ranger_21':
        optimizer = optimizers.Ranger21(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
                                        num_epochs=150, num_batches_per_epoch=num_batches_per_epoch)
    elif optimizer_name == 'swats':
        optimizer = optimizers.SWATS(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'yogi':
        optimizer = optimizers.Yogi(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimizer_name: {}.'.format(optimizer_name))

    if use_lookahead:
        return optimizers.Lookahead(optimizer, k=lookahead_k, alpha=lookahead_alpha)

    return optimizer


def get_scheduler(scheduler_name, optimizer, optimal_lr, T_0, T_mult, eta_min, step_size_up, gamma,
                  step_size, warmup_batches, num_batches_per_epoch, lr_finder_num_iter, logger):
    scheduler = None

    if scheduler_name == 'cosine':
        # Source: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T_0,
                                                                         T_mult=T_mult,
                                                                         eta_min=eta_min)
    elif scheduler_name == 'cyclic':
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR
        if lr_finder_num_iter > 0:
            base_lr = 0.5 * optimal_lr  # 0.8 * optimal_lr
            max_lr = 2 * optimal_lr  # 1.2 * optimal_lr
        else:
            base_lr = config.base_lr
            max_lr = config.max_lr
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=base_lr, max_lr=max_lr,
                                                      step_size_up=step_size_up, cycle_momentum=False)

    elif scheduler_name == 'exponential':
        # Source: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

    elif scheduler_name == 'step':
        # Source: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    # Wrap LearningRateWarmUp
    # Source: https://github.com/developer0hye/Learning-Rate-WarmUp
    # Source: https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
    if (warmup_batches > 0) and (optimal_lr is not None):
        scheduler = LearningRateWarmUp(optimizer=optimizer, warmup_batches=warmup_batches,
                                       num_batches_per_epoch=num_batches_per_epoch, target_lr=optimal_lr,
                                       after_scheduler=scheduler)

    return scheduler


def get_df_stats(df, groups, mode, frac, total_size, logger, decimals=config.nr_of_decimals):
    """
    Print dataframe statistics, useful for printing results of Stratified Sampling

    Args:
        df:
        groups:
        mode:
        frac:
        total_size:
        logger,
        decimals:

    Returns:

    """
    size = len(df)
    logger.my_print('>>> {} <<<'.format(mode.upper()))
    logger.my_print('Expected size (fraction): {} ({}%)'.format(int(total_size * frac), round(frac * 100, decimals)))
    logger.my_print('Size (fraction): {} ({}%)'.format(size, round(size / total_size * 100, decimals)))
    for g in groups:
        logger.my_print('Group: {}'.format(g))

        # Non-NaNs
        g_values = [x for x in set(df[g]) if not pd.isna(x)]
        for v in g_values:
            g_count = sum(df[g] == v)
            g_perc = sum(df[g] == v) / size * 100
            logger.my_print('\tValue = {}, \tProportion = {}% ({})'.format(v, round(g_perc, decimals), g_count))

        # NaNs (if any)
        g_count = sum(df[g].isnull())
        if g_count > 0:
            g_perc = sum(df[g].isnull()) / size * 100
            logger.my_print('\tValue = {}, \tProportion = {}% ({})'.format(np.nan, round(g_perc, decimals), g_count))
        logger.my_print('')

        logger.my_print('')


def intersection(list_1, list_2):
    """
    Intersection of two lists.

    Source: https://stackoverflow.com/questions/3697432/how-to-find-list-intersection

    Args:
        list_1:
        list_2:

    Returns:

    """
    return [x for x in list_1 if x in list_2]


def learning_rate_finder(model, train_dataloader, val_dataloader, optimizer, optimizer_name, loss_function, data_aug_p,
                         data_aug_strength, perform_augmix, mixture_width, mixture_depth, augmix_strength,
                         mean, std, grad_max_norm, lr_finder_num_iter, device, logger, save_filename=None):
    """
    Find optimal learning rate.
    Note: At the end of the whole test: the model and optimizer are restored to their initial states.

    Args:
        model:
        train_dataloader:
        val_dataloader:
        optimizer:
        optimizer_name:
        loss_function:
        data_aug_p:
        data_aug_strength:
        perform_augmix:
        mixture_width:
        mixture_depth:
        augmix_strength:
        mean: normalization: list of means, one value for each channel (first value for CT, second value for RTDOSE)
        std: normalization: list of stds, one value for each channel (first value for CT, second value for RTDOSE)
        grad_max_norm:
        lr_finder_num_iter:
        device:
        logger:
        save_filename:

    Returns:

    """
    # Store initial warmup value and temporarily set off for LR_Finder procedure
    if optimizer_name in ['apollo']:
        for param_group in optimizer.param_groups:
            init_warmup = param_group['warmup']
            param_group['warmup'] = 0
    elif optimizer_name in ['ranger_21']:
        init_use_warmup = optimizer.use_warmup
        optimizer.use_warmup = False

    # Apply lower learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = config.lr_finder_lower_lr

    # Initiate learning rate finder
    lr_finder = LearningRateFinder(model=model, optimizer=optimizer, optimizer_name=optimizer_name,
                                   criterion=loss_function, data_aug_p=data_aug_p, data_aug_strength=data_aug_strength,
                                   perform_augmix=perform_augmix, mixture_width=mixture_width,
                                   mixture_depth=mixture_depth, augmix_strength=augmix_strength,
                                   mean=mean, std=std,
                                   grad_max_norm=grad_max_norm, device=device, logger=logger)

    # Run test
    lr_finder.range_test(train_loader=train_dataloader, val_loader=val_dataloader, end_lr=config.lr_finder_upper_lr,
                         num_iter=lr_finder_num_iter)
    optimal_lr, _ = lr_finder.get_steepest_gradient()
    logger.my_print('Optimal learning rate: {}.'.format(optimal_lr))

    # Apply optimal learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = optimal_lr

    # Apply initial warmup
    if optimizer_name in ['apollo']:
        for param_group in optimizer.param_groups:
            param_group['warmup'] = init_warmup
    elif optimizer_name in ['ranger_21']:
        optimizer.user_warmup = init_use_warmup

    # Plot optimal learning rate
    if save_filename is not None:
        ax = plt.subplots(1, 1, figsize=config.figsize, facecolor='white')[1]
        lr_finder.plot(ax=ax)
        plt.savefig(os.path.join(config.exp_dir, save_filename))

    return optimal_lr, optimizer


class LearningRateWarmUp(object):
    """
    From https://github.com/developer0hye/Learning-Rate-WarmUp.
    Source: https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean

    Note: for LearningRateWarmUp we need scheduler.step() to be applied after every batch in main.py, otherwise
        the warmup will not be effective.
    """

    def __init__(self, optimizer, warmup_batches, num_batches_per_epoch, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_batches = warmup_batches
        self.num_batches_per_epoch = num_batches_per_epoch
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler

        self.warmup_epochs = warmup_batches / num_batches_per_epoch  # as float
        self.cur_iteration = 0
        self.step()

    def warmup_learning_rate(self):
        warmup_lr = self.target_lr * float(self.cur_iteration) / float(self.warmup_batches)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, cur_epoch=None):
        if cur_epoch is None:
            # E.g. cyclic scheduler: scheduler.step()
            self.cur_iteration += 1
        else:
            # E.g. cosine/exponential/step scheduler: scheduler.step(epoch + (i + 1) / train_num_iterations)
            self.cur_iteration = cur_epoch * self.num_batches_per_epoch

        if self.cur_iteration <= self.warmup_batches:
            self.warmup_learning_rate()
        else:
            self.after_scheduler.step(cur_epoch)

    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict)


def normalize(image, new_max, new_min):
    """
    Normalize image to max = new_max and min = new_min.

    """
    image_max = image.max()
    image_min = image.min()

    return (new_max - new_min) * (image - image_min) / (image_max - image_min) + new_min


def plot_values(y_list, y_label_list, best_epoch, legend_list, figsize, save_filename):
    """
    Create and save line plot of a list of loss values (per epoch).

    Args:
        y_list (list): list of list of values to be plotted (e.g. list of loss values and list of AUC metric)
        y_label_list (list): list of label of y
        best_epoch (int): draw vertical line at x=best_epoch
        legend_list (list): list of legend names
        figsize (tuple): size of output figure
        save_filename (str): filename to save to, including path

    Returns:

    """
    fig, ax = plt.subplots(nrows=len(y_list), ncols=1, figsize=tuple(figsize))
    ax = ax.flatten()

    for i, (y, y_label, legend) in enumerate(zip(y_list, y_label_list, legend_list)):
        for y_i in y:
            epochs = [e + 1 for e in range(len(y_i))]
            ax[i].plot(epochs, y_i)
        ax[i].set_ylim(bottom=0)
        ax[i].set_xlabel('Epoch')
        ax[i].axvline(x=best_epoch, color='red', linestyle='--')
        if y_label is not None:
            ax[i].set_title(y_label)
        if legend is not None:
            ax[i].legend(legend, bbox_to_anchor=(1, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_filename)
    plt.close(fig)


def save_predictions(patient_ids, y_pred_list, y_true_list, mode_list, num_classes, model_name, exp_dir, logger):
    """
    Save prediction and corresponding true labels to csv.

    Args:
        patient_ids (list): list of patient ids
        y_pred_list (list): list of PyTorch tensors
        y_true_list (list): list of PyTorch tensors
        mode_list (list): list of strings
        num_classes (int):
        model_name (str):
        exp_dir (str):
        logger:

    Returns:

    """
    # Print outputs
    logger.my_print('Model_name: {}.'.format(model_name))
    logger.my_print('patient_ids: {}.'.format(patient_ids))
    logger.my_print('y_pred_list: {}.'.format(y_pred_list))
    logger.my_print('y_true_list: {}.'.format(y_true_list))

    # Convert to CPU
    y_pred = [x.cpu().numpy() for x in y_pred_list]
    y_true = [x.cpu().numpy() for x in y_true_list]

    # Save to DataFrame
    num_cols = y_pred[0].shape[0]
    df_patient_ids = pd.DataFrame(patient_ids, columns=['PatientID'])
    df_y_pred = pd.DataFrame(y_pred, columns=['pred_{}'.format(c) for c in range(num_cols)])
    df_y_true = pd.DataFrame(y_true, columns=['true_{}'.format(c) for c in range(num_cols)])
    df_mode = pd.DataFrame(mode_list, columns=['Mode'])
    df_y = pd.concat([df_patient_ids, df_y_pred, df_y_true, df_mode], axis=1)

    # Save to file
    df_y.to_csv(os.path.join(exp_dir, config.filename_i_outputs_csv.format(i=model_name)), sep=';', index=False)


def stratified_sampling_split(df, groups, frac, seed):
    """
    Stratified Sampling.

    Source: https://github.com/pandas-dev/pandas/blob/v1.4.2/pandas/core/groupby/groupby.py#L3687-L3813
    Source: sample.sample(): https://github.com/pandas-dev/pandas/blob/v1.4.2/pandas/core/sample.py
    Source: random_state.choice(): https://github.com/numpy/numpy/blob/main/numpy/random/_generator.pyx#L601-L862

    Args:
        df:
        groups:
        frac:
        seed:

    Returns:

    """
    # Note: x.sample() accepts argument 'weights', but this should be one column name, where the rows with larger value
    # in the column are more likely to be sampled. So these are not the weights for 'groups'.
    df_sample = df.groupby(groups, group_keys=False).apply(lambda x: x.sample(frac=frac, replace=False,
                                                                              random_state=seed))
    df_other = df.loc[~df.index.isin(df_sample.index)]

    return df_sample, df_other


def weights_init(m, weight_init_name, kaiming_a, kaiming_mode, kaiming_nonlinearity, gain, logger):
    """
    Custom weights initialization.

    The weights_init function takes an initialized model as input and re-initializes all convolutional,
    batch normalization and instance normalization layers.

    Source: https://pytorch.org/docs/stable/nn.init.html
    Source: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    Source: https://github.com/pytorch/pytorch/blob/029a968212b018192cb6fc64075e68db9985c86a/torch/nn/modules/conv.py#L49
    Source: https://github.com/pytorch/pytorch/blob/029a968212b018192cb6fc64075e68db9985c86a/torch/nn/modules/linear.py#L83
    Source: https://github.com/pytorch/pytorch/blob/029a968212b018192cb6fc64075e68db9985c86a/torch/nn/modules/batchnorm.py#L51
    Source: https://discuss.pytorch.org/t/initialization-and-batch-normalization/30655/2

    Args:
        m:
        weight_init_name:
        kaiming_a:
        kaiming_mode:
        kaiming_nonlinearity:
        gain:
        logger:

    Returns:

    """
    # Initialize variables
    classname = m.__class__.__name__

    # Initialize output layer, for which a sigmoid-function will be preceded.
    if 'Output' in classname:
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)  # if fan_in > 0 else 0
            torch.nn.init.uniform_(m.bias, -bound, bound)
        logger.my_print('Weights init of output layer: Xavier uniform.')

    elif ('Conv' in classname) or ('Linear' in classname):
        if weight_init_name == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(m.weight, a=kaiming_a, mode=kaiming_mode, nonlinearity=kaiming_nonlinearity)
        elif weight_init_name == 'uniform':
            torch.nn.init.uniform_(m.weight)
        elif weight_init_name == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        elif weight_init_name == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(m.weight, a=kaiming_a, mode=kaiming_mode, nonlinearity=kaiming_nonlinearity)
        elif weight_init_name == 'normal':
            torch.nn.init.normal_(m.weight)
        elif weight_init_name == 'xavier_normal':
            torch.nn.init.xavier_normal_(m.weight, gain=gain)
        elif weight_init_name == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight, gain=gain)
        else:
            raise ValueError('Invalid weight_init_name: {}.'.format(weight_init_name))

        if m.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)  # if fan_in > 0 else 0
            torch.nn.init.uniform_(m.bias, -bound, bound)
