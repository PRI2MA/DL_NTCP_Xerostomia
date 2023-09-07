"""
Same as ResNet_DCNN, but where every ReLU activation has been replaced by LeakyReLU.
"""
import math
import torch
from .layers import conv3d_padding_same, Output, pooling_conv


def conv3x3x3(in_planes, out_planes, stride=1):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicResBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, pad_value, lrelu_alpha, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.norm1 = torch.nn.InstanceNorm3d(planes)
        self.activation = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.conv2 = conv3x3x3(planes, planes * self.expansion)
        self.norm2 = torch.nn.InstanceNorm3d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out


class InvertedResidual(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, pad_value, lrelu_alpha, stride=1, downsample=None):
        super().__init__()
        interm_features = planes * self.expansion

        self.conv1 = conv1x1x1(in_planes, interm_features)
        self.norm1 = torch.nn.InstanceNorm3d(interm_features)
        self.conv2 = conv3x3x3(interm_features, interm_features, stride)
        self.norm2 = torch.nn.InstanceNorm3d(interm_features)
        self.conv3 = conv1x1x1(interm_features, planes)
        self.norm3 = torch.nn.InstanceNorm3d(planes)
        self.activation = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out


class conv_block(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, pad_value, lrelu_alpha, use_activation,
                 use_bias=False):
        super(conv_block, self).__init__()

        if ((type(kernel_size) == list) or (type(kernel_size) == tuple)) and (len(kernel_size) == 3):
            kernel_depth = kernel_size[0]
            kernel_height = kernel_size[1]
            kernel_width = kernel_size[2]
        elif type(kernel_size) == int:
            kernel_depth = kernel_size
            kernel_height = kernel_size
            kernel_width = kernel_size
        else:
            raise ValueError("Kernel_size is invalid:", kernel_size)

        self.pad = conv3d_padding_same(depth=kernel_depth, height=kernel_height, width=kernel_width,
                                       pad_value=pad_value)
        self.conv1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                     stride=strides, bias=use_bias)
        self.norm1 = torch.nn.InstanceNorm3d(filters)
        self.use_activation = use_activation
        self.activation1 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.norm1(x)
        if self.use_activation:
            x = self.activation1(x)
        return x


class ResNet_DCNN_LReLU_V2(torch.nn.Module):
    """
    Similar to ResNet + DCNN, but where residual block and downsampling blocks are interchanged.
    """

    def __init__(self, n_input_channels, depth, height, width, n_features, num_classes, filters, kernel_sizes, strides,
                 pad_value, n_down_blocks, lrelu_alpha, dropout_p, pooling_conv_filters, perform_pooling,
                 linear_units, use_bias=False):
        super(ResNet_DCNN_LReLU_V2, self).__init__()
        self.n_features = n_features
        self.pooling_conv_filters = pooling_conv_filters
        self.perform_pooling = perform_pooling

        # Determine Linear input channel size
        end_depth = math.ceil(depth / (2 ** n_down_blocks[0]))
        end_height = math.ceil(height / (2 ** n_down_blocks[1]))
        end_width = math.ceil(width / (2 ** n_down_blocks[2]))

        # Initialize conv blocks
        in_channels = [n_input_channels] + list(filters[:-1])
        self.blocks = torch.nn.ModuleList()
        for i in range(len(in_channels)):
            use_activation = True

            # Downsampling conv block
            self.blocks.add_module('conv_block%s' % i,
                                   conv_block(in_channels=in_channels[i], filters=filters[i],
                                              kernel_size=kernel_sizes[i], strides=strides[i], pad_value=pad_value,
                                              lrelu_alpha=lrelu_alpha, use_bias=use_bias,
                                              use_activation=use_activation))

            # Residual block
            self.blocks.add_module('resblock%s' % i, BasicResBlock(in_planes=filters[i], planes=filters[i],
                                                                   pad_value=pad_value, lrelu_alpha=lrelu_alpha))

        # Initialize pooling conv
        if self.pooling_conv_filters is not None:
            pooling_conv_kernel_size = [end_depth, end_height, end_width]
            self.pool = pooling_conv(in_channels=filters[-1], filters=pooling_conv_filters,
                                     kernel_size=pooling_conv_kernel_size, strides=1,
                                     lrelu_alpha=lrelu_alpha, use_bias=use_bias)
            end_depth, end_height, end_width = 1, 1, 1
            filters[-1] = self.pooling_conv_filters
        elif self.perform_pooling:
            # self.pool = torch.nn.AvgPool3d(kernel_size=(1, end_height, end_width))
            # end_depth, end_height, end_width = depth, 1, 1
            self.pool = torch.nn.AvgPool3d(kernel_size=(end_depth, end_height, end_width))
            # self.pool = torch.nn.MaxPool3d(kernel_size=(end_depth, end_height, end_width))
            end_depth, end_height, end_width = 1, 1, 1

        # Initialize flatten layer
        self.flatten = torch.nn.Flatten()

        # Initialize linear layers
        self.linear_layers = torch.nn.ModuleList()
        linear_units = [end_depth * end_height * end_width * filters[-1]] + linear_units
        for i in range(len(linear_units) - 1):
            self.linear_layers.add_module('dropout%s' % i, torch.nn.Dropout(dropout_p[i]))
            self.linear_layers.add_module('linear%s' % i,
                                          torch.nn.Linear(in_features=linear_units[i], out_features=linear_units[i + 1],
                                                          bias=use_bias))
            self.linear_layers.add_module('lrelu%s' % i, torch.nn.LeakyReLU(negative_slope=lrelu_alpha))

        # Initialize output layer
        self.out_layer = Output(in_features=linear_units[-1] + self.n_features, out_features=num_classes, bias=use_bias)
        # self.out_layer.__class__.__name__ = 'Output'

    def forward(self, x, features):
        # Blocks
        for block in self.blocks:
            x = block(x)

        # Pooling layers
        if (self.pooling_conv_filters is not None) or self.perform_pooling:
            x = self.pool(x)

        x = self.flatten(x)

        # Linear layers
        for layer in self.linear_layers:
            x = layer(x)

        # Add features
        if self.n_features > 0:
            x = torch.cat([x, features], dim=1)

        # Output
        x = self.out_layer(x)

        return x


