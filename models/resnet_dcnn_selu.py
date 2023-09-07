"""
Same as ResNet_DCNN, but where SELU activation has been applied whenever possible.
"""
import math
import torch
from .layers import conv3d_padding_same, Output


def conv3x3x3(in_planes, out_planes, stride=1):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicResBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, pad_value, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.activation = torch.nn.SELU()
        self.conv2 = conv3x3x3(planes, planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.activation(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out


class InvertedResidual(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, pad_value, stride=1, downsample=None):
        super().__init__()
        interm_features = planes * self.expansion

        self.conv1 = conv1x1x1(in_planes, interm_features)
        self.conv2 = conv3x3x3(interm_features, interm_features, stride)
        self.conv3 = conv1x1x1(interm_features, planes)
        self.activation = torch.nn.SELU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.activation(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out


class conv_block(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, pad_value, use_activation,
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
        self.use_activation = use_activation
        self.activation1 = torch.nn.SELU()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)

        if self.use_activation:
            x = self.activation1(x)
        return x


class pooling_conv(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, use_bias=False):
        super(pooling_conv, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                     stride=strides, bias=use_bias)
        self.activation1 = torch.nn.SELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        return x


class ResNet_DCNN_SELU(torch.nn.Module):
    """
    ResNet + DCNN
    """

    def __init__(self, n_input_channels, depth, height, width, n_features, num_classes, filters, kernel_sizes, strides,
                 pad_value, n_down_blocks, dropout_p, pooling_conv_filters, perform_pooling,
                 linear_units, use_bias=False):
        super(ResNet_DCNN_SELU, self).__init__()
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
            # Residual block
            self.blocks.add_module('resblock%s' % i, BasicResBlock(in_planes=in_channels[i], planes=in_channels[i],
                                                                   pad_value=pad_value))
            # Downsampling conv block
            self.blocks.add_module('conv_block%s' % i,
                                   conv_block(in_channels=in_channels[i], filters=filters[i],
                                              kernel_size=kernel_sizes[i], strides=strides[i], pad_value=pad_value,
                                              use_bias=use_bias,
                                              use_activation=use_activation))

        # Initialize pooling conv
        if self.pooling_conv_filters is not None:
            pooling_conv_kernel_size = [end_depth, end_height, end_width]
            self.pool = pooling_conv(in_channels=filters[-1], filters=pooling_conv_filters,
                                     kernel_size=pooling_conv_kernel_size, strides=1,
                                     use_bias=use_bias)
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
            self.linear_layers.add_module('dropout%s' % i, torch.nn.AlphaDropout(dropout_p[i]))
            self.linear_layers.add_module('linear%s' % i,
                                          torch.nn.Linear(in_features=linear_units[i], out_features=linear_units[i + 1],
                                                          bias=use_bias))
            self.linear_layers.add_module('selu%s' % i, torch.nn.SELU())

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


