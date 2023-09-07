import torch
from functools import reduce
from operator import __add__


class conv3d_padding_same(torch.nn.Module):
    """
    Padding so that the next Conv3d layer outputs an array with the same dimension as the input.
    Depth, height and width are the kernel dimensions.

    Example:
    import torch
    from functools import reduce
    from operator import __add__

    batch_size = 8
    in_channels = 3
    out_channel = 16
    kernel_size = (2, 3, 5)
    stride = 1  # could also be 2, or 3, etc.
    pad_value = 0
    conv = torch.nn.Conv3d(in_channels, out_channel, kernel_size, stride=stride)

    x = torch.empty(batch_size, in_channels, 100, 100, 100)
    conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
    y = torch.nn.functional.pad(x, conv_padding, 'constant', pad_value)

    out = conv(y)
    print(out.shape): torch.Size([8, 16, 100, 100, 100])

    Source: https://stackoverflow.com/questions/58307036/is-there-really-no-padding-same-option-for-pytorchs-conv2d
    Source: https://pytorch.org/docs/master/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
    """

    def __init__(self, depth, height, width, pad_value):
        super(conv3d_padding_same, self).__init__()
        self.kernel_size = (depth, height, width)
        self.pad_value = pad_value

    def forward(self, x):
        # Determine amount of padding
        # Internal parameters used to reproduce Tensorflow "Same" padding.
        # For some reasons, padding dimensions are reversed wrt kernel sizes.
        conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]])
        x_padded = torch.nn.functional.pad(x, conv_padding, 'constant', self.pad_value)

        return x_padded


class depthwise_separable_conv(torch.nn.Module):
    """
    Note: apply padding (if necessary) before applying depthwise_separable_conv().

    Source: https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch/blob/master/DepthwiseSeparableConvolution/DepthwiseSeparableConvolution.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = torch.nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                         stride=stride, groups=in_channels, bias=bias)
        self.pointwise = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                         bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Output(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Output, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.fc = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, x):
        out = self.fc(x)
        return out


class pooling_conv(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, lrelu_alpha, use_bias=False):
        super(pooling_conv, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                     stride=strides, bias=use_bias)
        # No InstanceNorm, because output size is (batch_size, filters, 1, 1, 1)
        # self.norm1 = torch.nn.InstanceNorm3d(filters)
        self.activation1 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.norm1(x)
        x = self.activation1(x)
        return x


class reshape_tensor(torch.nn.Module):
    """
    Reshape tensor.
    """
    def __init__(self, *args):
        super(reshape_tensor, self).__init__()
        self.output_dim = []
        for a in args:
            self.output_dim.append(a)

    def forward(self, x, batch_size):
        output_dim = [batch_size] + self.output_dim
        x = x.view(output_dim)

        return x
