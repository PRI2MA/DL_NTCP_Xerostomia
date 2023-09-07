import torch
from .layers import Output


class MLP(torch.nn.Module):
    """
    Traditional MLP
    """

    def __init__(self, n_features, num_classes, lrelu_alpha, dropout_p, linear_units, use_bias=False):
        super(MLP, self).__init__()
        self.n_features = n_features

        # Initialize linear layers
        self.linear_layers = torch.nn.ModuleList()
        linear_units = [self.n_features] + linear_units
        for i in range(len(linear_units) - 1):
            self.linear_layers.add_module('linear%s' % i,
                                          torch.nn.Linear(in_features=linear_units[i], out_features=linear_units[i+1],
                                                          bias=use_bias))
            self.linear_layers.add_module('lrelu%s' % i, torch.nn.LeakyReLU(negative_slope=lrelu_alpha))
            self.linear_layers.add_module('dropout%s' % i, torch.nn.Dropout(dropout_p[i]))

        # Initialize output layer
        self.out_layer = Output(in_features=linear_units[-1], out_features=num_classes, bias=use_bias)
        # self.out_layer.__class__.__name__ = 'Output'

    def forward(self, x, features):
        # Linear layers
        h = features

        for layer in self.linear_layers:
            h = layer(h)

        # Output
        h = self.out_layer(h)

        return h


