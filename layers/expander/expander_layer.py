import torch
import torch.nn as nn

from layers.expander.expander_module import ExpanderLinear, ExpanderScatterLinear


class LinearLayer(nn.Module):
    def __init__(self, indim, outdim, bias=True,
                 linear_type="expander", **kwargs):
        super(LinearLayer, self).__init__()
        self.linear_type = linear_type
        self.indim, self.outdim = indim, outdim
        self.bias = bias

        if self.linear_type == "scatter":
            self.layer = ExpanderScatterLinear(indim, outdim,
                                               bias=self.bias, **kwargs)
        elif self.linear_type == "expander":
            self.layer = ExpanderLinear(indim, outdim, bias=self.bias,
                                        **kwargs)
        elif self.linear_type == "regular":
            self.layer = nn.Linear(indim, outdim, bias=self.bias)
            self.reset_parameters()
        else:
            raise ValueError("Invalid linear transform type.")

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer.weight)
        # nn.init.kaiming_normal_(self.layer.weight, mode="fan_ins")
        if self.layer.bias is not None:
            nn.init.zeros_(self.layer.bias)
        pass

    def forward(self, _input):
        return self.layer(_input)


class MultiLinearLayer(nn.Module):
    def __init__(self, indim, outdim, activation, batch_norm, num_layers,
                 hiddim=None, bias=False, linear_type="expander",
                 **kwargs):
        super(MultiLinearLayer, self).__init__()
        self.linear_type = linear_type
        self.indim, self.outdim, self.hiddim, self.num_layers = (indim,
                                                                 outdim,
                                                                 hiddim,
                                                                 num_layers)
        self.bias = bias

        layers = []
        sizes = [self.indim]
        sizes.extend([self.hiddim]*(self.num_layers-1))
        sizes.append(self.outdim)
        for i in range(len(sizes)-1):
            if i == 0:
                layers.append(LinearLayer(sizes[0], sizes[1], bias=self.bias,
                              linear_type=self.linear_type, **kwargs))
            else:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(sizes[i]))
                if activation is not None:
                    layers.append(activation)
                layers.append(LinearLayer(sizes[i], sizes[i+1], bias=self.bias,
                              linear_type=self.linear_type, **kwargs))

        self.layers = nn.Sequential(*layers)

    def forward(self, _input):
        return self.layers(_input)
