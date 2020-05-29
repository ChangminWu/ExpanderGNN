import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add
from torch_sparse import spmm


class ExpanderLinearLayer(nn.Module):
    def __init__(self, input_feat, output_feat, sparsity=None, bias=True):
        super(ExpanderLinearLayer, self).__init__()
        self.indim, self.outdim = input_feat, output_feat
        self.sparsity = sparsity
        if self.sparsity is not None:
            self.n_weight_params = min(self.indim, self.outdim) * int(max(self.indim, self.outdim) * self.sparsity)
        else:
            self.n_weight_params = self.indim*self.outdim

        self.weight = nn.Parameter(data=torch.Tensor(self.n_weight_params))
        if bias:
            self.bias = nn.Parameter(data=torch.Tensor(self.outdim))
            self.n_params = self.n_weight_params+ self.outdim
        else:
            self.register_parameter("bias", None)
            self.n_params = self.n_weight_params

        self.register_buffer("mask", None)

        self.reset_parameters()

    def forward(self, input_):
        x = input_[:, self.ind_in]
        x = x * self.weight
        x = scatter_add(x, self.ind_out)
        if self.bias is not None:
            x += self.bias
        return x

    def reset_parameters(self):
        stdv = math.sqrt(2./self.indim)
        self.weight.data = torch.randn(self.n_params) * stdv
        if self.bias is not None:
            self.bias.data = torch.randn(self.outdim) * stdv

    def generate_mask(self, init=None):
        if init is None:
            locs = []
            if self.outdim < self.indim:
                for i in range(self.outdim):
                    x = torch.randperm(self.indim)
                    locs.extend([torch.tensor([x[j], i]).int().reshape(-1, 1)
                                 for j in range(int(self.indim*self.sparsity))])
            else:
                for i in range(self.indim):
                    x = torch.randperm(self.outdim)
                    locs.extend([torch.tensor([i, x[j]]).int().reshape(-1, 1)
                                 for j in range(int(self.outdim * self.sparsity))])

            self.mask = torch.cat(locs, dim=1)
        else:
            self.mask = init

        assert self.mask.size(1) == self.n_weight_params, "sparsity does not match"
        self.ind_in = self.mask[0,:]
        self.ind_out = self.mask[1,:]


class ExpanderDoubleLinearLayer(nn.Module):
    def __init__(self, input_features, output_features, hidden_features, sparsity=None, activation=True):
        super(ExpanderDoubleLinearLayer, self).__init__()
        self.indim = input_features
        self.outdim = output_features
        self.hdim = hidden_features
        self.activation = activation
        self.sparsity = sparsity
        self.layer1 = ExpanderLinearLayer(self.indim, self.hdim, self.sparsity)
        self.layer2 = ExpanderLinearLayer(self.hdim, self.outdim, self.sparsity)

    def forward(self, input_):
        input_ = self.layer1(input_)
        if self.activation:
            input_ = F.relu(input_)
        input_ = self.layer2(input_)
        return input_


class ExpanderMultiLinearLayer(nn.Module):
    def __init__(self, num_layers, input_features, hidden_features, output_features, sparsity=None,
                 activation=None, batchnorm=False):
        super(ExpanderMultiLinearLayer, self).__init__()
        self.indim = input_features
        self.outdim = output_features
        self.hdim = hidden_features
        self.sparsity = sparsity

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")

        elif num_layers == 1:
            layers_modulo = [ExpanderLinearLayer(self.indim, self.outdim, self.sparsity)]

        else:
            layers_modulo = [ExpanderLinearLayer(self.indim, self.hdim, self.sparsity)]
            if batchnorm:
                layers_modulo.append(nn.BatchNorm1d(self.hdim))
            if activation is not None:
                layers_modulo.append(activation)

            for i in range(num_layers-1):
                if i == num_layers-2:
                    layers_modulo.append(ExpanderLinearLayer(self.hdim, self.outdim, self.sparsity))
                else:
                    layers_modulo.append(ExpanderLinearLayer(self.hdim, self.hdim, self.sparsity))
                    if batchnorm:
                        layers_modulo.append(nn.BatchNorm1d(self.hdim))
                    if activation is not None:
                        layers_modulo.append(activation)


        self.layers = nn.Sequential(*layers_modulo)

    def forward(self, x):
        return self.layers(x)






















