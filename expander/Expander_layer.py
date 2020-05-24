import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class ExpanderLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, mask, bias=None):
        ctx.save_for_backward(input_, weight, bias)
        weight.mul_(mask)
        ctx.mask = mask
        output = input_.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        weight.mul_(ctx.mask)
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input_)
            grad_weight.mul_(ctx.mask)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


class ExpanderLinearLayer(nn.Module):
    def __init__(self, input_feat, output_feat, sparsity=None, bias=True):
        super(ExpanderLinearLayer, self).__init__()
        self.indim, self.outdim = input_feat, output_feat
        self.sparsity = sparsity
        if self.sparsity is not None:
            self.n_params = min(self.indim, self.outdim) * int(max(self.indim, self.outdim) * self.sparsity)
        else:
            self.n_params = self.indim*self.outdim

        self.weight = nn.Parameter(data=torch.Tensor(self.outdim, self.indim))
        if bias:
            self.bias = nn.Parameter(data=torch.Tensor(self.outdim))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("mask", None)

        self.reset_parameters()

    def forward(self, input_):
        return ExpanderLinear.apply(input_, self.weight, self.mask, self.bias)

    def reset_parameters(self):
        # nn.init.kaiming_normal_(self.weight, mode='fan_in')
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            # bound = 1 / math.sqrt(fan_in)
            # nn.init.uniform_(self.bias, -bound, bound)
            # nn.init.constant_(self.bias, 0.)
            nn.init.zeros_(self.bias)

    def generate_mask(self, init=None):
        if init is None:
            self.mask = torch.zeros(self.outdim, self.indim)
            if self.outdim < self.indim:
                for i in range(self.outdim):
                    x = torch.randperm(self.indim)
                    for j in range(int(self.indim*self.sparsity)):
                        self.mask[i][x[j]] = 1
            else:
                for i in range(self.indim):
                    x = torch.randperm(self.outdim)
                    for j in range(int(self.outdim*self.sparsity)):
                        self.mask[x[j]][i] = 1
        else:
            assert self.sparsity == int(torch.sum(init) / (self.indim * self.outdim)), "sparsity does not match"
            self.mask = init


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






















