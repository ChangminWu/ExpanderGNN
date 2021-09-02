import math
import torch
import torch.nn as nn

# from torch_scatter import scatter_add
# from torch_sparse import spmm

from layers.expander.samplers import sampler


class ExpanderLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, mask, bias=None):
        ctx.save_for_backward(_input, weight, bias)
        ctx.mask = mask
        weight.mul_(mask)

        if bias is not None:
            output = torch.addmm(bias, _input, weight.t())
        else:
            output = _input.mm(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        weight.mul_(ctx.mask)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(_input)
            grad_weight.mul_(ctx.mask)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, None, grad_bias


class ExpanderLinear(nn.Module):
    def __init__(self, indim, outdim, density=None, bias=True,
                 sampler="regular"):
        super(ExpanderLinear, self).__init__()
        self.indim, self.outdim, self.density, self.sampler = (indim,
                                                               outdim,
                                                               density,
                                                               sampler)

        self.weight = nn.Parameter(data=torch.Tensor(self.outdim, self.indim))

        if bias:
            self.bias = nn.Parameter(data=torch.Tensor(self.outdim))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("mask", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # nn.init.kaiming_normal_(self.weight, mode="fan_ins")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, _input):
        return ExpanderLinearFunction.apply(_input, self.weight,
                                            self.mask, self.bias)

    def generate_mask(self, init=None):
        if init is None:
            self.mask, self.n_params = sampler(self.outdim,
                                               self.indim,
                                               self.density,
                                               method=self.sampler)
        else:
            self.n_params = torch.sum(init)
            self.mask = init

        if self.bias is not None:
            self.n_params += self.outdim


class ExpanderScatterLinear(nn.Module):
    def __init__(self, indim, outdim, density=None,
                 bias=True, sampler="regular"):
        super(ExpanderScatterLinear, self).__init__()
        self.indim, self.outdim, self.density, self.sampler = (indim,
                                                               outdim,
                                                               density,
                                                               sampler)

        if bias:
            self.bias = nn.Parameter(data=torch.Tensor(self.outdim))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("mask", None)

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.weight)
        # # nn.init.kaiming_normal_(self.weight, mode="fan_ins")
        # if self.bias is not None:
        #     nn.init.zeros_(self.bias)
        stdv = math.sqrt(2./self.indim)
        self.weight.data = torch.randn(self.n_params) * stdv
        if self.bias is not None:
            self.bias.data = torch.randn(self.outdim) * stdv

    def forward(self, _input):
        x = _input[:, self.ind_in]
        x = x*self.weight
        x = scatter_add(x, self.ind_out.to(_input.device))
        # x = spmm(self.inds, self.weight,
        #          self.outdim, self.indim, _input.t()).t()

        if self.bias is not None:
            x += self.bias
        return x

    def generate_mask(self, init=None):
        if init is None:
            self.mask, self.n_params = sampler(self.outdim,
                                               self.indim,
                                               self.density,
                                               method=self.sampler)

        else:
            self.n_params == torch.sum(init)
            self.mask = init

        self.weight = nn.Parameter(data=torch.Tensor(self.n_params))
        self.reset_parameters()

        self.inds = torch.nonzero(self.mask, as_tuple=True)
        self.ind_in = self.inds[1]
        self.ind_out = self.inds[0]

        if self.bias is not None:
            self.n_params += self.outdim
