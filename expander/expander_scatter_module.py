import math
import torch
import torch.nn as nn

from torch_scatter import scatter_add
# from torch_sparse import spmm

from expander.samplers import sampler


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
        x = scatter_add(x, self.ind_out)
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

        self.inds = torch.nonzero(self.mask,
                                  as_tuple=True).to(self.weight.device)
        self.ind_in = self.inds[1]
        self.ind_out = self.inds[0]

        if self.bias is not None:
            self.n_params += self.outdim
