import torch 
import torch.nn as nn
from .expander_layer import ExpanderLinearFunction
from .samplers import sampler


class ExpanderLinear(nn.Module):
    def __init__(self, indim, outdim, density=None, bias=True, sampler="regular"):
        super(ExpanderLinear, self).__init__()
        self.indim, self.outdim, self.density, self.sampler = indim, outdim, density, sampler

        self.weight = nn.Parameter(data=torch.Tensor(self.outdim, self.indim))
        
        if bias:
            self.bias = nn.Parameter(data=torch.Tensor(self.outdim))
            self.n_params += self.outdim
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
        return ExpanderLinearFunction.apply(_input, self.weight, self.mask, self.bias)

    def generate_mask(self, init=None):
        if init is None:
            self.mask, self.n_params = sampler(self.outdim, self.indim, self.density, method=self.sampler)
        else:
            self.n_params == torch.sum(init)
            self.mask = init
        
    
        
