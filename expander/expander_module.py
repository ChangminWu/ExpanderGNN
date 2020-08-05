import torch 
import torch.nn as nn


class ExpanderLinear(nn.Module):
    def __init__(self, indim, outdim, density=None, bias=True, sampler="regular"):
        super(ExpanderLinear, self).__init__()
        self.indim, self.outdim, self.density = indim, outdim, density
        
        if self.density is not None:
            self.n_params = min(self.indim, self.outdim) * int(max(self.indim, self.outdim) * self.density)
        else:
            self.n_params = self.indim * self.outdim

        self.weight = nn.Parameter(data=torch.Tensor(self.outdim, self.indim))
        
        if bias:
            self.bias = nn.Parameter(data=torch.Tensor(self.outdim))
            self.n_params += self.outdim
        else:
            self.register_parameter("bias", None)
        
        self.register_buffer("mask", None)
        self.reset_parameters()

    def reset_parameters(self):
        
    
        
