import torch
import torch.nn as nn

from .expander_module import ExpanderLinear
from .expander_scatter_module import ExpanderScatterLinear


class LinearLayer(nn.Module):
    def __init__(self, indim, outdim, bias=True, expander=False, scatter=False, **kwargs):
        super(LinearLayer, self).__init__()
        self.expander, self.scatter = expander, scatter
        self.indim, self.outdim = indim, outdim
        
        if self.expander and self.scatter:
            self.layer = ExpanderScatterLinear(indim, outdim, bias=bias, **kwargs)
        
        elif self.expander:
            self.layer = ExpanderLinear(indim, outdim, bias=bias, **kwargs)
        
        else:
            self.layer = nn.Linear(indim, outdim, bias=bias)

    def forward(self, _input):
        return self.layer(_input)



