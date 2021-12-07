import torch
import torch.nn as nn   

class ExpanderLinear(nn.Module):
    def __init__(self, indim, outdim, bias=True, density=None):
        self.indim, self.outdim = indim, outdim
        
