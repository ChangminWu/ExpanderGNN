import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.expander.expander_layer import ExpanderLinear

"""
    MLP Layer used after graph vector representation
"""

class ExpanderMLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ ExpanderLinear( input_dim//2**l , input_dim//2**(l+1), expandSize=(input_dim//2**(l+1))//8 ) for l in range(L) ]
        list_FC_layers.append(ExpanderLinear( input_dim//2**L , output_dim , expandSize=(input_dim//2**L)//4))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y