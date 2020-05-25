import torch.nn as nn
import torch.nn.functional as F

from expander.Expander_layer import ExpanderLinearLayer, ExpanderDoubleLinearLayer


class MLPReadout(nn.Module):
    def __init__(self, input_features, output_features, n_layers=2):
        super().__init__()
        list_FC_layers = [nn.Linear(input_features // 2**l, input_features // 2**(l+1), bias=True)
                          for l in range(n_layers)]
        list_FC_layers.append(nn.Linear(input_features // 2**n_layers, output_features, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.n_layers = n_layers
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.FC_layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, input_):
        out = input_
        for l in range(self.n_layers):
            out = self.FC_layers[l](out)
            out = F.relu(out)
        out = self.FC_layers[self.n_layers](out)
        return out


class ExpanderMLPReadout(nn.Module):
    def __init__(self, input_features, output_features, sparsity, n_layers=2):
        super().__init__()
        list_FC_layers = [ExpanderLinearLayer(input_features // 2**l, input_features // 2**(l+1), sparsity)
                          for l in range(n_layers)]
        #list_FC_layers.append(ExpanderLinearLayer(input_features // 2**n_layers, output_features, sparsity))
        list_FC_layers.append(nn.Linear(input_features // 2**n_layers, output_features, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.n_layers = n_layers

    def forward(self, input_):
        out = input_
        for l in range(self.n_layers):
            out = self.FC_layers[l](out)
            out = F.relu(out)
        out = self.FC_layers[self.n_layers](out)
        return out


class ExpanderDoubleMLPReadout(nn.Module):
    def __init__(self, input_features, output_features, sparsity, n_layers=2):
        super().__init__()
        list_FC_layers = [ExpanderDoubleLinearLayer(input_features // 2**l, input_features // 2**l,
                                                    input_features // 2**(l+1), sparsity) for l in range(n_layers)]
        list_FC_layers.append(ExpanderDoubleLinearLayer(input_features // 2**n_layers, input_features // 2**n_layers,
                                        output_features, sparsity))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.n_layers = n_layers

    def forward(self, input_):
        out = input_
        for l in range(self.n_layers):
            out = self.FC_layers[l](out)
            out = F.relu(out)
        out = self.FC_layers[self.n_layers](out)
        return out
