import torch
import torch.nn as nn

# import numpy as np


class LinearActiveLayer(nn.Module):
    def __init__(self, indim):
        super(LinearActiveLayer, self).__init__()
        self.weight = nn.Parameter(data=torch.Tensor(1, indim))
        self.reset_parameters()

    def forward(self, _input):
        return _input*self.weight

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)


class BiasedRELULayer(nn.Module):
    def __init__(self, intercept=False):
        super(BiasedRELULayer, self).__init__()
        self.alpha = nn.Parameter(data=torch.Tensor(1))
        if intercept:
            self.beta = nn.Parameter(data=torch.Tensor(1))
        else:
            self.register_parameter("beta", None)
        self.reset_parameters()

    def forward(self, _input):
        if self.beta is None:
            return torch.max(_input-self.alpha, torch)
        else:
            return torch.max(_input-self.alpha, torch) + self.beta

    def reset_parameters(self):
        # nn.init.uniform_(self.alpha, a=-np.sqrt(6), b=np.sqrt(6))
        nn.init.zeros_(self.alpha)
        if self.beta is not None:
            nn.init.zeros_(self.beta)


class ConvActivLayer(nn.Module):
    def __init__(self):
        super(ConvActivLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)

    def forward(self, _input):
        _input = _input[:, None, :]
        _input = self.conv(_input)
        return _input.squeeze(1)
