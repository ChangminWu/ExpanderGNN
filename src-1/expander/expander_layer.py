import torch
import torch.nn as nn
from torch import Tensor

from src.expander.expander_module import ExpanderLinear, SparseLinear
from src.expander.helper import sampler

from typing import Optional


class LinearLayer(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, bias: bool = True, layer_type: str = "full",
                 sample_method: str = "prabhu", density: float = 1.0, mask: Optional[Tensor] = None) -> None:
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.type, self.sample_method = layer_type.lower(), sample_method.lower()
        self.density = density

        if self.type == "full":
            self.mask = torch.ones([out_features, in_features])
            self.linear = nn.Linear(in_features, out_features, bias=self.bias)

        elif self.type == "expander":
            self.mask = self.generate_mask(in_features, out_features, self.density, sample_method, mask)
            self.linear = ExpanderLinear(in_features, out_features, mask=self.mask, bias=self.bias)

        elif self.type == "sparse":
            self.mask = self.generate_mask(in_features, out_features, self.density, sample_method, mask)
            self.linear = SparseLinear(in_features, out_features, mask=self.mask, bias=self.bias)

        self.n_params = torch.nonzero(self.mask, as_tuple=False).size(0)
        if self.bias:
            self.n_params += out_features

        self.register_buffer("mask", self.mask)
        self.register_buffer("n_params", self.n_params)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, input: Tensor) -> Tensor:
        return self.linear(input)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, type={},' \
               'sample_method={}, density={}'.format(
                self.in_features, self.out_features, self.bias, self.type,
                self.sample_method, self.density)

    @staticmethod
    def generate_mask(in_features: int, out_features: int, density: float,
                      sample_method: str = "prabhu", mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            return mask
        else:
            return sampler(in_features, out_features, density, sample_method)













