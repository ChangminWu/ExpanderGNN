import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_sparse.tensor import SparseTensor


class ExpanderLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, mask: Tensor, bias: bool = True) -> None:
        super(ExpanderLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.mask = mask

        assert self.mask.size() == self.weight.size(), "weight size and mask size not match"

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        self.weight.mul_(self.mask)
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class SparseLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, mask: Tensor, bias: bool = True) -> None:
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        n_params = torch.nonzero(mask)
        self.weight = Parameter(torch.Tensor(n_params))
        inds = torch.nonzero(self.mask, as_tuple=True)
        self.sparse_weight = SparseTensor(row=inds[0], col=inds[1], sparse_sizes=(in_features, out_features),
                                          value=self.weight)

    def forward(self, _input: Tensor) -> Tensor:
        if self.bias is not None:
            return (self.sparse_weight @ _input.t()).t() + self.bias
        else:
            return (self.sparse_weight @ _input.t()).t()

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )