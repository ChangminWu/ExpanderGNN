import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch_sparse import SparseTensor

import math
from typing import Optional


class ExpanderLinear(nn.Module):
    __constants__ = ["indim", "outdim"]
    indim: int
    outdim: int
    weight: SparseTensor

    def __init__(self, indim: int, outdim: int, bias: bool=True, edge_index: Optional[Tensor]=None, weight_initializer: Optional[str]=None) -> None:
        super().__init__()

        self.indim, self.outdim, self.edge_index = indim, outdim, edge_index
        self.weight_initializer = weight_initializer
        row, col = self.edge_index

        if bias:
            self.bias = Parameter(torch.Tensor(outdim))
        else:
            self.register_parameter("bias", None)

        self.nnz_weight = Parameter(torch.cuda.FloatTensor(len(row)))
        self.weight = SparseTensor(row=row.to(torch.device('cuda')), col=col.to(torch.device('cuda')), sparse_sizes=(outdim, indim), value=self.nnz_weight)
        
    def reset_parameters(self) -> None:
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        if self.weight_initializer == "glorot":
            stdv = math.sqrt(6.0 / (self.indim + self.outdim))
            self.nnz_weight.data.uniform_(-stdv, stdv)
        elif self.weight_initializer == "glorot-full":
            stdv = math.sqrt(6.0 / self.weight.nnz())
            self.nnz_weight.data.uniform_(-stdv, stdv)
        elif self.weight_initializer == "uniform":
            bound = 1.0 / math.sqrt(self.indim)
            self.nnz_weight.data.uniform(-bound, bound)
        elif self.weight_initializer == "ones":
            nn.init.ones_(self.nnz_weight.data)
        elif self.weight_initializer == None:
            bound = 1.0 / math.sqrt(self.weight.nnz())
            self.nnz_weight.data.uniform(-bound, bound)
        self.weight.set_value_(self.nnz_weight)

    def forward(self, x: Tensor) -> Tensor:
        if self.bias is not None:
            return self.weight.matmul(x.t()).t() + self.bias
        else:
            return self.weight.matmul(x.t()).t()
            
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.indim}, '
                f'{self.outdim}, bias={self.bias is not None})')


        





        




