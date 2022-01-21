from typing import Optional, Union, Tuple
import torch
from torch import Tensor

from torch_geometric.nn.conv import GCNConv, SAGEConv
from torch_geometric.typing import Adj, OptTensor, PairTensor
from expander.expander import ExpanderLinear

from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]



class ExpanderGCNConv(GCNConv):
    def __init__(self, indim: int, outdim: int, improved: bool = False, cached: bool = False, add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, edge_index: Optional[Tensor] = None, weight_initializer: Optional[str] = None, **kwargs):
        super().__init__(indim, outdim, improved, cached, add_self_loops, normalize, bias, **kwargs)
        self.lin = ExpanderLinear(indim, outdim, False, edge_index, weight_initializer)
        self.reset_parameters()


class ActivationGCNConv(GCNConv):
    def __init__(self, indim: int, outdim: int, improved: bool = False, cached: bool = False, add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = False, **kwargs):
        super().__init__(indim, outdim, improved, cached, add_self_loops, normalize, False, **kwargs)
        self.lin = torch.nn.Identity()

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
            """"""

            if self.normalize:
                if isinstance(edge_index, Tensor):
                    cache = self._cached_edge_index
                    if cache is None:
                        edge_index, edge_weight = gcn_norm(  # yapf: disable
                            edge_index, edge_weight, x.size(self.node_dim),
                            self.improved, self.add_self_loops)
                        if self.cached:
                            self._cached_edge_index = (edge_index, edge_weight)
                    else:
                        edge_index, edge_weight = cache[0], cache[1]

                elif isinstance(edge_index, SparseTensor):
                    cache = self._cached_adj_t
                    if cache is None:
                        edge_index = gcn_norm(  # yapf: disable
                            edge_index, edge_weight, x.size(self.node_dim),
                            self.improved, self.add_self_loops)
                        if self.cached:
                            self._cached_adj_t = edge_index
                    else:
                        edge_index = cache

            print("before iden, ", x)
            x = self.lin(x)
            print("after iden, ", x)
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                size=None)
            print("after prop, ", out)
            if self.bias is not None:
                out += self.bias

            return out

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None


class ExpanderSAGEConv(SAGEConv):
    def __init__(self,  indim: Union[int, Tuple[int, int]], outdim: int, normalize: bool = False, root_weight: bool = True,
                 bias: bool = True, edge_index: Union[Optional[Tensor], Tuple[Tensor, Tensor]] = None, weight_initializer: Optional[str] = None, **kwargs):
        super().__init__(indim, outdim, normalize, root_weight, bias, **kwargs)
        if isinstance(indim, int):
            indim = (indim, indim)
        if isinstance(edge_index, Tensor):
            edge_index = (edge_index, edge_index)
        self.lin_l = ExpanderLinear(indim[0], outdim, bias, edge_index[0], weight_initializer)
        if root_weight:
            self.lin_r = ExpanderLinear(indim[1], outdim, False, edge_index[1], weight_initializer)
        self.reset_parameters()