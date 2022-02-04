from typing import Optional, Union, Tuple, Dict, List
import torch
from torch import Tensor
from torch.nn import ModuleList, ReLU, Sequential

from torch_geometric.nn.conv import GCNConv, SAGEConv, PNAConv
from torch_geometric.typing import Adj, OptTensor, PairTensor
from expander.expander import ExpanderLinear

from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


class ExpanderPNAConv(PNAConv):
    def __init__(self, indim: int, outdim: int, aggregators: List[str], scalers: List[str], deg: Tensor, edge_dim: Optional[int] = None, towers: int = 1,
                 pre_layers: int = 1, post_layers: int = 1, divide_input: bool = False, bias: bool = True, edge_index: Optional[List[Tensor]] = None, weight_initializer: Optional[str] = None, **kwargs):
        super().__init__(indim, outdim, aggregators, scalers, deg, edge_dim, towers, pre_layers, post_layers, divide_input, **kwargs)
        if self.edge_dim is not None:
            self.edge_encoder = ExpanderLinear(edge_dim, self.F_in, bias, edge_index[0], weight_initializer)
            edge_index.pop(0)
        
        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for i in range(towers):
            modules = [ExpanderLinear((3 if edge_dim else 2) * self.F_in, self.F_in, bias, edge_index[(pre_layers+post_layers)*i], weight_initializer)]
            for j in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [ExpanderLinear(self.F_in, self.F_in, bias, edge_index[j+1+(pre_layers+post_layers)*i], weight_initializer)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [ExpanderLinear(in_channels, self.F_out, bias, edge_index[pre_layers+(pre_layers+post_layers)*i], weight_initializer)]
            for k in range(post_layers - 1):
                modules += [ReLU()]
                modules += [ExpanderLinear(self.F_out, self.F_out, bias, edge_index[pre_layers+k+1+(pre_layers+post_layers)*i], weight_initializer)]
            self.post_nns.append(Sequential(*modules))

        self.lin = ExpanderLinear(outdim, outdim, False, edge_index[-1], weight_initializer)
        self.reset_parameters()


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