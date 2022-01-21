from typing import Optional, Union, Tuple
import torch
from torch import Tensor

from torch_geometric.nn.conv import GCNConv, SAGEConv
from expander.expander import ExpanderLinear
    

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