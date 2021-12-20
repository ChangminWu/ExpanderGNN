from typing import Optional
from torch import Tensor

from torch_geometric.nn.conv import GCNConv
from expander.expander import ExpanderLinear
    

class ExpanderGCN(GCNConv):
    def __init__(self, indim: int, outdim: int, improved: bool = False, cached: bool = False, add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, edge_index: Optional[Tensor] = None, weight_initializer: Optional[str] = None, **kwargs):
        super().__init__(indim, outdim, improved, cached, add_self_loops, normalize, bias, **kwargs)
        self.lin = ExpanderLinear(indim, outdim, False, edge_index, weight_initializer)

