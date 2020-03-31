"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.gated_gcn_net import GatedGCNNet
from nets.gcn_net import GCNNet
from nets.graphsage_net import GraphSageNet
from nets.gin_net import GINNet
from nets.mo_net import MoNet as MoNet_
from nets.diffpool_net import DiffPoolNet
from nets.mlp_net import MLPNet
from nets.expander_gin_net import ExpanderGINNet
from nets.expander_gcn_net import ExpanderGCNNet
from nets.expander_mlp_net import ExpanderMLPNet


def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)

def DiffPool(net_params):
    return DiffPoolNet(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def ExpanderMLP(net_params):
    return ExpanderMLPNet(net_params)

def ExpanderGCN(net_params):
    return ExpanderGCNNet(net_params)

def ExpanderGIN(net_params):
    return ExpanderGINNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'MoNet': MoNet_,
        'DiffPool': DiffPool,
        'MLP': MLP,
        'ExpanderGCN': ExpanderGCN,
        'ExpanderGIN': ExpanderGIN,
        'ExpanderMLP': ExpanderMLP
        
    }
        
    return models[MODEL_NAME](net_params)