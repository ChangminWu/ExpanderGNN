from baselines.nets.gcn_net import GCNNet
from baselines.nets.gated_gcn_net import GatedGCNNet
from baselines.nets.mlp_net import MLPNet
from baselines.nets.gin_net import GINNet
from baselines.nets.simple_gcn import SimpleGCN
from baselines.nets.graphsage_net import GraphSageNet

from models.expander_gcn import ExpanderGCNNet
from models.expander_gatedgcn import ExpanderGatedGCNNet
from models.expander_mlp import ExpanderMLPNet
from models.expander_gin import ExpanderGINNet
from models.expander_simple_gcn import ExpanderSimpleGCN
from models.expander_graphsage import ExpanderGraphSageNet


def GCN(net_params):
    return GCNNet(net_params)

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def SGCN(net_params):
    return SimpleGCN(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def ExpanderGCN(net_params):
    return ExpanderGCNNet(net_params)

def ExpanderGatedGCN(net_params):
    return ExpanderGatedGCNNet(net_params)

def ExpanderMLP(net_params):
    return ExpanderMLPNet(net_params)

def ExpanderGIN(net_params):
    return ExpanderGINNet(net_params)

def ExpanderSGCN(net_params):
    return ExpanderSimpleGCN(net_params)

def ExpanderGraphSage(net_params):
    return ExpanderGraphSageNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GCN': GCN,
        'GatedGCN': GatedGCN,
        'MLP': MLP,
        'SimpleGCN': SGCN,
        'GIN': GIN,
        'GraphSage': GraphSage,
        'ExpanderGCN': ExpanderGCN,
        'ExpanderGatedGCN': ExpanderGatedGCN,
        'ExpanderMLP': ExpanderMLP,
        'ExpanderGIN': ExpanderGIN,
        'ExpanderSimpleGCN': ExpanderSGCN,
        'ExpanderGraphSage': ExpanderGraphSage
    }
    return models[MODEL_NAME](net_params)