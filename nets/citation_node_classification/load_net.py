from nets.citation_node_classification.mlp_net import MLPNet
from nets.citation_node_classification.gcn_net import GCNNet
from nets.citation_node_classification.gin_net import GINNet
from nets.citation_node_classification.graphsage_net import GraphSageNet
from nets.citation_node_classification.pna_net import PNANet

from nets.citation_node_classification.simple_gcn_net import SimpleGCNNet

from nets.citation_node_classification.activation_gcn_net import ActivationGCNNet
from nets.citation_node_classification.activation_gin_net import ActivationGINNet
from nets.citation_node_classification.activation_graphsage_net import ActivationGraphSageNet
from nets.citation_node_classification.activation_pna_net import ActivationPNANet


def MLP(net_params):
    return MLPNet(net_params)


def GCN(net_params):
    return GCNNet(net_params)


def GIN(net_params):
    return GINNet(net_params)


def GraphSage(net_params):
    return GraphSageNet(net_params)


def PNA(net_params):
    return PNANet(net_params)


def SimpleGCN(net_params):
    return SimpleGCNNet(net_params)


def ActivationGCN(net_params):
    return ActivationGCNNet(net_params)


def ActivationGIN(net_params):
    return ActivationGINNet(net_params)


def ActivationGraphSage(net_params):
    return ActivationGraphSageNet(net_params)


def ActivationPNA(net_params):
    return ActivationPNANet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        "MLP": MLP,
        "GCN": GCN,
        "GIN": GIN,
        "GraphSage": GraphSage,
        "PNA": PNA,
        "SimpleGCN": SimpleGCN,
        "ActivationGCN": ActivationGCN,
        "ActivationGIN": ActivationGIN,
        "ActivationGraphSage": ActivationGraphSage,
        "ActivationPNA": ActivationPNA
    }

    return models[MODEL_NAME](net_params)