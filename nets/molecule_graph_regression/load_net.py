from nets.molecule_graph_regression.mlp_net import MLPNet
from nets.molecule_graph_regression.gcn_net import GCNNet
from nets.molecule_graph_regression.gin_net import GINNet
from nets.molecule_graph_regression.graphsage_net import GraphSageNet
from nets.molecule_graph_regression.gated_gcn_net import GatedGCNNet
from nets.molecule_graph_regression.gat_net import GATNet
from nets.molecule_graph_regression.pna_net import PNANet

from nets.molecule_graph_regression.simple_mlp_net import SimpleMLPNet
from nets.molecule_graph_regression.simple_gcn_net import SimpleGCNNet
from nets.molecule_graph_regression.simple_gin_net import SimpleGINNet
from nets.molecule_graph_regression.simple_graphsage_net\
    import SimpleGraphSageNet
from nets.molecule_graph_regression.simple_gated_gcn_net import\
    SimpleGatedGCNNet
from nets.molecule_graph_regression.simple_gat_net\
    import SimpleGATNet
from nets.molecule_graph_regression.simple_pna_net\
    import SimplePNANet

from nets.molecule_graph_regression.activation_mlp_net\
    import ActivationMLPNet
from nets.molecule_graph_regression.activation_gcn_net\
    import ActivationGCNNet
from nets.molecule_graph_regression.activation_gin_net\
    import ActivationGINNet
from nets.molecule_graph_regression.activation_graphsage_net\
    import ActivationGraphSageNet
from nets.molecule_graph_regression.activation_gated_gcn_net\
    import ActivationGatedGCNNet
from nets.molecule_graph_regression.activation_gat_net\
    import ActivationGATNet
from nets.molecule_graph_regression.activation_pna_net\
    import ActivationPNANet


def MLP(net_params):
    return MLPNet(net_params)


def GCN(net_params):
    return GCNNet(net_params)


def GIN(net_params):
    return GINNet(net_params)


def GraphSage(net_params):
    return GraphSageNet(net_params)


def GatedGCN(net_params):
    return GatedGCNNet(net_params)


def GAT(net_params):
    return GATNet(net_params)


def PNA(net_params):
    return PNANet(net_params)


def SimpleMLP(net_params):
    return SimpleMLPNet(net_params)


def SimpleGCN(net_params):
    return SimpleGCNNet(net_params)


def SimpleGIN(net_params):
    return SimpleGINNet(net_params)


def SimpleGraphSage(net_params):
    return SimpleGraphSageNet(net_params)


def SimpleGatedGCN(net_params):
    return SimpleGatedGCNNet(net_params)


def SimpleGAT(net_params):
    return SimpleGATNet(net_params)


def SimplePNA(net_params):
    return SimplePNANet(net_params)


def ActivationMLP(net_params):
    return ActivationMLPNet(net_params)


def ActivationGCN(net_params):
    return ActivationGCNNet(net_params)


def ActivationGIN(net_params):
    return ActivationGINNet(net_params)


def ActivationGraphSage(net_params):
    return ActivationGraphSageNet(net_params)


def ActivationGatedGCN(net_params):
    return ActivationGatedGCNNet(net_params)


def ActivationGAT(net_params):
    return ActivationGATNet(net_params)


def ActivationPNA(net_params):
    return ActivationPNANet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        "MLP": MLP,
        "GCN": GCN,
        "GIN": GIN,
        "GraphSage": GraphSage,
        "GatedGCN": GatedGCN,
        "GAT": GAT,
        "PNA": PNA,
        "SimpleMLP": SimpleMLP,
        "SimpleGCN": SimpleGCN,
        "SimpleGIN": SimpleGIN,
        "SimpleGraphSage": SimpleGraphSage,
        "SimpleGatedGCN": SimpleGatedGCN,
        "SimpleGAT": SimpleGAT,
        "SimplePNA": SimplePNA,
        "ActivationMLP": ActivationMLP,
        "ActivationGCN": ActivationGCN,
        "ActivationGIN": ActivationGIN,
        "ActivationGraphSage": ActivationGraphSage,
        "ActivationGatedGCN": ActivationGatedGCN,
        "ActivationGAT": ActivationGAT,
        "ActivationPNA": ActivationPNA
    }

    return models[MODEL_NAME](net_params)
