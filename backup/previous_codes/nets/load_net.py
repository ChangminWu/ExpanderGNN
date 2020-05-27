"""
    Utility file to select GraphNN model as
    selected by the user
"""
from collections import OrderedDict

from nets.diffpool_net import DiffPoolNet
from nets.expander_gcn_net import ExpanderGCNNet
from nets.expander_gin_net import ExpanderGINNet
from nets.expander_mlp_net import ExpanderMLPNet
from nets.gated_gcn_net import GatedGCNNet
from nets.gcn_net import GCNNet
from nets.gin_net import GINNet
from nets.graphsage_net import GraphSageNet
from nets.mlp_net import MLPNet
from nets.mo_net import MoNet as MoNet_


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


def init_expander(net, saved_mask=None, saved_layers=None, expand_size=2):
    num_children = len(list(net.children()))
    layer_name = str(net.__class__).split(".")[-1].split("'")[0]  #str(net)[:str(net).find('(')]
    if num_children == 0:
        label = layer_name + "_" + str(len(saved_layers))
        if label in saved_mask:
            # print(label)
            net._init_mask(saved_mask[label])
            # net.mask = saved_mask[label]
            saved_layers.append(label)
        elif "ExpanderLinear" in label:
            net.expand_size = expand_size
            # print("net expander size", expand_size)
            net._init_mask()
            # output_features, input_features = net.output_features, net.input_features  # list(net.parameters())[0].size()
            # mask = torch.zeros(output_features, input_features)
            # if output_features < input_features:
            #     for i in range(output_features):
            #         x = torch.randperm(input_features)
            #         for j in range(expand_size):
            #             mask[i][x[j]] = 1
            # else:
            #     for i in range(input_features):
            #         x = torch.randperm(output_features)
            #         for j in range(expand_size):
            #             mask[x[j]][i] = 1
            # net.mask = mask.cuda()
            saved_mask[label] = net.mask  # ._indices().cpu().detach().numpy()
            saved_layers.append(label)
    else:
        if layer_name not in saved_mask:
            saved_mask[layer_name] = OrderedDict()
        for child in net.children():
            # label_child = str(child)[:str(child).find('(')]
            # if "Expander" in label_child:
            saved_mask[layer_name], saved_layers = init_expander(child, saved_mask[layer_name], saved_layers, expand_size)
    return saved_mask, saved_layers







