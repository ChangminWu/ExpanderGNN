import torch
import torch.nn as nn

import dgl

from layers.gcn_layer import GCNLayer
from expander.expander_layer import LinearLayer
from utils import activations


class GCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        indim = net_params["in_dim"]
        hiddim = net_params["hidden_dim"]
        outdim = net_params["out_dim"]

        n_classes = net_params["n_classes"]
        in_feat_dropout = net_params["in_feat_dropout"]
        dropout = net_params["dropout"]
        n_layers = net_params["L"]

        self.graph_pool = net_params["graph_pool"]
        self.neighbor_pool = net_params["neighbor_pool"]
        
        
        self.residual = net_params["residual"]
        self.batch_norm = net_params["batch_norm"]

        self.density = net_params["density"]

        self.activation = activations(net_params["activation"])

        self.node_encoder = LinearLayer(indim, hiddim, )