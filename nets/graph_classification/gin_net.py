import torch.nn as nn

import dgl 
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

from layers.gin_layer import GINLayer
from expander.expander_layer import LinearLayer, MultiLinearLayer
from utils import activations


class GINNet(nn.Module):
    def __init__(self, net_params):
        super(GINNet, self).__init__()
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
        self.n_mlp_layer = net_params["mlp_layers"]

        self.activation = activations(net_params["activation"])
        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]

        linear_params = {"density": self.density, "sampler": self.sampler}

        self.node_encoder = LinearLayer(indim, hiddim, bias=self.bias,
                                        linear_type=self.linear_type,
                                        **linear_params)

