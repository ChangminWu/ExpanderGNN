import torch
import torch.nn as nn

import dgl

from layers.activation_pna_layer import ActivationPNALayer,\
    ActivationPNASimplifiedLayer
from expander.expander_layer import LinearLayer
from utils import activations


"""
    PNA: Principal Neighbourhood Aggregation
    Gabriele Corso, Luca Cavalleri, Dominique Beaini,
    Pietro Lio, Petar Velickovic
    https://arxiv.org/abs/2004.05718
    Architecture follows that in
    https://github.com/graphdeeplearning/benchmarking-gnns
"""


class ActivationPNANet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hiddim = net_params["hidden_dim"]
        outdim = net_params["out_dim"]

        dropout = net_params["dropout"]
        n_layers = net_params["L"]

        self.graph_pool = net_params["graph_pool"]

        self.batch_norm = net_params["batch_norm"]

        self.activation = activations(net_params["activation"])
        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]
        linear_params = {"density": self.density, "sampler": self.sampler}

        self.aggregators = net_params["aggregators"]
        self.scalers = net_params["scalers"]
        self.avg_d = net_params["avg_d"]
        self.num_tower = net_params["num_tower"]
        self.edge_feat = net_params["edge_feat"]
        edge_dim = net_params["edge_dim"]
        self.divide_input = net_params["divide_input"]

        self.simplified = net_params["use_simplified_version"]

        self.node_encoder = nn.Embedding(num_atom_type, hiddim)
        if self.edge_feat:
            self.edge_encoder = nn.Embedding(num_bond_type, edge_dim)
            self.simplified = False

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == n_layers-1:
                if self.simplified:
                    new_layer = ActivationPNASimplifiedLayer(
                                    indim=hiddim, outdim=outdim, hiddim=hiddim,
                                    activation=self.activation,
                                    dropout=dropout,
                                    batch_norm=self.batch_norm,
                                    aggregators=self.aggregators,
                                    scalers=self.scalers, avg_d=self.avg_d)
                else:
                    new_layer = ActivationPNALayer(
                                    indim=hiddim, outdim=outdim, hiddim=hiddim,
                                    activation=self.activation,
                                    dropout=dropout,
                                    batch_norm=self.batch_norm,
                                    aggregators=self.aggregators,
                                    scalers=self.scalers, avg_d=self.avg_d,
                                    num_tower=self.num_tower,
                                    divide_input=False,
                                    edge_features=self.edge_feat,
                                    edge_dim=edge_dim)

            else:
                if self.simplified:
                    new_layer = ActivationPNASimplifiedLayer(
                                    indim=hiddim, outdim=hiddim, hiddim=hiddim,
                                    activation=self.activation,
                                    dropout=dropout,
                                    batch_norm=self.batch_norm,
                                    aggregators=self.aggregators,
                                    scalers=self.scalers, avg_d=self.avg_d)
                else:
                    new_layer = ActivationPNALayer(
                                    indim=hiddim, outdim=hiddim, hiddim=hiddim,
                                    activation=self.activation,
                                    dropout=dropout,
                                    batch_norm=self.batch_norm,
                                    aggregators=self.aggregators,
                                    scalers=self.scalers, avg_d=self.avg_d,
                                    num_tower=self.num_tower,
                                    divide_input=self.divide_input,
                                    edge_features=self.edge_feat,
                                    edge_dim=edge_dim)
            self.layers.append(new_layer)

        outdim = hiddim * (1+len(self.aggregators)*len(self.scalers))
        self.readout = nn.Sequential(LinearLayer(hiddim, hiddim//2,
                                                 bias=True,
                                                 linear_type="regular"),
                                     nn.ReLU(),
                                     LinearLayer(hiddim//2, hiddim//4,
                                                 bias=True,
                                                 linear_type="regular"),
                                     nn.ReLU(),
                                     LinearLayer(hiddim//4, 1,
                                                 bias=True,
                                                 linear_type="regular"))

    def forward(self, g, h, e):
        with g.local_scope():
            g = g.to(h.device)
            h = self.node_encoder(h)
            if self.edge_feat:
                e = self.edge_encoder(e)

            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(h.device).unsqueeze(1)

            for i, conv in enumerate(self.layers):
                h = conv(g, h, e, norm)
            g.ndata['h'] = h

            if self.graph_pool == "sum":
                hg = dgl.sum_nodes(g, "h")
            elif self.graph_pool == "max":
                hg = dgl.max_nodes(g, "h")
            elif self.graph_pool == "mean":
                hg = dgl.mean_nodes(g, "h")
            else:
                hg = dgl.mean_nodes(g, "h")  # default readout is mean nodes

            return self.readout(hg)

    def loss(self, scores, targets):
        loss = nn.L1Loss()(scores, targets)
        return loss
