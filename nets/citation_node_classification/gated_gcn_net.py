import torch
import torch.nn as nn

import dgl

from layers.gated_gcn_layer import GatedGCNLayer
from expander.expander_layer import LinearLayer
from utils import activations


class GatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super(GatedGCNNet, self).__init__()
        indim = net_params["in_dim"]
        indim_edge = net_params["in_dim_edge"]
        hiddim = net_params["hidden_dim"]
        outdim = net_params["out_dim"]

        n_classes = net_params["n_classes"]
        dropout = net_params["dropout"]
        n_layers = net_params["L"]

        self.graph_pool = net_params["graph_pool"]

        self.residual = net_params["residual"]
        self.batch_norm = net_params["batch_norm"]
        self.n_mlp_layer = net_params["mlp_layers"]

        self.activation = activations(net_params["activation"])
        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]
        self.edge_feat = net_params["edge_feat"]
        self.device = net_params["device"]

        linear_params = {"density": self.density, "sampler": self.sampler}

        self.node_encoder = LinearLayer(indim, hiddim, bias=self.bias,
                                        linear_type=self.linear_type,
                                        **linear_params)

        self.edge_encoder = LinearLayer(indim_edge, hiddim, bias=self.bias,
                                        linear_type=self.linear_type,
                                        **linear_params)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == n_layers-1:
                self.layers.append(GatedGCNLayer(self.n_mlp_layer, hiddim,
                                                 outdim, hiddim,
                                                 activation=self.activation,
                                                 dropout=dropout,
                                                 batch_norm=self.batch_norm,
                                                 bias=self.bias,
                                                 residual=self.residual,
                                                 linear_type=self.linear_type,
                                                 **linear_params))
            else:
                self.layers.append(GatedGCNLayer(self.n_mlp_layer, hiddim,
                                                 hiddim, hiddim,
                                                 activation=self.activation,
                                                 dropout=dropout,
                                                 batch_norm=self.batch_norm,
                                                 bias=self.bias,
                                                 residual=self.residual,
                                                 linear_type=self.linear_type,
                                                 **linear_params))

        self.readout = LinearLayer(outdim, n_classes, bias=True,
                                   linear_type=self.linear_type,
                                   **linear_params)

    def forward(self, g, h, e):
        with g.local_scope():
            g = g.to(h.device)
            h = self.node_encoder(h)
            if not self.edge_feat:
                e = torch.ones_like(e).to(self.device)
            e = self.edge_encoder(e)

            for conv in self.layers:
                h, e = conv(g, h, e)
            g.ndata["h"] = h

            return self.readout(h)

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
