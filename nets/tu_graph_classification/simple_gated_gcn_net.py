import torch
import torch.nn as nn

import dgl

from layers.simple_gated_gcn_layer import SimpleGatedGCNLayer
from expander.expander_layer import LinearLayer


class SimpleGatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super(SimpleGatedGCNNet, self).__init__()
        indim = net_params["in_dim"]
        hiddim = net_params["hidden_dim"]
        outdim = net_params["out_dim"]

        n_classes = net_params["n_classes"]
        dropout = net_params["dropout"]
        n_layers = net_params["L"]

        self.graph_pool = net_params["graph_pool"]

        self.residual = net_params["residual"]
        self.batch_norm = net_params["batch_norm"]
        self.n_mlp_layer = net_params["mlp_layers"]

        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]

        linear_params = {"density": self.density, "sampler": self.sampler}

        self.node_encoder = LinearLayer(indim, hiddim, bias=self.bias,
                                        linear_type=self.linear_type,
                                        **linear_params)

        self.edge_encoder = LinearLayer(indim, hiddim, bias=self.bias,
                                        linear_type=self.linear_type,
                                        **linear_params)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == n_layers-1:
                self.layers.append(
                    SimpleGatedGCNLayer(self.n_mlp_layer, hiddim,
                                        outdim, hiddim,
                                        dropout=dropout,
                                        batch_norm=self.batch_norm,
                                        bias=self.bias,
                                        residual=self.residual,
                                        linear_type=self.linear_type,
                                        **linear_params))
            else:
                self.layers.append(
                    SimpleGatedGCNLayer(self.n_mlp_layer, hiddim,
                                        hiddim, hiddim,
                                        dropout=dropout,
                                        batch_norm=self.batch_norm,
                                        bias=self.bias,
                                        residual=self.residual,
                                        linear_type=self.linear_type,
                                        **linear_params))

        self.readout = nn.Sequential(LinearLayer(outdim, outdim//2,
                                                 bias=True,
                                                 linear_type="regular"),
                                     nn.ReLU(),
                                     LinearLayer(outdim//2, outdim//4,
                                                 bias=True,
                                                 linear_type="regular"),
                                     nn.ReLU(),
                                     LinearLayer(outdim//4, n_classes,
                                                 bias=True,
                                                 linear_type="regular"))

    def forward(self, g, h, e):
        with g.local_scope():
            g = g.to(h.device)
            h = self.node_encoder(h)
            e = self.edge_encoder(e)

            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(h.device).unsqueeze(1)

            for conv in self.layers:
                h, e = conv(g, h, e, norm)
            g.ndata["h"] = h

            if self.graph_pool == "sum":
                hg = dgl.sum_nodes(g, "h")
            elif self.graph_pool == "max":
                hg = dgl.max_nodes(g, "h")
            elif self.graph_pool == "mean":
                hg = dgl.mean_nodes(g, "h")
            else:
                hg = dgl.mean_nodes(g, "h")  # default readout is mean nodes

            return self.readout(hg)

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
