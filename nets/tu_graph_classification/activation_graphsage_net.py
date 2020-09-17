import torch
import torch.nn as nn 

import dgl

from layers.activation_graphsage_layer import ActivationGraphSageLayer
from expander.expander_layer import LinearLayer
from utils import activations


class ActivationGraphSageNet(nn.Module):
    def __init__(self, net_params):
        super(ActivationGraphSageNet, self).__init__()
        indim = net_params["in_dim"]
        hiddim = net_params["hidden_dim"]

        n_classes = net_params["n_classes"]
        in_feat_dropout = net_params["in_feat_dropout"]
        dropout = net_params["dropout"]
        n_layers = net_params["L"]

        self.graph_pool = net_params["graph_pool"]
        self.neighbor_pool = net_params["neighbor_pool"]

        self.batch_norm = net_params["batch_norm"]

        self.activation = activations(net_params["activation"], param=hiddim)
        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]

        linear_params = {"density": self.density, "sampler": self.sampler}
        # if use expander linear, aggregation is set to mean,
        # in order to avoid input of 3 dimensions
        if self.linear_type == "expander":
            self.neighbor_pool = "mean"

        self.node_encoder = LinearLayer(indim, hiddim, bias=self.bias,
                                        linear_type=self.linear_type,
                                        **linear_params)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.batchnorm_h = nn.BatchNorm1d((n_layers+1)*hiddim)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                ActivationGraphSageLayer(hiddim, hiddim,
                                         aggr_type=self.neighbor_pool,
                                         activation=self.activation,
                                         bundle_activation=activations(net_params["activation"], param=(i+2)*hiddim),
                                         dropout=dropout,
                                         batch_norm=self.batch_norm))

        self.readout = nn.Sequential(LinearLayer((n_layers+1)*hiddim,
                                                 (n_layers+1)*hiddim//2,
                                                 bias=True,
                                                 linear_type="regular"),
                                     nn.ReLU(),
                                     LinearLayer((n_layers+1)*hiddim//2,
                                                 (n_layers+1)*hiddim//4,
                                                 bias=True,
                                                 linear_type="regular"),
                                     nn.ReLU(),
                                     LinearLayer((n_layers+1)*hiddim//4,
                                                 n_classes,
                                                 bias=True,
                                                 linear_type="regular"))

    def forward(self, g, h, e):
        with g.local_scope():
            g = g.to(h.device)
            h = self.node_encoder(h)
            h = self.in_feat_dropout(h)

            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(h.device).unsqueeze(1)

            for conv in self.layers:
                h, b = conv(g, h, norm)

            if self.batch_norm:
                b = self.batchnorm_h(b)

            if self.activation is not None:
                b = self.activation(b)

            g.ndata["h"] = b

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
