import torch
import torch.nn as nn 

import dgl 

from expander.expander_layer import LinearLayer
from utils import activations


class ActivationMLPNet(nn.Module):
    def __init__(self, net_params):
        super(ActivationMLPNet, self).__init__()
        indim = net_params["in_dim"]
        hiddim = net_params["hidden_dim"]

        n_classes = net_params["n_classes"]
        in_feat_dropout = net_params["in_feat_dropout"]
        dropout = net_params["dropout"]
        n_layers = net_params["L"]
        self.gated = net_params["gated"]

        self.batch_norm = net_params["batch_norm"]
        self.n_mlp_layer = net_params["mlp_layers"]

        self.activation = activations(net_params["activation"], param=hiddim)
        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]

        linear_params = {"density": self.density, "sampler": self.sampler}

        self.node_encoder = LinearLayer(indim, hiddim, bias=self.bias,
                                        linear_type=self.linear_type,
                                        **linear_params)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList()
        for _ in n_layers:
            if self.batch_norm is not None:
                self.layers.append(nn.BatchNorm1d(hiddim))

            if self.activation is not None:
                self.layers.append(self.activation)

            self.layers.append(nn.Dropout(dropout))

        if self.gated:
            self.gates = LinearLayer(hiddim, hiddim, bias=self.bias,
                                     linear_type=self.linear_type,
                                     **linear_params)

        self.readout = nn.Sequential(LinearLayer(hiddim, hiddim//2,
                                                 bias=True,
                                                 linear_type="regular"),
                                     nn.ReLU(),
                                     LinearLayer(hiddim//2, hiddim//4,
                                                 bias=True,
                                                 linear_type="regular"),
                                     nn.ReLU(),
                                     LinearLayer(hiddim//4, n_classes,
                                                 bias=True,
                                                 linear_type="regular"))

    def forward(self, g, h, e):
        with g.local_scope():
            h = self.node_encoder(h)
            h = self.in_feat_dropout(h)
            for layer in self.layers:
                h = layer(h)
            if self.gated:
                h = torch.sigmoid(self.gates(h))*h
                g.ndata["h"] = h
                hg = dgl.sum_nodes(g, "h")
                # hg = torch.cat(
                #     (
                #         dgl.sum_nodes(g, 'h'),
                #         dgl.max_nodes(g, 'h')
                #     ),
                #     dim=1
                # )
            else:
                g.ndata["h"] = h
                hg = dgl.mean_nodes(g, "h")

        return self.readout(hg)

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
