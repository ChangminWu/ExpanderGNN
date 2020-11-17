import torch
import torch.nn as nn

import dgl

from layers.expander.expander_layer import LinearLayer, MultiLinearLayer
from utils import activations


class MLPNet(nn.Module):
    def __init__(self, net_params):
        super(MLPNet, self).__init__()
        indim = net_params["in_dim"]
        hiddim = net_params["hidden_dim"]
        outdim = net_params["out_dim"]

        n_classes = net_params["n_classes"]
        in_feat_dropout = net_params["in_feat_dropout"]
        dropout = net_params["dropout"]
        n_layers = net_params["L"]
        self.gated = net_params["gated"]

        self.batch_norm = net_params["batch_norm"]
        self.n_mlp_layer = net_params["mlp_layers"]

        self.activation = activations(net_params["activation"])
        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]

        linear_params = {"density": self.density, "sampler": self.sampler}

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList()
        sizes = [indim]
        sizes.extend([hiddim]*(n_layers-1))
        sizes.append(outdim)

        for i in range(len(sizes)-1):
            self.layers.append(MultiLinearLayer(indim=sizes[i],
                                                outdim=sizes[i+1],
                                                activation=self.activation,
                                                batch_norm=self.batch_norm,
                                                num_layers=self.n_mlp_layer,
                                                hiddim=hiddim,
                                                bias=self.bias,
                                                linear_type=self.linear_type,
                                                **linear_params))

            if self.batch_norm is not None:
                self.layers.append(nn.BatchNorm1d(sizes[i+1]))

            if self.activation is not None:
                self.layers.append(self.activation)

            self.layers.append(nn.Dropout(dropout))

        if self.gated:
            self.gates = LinearLayer(outdim, outdim, bias=self.bias,
                                     linear_type=self.linear_type,
                                     **linear_params)

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
            h = self.in_feat_dropout(h)
            for conv in self.layers:
                h = conv(h)
            if self.gated:
                h = torch.sigmoid(self.gates(h))*h
                g.ndata["h"] = h
                hg = dgl.sum_nodes(g, "h")
            else:
                g.ndata["h"] = h
                hg = dgl.mean_nodes(g, "h")

        return self.readout(hg)

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss