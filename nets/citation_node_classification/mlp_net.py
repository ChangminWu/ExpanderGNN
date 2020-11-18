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

        self.layers.append(MultiLinearLayer(indim=sizes[i+1],
                                            outdim=n_classes,
                                            activation=self.activation,
                                            batch_norm=self.batch_norm,
                                            num_layers=self.n_mlp_layer,
                                            hiddim=hiddim,
                                            bias=self.bias,
                                            linear_type="regular",
                                            **linear_params))

        if self.gated:
            self.gates = LinearLayer(hiddim, hiddim, bias=self.bias,
                                     linear_type=self.linear_type,
                                     **linear_params)

    def forward(self, g, h, e):
        with g.local_scope():
            g = g.to(h.device)
            h = self.in_feat_dropout(h)
            for i, conv in enumerate(self.layers):
                h = conv(h)
                if i == len(self.layers)-2 and self.gated:
                    h = torch.sigmoid(self.gates(h))*h
            g.ndata["h"] = h

        return h

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss