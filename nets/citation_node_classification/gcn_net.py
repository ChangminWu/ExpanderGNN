import torch
import torch.nn as nn

import dgl

from layers.gcn_layer import GCNLayer
from layers.expander.expander_layer import LinearLayer, MultiLinearLayer
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
        self.n_mlp_layer = net_params["mlp_layers"]

        self.activation = activations(net_params["activation"])
        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]
        self.dgl_builtin = net_params["dgl_builtin"]

        linear_params = {"density": self.density, "sampler": self.sampler}

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList()
        linear_transform = MultiLinearLayer(indim, hiddim,
                                            activation=self.activation,
                                            batch_norm=self.batch_norm,
                                            num_layers=self.n_mlp_layer,
                                            hiddim=hiddim,
                                            bias=self.bias,
                                            linear_type=self.linear_type,
                                            **linear_params)

        self.layers.append(GCNLayer(linear_transform,
                                    aggr_type=self.neighbor_pool,
                                    activation=self.activation,
                                    dropout=dropout,
                                    batch_norm=self.batch_norm,
                                    residual=self.residual,
                                    dgl_builtin=self.dgl_builtin))

        for i in range(n_layers-1):
            if i == n_layers-2:
                linear_transform = \
                                MultiLinearLayer(hiddim, n_classes,
                                                 activation=self.activation,
                                                 batch_norm=self.batch_norm,
                                                 num_layers=self.n_mlp_layer,
                                                 hiddim=hiddim,
                                                 bias=self.bias,
                                                 linear_type="regular",
                                                 **linear_params)
                self.layers.append(GCNLayer(linear_transform,
                                   aggr_type=self.neighbor_pool,
                                   activation=None,
                                   dropout=dropout,
                                   batch_norm=False,
                                   residual=self.residual,
                                   dgl_builtin=self.dgl_builtin))

            else:
                linear_transform = \
                                MultiLinearLayer(hiddim, hiddim,
                                                 activation=self.activation,
                                                 batch_norm=self.batch_norm,
                                                 num_layers=self.n_mlp_layer,
                                                 hiddim=hiddim,
                                                 bias=self.bias,
                                                 linear_type=self.linear_type,
                                                 **linear_params)

                self.layers.append(GCNLayer(linear_transform,
                                            aggr_type=self.neighbor_pool,
                                            activation=self.activation,
                                            dropout=dropout,
                                            batch_norm=self.batch_norm,
                                            residual=self.residual,
                                            dgl_builtin=self.dgl_builtin))

    def forward(self, g, h, e):
        with g.local_scope():
            g = g.int().to(h.device)
            h = self.in_feat_dropout(h)

            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(h.device).unsqueeze(1)

            for conv in self.layers:
                h = conv(g, h, norm)
            g.ndata["h"] = h

            return h.log_softmax(dim=-1)

    def loss(self, pred, label):
        criterion = nn.NLLLoss()
        #criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss