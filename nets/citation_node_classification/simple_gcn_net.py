import torch
import torch.nn as nn

import dgl.function as fn

from layers.expander.expander_layer import LinearLayer, MultiLinearLayer


class SimpleGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        indim = net_params["in_dim"]
        hiddim = net_params["hidden_dim"]
        outdim = net_params["out_dim"]

        n_classes = net_params["n_classes"]
        in_feat_dropout = net_params["in_feat_dropout"]
        self.n_layers = net_params["L"]

        self.graph_pool = net_params["graph_pool"]
        self.neighbor_pool = net_params["neighbor_pool"]

        self.batch_norm = net_params["batch_norm"]
        self.n_mlp_layer = net_params["mlp_layers"]

        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]

        if self.neighbor_pool == "sum":
            self._reducer = fn.sum
        elif self.neighbor_pool == "max":
            self._reducer = fn.max
        elif self.neighbor_pool == "mean":
            self._reducer = fn.mean
        else:
            raise KeyError("Aggregator type {} not recognized."
                           .format(self.neighbor_pool))

        linear_params = {"density": self.density, "sampler": self.sampler}

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.linear = MultiLinearLayer(indim, n_classes,
                                       activation=None,
                                       batch_norm=self.batch_norm,
                                       num_layers=self.n_mlp_layer,
                                       hiddim=hiddim,
                                       bias=self.bias,
                                       linear_type=self.linear_type,
                                       **linear_params)

    def forward(self, g, h, e):
        with g.local_scope():
            g = g.to(h.device)
            h = self.in_feat_dropout(h)

            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(h.device).unsqueeze(1)

            # compute (D^-1 A^k D)^k X
            for _ in range(self.n_layers):
                h = h * norm
                g.ndata['h'] = h
                g.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'h'))

                h = g.ndata.pop('h')
                h = h * norm

            h = self.linear(h)
            return h

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss