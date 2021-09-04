import torch.nn as nn

import dgl

from layers.graphsage_layer import GraphSageLayer
from layers.expander.expander_layer import LinearLayer, MultiLinearLayer
from utils import activations


class GraphSageNet(nn.Module):
    def __init__(self, net_params):
        super(GraphSageNet, self).__init__()
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

        # if use expander linear, aggregation is set to mean,
        # in order to avoid input of 3 dimensions
        if self.linear_type == "expander":
            self.neighbor_pool = "mean"

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList()
        apply_func = MultiLinearLayer(2*indim, hiddim,
                                      activation=self.activation,
                                      batch_norm=self.batch_norm,
                                      num_layers=self.n_mlp_layer,
                                      hiddim=indim,
                                      bias=self.bias,
                                      linear_type=self.linear_type,
                                      **linear_params)

        self.layers.append(GraphSageLayer(apply_func,
                                          aggr_type=self.neighbor_pool,
                                          activation=self.activation,
                                          dropout=dropout,
                                          batch_norm=self.batch_norm,
                                          residual=self.residual,
                                          dgl_builtin=self.dgl_builtin,
                                          **linear_params))

        for i in range(n_layers-1):
            if i == n_layers-2:
                apply_func = MultiLinearLayer(2*hiddim, n_classes,
                                              activation=self.activation,
                                              batch_norm=self.batch_norm,
                                              num_layers=self.n_mlp_layer,
                                              hiddim=hiddim,
                                              bias=self.bias,
                                              linear_type="regular",
                                              **linear_params)

                self.layers.append(GraphSageLayer(apply_func,
                                                  aggr_type=self.neighbor_pool,
                                                  activation=None,
                                                  dropout=dropout,
                                                  batch_norm=False,
                                                  residual=self.residual,
                                                  dgl_builtin=self.dgl_builtin,
                                                  **linear_params))

            else:
                apply_func = MultiLinearLayer(2*hiddim, hiddim,
                                              activation=self.activation,
                                              batch_norm=self.batch_norm,
                                              num_layers=self.n_mlp_layer,
                                              hiddim=hiddim,
                                              bias=self.bias,
                                              linear_type=self.linear_type,
                                              **linear_params)

                self.layers.append(GraphSageLayer(apply_func,
                                                  aggr_type=self.neighbor_pool,
                                                  activation=self.activation,
                                                  dropout=dropout,
                                                  batch_norm=self.batch_norm,
                                                  residual=self.residual,
                                                  dgl_builtin=self.dgl_builtin,
                                                  **linear_params))

    def forward(self, g, h, e):
        with g.local_scope():
            g = g.to(h.device)
            h = self.in_feat_dropout(h)

            for conv in self.layers:
                h = conv(g, h)

            return h

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss