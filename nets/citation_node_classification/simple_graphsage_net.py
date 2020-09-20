import torch
import torch.nn as nn

import dgl

from layers.simple_graphsage_layer import SimpleGraphSageLayer
from expander.expander_layer import LinearLayer, MultiLinearLayer


class SimpleGraphSageNet(nn.Module):
    def __init__(self, net_params):
        super(SimpleGraphSageNet, self).__init__()
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

        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]

        linear_params = {"density": self.density, "sampler": self.sampler}

        # if use expander linear, aggregation is set to mean,
        # in order to avoid input of 3 dimensions
        if self.linear_type == "expander":
            self.neighbor_pool = "mean"

        # self.node_encoder = LinearLayer(indim, hiddim, bias=self.bias,
        #                                 linear_type=self.linear_type,
        #                                 **linear_params)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList()
        apply_func = MultiLinearLayer(indim+hiddim, hiddim,
                                      activation=None,
                                      batch_norm=self.batch_norm,
                                      num_layers=self.n_mlp_layer,
                                      hiddim=indim,
                                      bias=self.bias,
                                      linear_type=self.linear_type,
                                      **linear_params)
        self.layers.append(
            SimpleGraphSageLayer(indim, hiddim, apply_func,
                                 aggr_type=self.neighbor_pool,
                                 dropout=dropout,
                                 batch_norm=self.batch_norm,
                                 residual=self.residual,
                                 bias=self.bias,
                                 linear_type=self.linear_type,
                                 **linear_params))

        for i in range(n_layers-1):
            if i == n_layers-2:
                apply_func = MultiLinearLayer(hiddim+n_classes, n_classes,
                                              activation=None,
                                              batch_norm=self.batch_norm,
                                              num_layers=self.n_mlp_layer,
                                              hiddim=hiddim,
                                              bias=self.bias,
                                              linear_type=self.linear_type,
                                              **linear_params)
                self.layers.append(
                    SimpleGraphSageLayer(hiddim, n_classes, apply_func,
                                         aggr_type=self.neighbor_pool,
                                         dropout=dropout,
                                         batch_norm=False,
                                         residual=self.residual,
                                         bias=self.bias,
                                         linear_type=self.linear_type,
                                         **linear_params))
            else:
                apply_func = MultiLinearLayer(2*hiddim, hiddim,
                                              activation=None,
                                              batch_norm=self.batch_norm,
                                              num_layers=self.n_mlp_layer,
                                              hiddim=hiddim,
                                              bias=self.bias,
                                              linear_type=self.linear_type,
                                              **linear_params)
                self.layers.append(
                    SimpleGraphSageLayer(hiddim, hiddim, apply_func,
                                         aggr_type=self.neighbor_pool,
                                         dropout=dropout,
                                         batch_norm=self.batch_norm,
                                         residual=self.residual,
                                         bias=self.bias,
                                         linear_type=self.linear_type,
                                         **linear_params))

        # self.readout = LinearLayer(outdim, n_classes, bias=True,
        #                            linear_type=self.linear_type,
        #                            **linear_params)

    def forward(self, g, h, e):
        with g.local_scope():
            g = g.to(h.device)
            # h = self.node_encoder(h)
            h = self.in_feat_dropout(h)

            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(h.device).unsqueeze(1)

            for conv in self.layers:
                h = conv(g, h, norm)

            # g.ndata["h"] = h

            return h #self.readout(h)

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
