import torch
import torch.nn as nn

import dgl

from layers.simple_pna_layer import SimplePNALayer, SimplePNASimplifiedLayer
from expander.expander_layer import LinearLayer


"""
    PNA: Principal Neighbourhood Aggregation
    Gabriele Corso, Luca Cavalleri, Dominique Beaini,
    Pietro Lio, Petar Velickovic
    https://arxiv.org/abs/2004.05718
    Architecture follows that in
    https://github.com/graphdeeplearning/benchmarking-gnns
"""


class SimplePNANet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
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

        self.aggregators = net_params["aggregators"]
        self.scalers = net_params["scalers"]
        self.avg_d = net_params["avg_d"]
        self.num_tower = net_params["num_tower"]
        self.edge_feat = net_params["edge_feat"]
        edge_dim = net_params["edge_dim"]
        self.divide_input = net_params["divide_input"]

        num_pretrans_layer = net_params["num_pretrans_layer"]
        num_posttrans_layer = net_params["num_posttrans_layer"]
        self.simplified = net_params["use_simplified_version"]

        self.node_encoder = LinearLayer(indim, hiddim, bias=self.bias,
                                        linear_type=self.linear_type,
                                        **linear_params)
        if self.edge_feat:
            self.edge_encoder = LinearLayer(indim, hiddim, bias=self.bias,
                                            linear_type=self.linear_type,
                                            **linear_params)
            self.simplified = False

        self.layers = nn.ModuleList()
        if self.simplified:
            new_layer = SimplePNASimplifiedLayer(
                indim=hiddim, outdim=hiddim, hiddim=hiddim,
                dropout=dropout,
                batch_norm=self.batch_norm,
                aggregators=self.aggregators,
                scalers=self.scalers, avg_d=self.avg_d,
                num_posttrans_layer=num_posttrans_layer,
                residual=self.residual, bias=self.bias,
                linear_type=self.linear_type,
                **linear_params)
        else:
            new_layer = SimplePNALayer(
                indim=hiddim, outdim=hiddim, hiddim=hiddim,
                dropout=dropout,
                batch_norm=self.batch_norm,
                aggregators=self.aggregators,
                scalers=self.scalers, avg_d=self.avg_d,
                num_tower=self.num_tower,
                num_pretrans_layer=num_pretrans_layer,
                num_posttrans_layer=num_posttrans_layer,
                divide_input=False, residual=self.residual,
                edge_features=self.edge_feat,
                edge_dim=edge_dim, bias=self.bias,
                linear_type=self.linear_type,
                **linear_params)

        self.layers.append(new_layer)

        for i in range(n_layers-1):
            if i == n_layers-2:
                if self.simplified:
                    new_layer = SimplePNASimplifiedLayer(
                                    indim=hiddim, outdim=n_classes, hiddim=hiddim,
                                    dropout=dropout,
                                    batch_norm=False,
                                    aggregators=self.aggregators,
                                    scalers=self.scalers, avg_d=self.avg_d,
                                    num_posttrans_layer=num_posttrans_layer,
                                    residual=self.residual, bias=self.bias,
                                    linear_type=self.linear_type,
                                    **linear_params)
                else:
                    new_layer = SimplePNALayer(
                                    indim=hiddim, outdim=n_classes, hiddim=hiddim,
                                    dropout=dropout,
                                    batch_norm=False,
                                    aggregators=self.aggregators,
                                    scalers=self.scalers, avg_d=self.avg_d,
                                    num_tower=1,
                                    num_pretrans_layer=num_pretrans_layer,
                                    num_posttrans_layer=num_posttrans_layer,
                                    divide_input=False, residual=self.residual,
                                    edge_features=self.edge_feat,
                                    edge_dim=edge_dim, bias=self.bias,
                                    linear_type=self.linear_type,
                                    **linear_params)

            else:
                if self.simplified:
                    new_layer = SimplePNASimplifiedLayer(
                                    indim=hiddim, outdim=hiddim, hiddim=hiddim,
                                    dropout=dropout,
                                    batch_norm=self.batch_norm,
                                    aggregators=self.aggregators,
                                    scalers=self.scalers, avg_d=self.avg_d,
                                    num_posttrans_layer=num_posttrans_layer,
                                    residual=self.residual, bias=self.bias,
                                    linear_type=self.linear_type,
                                    **linear_params)
                else:
                    new_layer = SimplePNALayer(
                                    indim=hiddim, outdim=hiddim, hiddim=hiddim,
                                    dropout=dropout,
                                    batch_norm=self.batch_norm,
                                    aggregators=self.aggregators,
                                    scalers=self.scalers, avg_d=self.avg_d,
                                    num_tower=self.num_tower,
                                    num_pretrans_layer=num_pretrans_layer,
                                    num_posttrans_layer=num_posttrans_layer,
                                    divide_input=self.divide_input,
                                    residual=self.residual,
                                    edge_features=self.edge_feat,
                                    edge_dim=edge_dim, bias=self.bias,
                                    linear_type=self.linear_type,
                                    **linear_params)
            self.layers.append(new_layer)

        # self.readout = LinearLayer(outdim, n_classes, bias=True,
        #                            linear_type=self.linear_type,
        #                            **linear_params)

    def forward(self, g, h, e):
        with g.local_scope():
            g = g.to(h.device)
            h = self.node_encoder(h)
            if self.edge_feat:
                e = self.edge_encoder(e)

            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(h.device).unsqueeze(1)

            for i, conv in enumerate(self.layers):
                h = conv(g, h, e, norm)
            # g.ndata['h'] = h

            return h #self.readout(h)

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
