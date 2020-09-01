import torch
import torch.nn as nn

from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

from layers.simple_gin_layer import SimpleGINLayer
from expander.expander_layer import LinearLayer


class SimpleGINNet(nn.Module):
    def __init__(self, net_params):
        super(SimpleGINNet, self).__init__()
        indim = net_params["in_dim"]
        hiddim = net_params["hidden_dim"]
        outdim = net_params["out_dim"]

        n_classes = net_params["n_classes"]
        in_feat_dropout = net_params['in_feat_dropout']
        self.n_layers = net_params["L"]

        self.graph_pool = net_params["graph_pool"]
        self.neighbor_pool = net_params["neighbor_pool"]

        self.residual = net_params["residual"]
        self.batch_norm = net_params["batch_norm"]

        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]
        self.learn_eps = net_params["learn_eps"]

        linear_params = {"density": self.density, "sampler": self.sampler}

        self.node_encoder = LinearLayer(indim, hiddim, bias=self.bias,
                                        linear_type=self.linear_type,
                                        **linear_params)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList()
        self.linears = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(SimpleGINLayer(hiddim, hiddim,
                                              aggr_type=self.neighbor_pool,
                                              batch_norm=self.batch_norm,
                                              residual=self.residual,
                                              learn_eps=self.learn_eps))
            self.linears.append(LinearLayer(hiddim, outdim, bias=self.bias,
                                            linear_type=self.linear_type,
                                            **linear_params))

        self.linear_predictions = nn.ModuleList()
        for layer in range(self.n_layers+1):
            self.linear_predictions.append(
                LinearLayer(outdim, n_classes, bias=True,
                            linear_type=self.linear_type,
                            **linear_params))

    def forward(self, g, h, e):
        with g.local_scope():
            g = g.to(h.device)
            h = self.node_encoder(h)
            h = self.in_feat_dropout(h)

            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(h.device).unsqueeze(1)

            hidden_rep = [h]

            for i in range(self.n_layers):
                h = self.layers[i](g, h, norm)
                h = self.linears[i](h)

                hidden_rep.append(h)

            score_over_layer = 0
            for i, h in enumerate(hidden_rep):
                score_over_layer += self.linear_predictions[i](h)

        return score_over_layer

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss