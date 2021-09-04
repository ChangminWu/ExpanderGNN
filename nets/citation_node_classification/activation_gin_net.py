import torch
import torch.nn as nn

from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

from layers.activation_gin_layer import ActivationGINLayer
from layers.expander.expander_layer import LinearLayer

from utils import activations


class ActivationGINNet(nn.Module):
    def __init__(self, net_params):
        super(ActivationGINNet, self).__init__()
        indim = net_params["in_dim"]
        hiddim = indim
        n_classes = net_params["n_classes"]
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params["dropout"]
        self.n_layers = net_params["L"]

        self.graph_pool = net_params["graph_pool"]
        self.neighbor_pool = net_params["neighbor_pool"]

        self.batch_norm = net_params["batch_norm"]

        self.activation = activations(net_params["activation"])

        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]
        self.learn_eps = net_params["learn_eps"]

        linear_params = {"density": self.density, "sampler": self.sampler}

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList()
        self.layers.append(ActivationGINLayer(indim, hiddim,
                                              aggr_type=self.neighbor_pool,
                                              activation=self.activation,
                                              dropout=dropout,
                                              batch_norm=self.batch_norm,
                                              learn_eps=self.learn_eps))
        for i in range(self.n_layers-1):
            self.layers.append(ActivationGINLayer(hiddim, hiddim,
                                                  aggr_type=self.neighbor_pool,
                                                  activation=self.activation,
                                                  dropout=dropout,
                                                  batch_norm=self.batch_norm,
                                                  learn_eps=self.learn_eps))

        self.linear_predictions = nn.ModuleList()
        self.linear_predictions.append(
            LinearLayer(indim,
                        n_classes, bias=self.bias,
                        linear_type=self.linear_type,
                        **linear_params))

        for _ in range(self.n_layers):
            self.linear_predictions.append(
                        LinearLayer(hiddim,
                                    n_classes, bias=self.bias,
                                    linear_type=self.linear_type,
                                    **linear_params))

    def forward(self, g, h, e):
        with g.local_scope():
            g = g.to(h.device)
            h = self.in_feat_dropout(h)

            # degs = g.in_degrees().float().clamp(min=1)
            # norm = torch.pow(degs, -0.5)
            # norm = norm.to(h.device).unsqueeze(1)

            hidden_rep = [h]

            for i in range(self.n_layers):
                h = self.layers[i](g, h, norm=None)
                hidden_rep.append(h)

            score_over_layer = 0
            for i, h in enumerate(hidden_rep):
                score_over_layer += self.linear_predictions[i](h)

        return score_over_layer

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss