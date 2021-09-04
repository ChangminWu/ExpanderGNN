import torch.nn as nn

from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

from layers.gin_layer import GINLayer
from layers.expander.expander_layer import LinearLayer, MultiLinearLayer
from utils import activations


class GINNet(nn.Module):
    def __init__(self, net_params):
        super(GINNet, self).__init__()
        indim = net_params["in_dim"]
        hiddim = net_params["hidden_dim"]

        n_classes = net_params["n_classes"]
        dropout = net_params["dropout"]
        self.n_layers = net_params["L"]

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
        self.learn_eps = net_params["learn_eps"]

        linear_params = {"density": self.density, "sampler": self.sampler}

        self.layers = nn.ModuleList()
        linear_transform = \
            MultiLinearLayer(indim, hiddim,
                             activation=self.activation,
                             batch_norm=self.batch_norm,
                             num_layers=self.n_mlp_layer,
                             hiddim=hiddim,
                             bias=self.bias,
                             linear_type=self.linear_type,
                             **linear_params)
        self.layers.append(GINLayer(linear_transform,
                                    aggr_type=self.neighbor_pool,
                                    activation=self.activation,
                                    dropout=dropout,
                                    batch_norm=self.batch_norm,
                                    residual=self.residual,
                                    learn_eps=self.learn_eps))

        for i in range(self.n_layers-1):
            linear_transform = \
                            MultiLinearLayer(hiddim, hiddim,
                                             activation=self.activation,
                                             batch_norm=self.batch_norm,
                                             num_layers=self.n_mlp_layer,
                                             hiddim=hiddim,
                                             bias=self.bias,
                                             linear_type=self.linear_type,
                                             **linear_params)
            self.layers.append(GINLayer(linear_transform,
                                        aggr_type=self.neighbor_pool,
                                        activation=self.activation,
                                        dropout=dropout,
                                        batch_norm=self.batch_norm,
                                        residual=self.residual,
                                        learn_eps=self.learn_eps))



        self.linear_predictions = nn.ModuleList()
        self.linear_predictions.append(
            LinearLayer(indim,
                        n_classes, bias=self.bias,
                        linear_type="regular",
                        **linear_params))

        for _ in range(self.n_layers):
            self.linear_predictions.append(
                LinearLayer(hiddim,
                            n_classes, bias=self.bias,
                            linear_type="regular",
                            **linear_params))

    def forward(self, g, h, e):
        with g.local_scope():
            g = g.to(h.device)

            hidden_rep = [h]

            for i in range(self.n_layers):
                h = self.layers[i](g, h)
                hidden_rep.append(h)

            score_over_layer = 0
            for i, h in enumerate(hidden_rep):
                score_over_layer += self.linear_predictions[i](h)

        return score_over_layer

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss