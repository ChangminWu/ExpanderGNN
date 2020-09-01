import torch
import torch.nn as nn 

from expander.expander_layer import LinearLayer, MultiLinearLayer


class SimpleMLPNet(nn.Module):
    def __init__(self, net_params):
        super(SimpleMLPNet, self).__init__()
        indim = net_params["in_dim"]
        hiddim = net_params["hidden_dim"]
        outdim = net_params["out_dim"]

        n_classes = net_params["n_classes"]
        in_feat_dropout = net_params["in_feat_dropout"]
        self.gated = net_params["gated"]

        self.batch_norm = net_params["batch_norm"]
        self.n_mlp_layer = net_params["mlp_layers"]

        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]

        linear_params = {"density": self.density, "sampler": self.sampler}

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.linear = MultiLinearLayer(indim=indim,
                                       outdim=outdim,
                                       activation=None,
                                       batch_norm=self.batch_norm,
                                       num_layers=self.n_mlp_layer,
                                       hiddim=hiddim,
                                       bias=self.bias,
                                       linear_type=self.linear_type,
                                       **linear_params)

        if self.gated:
            self.gates = LinearLayer(outdim, outdim, bias=self.bias,
                                     linear_type=self.linear_type,
                                     **linear_params)

        self.readout = LinearLayer(outdim, n_classes, bias=True,
                                   linear_type=self.linear_type,
                                   **linear_params)

    def forward(self, g, h, e):
        with g.local_scope():
            g = g.to(h.device)
            h = self.in_feat_dropout(h)
            h = self.linear(h)
            if self.gated:
                h = torch.sigmoid(self.gates(h))*h
            g.ndata["h"] = h

        return self.readout(h)

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
