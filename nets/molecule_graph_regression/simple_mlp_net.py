import torch
import torch.nn as nn 

import dgl 

from expander.expander_layer import LinearLayer, MultiLinearLayer


class SimpleMLPNet(nn.Module):
    def __init__(self, net_params):
        super(SimpleMLPNet, self).__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hiddim = net_params["hidden_dim"]
        outdim = net_params["out_dim"]

        in_feat_dropout = net_params["in_feat_dropout"]
        self.gated = net_params["gated"]

        self.batch_norm = net_params["batch_norm"]
        self.n_mlp_layer = net_params["mlp_layers"]

        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]

        linear_params = {"density": self.density, "sampler": self.sampler}
        self.node_encoder = nn.Embedding(num_atom_type, hiddim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.linear = MultiLinearLayer(indim=hiddim,
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

        self.readout = nn.Sequential(LinearLayer(outdim, outdim//2,
                                                 bias=True,
                                                 linear_type="regular"),
                                     nn.ReLU(),
                                     LinearLayer(outdim//2, outdim//4,
                                                 bias=True,
                                                 linear_type="regular"),
                                     nn.ReLU(),
                                     LinearLayer(outdim//4, 1,
                                                 bias=True,
                                                 linear_type="regular"))

    def forward(self, g, h, e):
        with g.local_scope():
            g = g.to(h.device)
            h = self.node_encoder(h)
            h = self.in_feat_dropout(h)
            h = self.linear(h)
            if self.gated:
                h = torch.sigmoid(self.gates(h))*h
                g.ndata["h"] = h
                hg = dgl.sum_nodes(g, "h")
                # hg = torch.cat(
                #     (
                #         dgl.sum_nodes(g, 'h'),
                #         dgl.max_nodes(g, 'h')
                #     ),
                #     dim=1
                # )
            else:
                g.ndata["h"] = h
                hg = dgl.mean_nodes(g, "h")

        return self.readout(hg)

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss