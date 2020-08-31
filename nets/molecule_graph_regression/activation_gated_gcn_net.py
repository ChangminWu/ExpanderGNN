import torch
import torch.nn as nn

import dgl

from layers.activation_gated_gcn_layer import ActivationGatedGCNLayer
from expander.expander_layer import LinearLayer
from utils import activations


class ActivationGatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super(ActivationGatedGCNNet, self).__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hiddim = net_params["hidden_dim"]

        dropout = net_params["dropout"]
        n_layers = net_params["L"]

        self.graph_pool = net_params["graph_pool"]

        self.residual = net_params["residual"]
        self.batch_norm = net_params["batch_norm"]
        self.n_mlp_layer = net_params["mlp_layers"]

        self.activation = activations(net_params["activation"], param=hiddim)
        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]
        self.edge_feat = net_params["edge_feat"]
        self.device = net_params["device"]
        linear_params = {"density": self.density, "sampler": self.sampler}

        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.pos_encoder = LinearLayer(pos_enc_dim, hiddim,
                                           bias=self.bias,
                                           linear_type=self.linear_type,
                                           **linear_params)

        self.node_encoder = nn.Embedding(num_atom_type, hiddim)

        if self.edge_feat:
            self.edge_encoder = nn.Embedding(num_bond_type, hiddim)
        else:
            self.edge_encoder = nn.Linear(1, hiddim)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                ActivationGatedGCNLayer(hiddim, hiddim, hiddim,
                                        activation=self.activation,
                                        dropout=dropout,
                                        batch_norm=self.batch_norm))

        self.readout = nn.Sequential(LinearLayer(hiddim, hiddim//2,
                                                 bias=True,
                                                 linear_type="regular"),
                                     nn.ReLU(),
                                     LinearLayer(hiddim//2, hiddim//4,
                                                 bias=True,
                                                 linear_type="regular"),
                                     nn.ReLU(),
                                     LinearLayer(hiddim//4, 1,
                                                 bias=True,
                                                 linear_type="regular"))

    def forward(self, g, h, e, h_pos_enc=None):
        with g.local_scope():
            g = g.to(h.device)
            h = self.node_encoder(h)
            if self.pos_enc:
                h_pos_enc = self.pos_encoder(h_pos_enc.float())
            if not self.edge_feat:
                e = torch.ones(e.size(0), 1).to(self.device)
            e = self.edge_encoder(e)

            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(h.device).unsqueeze(1)

            for conv in self.layers:
                h, e = conv(g, h, e, norm)

            g.ndata["h"] = h

            if self.graph_pool == "sum":
                hg = dgl.sum_nodes(g, "h")
            elif self.graph_pool == "max":
                hg = dgl.max_nodes(g, "h")
            elif self.graph_pool == "mean":
                hg = dgl.mean_nodes(g, "h")
            else:
                hg = dgl.mean_nodes(g, "h")  # default readout is mean nodes

            return self.readout(hg)

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss