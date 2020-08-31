import torch.nn as nn

import dgl

from layers.gat_layer import GATLayer
from expander.expander_layer import LinearLayer
from utils import activations


class GATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hiddim = net_params["hidden_dim"]
        outdim = net_params["out_dim"]
        num_heads = net_params['num_heads']
        in_feat_dropout = net_params["in_feat_dropout"]
        dropout = net_params["dropout"]
        n_layers = net_params["L"]

        self.graph_pool = net_params["graph_pool"]
        self.merge_type = net_params["merge_type"]

        self.residual = net_params["residual"]
        self.batch_norm = net_params["batch_norm"]
        self.n_mlp_layer = net_params["mlp_layers"]

        self.activation = activations(net_params["activation"])
        self.linear_type = net_params["linear_type"]
        self.density = net_params["density"]
        self.sampler = net_params["sampler"]
        self.bias = net_params["bias"]

        linear_params = {"density": self.density, "sampler": self.sampler}

        self.node_encoder = nn.Embedding(num_atom_type, hiddim*num_heads)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == n_layers-1:
                self.layers.append(
                    GATLayer(merge_type=self.merge_type,
                             num_heads=1,
                             n_mlp_layer=self.n_mlp_layer,
                             indim=hiddim*num_heads,
                             outdim=outdim,
                             hiddim=hiddim,
                             activation=self.activation,
                             dropout=dropout,
                             batch_norm=self.batch_norm,
                             residual=self.residual,
                             bias=self.bias,
                             linear_type=self.linear_type,
                             **linear_params))
            else:
                self.layers.append(
                    GATLayer(merge_type=self.merge_type,
                             num_heads=num_heads,
                             n_mlp_layer=self.n_mlp_layer,
                             indim=hiddim*num_heads,
                             outdim=hiddim,
                             hiddim=hiddim,
                             activation=self.activation,
                             dropout=dropout,
                             batch_norm=self.batch_norm,
                             residual=self.residual,
                             bias=self.bias,
                             linear_type=self.linear_type,
                             **linear_params))

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
            for conv in self.layers:
                h, e = conv(g, h, e)
            g.ndata["h"] = h

            if self.graph_pool == "sum":
                hg = dgl.sum_nodes(g, "h")
            elif self.graph_pool == "mean":
                hg = dgl.mean_nodes(g, "h")
            elif self.graph_pool == "max":
                hg = dgl.max_nodes(g, "h")
            else:
                hg = dgl.mean_nodes(g, "h")

            return self.readout(hg)

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss
