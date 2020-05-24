import torch
from torch import nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

from baselines.layers.mlp_readout_layer import MLPReadout

class SimpleGCN(nn.Module):
    def __init__(self, net_params):
        super(SimpleGCN, self).__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        neighbor_aggr_type = net_params['neighbor_aggr_SGCN']
        self.n_layers = net_params['L']
        self.batch_norm = net_params['batch_norm']
        self.readout = net_params['readout']
        if net_params['activation'] == "relu":
            self.activation = F.relu
        elif net_params['activation'] is None:
            self.activation = None
        else:
            raise ValueError("Invalid activation type.")

        if neighbor_aggr_type == 'sum':
            self._reducer = fn.sum
        elif neighbor_aggr_type == 'max':
            self._reducer = fn.max
        elif neighbor_aggr_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(neighbor_aggr_type))

        self.mlp_readout = net_params["mlp_readout"]

        self.embedding_h = nn.Linear(in_dim, hidden_dim, bias=True)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.batchnorm_h = nn.BatchNorm1d(out_dim)

        if self.mlp_readout:
            self.readout = MLPReadout(out_dim, n_classes)
        else:
            self.readout = nn.Linear(out_dim, n_classes)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.embedding_h.weight)

        if self.embedding_h.bias is not None:
            nn.init.zeros_(self.embedding_h.bias)

        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, g, feat, e, snorm_n, snorm_e):
        with g.local_scope():
            h = self.embedding_h(feat)
            h = self.in_feat_dropout(h)

            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(feat.device).unsqueeze(1)

            # compute (D^-1 A^k D)^k X
            for _ in range(self.n_layers):
                h = h * norm
                g.ndata['h'] = h
                g.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'h'))

                h = g.ndata.pop('h')
                h = h * norm

            h = self.linear(h)

            if self.batch_norm:
                h = self.batchnorm_h(h)

            if self.activation is not None:
                h = self.activation(h)

            g.ndata['h'] = h

            if self.readout == "sum":
                hg = dgl.sum_nodes(g, 'h')
            elif self.readout == "max":
                hg = dgl.max_nodes(g, 'h')
            elif self.readout == "mean":
                hg = dgl.mean_nodes(g, 'h')
            else:
                hg = dgl.mean_nodes(g, 'h')

            return self.readout(hg)