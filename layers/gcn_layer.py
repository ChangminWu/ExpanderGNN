import torch
import torch.nn as nn

import dgl.function as fn
from dgl.nn.pytorch import GraphConv


class UpdateModule(nn.Module):
    def __init__(self, apply_func):
        super().__init__()
        self.apply_func = apply_func

    def forward(self, node):
        h = self.apply_func(node.data["h"])
        return {"h": h}


class GCNLayer(nn.Module):
    def __init__(self, apply_func,
                 aggr_type, activation, dropout,
                 batch_norm, residual=False, dgl_builtin=False):
        super().__init__()
        if aggr_type == "sum":
            self._reducer = fn.sum
        elif aggr_type == "max":
            self._reducer = fn.max
        elif aggr_type == "mean":
            self._reducer = fn.mean
        else:
            raise KeyError("Aggregator type {} not recognized."
                           .format(aggr_type))

        self.batch_norm, self.residual = batch_norm, residual
        self.dgl_builtin = dgl_builtin

        if apply_func.indim != apply_func.outdim:
            self.residual = False

        self.activation = activation
        self.batchnorm_h = nn.BatchNorm1d(apply_func.outdim)
        self.dropout = nn.Dropout(dropout)

        if self.dgl_builtin:
            self.conv = GraphConv(apply_func.indim,
                                  apply_func.outdim)
        else:
            self.apply_mod = UpdateModule(apply_func)

    def forward(self, g, features):
        h_in = features

        if self.dgl_builtin:
            h = self.conv(g, features)
        else:
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(features.device).unsqueeze(1)
            features = features * norm
            g.ndata["h"] = features
            g.update_all(fn.copy_src(src="h", out="m"),
                         self._reducer("m", "h"))
            g.apply_nodes(func=self.apply_mod)
            h = g.ndata["h"]
            h = h*norm

        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.activation is not None:
            h = self.activation(h)

        if self.residual:
            h = h_in + h

        return self.dropout(h)
