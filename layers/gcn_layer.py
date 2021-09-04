import torch
import torch.nn as nn

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv


class UpdateModule(nn.Module):
    """
    update node features
    """
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
        """
        Parameters
        ----------
        apply_func: callable, linear transform function to update node features
        aggr_type: string, neighborhood aggregation types
        activation: callable, activation function
        dropout: bool, whether or not use dropout on input features
        batch_norm: bool, whether or not add batch normalization before activation, after aggregation
        and linear transform
        residual: bool, whether or not use residual connection
        dgl_builtin: bool, whether or not use dgl builtin GCN convolution layer
        """
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
            if dgl.__version__ < "0.5":
                self.conv = GraphConv(apply_func.indim,
                                      apply_func.outdim)
            else:
                self.conv = GraphConv(apply_func.indim,
                                      apply_func.outdim, allow_zero_in_degree=True)
        else:
            self.apply_mod = UpdateModule(apply_func)

    def forward(self, g, features, norm=None):
        # norm is the square-root of degrees, used for symmetrically normalize the adjacency matrix
        h_in = features

        if norm is None:
            norm = 1

        if self.dgl_builtin:
            h = self.conv(g, features)
        else:
            features = features * norm
            g.ndata["h"] = features
            g.update_all(fn.copy_src(src="h", out="m"),
                         self._reducer("m", "h"))
            g.apply_nodes(func=self.apply_mod)
            h = g.ndata.pop('h')
            h = h*norm

        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.activation is not None:
            h = self.activation(h)

        if self.residual:
            h = h_in + h

        return self.dropout(h)
