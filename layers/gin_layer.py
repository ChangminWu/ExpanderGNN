import torch
import torch.nn as nn

import dgl.function as fn


class GINLayer(nn.Module):
    def __init__(self, apply_func, aggr_type, activation, dropout,
                 batch_norm, residual=False,
                 init_eps=0, learn_eps=False):
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
        init_eps: float, initial coefficient for central node feature in update step
        learn_eps: bool, whether or not the aforementioned coefficient is learnable
        """
        super(GINLayer, self).__init__()
        self.apply_func = apply_func

        if aggr_type == "sum":
            self._reducer = fn.sum
        elif aggr_type == "mean":
            self._reducer = fn.mean
        elif aggr_type == "max":
            self._reducer = fn.max
        else:
            raise KeyError("Aggregator type {} not recognized."
                           .format(aggr_type))

        self.batch_norm, self.residual = batch_norm, residual
        if self.apply_func.indim != self.apply_func.outdim:
            self.residual = False

        self.activation = activation
        self.batchnorm_h = nn.BatchNorm1d(self.apply_func.outdim)
        self.dropout = nn.Dropout(dropout)

        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))

    def forward(self, g, h):
        h_in = h

        g = g.local_var()
        g.ndata["h"] = h
        g.update_all(fn.copy_u("h", "m"), self._reducer("m", "neigh"))
        h = g.ndata["h"]

        h = (1+self.eps)*h + g.ndata["neigh"]
        if self.apply_func is not None:
            h = self.apply_func(h)

        if self.batch_norm:
            h = self.batchnorm_h(h)
        if self.activation is not None:
            h = self.activation(h)

        if self.residual:
            h = h_in + h

        return self.dropout(h)