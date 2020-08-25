import torch
import torch.nn as nn

import dgl.function as fn 


class ActivationGINLayer(nn.Module):
    def __init__(self, indim, outdim, aggr_type, activation, dropout,
                 batch_norm,
                 init_eps=0, learn_eps=False):
        super(ActivationGINLayer, self).__init__()

        if aggr_type == "sum":
            self._reducer = fn.sum
        elif aggr_type == "mean":
            self._reducer = fn.mean
        elif aggr_type == "max":
            self._reducer = fn.max
        else:
            raise KeyError("Aggregator type {} not recognized."
                           .format(aggr_type))

        self.batch_norm = batch_norm

        self.activation = activation
        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.dropout = nn.Dropout(dropout)

        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))

    def forward(self, g, h, norm):
        h = h*norm
        g = g.local_var()
        g.ndata["h"] = h
        g.update_all(fn.copy_u("h", "m"), self._reducer("m", "neigh"))
        h = g.ndata["h"]
        h = (1+self.eps)*h + g.ndata["neigh"]
        h = h*norm

        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.activation is not None:
            h = self.activation(h)

        return self.dropout(h)
