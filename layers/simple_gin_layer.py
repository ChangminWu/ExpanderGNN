import torch
import torch.nn as nn

import dgl.function as fn 


class SimpleGINLayer(nn.Module):
    def __init__(self, indim, outdim, aggr_type,
                 batch_norm, residual=False,
                 init_eps=0, learn_eps=False):
        super(SimpleGINLayer, self).__init__()

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
        if indim != outdim:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(outdim)

        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))

    def forward(self, g, h, norm=None):
        h_in = h
        if norm is not None:
            h = h*norm
        g = g.local_var()
        g.ndata["h"] = h
        g.update_all(fn.copy_u("h", "m"), self._reducer("m", "neigh"))
        h = g.ndata["h"]
        h = (1+self.eps)*h + g.ndata["neigh"]
        if norm is not None:
            h = h*norm

        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.residual:
            h = h_in + h

        return h
