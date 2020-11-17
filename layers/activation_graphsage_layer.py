import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn


class Aggregator(nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}

    def aggre(self, neighbour):
        raise NotImplementedError


class MeanAggregator(Aggregator):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def aggre(self, neighbour):
        mean_neighbour = torch.mean(neighbour, dim=1)
        return mean_neighbour


class MaxPoolAggregator(Aggregator):
    def __init__(self, activation):
        super(MaxPoolAggregator, self).__init__()
        self.activation = activation

    def aggre(self, neighbour):
        if self.activation is not None:
            neighbour = self.activation(neighbour)
        maxpool_neighbour = torch.max(neighbour, dim=1)[0]
        return maxpool_neighbour


class LSTMAggregator(Aggregator):
    def __init__(self, indim, hiddim):
        super(LSTMAggregator, self).__init__()
        self.lstm = nn.LSTM(indim, hiddim, batch_first=True)
        self.hiddim = hiddim
        self.hidden = self.init_hidden()

        nn.init.xavier_uniform_(self.lstm.weight,
                                gain=nn.init.calculate_gain('relu'))

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hiddim),
                torch.zeros(1, 1, self.hiddim))

    def aggre(self, neighbours):
        rand_order = torch.randperm(neighbours.size()[1])
        neighbours = neighbours[:, rand_order, :]

        (lstm_out, self.hidden) = self.lstm(neighbours.view(
                                            neighbours.size()[0],
                                            neighbours.size()[1], -1))
        return lstm_out[:, -1, :]

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}


class UpdateModule(nn.Module):
    def __init__(self, activation, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

    def concat(self, h, aggre_result):
        bundle = h + aggre_result
        return bundle

    def forward(self, node):
        h = node.data['h']
        c = node.data['c']
        bundle = self.concat(h, c)
        bundle = F.normalize(bundle, p=2, dim=1)
        if self.activation is not None:
            bundle = self.activation(bundle)
        return {"h": bundle}


class ActivationGraphSageLayer(nn.Module):
    def __init__(self, indim, outdim,
                 aggr_type, activation, dropout,
                 batch_norm):
        """
        Parameters
        ----------
        indim: int, input feature dimension
        outdim: int, output feature dimension
        aggr_type: string, neighborhood aggregation types
        activation: callable, activation function
        dropout: bool, whether or not use dropout on input features
        batch_norm: bool, whether or not add batch normalization before activation, after aggregation and linear transform
        """
        super(ActivationGraphSageLayer, self).__init__()

        self.batch_norm = batch_norm

        self.activation = activation
        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.dropout = nn.Dropout(dropout)

        self.apply_mod = UpdateModule(activation=self.activation,
                                      dropout=dropout)
        if aggr_type == "max":
            self.reducer = MaxPoolAggregator(activation=None)
        elif aggr_type == "mean":
            self.reducer = MeanAggregator()
        elif aggr_type == "LSTM":
            self.reducer = LSTMAggregator(indim=indim,
                                          hiddim=indim)
        else:
            raise KeyError("Aggregator type {} not recognized."
                           .format(aggr_type))

    def forward(self, g, h, norm=None):
        if norm is None:
            norm = 1

        h = self.dropout(h)
        h = h*norm

        g.ndata["h"] = h
        g.update_all(fn.copy_src(src="h", out="m"), self.reducer,
                     self.apply_mod)
        h = g.ndata["h"]

        h = h*norm

        if self.batch_norm:
            h = self.batchnorm_h(h)
        return h
