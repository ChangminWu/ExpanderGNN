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
        bundle = torch.cat((h, aggre_result), 1)
        return bundle

    def forward(self, node):
        h = node.data['h']
        c = node.data['c']
        if "b" in node.data:
            b = node.data['b']
            bundle = self.concat(b, c)
        else:
            bundle = self.concat(h, c)
        bundle = F.normalize(bundle, p=2, dim=1)
        if self.activation is not None:
            bundle = self.activation(bundle)
        return {"b": bundle, "h": c}


class ActivationGraphSageLayer(nn.Module):
    def __init__(self, indim, outdim,
                 aggr_type, activation, dropout,
                 batch_norm):
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

    def forward(self, g, h, norm):
        h = self.dropout(h)
        h = h*norm
        g.ndata["h"] = h
        g.update_all(fn.copy_src(src="h", out="m"), self.reducer,
                     self.apply_mod)
        h = g.ndata["h"]
        h = h*norm
        b = g.ndata["b"]

        if self.batch_norm:
            h = self.batchnorm_h(h)
        return h, b


class ActivationGraphSageEdgeLayer(nn.Module):
    def __init__(self, indim, outdim,
                 aggr_type, activation, dropout,
                 batch_norm):
        super(ActivationGraphSageEdgeLayer, self).__init__()

        self.activation = activation
        self.batch_norm = batch_norm

        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.dropout = nn.Dropout(dropout)

        self.apply_mod = UpdateModule(self.activation, dropout)

    def message_func(self, edges):
        Ah_j = edges.src["Ah"]
        e_ij = edges.src["Bh"] + edges.dst["Bh"]
        edges.data["e"] = e_ij
        return {"Ah_j": Ah_j, "e_ij": e_ij}

    def reduce_func(self, nodes):
        Ah_j = nodes.mailbox["Ah_j"]
        e = nodes.mailbox["e_ij"]
        sigma_ij = torch.sigmoid(e)
        Ah_j = sigma_ij*Ah_j

        if self.activation is not None:
            Ah_j = self.activation(Ah_j)
        c = torch.max(Ah_j, dim=1)[0]
        return {"c": c}

    def forward(self, g, h, norm):
        h = self.dropout(h)

        h = h*norm
        g.ndata["h"] = h
        g.ndata["Ah"] = h
        g.ndata["Bh"] = h
        g.update_all(self.message_func, self.reduce_func, self.apply_mod)

        b = g.ndata["b"]
        h = g.ndata["h"]
        h = h*norm

        if self.batch_norm:
            h = self.bn_node_h(h)

        return h, b


class ActivationGraphSageEdgeReprLayer(nn.Module):
    def __init__(self, indim, outdim,
                 aggr_type, activation, dropout,
                 batch_norm):
        super(ActivationGraphSageEdgeReprLayer, self).__init__()

        self.activation = activation
        self.batch_norm = batch_norm

        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.batchnorm_e = nn.BatchNorm1d(outdim)

        self.dropout = nn.Dropout(dropout)

        self.apply_mod = UpdateModule(self.activation, dropout)

    def message_func(self, edges):
        Ah_j = edges.src["Ah"]
        e_ij = edges.data["Ce"]+edges.src["Bh"]+edges.dst["Bh"]
        edges.data["e"] = e_ij
        return {"Ah_j": Ah_j, "e_ij": e_ij}

    def reduce_func(self, nodes):
        Ah_j = nodes.mailbox["Ah_j"]
        e = nodes.mailbox["e_ij"]
        sigma_ij = torch.sigmoid(e)
        Ah_j = sigma_ij*Ah_j

        if self.activation is not None:
            Ah_j = self.activation(Ah_j)
        c = torch.max(Ah_j, dim=1)[0]
        return {"c": c}

    def forward(self, g, h, e, norm):
        h = self.dropout(h)

        h = h*norm
        g.ndata["h"] = h
        g.ndata["Ah"] = h
        g.ndata["Bh"] = h
        g.ndata["e"] = e
        g.ndata["Ce"] = e
        g.update_all(self.message_func, self.reduce_func, self.apply_mod)

        b = g.ndata["b"]
        h = g.ndata["h"]
        h = h*norm
        e = g.ndata["e"]

        if self.activation is not None:
            e = self.activation(e)

        if self.batch_norm:
            h = self.batchnorm_h(h)
            e = self.batchnorm_e(e)

        return h, b, e
