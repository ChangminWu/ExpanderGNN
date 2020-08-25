import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

from expander.expander_layer import LinearLayer, MultiLinearLayer


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
    def __init__(self, indim, outdim, activation, bias, linear_type, **kwargs):
        super(MaxPoolAggregator, self).__init__()
        self.linear = LinearLayer(indim, outdim, bias=bias,
                                  linear_type=linear_type, **kwargs)
        self.activation = activation

    def aggre(self, neighbour):
        neighbour = self.linear(neighbour)
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
    def __init__(self, apply_func=None):
        super().__init__()
        self.apply_func = apply_func

    def concat(self, h, aggre_result):
        bundle = torch.cat((h, aggre_result), 1)
        if self.apply_func is not None:
            bundle = self.apply_func(bundle)
        return bundle

    def forward(self, node):
        h = node.data['h']
        c = node.data['c']
        bundle = self.concat(h, c)
        bundle = F.normalize(bundle, p=2, dim=1)
        return {"h": bundle}


class SimpleGraphSageLayer(nn.Module):
    def __init__(self, indim, outdim, apply_func, aggr_type, dropout,
                 batch_norm, residual=False, bias=True,
                 linear_type="expander", **kwargs):
        super(SimpleGraphSageLayer, self).__init__()

        self.batch_norm, self.linear_type, self.bias = (batch_norm,
                                                        linear_type,
                                                        bias)

        self.residual = residual
        if indim != outdim:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.dropout = nn.Dropout(dropout)

        self.apply_mod = UpdateModule(apply_func)

        if aggr_type == "max":
            self.reducer = MaxPoolAggregator(indim=indim,
                                             outdim=indim,
                                             activation=None,
                                             bias=self.bias,
                                             linear_type=self.linear_type,
                                             **kwargs)
        elif aggr_type == "mean":
            self.reducer = MeanAggregator()
        elif aggr_type == "LSTM":
            self.reducer = LSTMAggregator(indim=indim,
                                          hiddim=indim)
        else:
            raise KeyError("Aggregator type {} not recognized."
                           .format(aggr_type))

    def forward(self, g, h, norm):
        h_in = h
        h = self.dropout(h)

        h = h*norm
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src="h", out="m"), self.reducer,
                     self.apply_mod)

        h = g.ndata.pop["h"]
        h = h*norm

        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.residual:
            h = h_in+h
        return h


class SimpleGraphSageEdgeLayer(nn.Module):
    def __init__(self, indim, outdim, hiddim, apply_func, aggr_type,
                 n_mlp_layer, dropout,
                 batch_norm, residual=False, bias=True,
                 linear_type="expander", **kwargs):
        super(SimpleGraphSageEdgeLayer, self).__init__()

        self.linear_type, self.bias = linear_type, bias
        self.batch_norm = batch_norm

        self.residual = residual
        if indim != outdim:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.dropout = nn.Dropout(dropout)

        self.A = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)

        self.B = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)

        self.apply_mod = UpdateModule(apply_func)

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

        c = torch.max(Ah_j, dim=1)[0]
        return {"c": c}

    def forward(self, g, h, norm):
        h_in = h
        h = self.dropout(h)

        h = h*norm
        g.ndata["h"] = h
        g.ndata["Ah"] = self.A(h)
        g.ndata["Bh"] = self.B(h)
        g.update_all(self.message_func, self.reduce_func, self.apply_mod)
        h = g.ndata.pop["h"]
        h = h*norm

        if self.batch_norm:
            h = self.bn_node_h(h)

        if self.residual:
            h = h_in+h

        return h


class SimpleGraphSageEdgeReprLayer(nn.Module):
    def __init__(self, indim, outdim, hiddim, apply_func, aggr_type,
                 n_mlp_layer, dropout,
                 batch_norm, residual=False, bias=True,
                 linear_type="expander", **kwargs):
        super(SimpleGraphSageEdgeReprLayer, self).__init__()

        self.linear_type, self.bias = linear_type, bias
        self.batch_norm = batch_norm

        self.residual = residual
        if indim != outdim:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.batchnorm_e = nn.BatchNorm1d(outdim)
        self.dropout = nn.Dropout(dropout)

        self.A = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)
        self.B = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)
        self.C = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)

        self.apply_mod = UpdateModule(apply_func)

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

        c = torch.max(Ah_j, dim=1)[0]
        return {"c": c}

    def forward(self, g, h, e, norm):
        h_in = h
        e_in = e
        h = self.dropout(h)

        h = h*norm
        g.ndata["h"] = h
        g.ndata["Ah"] = self.A(h)
        g.ndata["Bh"] = self.B(h)
        g.ndata["e"] = e
        g.ndata["Ce"] = self.C(e)
        g.update_all(self.message_func, self.reduce_func, self.apply_mod)
        h = g.ndata.pop["h"]
        e = g.ndata.pop["e"]
        h = h*norm

        if self.batch_norm:
            h = self.batchnorm_h(h)
            e = self.batchnorm_e(e)

        if self.residual:
            h = h_in+h
            e = e_in+e

        return h, e
