import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch import SAGEConv

from layers.expander.expander_layer import LinearLayer, MultiLinearLayer


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
    def __init__(self, apply_func, activation, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.apply_func = apply_func
        self.activation = activation

    def concat(self, h, aggre_result):
        bundle = torch.cat((h, aggre_result), 1)
        bundle = self.apply_func(bundle)
        return bundle

    def forward(self, node):
        h = node.data['h']
        c = node.data['c']
        bundle = self.concat(h, c)
        bundle = F.normalize(bundle, p=2, dim=1)
        if self.activation is not None:
            bundle = self.activation(bundle)
        return {"h": bundle}


class GraphSageLayer(nn.Module):
    def __init__(self, apply_func,
                 aggr_type, activation, dropout,
                 batch_norm, residual=False, dgl_builtin=False, **kwargs):
        """
        Parameters
        ----------
        apply_func: callable, linear transform function to update node features
        aggr_type: string, neighborhood aggregation types
        activation: callable, activation function
        dropout: bool, whether or not use dropout on input features
        batch_norm: bool, whether or not add batch normalization before activation, after aggregation and linear transform
        residual: bool, whether or not use residual connection
        dgl_builtin: bool,
        kwargs
        """
        super(GraphSageLayer, self).__init__()
        self.dgl_builtin = dgl_builtin

        self.batch_norm, self.residual = batch_norm, residual
        self.dgl_builtin = dgl_builtin

        indim, outdim = apply_func.indim // 2, apply_func.outdim
        if indim != outdim:
            self.residual = False

        self.activation = activation
        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.dropout = nn.Dropout(dropout)

        if not self.dgl_builtin:
            self.apply_mod = UpdateModule(apply_func,
                                          activation=self.activation,
                                          dropout=dropout)
            if aggr_type == "max":
                self.reducer = MaxPoolAggregator(indim=indim,
                                                 outdim=indim,
                                                 activation=self.activation,
                                                 bias=apply_func.bias,
                                                 linear_type=apply_func
                                                 .linear_type,
                                                 **kwargs)
            elif aggr_type == "mean":
                self.reducer = MeanAggregator()
            elif aggr_type == "LSTM":
                self.reducer = LSTMAggregator(indim=apply_func.indim,
                                              hiddim=apply_func.indim)
            else:
                raise KeyError("Aggregator type {} not recognized."
                               .format(aggr_type))
        else:
            self.sageconv = SAGEConv(apply_func.indim,
                                     apply_func.outdim,
                                     aggr_type, dropout,
                                     self.activation)

    def forward(self, g, h):
        h_in = h
        if not self.dgl_builtin:
            h = self.dropout(h)
            g.ndata["h"] = h
            g.update_all(fn.copy_src(src="h", out="m"), self.reducer,
                         self.apply_mod)
            h = g.ndata["h"]
        else:
            h = self.sageconv(h)

        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.residual:
            h = h_in+h
        return h


class GraphSageEdgeLayer(nn.Module):
    def __init__(self, apply_func,
                 aggr_type, activation, dropout,
                 batch_norm, residual=False, dgl_builtin=False, **kwargs):
        super(GraphSageEdgeLayer, self).__init__()

        indim, outdim, hiddim, n_mlp_layer = (apply_func.indim - apply_func.outdim,
                                              apply_func.outdim,
                                              apply_func.hiddim,
                                              apply_func.num_layers)
        linear_type, self.bias = apply_func.linear_type, apply_func.bias

        self.activation = activation
        self.batch_norm, self.residual = batch_norm, residual
        if indim != outdim:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.dropout = nn.Dropout(dropout)

        self.A = MultiLinearLayer(indim, outdim,
                                  activation=self.activation,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)
        self.B = MultiLinearLayer(indim, outdim,
                                  activation=self.activation,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)

        self.apply_mod = UpdateModule(apply_func, self.activation, dropout)

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

    def forward(self, g, h):
        h_in = h
        h = self.dropout(h)

        g.ndata["h"] = h
        g.ndata["Ah"] = self.A(h)
        g.ndata["Bh"] = self.B(h)
        g.update_all(self.message_func, self.reduce_func, self.apply_mod)
        h = g.ndata["h"]

        if self.batch_norm:
            h = self.bn_node_h(h)

        if self.residual:
            h = h_in+h

        return h


class GraphSageEdgeReprLayer(nn.Module):
    def __init__(self, apply_func,
                 aggr_type, activation, dropout,
                 batch_norm, residual=False, dgl_builtin=False, **kwargs):
        super(GraphSageEdgeReprLayer, self).__init__()

        indim, outdim, hiddim, n_mlp_layer = (apply_func.indim-apply_func.outdim,
                                              apply_func.outdim,
                                              apply_func.hiddim,
                                              apply_func.num_layers)
        linear_type, self.bias = apply_func.linear_type, apply_func.bias

        self.activation = activation
        self.batch_norm, self.residual = batch_norm, residual
        if indim != outdim:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.batchnorm_e = nn.BatchNorm1d(outdim)

        self.dropout = nn.Dropout(dropout)

        self.A = MultiLinearLayer(indim, outdim,
                                  activation=self.activation,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)
        self.B = MultiLinearLayer(indim, outdim,
                                  activation=self.activation,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)
        self.C = MultiLinearLayer(indim, outdim,
                                  activation=self.activation,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)

        self.apply_mod = UpdateModule(apply_func, self.activation, dropout)

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

    def forward(self, g, h, e):
        h_in = h
        e_in = e
        h = self.dropout(h)

        g.ndata["h"] = h
        g.ndata["Ah"] = self.A(h)
        g.ndata["Bh"] = self.B(h)
        g.ndata["e"] = e
        g.ndata["Ce"] = self.C(e)
        g.update_all(self.message_func, self.reduce_func, self.apply_mod)
        h = g.ndata["h"]
        e = g.ndata["e"]

        if self.activation is not None:
            e = self.activation(e)

        if self.batch_norm:
            h = self.batchnorm_h(h)
            e = self.batchnorm_e(e)

        if self.residual:
            h = h_in+h
            e = e_in+e

        return h, e