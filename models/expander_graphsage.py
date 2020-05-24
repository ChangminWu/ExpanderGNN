import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

from expander.Expander_layer import ExpanderLinearLayer, ExpanderMultiLinearLayer
from .readout import MLPReadout, ExpanderMLPReadout


class Aggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}

    def aggre(self, neighbour):
        raise NotImplementedError


class MeanAggregator(Aggregator):
    def __init__(self):
        super().__init__()

    def aggre(self, neighbour):
        mean_neighbour = torch.mean(neighbour, dim=1)
        return mean_neighbour


class MaxPoolAggregator(Aggregator):
    def __init__(self, in_feats, out_feats, activation, sparsity):
        super().__init__()
        self.linear = ExpanderLinearLayer(in_feats, out_feats, sparsity)
        self.activation = activation

    def aggre(self, neighbour):
        neighbour = self.linear(neighbour)
        if self.activation is not None:
            neighbour = self.activation(neighbour)
        maxpool_neighbour = torch.max(neighbour, dim=1)[0]
        return maxpool_neighbour


class LSTMAggregator(Aggregator):
    def __init__(self, in_feats, hidden_feats):
        super().__init__()
        self.lstm = nn.LSTM(in_feats, hidden_feats, batch_first=True)
        self.hidden_dim = hidden_feats
        self.hidden = self.init_hidden()

        nn.init.xavier_uniform_(self.lstm.weight,
                                gain=nn.init.calculate_gain('relu'))

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def aggre(self, neighbours):
        rand_order = torch.randperm(neighbours.size()[1])
        neighbours = neighbours[:, rand_order, :]

        (lstm_out, self.hidden) = self.lstm(neighbours.view(neighbours.size()[0],
                                                            neighbours.size()[1], -1))
        return lstm_out[:, -1, :]

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}


class NodeApplyModule(nn.Module):
    def __init__(self, n_mlp_layers, in_feats, hidden_feats, out_feats, sparsity, activation, dropout, batchnorm):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        self.linear = ExpanderMultiLinearLayer(n_mlp_layers, in_feats, hidden_feats, out_feats, sparsity,
                                               self.activation, batchnorm)


    def concat(self, h, aggre_result):
        bundle = torch.cat((h, aggre_result), 1)
        bundle = self.linear(bundle)
        return bundle

    def forward(self, node):
        h = node.data['h']
        c = node.data['c']
        bundle = self.concat(h, c)
        bundle = F.normalize(bundle, p=2, dim=1)
        if self.activation is not None:
            bundle = self.activation(bundle)
        return {"h": bundle}


class ExpanderGraphSageLayer(nn.Module):
    def __init__(self, n_mlp_layers, aggregator_type, in_feats, hidden_feats, out_feats, dropout,
                 sparsity, batchnorm=False, activation=None, residual=False):
        super().__init__()
        self.in_channels = in_feats
        self.hidden_channels = hidden_feats
        self.out_channels = out_feats
        self.aggregator_type = aggregator_type

        self.sparsity = sparsity
        self.residual = residual
        if in_feats != out_feats:
            self.residual = False

        self.batchnorm = batchnorm

        self.nodeapply = NodeApplyModule(n_mlp_layers, in_feats, hidden_feats, out_feats,
                                         self.sparsity, activation, dropout, self.batchnorm)
        self.dropout = nn.Dropout(p=dropout)

        if aggregator_type == "maxpool":
            self.aggregator = MaxPoolAggregator(in_feats, in_feats, activation, self.sparsity)
        elif aggregator_type == "lstm":
            self.aggregator = LSTMAggregator(in_feats, in_feats)
        else:
            self.aggregator = MeanAggregator()

        self.batchnorm_h = nn.BatchNorm1d(out_feats)

    def forward(self, g, h, snorm_n=None):
        h_in = h  # for residual connection
        h = self.dropout(h)
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'), self.aggregator, self.nodeapply)
        h = g.ndata['h']

        if snorm_n is not None:
            h = h * snorm_n

        if self.batchnorm:
            h = self.batchnorm_h(h)

        if self.residual:
            h = h_in + h

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, aggregator={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.aggregator_type, self.residual)


class ExpanderGraphSageNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']
        self.readout = net_params['readout']

        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.sparsity = net_params['sparsity']
        if net_params['activation'] == "relu":
            self.activation = nn.ReLU()
        elif net_params['activation'] is None:
            self.activation = None
        else:
            raise ValueError("Invalid activation type.")
        sparse_readout = net_params["sparse_readout"]
        mlp_readout = net_params["mlp_readout"]
        n_mlp_layers = net_params['n_mlp']

        self.embedding_h = ExpanderLinearLayer(in_dim, hidden_dim, self.sparsity)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([ExpanderGraphSageLayer(n_mlp_layers, aggregator_type, hidden_dim, hidden_dim,
                                                            hidden_dim, dropout, self.sparsity, self.batch_norm,
                                                            self.activation, self.residual) for _ in range(n_layers - 1)])
        self.layers.append(ExpanderGraphSageLayer(n_mlp_layers, aggregator_type, hidden_dim, hidden_dim,
                                                            out_dim, dropout, self.sparsity, self.batch_norm,
                                                            self.activation, self.residual))

        if sparse_readout:
            if mlp_readout:
                self.readout = ExpanderMLPReadout(out_dim, n_classes, sparsity=self.sparsity)
            else:
                self.readout = ExpanderLinearLayer(out_dim, n_classes, sparsity=self.sparsity)
        else:
            if mlp_readout:
                self.readout = MLPReadout(out_dim, n_classes)
            else:
                self.readout = nn.Linear(out_dim, n_classes)
                self.readout.reset_parameters()

    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(g, h, snorm_n)
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return self.readout(hg)

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss