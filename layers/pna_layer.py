import torch
import torch.nn as nn

import dgl.function as fn

from layers.pna_utils.aggregators import AGGREGATORS
from layers.pna_utils.scalers import SCALERS
from layers.expander.expander_layer import LinearLayer, MultiLinearLayer


"""
    Code taken and adapted from https://github.com/lukecavabarrett/pna
    PNA: Principal Neighbourhood Aggregation
    Gabriele Corso, Luca Cavalleri,
    Dominique Beaini, Pietro Lio, Petar Velickovic
    https://arxiv.org/abs/2004.05718
"""


class PNATower(nn.Module):
    def __init__(self, indim, outdim, activation, dropout, batch_norm,
                 aggregators, scalers, avg_d,
                 num_pretrans_layer, num_posttrans_layer,
                 edge_features, edge_dim,
                 bias=True, linear_type="expander", **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.edge_features = edge_features
        self.activation = activation

        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.aggregators = aggregators
        self.scalers = scalers

        self.pretrans = MultiLinearLayer(
                                 2*indim + (edge_dim if edge_features else 0),
                                 indim,
                                 activation=self.activation,
                                 batch_norm=self.batch_norm,
                                 num_layers=num_pretrans_layer,
                                 hiddim=indim,
                                 bias=bias,
                                 linear_type=linear_type,
                                 **kwargs)

        self.posttrans = MultiLinearLayer(
                                 indim*(1+len(aggregators)*len(scalers)),
                                 outdim,
                                 activation=self.activation,
                                 batch_norm=self.batch_norm,
                                 num_layers=num_posttrans_layer,
                                 hiddim=outdim,
                                 bias=bias,
                                 linear_type=linear_type,
                                 **kwargs)

        self.avg_d = avg_d

    def pretrans_edges(self, edges):
        if self.edge_features:
            z2 = torch.cat([edges.src['h'],
                            edges.dst['h'],
                            edges.data['ef']], dim=1)
        else:
            z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        return {'e': self.pretrans(z2)}

    def message_func(self, edges):
        return {'e': edges.data['e']}

    def reduce_func(self, nodes):
        h = nodes.mailbox['e']
        D = h.shape[-2]
        h = torch.cat([aggregate(h) for aggregate in self.aggregators], dim=1)
        h = torch.cat([scale(h, D=D, avg_d=self.avg_d)
                       for scale in self.scalers], dim=1)
        return {'h': h}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data['h'])

    def forward(self, g, h, e):
        g.ndata['h'] = h
        # add the edges information only if edge_features = True
        if self.edge_features:
            g.edata['ef'] = e

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata['h']], dim=1)

        # posttransformation
        h = self.posttrans(h)

        # graph and batch normalization
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = self.dropout(h)
        return h


class PNALayer(nn.Module):
    def __init__(self, indim, outdim, hiddim, activation, dropout, batch_norm,
                 aggregators, scalers, avg_d,
                 num_tower=1, num_pretrans_layer=1, num_posttrans_layer=1,
                 divide_input=True, residual=False, edge_features=False,
                 edge_dim=0, bias=True, linear_type="expander", **kwargs):
        """
        :param in_dim:                 size of the input per node
        :param out_dim:                size of the output per node
        :param aggregators:            set of aggregation function identifiers
        :param scalers:                set of scaling functions identifiers
        :param avg_d:                  average degree of nodes in the
                                       training set,
                                       used by scalers to normalize
        :param dropout:                dropout used
        :param batch_norm:             whether to use batch normalisation
        :param num_tower:              number of towers to use
        :param num_pretrans_layer:     number of layers in the transformation
                                       before the aggregation
        :param num_posttrans_layer:    number of layers in the transformation
                                       after the aggregation
        :param divide_input:           whether the input features should be
                                       split between towers or not
        :param residual:               whether to add a residual connection
        :param edge_features:          whether to use the edge features
        :param edge_dim:               size of the edge features
        """
        super().__init__()
        assert ((not divide_input) or indim % num_tower == 0),\
            "if divide_input is set the number of towers has to divide indim"
        assert (outdim % num_tower == 0),\
            "the number of towers has to divide the outdim"
        assert avg_d is not None

        # retrieve the aggregators and scalers functions
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators.split()]
        scalers = [SCALERS[scale] for scale in scalers.split()]

        self.divide_input = divide_input
        self.input_tower = indim // num_tower if divide_input else indim
        self.output_tower = outdim // num_tower
        self.edge_features = edge_features
        self.residual = residual
        if indim != outdim:
            self.residual = False

        # convolution
        self.towers = nn.ModuleList()
        for _ in range(num_tower):
            self.towers.append(
                PNATower(indim=self.input_tower,
                         outdim=self.output_tower,
                         activation=activation,
                         dropout=dropout,
                         batch_norm=batch_norm,
                         aggregators=aggregators,
                         scalers=scalers,
                         avg_d=avg_d,
                         num_pretrans_layer=num_pretrans_layer,
                         num_posttrans_layer=num_posttrans_layer,
                         edge_features=self.edge_features,
                         edge_dim=edge_dim,
                         bias=bias, linear_type=linear_type, **kwargs))

        # mixing network
        self.mixing_network = LinearLayer(outdim, outdim, bias=bias,
                                          linear_type=linear_type, **kwargs)

        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.activation = activation  # nn.LeakyReLU()

    def forward(self, g, h, e):
        h_in = h  # for residual connection

        if self.divide_input:
            h_cat = torch.cat(
                [tower(g, h[:, n_tower * self.input_tower:
                            (n_tower + 1) * self.input_tower], e)
                 for n_tower, tower in enumerate(self.towers)], dim=1)
        else:
            h_cat = torch.cat([tower(g, h, e) for tower in self.towers], dim=1)

        h_out = self.mixing_network(h_cat)

        if self.activation is not None:
            h_out = self.activation(h_out)

        h_out = self.dropout(h_out)

        if self.batch_norm:
            h_out = self.batchnorm_h(h_out)

        if self.residual:
            h_out = h_in + h_out  # residual connection
        return h_out


class PNASimplifiedLayer(nn.Module):
    def __init__(self, indim, outdim, hiddim, activation, dropout, batch_norm,
                 aggregators, scalers, avg_d,
                 num_posttrans_layer=1,
                 residual=False, bias=True, linear_type="expander", **kwargs):
        """
        A simpler version of PNA layer that simply aggregates
        the neighbourhood (similar to GCN and GIN),
        without using the pretransformation or the tower mechanisms
        of the MPNN. It does not support edge features.
        :param indim:                  size of the input per node
        :param outdim:                 size of the output per node
        :param aggregators:            set of aggregation function identifiers
        :param scalers:                set of scaling functions identifiers
        :param avg_d:                  average degree of nodes in the
                                       training set, used by scalers
                                       to normalize
        :param dropout:                dropout used
        :param batch_norm:             whether to use batch normalisation
        :param num_posttrans_layer:    number of layers in the transformation
                                       after the aggregation
        """
        super().__init__()

        # retrieve the aggregators and scalers functions
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators.split()]
        scalers = [SCALERS[scale] for scale in scalers.split()]

        self.aggregators = aggregators
        self.scalers = scalers
        self.residual = residual
        if indim != outdim:
            self.residual = False

        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.activation = activation

        self.posttrans = MultiLinearLayer(
                                 indim*(1+len(aggregators)*len(scalers)),
                                 outdim,
                                 activation=self.activation,
                                 batch_norm=self.batch_norm,
                                 num_layers=num_posttrans_layer,
                                 hiddim=hiddim,
                                 bias=bias,
                                 linear_type=linear_type,
                                 **kwargs)
        self.avg_d = avg_d

    def reduce_func(self, nodes):
        h = nodes.mailbox['m']
        D = h.shape[-2]
        h = torch.cat([aggregate(h) for aggregate in self.aggregators], dim=1)
        h = torch.cat([scale(h, D=D, avg_d=self.avg_d)
                       for scale in self.scalers], dim=1)
        return {'h': h}

    def forward(self, g, h, e):
        h_in = h  # for residual connection
        g.ndata['h'] = h

        # aggregation
        g.update_all(fn.copy_u('h', 'm'), self.reduce_func)
        h = torch.cat([h, g.ndata['h']], dim=1)

        # posttransformation
        h = self.posttrans(h)

        if self.activation is not None:
            h = self.activation(h)
        h = self.dropout(h)
        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.residual:
            h = h_in + h  # residual connection
        return h