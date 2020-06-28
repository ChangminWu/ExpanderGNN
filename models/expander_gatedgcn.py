import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from expander.Expander_layer import ExpanderLinearLayer, ExpanderMultiLinearLayer
from .readout import MLPReadout, ExpanderMLPReadout


class ExpanderGatedGCNLayer(nn.Module):
    def __init__(self, n_mlp_layers, input_dim, hidden_dim, output_dim, dropout, sparsity, graph_norm, batch_norm,
                 activation=None, residual=False):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm

        self.activation = activation
        self.residual = residual
        if input_dim != output_dim:
            self.residual = False
        self.sparsity = sparsity

        self.A = ExpanderMultiLinearLayer(n_mlp_layers, input_dim, hidden_dim,
                                          output_dim, self.sparsity, self.activation, self.batch_norm)
        self.B = ExpanderMultiLinearLayer(n_mlp_layers, input_dim, hidden_dim,
                                          output_dim, self.sparsity, self.activation, self.batch_norm)
        self.C = ExpanderMultiLinearLayer(n_mlp_layers, input_dim, hidden_dim,
                                          output_dim, self.sparsity, self.activation, self.batch_norm)
        self.D = ExpanderMultiLinearLayer(n_mlp_layers, input_dim, hidden_dim,
                                          output_dim, self.sparsity, self.activation, self.batch_norm)
        self.E = ExpanderMultiLinearLayer(n_mlp_layers, input_dim, hidden_dim,
                                          output_dim, self.sparsity, self.activation, self.batch_norm)

        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']
        e_ij = edges.data['Ce'] + edges.src['Dh'] + edges.dst['Eh']  # e_ij = Ce_ij + Dhi + Ehj
        edges.data['e'] = e_ij
        return {'Bh_j': Bh_j, 'e_ij': e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij']
        sigma_ij = torch.sigmoid(e)  # sigma_ij = sigmoid(e_ij)
        # h = Ah_i + torch.mean( sigma_ij * Bh_j, dim=1 ) # hi = Ahi + mean_j alpha_ij * Bhj
        h = Ah_i + torch.sum(sigma_ij * Bh_j, dim=1) / (torch.sum(sigma_ij,
                                                                  dim=1) + 1e-6)  # hi = Ahi + sum_j eta_ij/sum_j' eta_ij' * Bhj <= dense attention
        return {'h': h}

    def forward(self, g, h, e, snorm_n, snorm_e):

        h_in = h  # for residual connection
        e_in = e  # for residual connection

        g.ndata['h'] = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['e'] = e
        g.edata['Ce'] = self.C(e)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']  # result of graph convolution
        e = g.edata['e']  # result of graph convolution

        if self.graph_norm:
            h = h * snorm_n  # normalize activation w.r.t. graph size
            e = e * snorm_e  # normalize activation w.r.t. graph size

        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization
            e = self.bn_node_e(e)  # batch normalization

        if self.activation is not None:
            h = self.activation(h)  # non-linear activation
            e = self.activation(e)  # non-linear activation

        if self.residual:
            h = h_in + h  # residual connection
            e = e_in + e  # residual connection

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        return h, e

    # def __repr__(self):
    #     return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
    #                                                         self.in_channels,
    #                                                         self.out_channels)


class ExpanderGatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
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
        self.embedding_e = ExpanderLinearLayer(in_dim, hidden_dim, self.sparsity)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == n_layers - 1:
                self.layers.append(ExpanderGatedGCNLayer(n_mlp_layers, hidden_dim, hidden_dim, out_dim, dropout,
                                                         self.sparsity, self.graph_norm, self.batch_norm, self.activation, self.residual))
            else:
                self.layers.append(ExpanderGatedGCNLayer(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim, dropout,
                                                         self.sparsity, self.graph_norm, self.batch_norm,
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
        with g.local_scope():
            h = self.embedding_h(h)
            e = self.embedding_e(e)

            for conv in self.layers:
                h, e = conv(g, h, e, snorm_n, snorm_e)
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