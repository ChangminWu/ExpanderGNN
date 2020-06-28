import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

from expander.Expander_layer import ExpanderLinearLayer, ExpanderMultiLinearLayer
from .readout import MLPReadout, ExpanderMLPReadout


class NodeApplyModule(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, h):
        h = self.mlp(h)
        return h


class ExpanderGINLayer(nn.Module):
    def __init__(self, apply_func, aggr_type, dropout, graph_norm, batch_norm,
                 activation=None, residual=False, init_eps=0, learn_eps=False):
        super().__init__()
        self.apply_func = apply_func

        if aggr_type == 'sum':
            self._reducer = fn.sum
        elif aggr_type == 'max':
            self._reducer = fn.max
        elif aggr_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggr_type))

        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout
        self.activation = activation

        indim = apply_func.mlp.indim
        outdim = apply_func.mlp.outdim

        if indim != outdim:
            self.residual = False
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

        self.bn_node_h = nn.BatchNorm1d(outdim)

    def forward(self, g, h, snorm_n):
        h_in = h  # for residual connection

        g = g.local_var()
        g.ndata['h'] = h
        g.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'neigh'))
        h = (1 + self.eps) * h + g.ndata['neigh']
        if self.apply_func is not None:
            h = self.apply_func(h)

        if self.graph_norm:
            h = h * snorm_n  # normalize activation w.r.t. graph size

        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization

        if self.activation is not None:
            h = self.activation(h)  # non-linear activation

        if self.residual:
            h = h_in + h  # residual connection

        h = F.dropout(h, self.dropout, training=self.training)
        return h


class ExpanderGINNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp']  # GIN
        learn_eps = net_params['learn_eps_GIN']  # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN']  # GIN
        readout = net_params['readout']  # this is graph_pooling_type
        graph_norm = net_params['graph_norm']
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']

        self.sparsity = net_params['sparsity']
        if net_params['activation'] == "relu":
            self.activation = nn.ReLU()
        elif net_params['activation'] is None:
            self.activation = None
        else:
            raise ValueError("Invalid activation type.")
        sparse_readout = net_params["sparse_readout"]
        mlp_readout = net_params["mlp_readout"]

        self.ginlayers = nn.ModuleList()
        self.embedding_h = ExpanderLinearLayer(in_dim, hidden_dim, self.sparsity)

        for layer in range(self.n_layers):
            mlp = ExpanderMultiLinearLayer(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim,
                                           self.sparsity, self.activation, batch_norm)
            self.ginlayers.append(ExpanderGINLayer(NodeApplyModule(mlp), neighbor_aggr_type, dropout, graph_norm,
                                                   batch_norm, self.activation, residual, 0, learn_eps))

        # Linear function for graph poolings (readout) of output of each layer
        # which maps the output of different layers into a prediction score
        if readout == 'sum':
            self.pool = SumPooling()
        elif readout == 'mean':
            self.pool = AvgPooling()
        elif readout == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

        self.linears_prediction = nn.ModuleList()
        for layer in range(self.n_layers+1):
            if sparse_readout:
                if mlp_readout:
                    self.linears_prediction.append(ExpanderMLPReadout(hidden_dim, n_classes, self.sparsity))
                else:
                    self.linears_prediction.append(ExpanderLinearLayer(hidden_dim, n_classes, self.sparsity))
            else:
                if mlp_readout:
                    self.linears_prediction.append(MLPReadout(hidden_dim, n_classes))
                else:
                    self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))

        if not sparse_readout and mlp_readout:
            for layer in self.linears_prediction:
                layer.reset_parameters()

    def forward(self, g, h, e, snorm_n, snorm_e):
        with g.local_scope():
            h = self.embedding_h(h)

            # list of hidden representation at each layer (including input)
            hidden_rep = [h]

            for i in range(self.n_layers):
                h = self.ginlayers[i](g, h, snorm_n)
                hidden_rep.append(h)

            score_over_layer = 0
            # perform pooling over all nodes in each graph in every layer
            for i, h in enumerate(hidden_rep):
                pooled_h = self.pool(g, h)
                score_over_layer += self.linears_prediction[i](pooled_h)

            return score_over_layer

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss