import dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F

from expander.Expander_layer import ExpanderLinearLayer, ExpanderMultiLinearLayer
from .readout import MLPReadout, ExpanderMLPReadout


class NodeApplyModule(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, node):
        h = self.mlp(node.data['h'])
        return {'h': h}


class ExpanderGCNLayer(nn.Module):
    def __init__(self, apply_func, aggr_type, dropout, graph_norm, batch_norm,
                 activation=None, residual=False):
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

        self.batchnorm_h = nn.BatchNorm1d(outdim)

    def forward(self, g, feature, snorm_n):
        h_in = feature

        # g = g.local_var()
        g.ndata["h"] = feature
        g.update_all(fn.copy_src(src="h", out="m"), self._reducer('m', 'h'))
        g.apply_nodes(func=self.apply_func)

        h = g.ndata["h"]

        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.activation is not None:
            h = self.activation(h)

        if self.residual:
            h = h_in + h

        h = F.dropout(h, self.dropout, training=self.training)
        return h

    # def __repr__(self):
    #     return '{}(input_features={}, output_features={}, residual={})'.format(self.__class__.__name__, self.indim,
    #                                                                      self.outdim, self.residual)


class ExpanderGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        input_features = net_params['in_dim']
        hidden_features = net_params['hidden_dim']
        output_features = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
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
        neighbor_aggr_type = net_params['neighbor_aggr_GCN']

        self.embedding_h = ExpanderLinearLayer(input_features, hidden_features, self.sparsity)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == n_layers-1:
                mlp = ExpanderMultiLinearLayer(n_mlp_layers, hidden_features, hidden_features, output_features,
                                            self.sparsity, self.activation, self.batch_norm)
            else:
                mlp = ExpanderMultiLinearLayer(n_mlp_layers, hidden_features, hidden_features, hidden_features,
                                            self.sparsity, self.activation, self.batch_norm)

            self.layers.append(ExpanderGCNLayer(NodeApplyModule(mlp), neighbor_aggr_type, dropout, self.graph_norm,
                                                   self.batch_norm, self.activation, self.residual))

        if sparse_readout:
            if mlp_readout:
                self.readout = ExpanderMLPReadout(output_features, n_classes, sparsity=self.sparsity)
            else:
                self.readout = ExpanderLinearLayer(output_features, n_classes, sparsity=self.sparsity)
        else:
            if mlp_readout:
                self.readout = MLPReadout(output_features, n_classes)
            else:
                self.readout = nn.Linear(output_features, n_classes)
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










