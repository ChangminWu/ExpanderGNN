import dgl
import torch
import torch.nn as nn

from expander.Expander_layer import ExpanderLinearLayer, ExpanderMultiLinearLayer
from .readout import MLPReadout, ExpanderMLPReadout


class ExpanderMLPLayer(nn.Module):
    def __init__(self, n_mlp_layers, input_features, hidden_features, output_features, sparsity, activation,
                 dropout, batchnorm=False):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear = ExpanderMultiLinearLayer(n_mlp_layers, input_features, hidden_features,
                                               output_features, sparsity, activation, batchnorm)

        self.batch_norm = batchnorm
        self.batchnorm_h = nn.BatchNorm1d(output_features)

    def forward(self, h):
        h = self.linear(h)
        if self.batch_norm:
            h = self.batchnorm_h(h)
        if self.activation is not None:
            h = self.activation(h)
        h = self.dropout(h)
        return h


class ExpanderMLPNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        input_features = net_params['in_dim']
        hidden_features = net_params['hidden_dim']
        output_features = net_params['out_dim']
        batchnorm = net_params['batch_norm']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp']
        self.gated = net_params['gated']
        self.sparsity = net_params['sparsity']
        if net_params['activation'] == "relu":
            self.activation = nn.ReLU()
        elif net_params['activation'] is None:
            self.activation = None
        sparse_readout = net_params["sparse_readout"]
        mlp_readout = net_params["mlp_readout"]
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        feat_mlp_modules = [ExpanderMLPLayer(n_mlp_layers, input_features, hidden_features, hidden_features,
                                             self.sparsity, self.activation, dropout, batchnorm)]
        for _ in range(n_layers - 2):
            feat_mlp_modules.append(ExpanderMLPLayer(n_mlp_layers, hidden_features, hidden_features, hidden_features,
                                                     self.sparsity, self.activation, dropout, batchnorm))

        feat_mlp_modules.append(ExpanderMLPLayer(n_mlp_layers, hidden_features, hidden_features, output_features,
                                                 self.sparsity, self.activation, dropout, batchnorm))

        self.feat_mlp = nn.Sequential(*feat_mlp_modules)

        if self.gated:
            self.gates = ExpanderLinearLayer(output_features, output_features, self.sparsity)

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
        with g.local_scope():
            h = self.in_feat_dropout(h)
            h = self.feat_mlp(h)
            if self.gated:
                h = torch.sigmoid(self.gates(h)) * h
                g.ndata['h'] = h
                hg = dgl.sum_nodes(g, 'h')
                # hg = torch.cat(
                #     (
                #         dgl.sum_nodes(g, 'h'),
                #         dgl.max_nodes(g, 'h')
                #     ),
                #     dim=1
                # )

            else:
                g.ndata['h'] = h
                hg = dgl.mean_nodes(g, 'h')

            return self.readout(hg)

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss

