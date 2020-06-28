import dgl
import torch
import torch.nn as nn

from baselines.layers.mlp_readout_layer import MLPReadout

class SimpleMLPNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        input_features = net_params['in_dim']
        hidden_features = net_params['hidden_dim']
        output_features = net_params['out_dim']
        self.batchnorm = net_params['batch_norm']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp']
        self.gated = net_params['gated']
        self.sparsity = net_params['sparsity']
        sparse_readout = net_params["sparse_readout"]
        mlp_readout = net_params["mlp_readout"]
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        if n_mlp_layers == 1:
            feat_mlp_modules = [nn.Linear(input_features, output_features, bias=True)]
        elif n_mlp_layers == 2:
            feat_mlp_modules = [nn.Linear(input_features, hidden_features, bias=True),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_features, output_features, bias=True)]
        elif n_mlp_layers >= 3:
            feat_mlp_modules = [nn.Linear(input_features, hidden_features, bias=True), nn.Dropout(dropout)]
            for _ in range(n_mlp_layers-2):
                feat_mlp_modules.append(nn.Linear(hidden_features, hidden_features, bias=True))
                feat_mlp_modules.append(nn.Dropout(dropout))
            feat_mlp_modules.append(nn.Linear(hidden_features, output_features, bias=True))

        self.feat_mlp = nn.Sequential(*feat_mlp_modules)

        self.batchnorm_h = nn.BatchNorm1d(input_features)

        if self.gated:
            self.gates = nn.Linear(output_features, output_features)
            self.gates.reset_parameters()

        if mlp_readout:
            self.readout = MLPReadout(output_features, n_classes)
        else:
            self.readout = nn.Linear(output_features, n_classes)
            self.readout.reset_parameters()

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.feat_mlp:
            if layer.__class__ == nn.Linear:
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, g, h, e, snorm_n, snorm_e):
        with g.local_scope():
            h = self.in_feat_dropout(h)
            if self.batchnorm:
                h = self.batchnorm_h(h)
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