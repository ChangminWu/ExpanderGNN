import dgl
import torch
import torch.nn as nn

from backup.previous_codes.layers.expander.expander_layer import ExpanderLinearLayer
from backup.previous_codes.layers.expander.expander_mlp_readout_layer import ExpanderMLPReadout


class ExpanderMLPNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.gated = net_params['gated']
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        feat_mlp_modules = [
            ExpanderLinearLayer(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_layers-1):
            feat_mlp_modules.append(ExpanderLinearLayer(hidden_dim, hidden_dim))
            feat_mlp_modules.append(nn.ReLU())
            feat_mlp_modules.append(nn.Dropout(dropout))
        self.feat_mlp = nn.Sequential(*feat_mlp_modules)
        
        if self.gated:
            self.gates = ExpanderLinearLayer(hidden_dim, hidden_dim)
        
        self.readout_mlp = ExpanderMLPReadout(hidden_dim, n_classes)

    def forward(self, g, h, e, snorm_n, snorm_e):
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
        
        return self.readout_mlp(hg)

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
