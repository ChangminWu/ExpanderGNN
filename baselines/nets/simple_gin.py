import dgl
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
import torch
from torch import nn

from baselines.layers.mlp_readout_layer import MLPReadout


class SimpleGINLayer(nn.Module):
    def __init__(self, indim, outdim, aggr_type, residual=False, init_eps=0, learn_eps=False):
        super().__init__()

        if aggr_type == 'sum':
            self._reducer = fn.sum
        elif aggr_type == 'max':
            self._reducer = fn.max
        elif aggr_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggr_type))

        self.residual = residual

        if indim != outdim:
            self.residual = False

        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

    def forward(self, g, h, norm):
        h_in = h  # for residual connection

        h = h * norm
        g.ndata['h'] = h
        g.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'neigh'))
        h = (1 + self.eps) * h + g.ndata['neigh']
        h = h * norm

        if self.residual:
            h = h_in + h  # residual connection
        return h

class SimpleGIN(nn.Module):
    def __init__(self, net_params):
        super(SimpleGIN, self).__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        neighbor_aggr_type = net_params['neighbor_aggr_SGIN']
        self.n_layers = net_params['L']
        n_mlp_layers = net_params["n_mlp"]
        self.batch_norm = net_params["batch_norm"]
        learn_eps = net_params['learn_eps_GIN']
        residual = net_params['residual']

        self.readout = net_params['readout']
        self.sparsity = net_params['sparsity']

        if neighbor_aggr_type == 'sum':
            self._reducer = fn.sum
        elif neighbor_aggr_type == 'max':
            self._reducer = fn.max
        elif neighbor_aggr_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(neighbor_aggr_type))

        if self.readout == 'sum':
            self.pool = SumPooling()
        elif self.readout == 'mean':
            self.pool = AvgPooling()
        elif self.readout == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

        sparse_readout = net_params["sparse_readout"]
        mlp_readout = net_params["mlp_readout"]

        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.batchnorm_h = nn.BatchNorm1d(hidden_dim)

        self.sgin = nn.ModuleList()
        self.linears_prediction = nn.ModuleList()
        for layer in range(self.n_layers):
            self.sgin.append(SimpleGINLayer(hidden_dim, hidden_dim, neighbor_aggr_type, residual, 0, learn_eps))
            self.linears_prediction.append(nn.Linear(hidden_dim, out_dim))

        self.linears_readout = nn.ModuleList()
        for layer in range(self.n_layers+1):
            if mlp_readout:
                self.linears_readout.append(MLPReadout(out_dim, n_classes))
            else:
                self.linears_readout.append(nn.Linear(out_dim, n_classes))


        if not mlp_readout:
            for layer in self.linears_readout:
                layer.reset_parameters()

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        for layer in self.linears_prediction:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.embedding_h.weight)
        if self.embedding_h.bias is not None:
            nn.init.zeros_(self.embedding_h.bias)

    def forward(self, g, feat, e, snorm_n, snorm_e):
        with g.local_scope():
            h = self.embedding_h(feat)
            h = self.in_feat_dropout(h)

            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(feat.device).unsqueeze(1)

            hidden_rep = [h]
            for i in range(self.n_layers):
                h = self.sgin[i](g, h, norm)
                if self.batch_norm:
                    h = self.batchnorm_h(h)
                h = self.linears_prediction[i](h)
                hidden_rep.append(h)

            score_over_layer = 0
            # perform pooling over all nodes in each graph in every layer
            for i, h in enumerate(hidden_rep):
                pooled_h = self.pool(g, h)
                score_over_layer += self.linears_readout[i](pooled_h)

            return score_over_layer

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss