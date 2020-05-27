import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.layers.mlp_readout_layer import MLPReadout

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
    def __init__(self, in_feats, out_feats, activation):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
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
    def __init__(self):
        super().__init__()

    def concat(self, h, aggre_result):
        bundle = torch.cat((h, aggre_result), 1)
        return bundle

    def forward(self, node):
        try:
            print("test")
            b = node.data['b']
        except:
            b = node.data['h']
        c = node.data['c']
        bundle = self.concat(b, c)
        bundle = F.normalize(bundle, p=2, dim=1)
        return {"b": bundle, "h": c}


class SimpleGraphSageLayer(nn.Module):
    def __init__(self, aggregator_type, in_feats, out_feats, dropout, residual=False):
        super().__init__()
        self.in_channels = in_feats
        self.aggregator_type = aggregator_type

        self.residual = residual
        if in_feats != out_feats:
            self.residual = False

        self.nodeapply = NodeApplyModule()
        self.dropout = nn.Dropout(p=dropout)

        if aggregator_type == "maxpool":
            self.aggregator = MaxPoolAggregator(in_feats, in_feats, activation=None)
        elif aggregator_type == "lstm":
            self.aggregator = LSTMAggregator(in_feats, in_feats)
        else:
            self.aggregator = MeanAggregator()

    def forward(self, g, h, norm):
        h_in = h
        h = self.dropout(h)

        h = h * norm
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'), self.aggregator, self.nodeapply)
        h = g.ndata.pop('h')
        b = g.ndata.pop('b')
        h = h * norm

        if self.residual:
            h = h_in + h

        return b, h

    # def __repr__(self):
    #     return '{}(in_channels={}, out_channels={}, aggregator={}, residual={})'.format(self.__class__.__name__,
    #                                          self.in_channels,
    #                                          self.out_channels, self.aggregator_type, self.residual)


class SimpleGraphSageNet(nn.Module):
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

        sparse_readout = net_params["sparse_readout"]
        mlp_readout = net_params["mlp_readout"]
        n_mlp_layers = net_params['n_mlp']

        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(SimpleGraphSageLayer(aggregator_type, hidden_dim,
                                                            hidden_dim, dropout, self.residual))

        self.linear = nn.Linear((n_layers+1)*hidden_dim, out_dim, bias=True)

        self.batchnorm_h = nn.BatchNorm1d((n_layers+1)*hidden_dim)

        if mlp_readout:
            self.readout = MLPReadout(out_dim, n_classes)
        else:
            self.readout = nn.Linear(out_dim, n_classes)
            self.readout.reset_parameters()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        nn.init.xavier_uniform_(self.embedding_h.weight)
        if self.embedding_h.bias is not None:
            nn.init.zeros_(self.embedding_h.bias)

    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(h.device).unsqueeze(1)

        for sconv in self.layers:
            b, h = sconv(g, h, norm)
        print(b.size)
        if self.batch_norm:
            b = self.batchnorm_h(b)

        b = self.linear(b)

        g.ndata['h'] = b

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