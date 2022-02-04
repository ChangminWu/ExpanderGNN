import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, PNAConv, global_add_pool, BatchNorm, inits
from layers.convs import ExpanderGCNConv, ExpanderSAGEConv, ActivationGCNConv, ExpanderPNAConv
from expander.samplers import sampler
from expander.expander import ExpanderLinear


class ExpanderGCN(torch.nn.Module):
    def __init__(self, indim, hiddim, outdim, num_layers, dropout, density, sample_method, weight_initializer, dense_output=False):
        super().__init__()

        self.edge_index_list = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.edge_index_list.append(sampler(indim, hiddim, density, sample_method))
        self.convs.append(ExpanderGCNConv(indim, hiddim, cached=True, edge_index=self.edge_index_list[-1], weight_initializer=weight_initializer))
        self.bns.append(torch.nn.BatchNorm1d(hiddim))

        for _ in range(num_layers - 2):
            self.edge_index_list.append(sampler(hiddim, hiddim, density, sample_method))
            self.convs.append(ExpanderGCNConv(hiddim, hiddim, cached=True, edge_index=self.edge_index_list[-1], weight_initializer=weight_initializer))
            self.bns.append(torch.nn.BatchNorm1d(hiddim))
        
        if not dense_output:
            self.edge_index_list.append(sampler(hiddim, outdim, density, sample_method))
            self.convs.append(ExpanderGCNConv(hiddim, outdim, cached=True, edge_index=self.edge_index_list[-1], weight_initializer=weight_initializer))
        else:
            self.convs.append(GCNConv(hiddim, outdim, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class ActivationGCN(torch.nn.Module):
    def __init__(self, indim, hiddim, outdim, num_layers, dropout):
        super().__init__()

        self.edge_index_list = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(ActivationGCNConv(indim, indim, cached=True))
        self.bns.append(torch.nn.BatchNorm1d(indim))

        for _ in range(num_layers - 2):
            self.convs.append(ActivationGCNConv(indim, indim, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(indim))
        
        self.convs.append(GCNConv(indim, outdim, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        print("initial: ", x)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            #print("after prop: ", x)
            #x = self.bns[i](x)
            if i == 0:
                x = F.relu(x)
                print("after activ: ", x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class ExpanderSAGE(torch.nn.Module):
    def __init__(self, indim, hiddim, outdim, num_layers, dropout, density, sample_method, weight_initializer, dense_output=False):
        super().__init__()

        self.edge_index_list = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.edge_index_list.append(sampler(indim, hiddim, density, sample_method))
        self.convs.append(ExpanderSAGEConv(indim, hiddim, edge_index=self.edge_index_list[-1], weight_initializer=weight_initializer))
        self.bns.append(torch.nn.BatchNorm1d(hiddim))

        for _ in range(num_layers - 2):
            self.edge_index_list.append(sampler(hiddim, hiddim, density, sample_method))
            self.convs.append(ExpanderSAGEConv(hiddim, hiddim, edge_index=self.edge_index_list[-1], weight_initializer=weight_initializer))
            self.bns.append(torch.nn.BatchNorm1d(hiddim))
        
        if not dense_output:
            self.edge_index_list.append(sampler(hiddim, outdim, density, sample_method))
            self.convs.append(ExpanderSAGEConv(hiddim, outdim, edge_index=self.edge_index_list[-1], weight_initializer=weight_initializer))
        else:
            self.convs.append(SAGEConv(hiddim, outdim))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
        

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class PNA(torch.nn.Module):
    def __init__(self, deg, readout_layer=3):
        super().__init__()

        self.node_emb = torch.nn.Embedding(21, 75)
        self.edge_emb = torch.nn.Embedding(4, 50)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=50, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        if readout_layer == 3:
            self.mlp = torch.nn.Sequential(torch.nn.Linear(75, 50), torch.nn.ReLU(), torch.nn.Linear(50, 25), torch.nn.ReLU(), torch.nn.Linear(25, 1))
        else:
            self.mlp = torch.nn.Sequential(torch.nn.Linear(75, 1))

    def reset_parameters(self):
        self.node_emb.reset_parameters()
        self.edge_emb.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        inits.reset(self.mlp)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        return self.mlp(x)


class ExpanderPNA(torch.nn.Module):
    def __init__(self, deg, readout_layer, density, sample_method, weight_initializer, dense_output=False):
        super().__init__()

        self.node_emb = torch.nn.Embedding(21, 75)
        self.edge_emb = torch.nn.Embedding(4, 50)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.edge_index_list = []
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for _ in range(4):
            edge_index_list = []
            edge_index_list.append(sampler(50, 75, density, sample_method))
            for _ in range(5):
                edge_index_list.append(sampler(225, 75, density, sample_method))
                edge_index_list.append(sampler(975, 15, density, sample_method))
            edge_index_list.append(sampler(75, 75, density, sample_method))
            self.edge_index_list.append(edge_index_list)
            
            conv = ExpanderPNAConv(indim=75, outdim=75, aggregators=aggregators, scalers=scalers, deg=deg, 
                                   edge_dim=50, towers=5, pre_layers=1, post_layers=1, divide_input=False, edge_index=edge_index_list, weight_initializer=weight_initializer)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        if not dense_output:
            if readout_layer == 3:
                self.edge_index_list.append(sampler(75, 50, density, sample_method))
                self.edge_index_list.append(sampler(50, 25, density, sample_method))
                self.edge_index_list.append(sampler(25, 1, density, sample_method))
                self.mlp = torch.nn.Sequential(ExpanderLinear(75, 50, edge_index=self.edge_index_list[-3], weight_initializer=weight_initializer), 
                                               torch.nn.ReLU(), 
                                               ExpanderLinear(50, 25, edge_index=self.edge_index_list[-2], weight_initializer=weight_initializer), 
                                               torch.nn.ReLU(), 
                                               ExpanderLinear(25, 1, edge_index=self.edge_index_list[-1], weight_initializer=weight_initializer))
            else:
                self.edge_index_list.append(sampler(75, 1, density, sample_method))
                self.mlp = torch.nn.Sequential(ExpanderLinear(75, 1, edge_index=self.edge_index_list[-1], weight_initializer=weight_initializer))
        else:
            if readout_layer == 3:
                self.mlp = torch.nn.Sequential(torch.nn.Linear(75, 50), torch.nn.ReLU(), torch.nn.Linear(50, 25), torch.nn.ReLU(), torch.nn.Linear(25, 1))
            else:
                self.mlp = torch.nn.Sequential(torch.nn.Linear(75, 1))

    def reset_parameters(self):
        self.node_emb.reset_parameters()
        self.edge_emb.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        inits.reset(self.mlp)
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        return self.mlp(x)


