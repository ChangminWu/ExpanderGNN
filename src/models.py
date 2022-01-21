import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv
from layers.convs import ExpanderGCNConv, ExpanderSAGEConv, ActivationGCNConv
from expander.samplers import sampler


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
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            #x = self.bns[i](x)
            x = F.relu(x)
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


