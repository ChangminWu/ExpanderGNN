import torch
import torch.nn as nn


class ActivationGATSingleHeadLayer(nn.Module):
    def __init__(self, indim, outdim, hiddim, activation, dropout,
                 batch_norm):
        super().__init__()
        self.activation = activation  # nn.ELU()
        self.attn_activation = activation  # nn.LeakyReLU()

        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm

        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.softmax = nn.Softmax(dim=1)

    def edge_attention(self, edges):
        # https://github.com/pytorch/pytorch/issues/18027
        b, s = edges.src["z"].size(0), edges.src["z"].size(1)
        a = torch.bmm(edges.src["z"].view(b, 1, s),
                      edges.dst['z'].view(b, s, 1)).reshape(-1)
        return {'e': self.attn_activation(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = self.softmax(nodes.mailbox['e'])
        alpha = self.dropout(alpha)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = h
        g = g.local_var()
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']

        if self.batch_norm:
            h = self.batchnorm_h(h)
        if self.activation is not None:
            h = self.activation(h)
        h = self.dropout(h)
        return h


class ActivationGATLayer(nn.Module):
    def __init__(self, merge_type, num_heads, indim, outdim,
                 hiddim, activation, dropout, batch_norm):
        super().__init__()

        self.num_heads = num_heads

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(
                ActivationGATSingleHeadLayer(indim, outdim,
                                             hiddim, activation, dropout,
                                             batch_norm))
        self.merge_type = merge_type

    def forward(self, g, h, e, norm):
        h = h*norm
        head_outs = []
        for attn_head in self.heads:
            h_temp = attn_head(g, h)
            h_temp = h_temp*norm
            head_outs.append(h_temp)

        if self.merge_type == "cat":
            h = torch.cat(head_outs, dim=1)
        elif self.merge_type == "mean":
            h = torch.mean(torch.stack(head_outs))
        else:
            raise KeyError("merge type {} not recognized."
                           .format(self.merge_type))
        return h, e


class ActivationGATSingleHeadLayerEdgeReprFeat(nn.Module):
    def __init__(self, indim, outdim, hiddim, activation, dropout,
                 batch_norm):
        super().__init__()
        self.activation = activation  # nn.ELU()
        self.attn_activation = activation  # nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm

        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.batchnorm_e = nn.BatchNorm1d(outdim)
        self.softmax = nn.Softmax(dim=1)

    def edge_attention(self, edges):
        b, s = edges.src["z"].size(0), edges.src["z"].size(1)
        a = torch.bmm(edges.src["z"].view(b, 1, s),
                      edges.dst['z'].view(b, s, 1)).reshape(-1)
        a = a*torch.norm(edges.data['z_e'], dim=1)
        e_proj = torch.stack([edges.data['z_e'], edges.src['z_h'],
                              edges.dst['z_h']], dim=0).sum(0)

        return {'attn': self.attn_activation(a), 'e_proj': e_proj}

    def message_func(self, edges):
        return {'z': edges.src['z_h'], 'attn': edges.data['attn']}

    def reduce_func(self, nodes):
        alpha = self.softmax(nodes.mailbox['attn'])
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, e, norm):
        z_h = h
        z_e = e
        g = g.local_var()
        g.ndata['z_h'] = z_h
        g.edata['z_e'] = z_e

        g.apply_edges(self.edge_attention)

        g.update_all(self.message_func, self.reduce_func)

        h = g.ndata['h']
        e = g.edata['e_proj']

        if self.batch_norm:
            h = self.batchnorm_h(h)
            e = self.batchnorm_e(e)

        if self.activation is not None:
            h = self.activation(h)
            e = self.activation(e)

        h = self.dropout(h)
        e = self.dropout(e)
        return h, e


class ActivationGATLayerEdgeReprFeat(nn.Module):
    def __init__(self, merge_type, num_heads, indim, outdim,
                 hiddim, activation, dropout, batch_norm):
        super().__init__()
        self.num_heads, self.merge_type = num_heads, merge_type

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(
                ActivationGATSingleHeadLayerEdgeReprFeat(indim,
                                                         outdim, hiddim,
                                                         activation, dropout,
                                                         batch_norm))

    def forward(self, g, h, e, norm):
        h = h*norm
        head_outs_h = []
        head_outs_e = []
        for attn_head in self.heads:
            h_temp, e_temp = attn_head(g, h, e)
            head_outs_h.append(h_temp*norm)
            head_outs_e.append(e_temp)

        if self.merge_type == "cat":
            h = torch.cat(head_outs_h, dim=1)
            e = torch.cat(head_outs_e, dim=1)
        elif self.merge_type == "mean":
            h = torch.mean(torch.stack(head_outs_h))
            e = torch.mean(torch.stack(head_outs_e))
        else:
            raise KeyError("merge type {} not recognized."
                           .format(self.merge_type))
        return h, e


class ActivationGATSingleHeadLayerIsotropic(nn.Module):
    def __init__(self, indim, outdim, hiddim, activation, dropout,
                 batch_norm):
        super().__init__()
        self.activation = activation  # nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm

        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.softmax = nn.Softmax(dim=1)

    def message_func(self, edges):
        return {'z': edges.src['z']}

    def reduce_func(self, nodes):
        h = torch.sum(nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = h
        g = g.local_var()
        g.ndata['z'] = z
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']

        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.activation is not None:
            h = self.activation(h)

        h = self.dropout(h)
        return h


class ActivationGATLayerIsotropic(nn.Module):
    def __init__(self, merge_type, num_heads, indim, outdim,
                 hiddim, activation, dropout, batch_norm):
        super().__init__()
        self.num_heads, self.merge_type = num_heads, merge_type

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(
                ActivationGATSingleHeadLayerIsotropic(indim,
                                                      outdim, hiddim,
                                                      activation, dropout,
                                                      batch_norm))

    def forward(self, g, h, e, norm):
        h = h*norm
        head_outs = []
        for attn_head in self.heads:
            h_temp = attn_head(g, h)
            h_temp = h_temp*norm
            head_outs.append(h_temp)

        if self.merge_type == "cat":
            h = torch.cat(head_outs, dim=1)
        elif self.merge_type == "mean":
            h = torch.mean(torch.stack(head_outs))
        else:
            raise KeyError("merge type {} not recognized."
                           .format(self.merge_type))

        return h, e
