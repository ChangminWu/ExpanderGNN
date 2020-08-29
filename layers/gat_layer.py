import torch
import torch.nn as nn

from expander.expander_layer import LinearLayer, MultiLinearLayer


class GATSingleHeadLayer(nn.Module):
    def __init__(self, n_mlp_layer, indim, outdim, hiddim, activation, dropout,
                 batch_norm,
                 bias=True, linear_type="expander", **kwargs):
        super().__init__()
        self.activation = activation  # nn.ELU()
        self.attn_activation = activation  # nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm

        self.linear = MultiLinearLayer(indim, outdim,
                                       activation=self.activation,
                                       batch_norm=self.batch_norm,
                                       num_layers=n_mlp_layer,
                                       hiddim=hiddim,
                                       bias=False,
                                       linear_type=linear_type,
                                       **kwargs)

        self.attn_linear = LinearLayer(2*outdim, 1, bias=False,
                                       linear_type=linear_type,
                                       **kwargs)

        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.softmax = nn.Softmax(dim=1)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_linear(z2)
        return {'e': self.attn_activation(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = self.softmax(nodes.mailbox['e'])
        alpha = self.dropout(alpha)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = self.linear(h)
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


class GATLayer(nn.Module):
    def __init__(self, merge_type, num_heads, n_mlp_layer, indim, outdim,
                 hiddim, activation, dropout, batch_norm,
                 residual=True, bias=True, linear_type="expander", **kwargs):
        super().__init__()

        self.num_heads = num_heads
        self.residual = residual
        if indim != (outdim*num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATSingleHeadLayer(n_mlp_layer, indim, outdim,
                                                 hiddim, activation, dropout,
                                                 batch_norm, bias, linear_type,
                                                 **kwargs))
        self.merge_type = merge_type

    def forward(self, g, h, e):
        h_in = h

        head_outs = [attn_head(g, h) for attn_head in self.heads]

        if self.merge_type == "cat":
            h = torch.cat(head_outs, dim=1)
        elif self.merge_type == "mean":
            h = torch.mean(torch.stack(head_outs))
        else:
            raise KeyError("merge type {} not recognized."
                           .format(self.merge_type))

        if self.residual:
            h = h_in + h
        return h, e


class GATSingleHeadLayerEdgeReprFeat(nn.Module):
    def __init__(self, n_mlp_layer, indim, outdim, hiddim, activation, dropout,
                 batch_norm,
                 bias=True, linear_type="expander", **kwargs):
        super().__init__()
        self.activation = activation  # nn.ELU()
        self.attn_activation = activation  # nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm

        self.linear_h = MultiLinearLayer(indim, outdim,
                                         activation=self.activation,
                                         batch_norm=self.batch_norm,
                                         num_layers=n_mlp_layer,
                                         hiddim=hiddim,
                                         bias=False,
                                         linear_type=linear_type,
                                         **kwargs)
        self.linear_e = MultiLinearLayer(indim, outdim,
                                         activation=self.activation,
                                         batch_norm=self.batch_norm,
                                         num_layers=n_mlp_layer,
                                         hiddim=hiddim,
                                         bias=False,
                                         linear_type=linear_type,
                                         **kwargs)
        self.proj = MultiLinearLayer(3*outdim, outdim,
                                     activation=self.activation,
                                     batch_norm=self.batch_norm,
                                     num_layers=n_mlp_layer,
                                     hiddim=hiddim,
                                     bias=bias,
                                     linear_type=linear_type,
                                     **kwargs)
        self.attn_linear = LinearLayer(3*outdim, 1, bias=False,
                                       linear_type=linear_type,
                                       **kwargs)

        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.batchnorm_e = nn.BatchNorm1d(outdim)
        self.softmax = nn.Softmax(dim=1)

    def edge_attention(self, edges):
        z = torch.cat([edges.data['z_e'], edges.src['z_h'],
                       edges.dst['z_h']], dim=1)
        e_proj = self.proj(z)
        attn = self.attn_linear(z)
        return {'attn': self.attn_activation(attn), 'e_proj': e_proj}

    def message_func(self, edges):
        return {'z': edges.src['z_h'], 'attn': edges.data['attn']}

    def reduce_func(self, nodes):
        alpha = self.softmax(nodes.mailbox['attn'])
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, e):
        z_h = self.linear_h(h)
        z_e = self.linear_e(e)
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


class GATLayerEdgeReprFeat(nn.Module):
    def __init__(self, merge_type, num_heads, n_mlp_layer, indim, outdim,
                 hiddim, activation, dropout, batch_norm,
                 residual=True, bias=True, linear_type="expander", **kwargs):
        super().__init__()
        self.num_heads, self.merge_type = num_heads, merge_type
        self.residual = residual
        if indim != (outdim*num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(
                GATSingleHeadLayerEdgeReprFeat(n_mlp_layer, indim, outdim,
                                               hiddim, activation, dropout,
                                               batch_norm, bias, linear_type,
                                               **kwargs))

    def forward(self, g, h, e):
        h_in = h
        e_in = e

        head_outs_h = []
        head_outs_e = []
        for attn_head in self.heads:
            h_temp, e_temp = attn_head(g, h, e)
            head_outs_h.append(h_temp)
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

        if self.residual:
            h = h_in + h
            e = e_in + e

        return h, e


class GATSingleHeadLayerIsotropic(nn.Module):
    def __init__(self, n_mlp_layer, indim, outdim, hiddim, activation, dropout,
                 batch_norm,
                 bias=True, linear_type="expander", **kwargs):
        super().__init__()
        self.activation = activation  # nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm

        self.linear = MultiLinearLayer(indim, outdim,
                                       activation=self.activation,
                                       batch_norm=self.batch_norm,
                                       num_layers=n_mlp_layer,
                                       hiddim=hiddim,
                                       bias=False,
                                       linear_type=linear_type,
                                       **kwargs)

        self.batchnorm_h = nn.BatchNorm1d(outdim)
        self.softmax = nn.Softmax(dim=1)

    def message_func(self, edges):
        return {'z': edges.src['z']}

    def reduce_func(self, nodes):
        h = torch.sum(nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = self.linear(h)
        g.ndata['z'] = z
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']

        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.activation is not None:
            h = self.activation(h)

        h = self.dropout(h)
        return h


class GATLayerIsotropic(nn.Module):
    def __init__(self, merge_type, num_heads, n_mlp_layer, indim, outdim,
                 hiddim, activation, dropout, batch_norm,
                 residual=True, bias=True, linear_type="expander", **kwargs):
        super().__init__()
        self.num_heads, self.merge_type = num_heads, merge_type
        self.residual = residual
        if indim != (outdim*num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(
                GATSingleHeadLayerIsotropic(n_mlp_layer, indim, outdim,
                                            hiddim, activation, dropout,
                                            batch_norm, bias, linear_type,
                                            **kwargs))

    def forward(self, g, h, e):
        h_in = h

        head_outs = [attn_head(g, h) for attn_head in self.heads]

        if self.merge_type == "cat":
            h = torch.cat(head_outs, dim=1)
        elif self.merge_type == "mean":
            h = torch.mean(torch.stack(head_outs))
        else:
            raise KeyError("merge type {} not recognized."
                           .format(self.merge_type))

        if self.residual:
            h = h_in + h
        return h, e
