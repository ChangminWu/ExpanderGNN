import torch
import torch.nn as nn

from expander.expander_layer import MultiLinearLayer


class SimpleGatedGCNLayer(nn.Module):
    def __init__(self, n_mlp_layer, indim, outdim, hiddim,
                 dropout, batch_norm,
                 bias=True, residual=False, linear_type="expander", **kwargs):
        super(SimpleGatedGCNLayer, self).__init__()

        self.batch_norm, self.residual, self.bias = batch_norm, residual, bias
        if indim != outdim:
            self.residual = False

        self.bn_node_h = nn.BatchNorm1d(outdim)
        self.bn_node_e = nn.BatchNorm1d(outdim)
        self.dropout = nn.Dropout(dropout)

        self.A = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)
        self.B = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)
        self.C = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)
        self.D = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)
        self.E = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)

    def message_func(self, edges):
        Bh_j = edges.src["Bh"]
        e_ij = edges.data["Ce"] + edges.src["Dh"] + edges.dst["Eh"]
        edges.data["e"] = e_ij
        return {"Bh_j": Bh_j, "e_ij": e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data["Ah"]
        Bh_j = nodes.mailbox["Bh_j"]
        e = nodes.mailbox["e_ij"]
        sigma_ij = torch.sigmoid(e)
        h = Ah_i + torch.sum(sigma_ij*Bh_j, dim=1) / (torch.sum(sigma_ij,
                                                                dim=1) + 1e-6)
        return {"h": h}

    def forward(self, g, h, e, norm):
        h_in = h
        e_in = e

        h = h*norm
        g.ndata["h"] = h
        g.ndata["Ah"] = self.A(h)
        g.ndata["Bh"] = self.B(h)
        g.ndata["Dh"] = self.D(h)
        g.ndata["Eh"] = self.E(h)
        g.edata["e"] = e
        g.edata["Ce"] = self.C(e)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata["h"]
        e = g.edata["e"]
        h = h*norm

        if self.batch_norm:
            h = self.bn_node_h(h)
            e = self.bn_node_e(e)

        if self.residual:
            h = h_in+h
            e = e_in+e

        h = self.dropout(h)
        e = self.dropout(e)
        return h, e


class SimpleGatedGCNEdgesLayer(nn.Module):
    def __init__(self, n_mlp_layer, indim, outdim, hiddim,
                 dropout, batch_norm,
                 bias=True, residual=False, linear_type="expander", **kwargs):
        super(SimpleGatedGCNEdgesLayer, self).__init__()

        self.batch_norm, self.residual, self.bias = batch_norm, residual, bias
        if indim != outdim:
            self.residual = False

        self.bn_node_h = nn.BatchNorm1d(outdim)
        self.dropout = nn.Dropout(dropout)

        self.A = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)
        self.B = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)
        self.D = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)
        self.E = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)

    def message_func(self, edges):
        Bh_j = edges.src["Bh"]
        e_ij = edges.src["Dh"] + edges.dst["Eh"]
        edges.data["e"] = e_ij
        return {"Bh_j": Bh_j, "e_ij": e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data["Ah"]
        Bh_j = nodes.mailbox["Bh_j"]
        e = nodes.mailbox["e_ij"]
        sigma_ij = torch.sigmoid(e)
        h = Ah_i + torch.sum(sigma_ij*Bh_j, dim=1) / (torch.sum(sigma_ij,
                                                                dim=1) + 1e-6)
        return {"h": h}

    def forward(self, g, h, e, norm):
        h_in = h

        h = h*norm
        g.ndata["h"] = h
        g.ndata["Ah"] = self.A(h)
        g.ndata["Bh"] = self.B(h)
        g.ndata["Dh"] = self.D(h)
        g.ndata["Eh"] = self.E(h)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata["h"]
        h = h*norm

        if self.batch_norm:
            h = self.bn_node_h(h)

        if self.residual:
            h = h_in+h

        h = self.dropout(h)
        return h, e


class SimpleGatedGCNIsotrophicLayer(nn.Module):
    def __init__(self, n_mlp_layer, indim, outdim, hiddim,
                 dropout, batch_norm,
                 bias=True, residual=False, linear_type="expander", **kwargs):
        super(SimpleGatedGCNIsotrophicLayer, self).__init__()
        self.batch_norm, self.residual, self.bias = batch_norm, residual, bias
        if indim != outdim:
            self.residual = False

        self.bn_node_h = nn.BatchNorm1d(outdim)
        self.dropout = nn.Dropout(dropout)

        self.A = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)
        self.B = MultiLinearLayer(indim, outdim,
                                  activation=None,
                                  batch_norm=self.batch_norm,
                                  num_layers=n_mlp_layer,
                                  hiddim=hiddim,
                                  bias=self.bias,
                                  linear_type=linear_type,
                                  **kwargs)

    def message_func(self, edges):
        Bh_j = edges.src["Bh"]
        return {"Bh_j": Bh_j}

    def reduce_func(self, nodes):
        Ah_i = nodes.data["Ah"]
        Bh_j = nodes.mailbox["Bh_j"]
        h = Ah_i + torch.sum(Bh_j, dim=1)
        return {"h": h}

    def forward(self, g, h, e, norm):
        h_in = h

        h = h*norm
        g.ndata["h"] = h
        g.ndata["Ah"] = self.A(h)
        g.ndata["Bh"] = self.B(h)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata["h"]
        h = h*norm

        if self.batch_norm:
            h = self.bn_node_h(h)

        if self.residual:
            h = h_in+h

        h = self.dropout(h)
        return h, e
