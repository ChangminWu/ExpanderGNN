import torch
import torch.nn as nn


class ActivationGatedGCNLayer(nn.Module):
    def __init__(self, indim, outdim, hiddim,
                 activation, dropout, batch_norm):
        super(ActivationGatedGCNLayer, self).__init__()

        self.activation = activation
        self.batch_norm = batch_norm

        self.bn_node_h = nn.BatchNorm1d(outdim)
        self.bn_node_e = nn.BatchNorm1d(outdim)
        self.dropout = nn.Dropout(dropout)

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
        h = h*norm
        g.ndata["h"] = h
        g.ndata["Ah"] = h
        g.ndata["Bh"] = h
        g.ndata["Dh"] = h
        g.ndata["Eh"] = h
        g.edata["e"] = e
        g.edata["Ce"] = e
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata["h"]
        e = g.edata["e"]
        h = h*norm

        if self.batch_norm:
            h = self.bn_node_h(h)
            e = self.bn_node_e(e)

        if self.activation is not None:
            h = self.activation(h)
            e = self.activation(e)

        h = self.dropout(h)
        e = self.dropout(e)
        return h, e


class ActivationGatedGCNEdgesLayer(nn.Module):
    def __init__(self, indim, outdim, hiddim,
                 activation, dropout, batch_norm):
        super(ActivationGatedGCNEdgesLayer, self).__init__()

        self.activation = activation
        self.batch_norm = batch_norm

        self.bn_node_h = nn.BatchNorm1d(outdim)
        self.dropout = nn.Dropout(dropout)

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
        h = h*norm
        g.ndata["h"] = h
        g.ndata["Ah"] = h
        g.ndata["Bh"] = h
        g.ndata["Dh"] = h
        g.ndata["Eh"] = h
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata["h"]
        h = h*norm

        if self.batch_norm:
            h = self.bn_node_h(h)

        if self.activation is not None:
            h = self.activation(h)

        h = self.dropout(h)
        return h, e


class ActivationGatedGCNIsotrophicLayer(nn.Module):
    def __init__(self, indim, outdim, hiddim,
                 activation, dropout, batch_norm):
        super(ActivationGatedGCNIsotrophicLayer, self).__init__()

        self.activation = activation
        self.batch_norm = batch_norm

        self.bn_node_h = nn.BatchNorm1d(outdim)
        self.dropout = nn.Dropout(dropout)

    def message_func(self, edges):
        Bh_j = edges.src["Bh"]
        return {"Bh_j": Bh_j}

    def reduce_func(self, nodes):
        Ah_i = nodes.data["Ah"]
        Bh_j = nodes.mailbox["Bh_j"]
        h = Ah_i + torch.sum(Bh_j, dim=1)
        return {"h": h}

    def forward(self, g, h, e, norm):
        h = h*norm
        g.ndata["h"] = h
        g.ndata["Ah"] = h
        g.ndata["Bh"] = h
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata["h"]
        h = h*norm

        if self.batch_norm:
            h = self.bn_node_h(h)

        if self.activation is not None:
            h = self.activation(h)

        h = self.dropout(h)
        return h, e
