import torch
import torch.nn as nn

import dgl 

from layers.gcn_layer import GCNLayer
from expander.expander_layer import LinearLayer

class GCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        indim = net_params["in_dim"]
        hiddim = net_params["hidden_dim"]
        outdim = net_params["out_dim"]

        n_classes = net_params["n_classes"]
        in_feat_dropout = net_params["in_feat_dropout"]
        dropout = net_params["dropout"]
        n_iters = net_params["L"]

        self.readout = net_params["readout"]
        self.batch_norm = net_params["residual"]
        




        

