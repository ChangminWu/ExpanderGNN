import torch
import torch.utils.data
import time
import numpy as np

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, CoraFullDataset, MUTAGDataset, \
    CoauthorCSDataset, CoauthorPhysicsDataset, RedditDataset, AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CitationGraphDataset
import networkx as nx

import random


def self_loop(g):
    """
        Utility function only, to be used only when necessary
        as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop()
        to not miss ndata['feat'] and edata['feat']
        This function is called inside a function
        in CitationGraphsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since
    # this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


class CitationsDataset(torch.utils.data.Dataset):
    def __init__(self, name):
        t0 = time.time()
        self.name = name.lower()

        if self.name == 'cora':
            dataset = CoraGraphDataset()
        elif self.name == 'citeseer':
            dataset = CiteseerGraphDataset()
        elif self.name == 'pubmed':
            dataset = PubmedGraphDataset()
        elif self.name == 'cora-full':
            dataset = CoraFullDataset()
        elif self.name == 'mutag':
            dataset = MUTAGDataset()
        elif self.name == 'coauthor-cs':
            dataset = CoauthorCSDataset()
        elif self.name == 'coauthor-physics':
            dataset = CoauthorPhysicsDataset()
        elif self.name == 'reddit':
            dataset = RedditDataset()
        elif self.name == 'amazon-photo':
            dataset = AmazonCoBuyPhotoDataset()
        elif self.name == 'amazon-computer':
            dataset = AmazonCoBuyComputerDataset()

        print("[!] Dataset: ", self.name)

        graph = dataset[0]
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        E = graph.number_of_edges()
        N = graph.number_of_nodes()
        D = graph.ndata['feat'].shape[1]
        # graph.ndata['feat'] = torch.FloatTensor(graph.ndata['feat'])
        # graph.edata['feat'] = torch.empty((E, D))
        graph.batch_num_nodes = [N]

        self.graph = graph
        self.train_mask = graph.ndata['train_mask']  #torch.BoolTensor(graph.ndata['train_mask'])
        self.val_mask = graph.ndata['val_mask'] #torch.BoolTensor(graph.ndata['val_mask'])
        self.test_mask = graph.ndata['test_mask'] #torch.BoolTensor(graph.ndata['test_mask'])
        self.labels = graph.ndata['label'] #torch.LongTensor(graph.ndata['label'])
        self.num_classes = dataset.num_classes
        self.num_dims = D

        print("[!] Dataset: ", self.name)
        print("Time taken: {:.4f}s".format(time.time()-t0))

    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True
        self.graph = self_loop(self.graph)