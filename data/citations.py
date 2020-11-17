import torch
import torch.utils.data
import time
import numpy as np

import dgl
from dgl.data import CoraDataset
from dgl.data import CitationGraphDataset
import networkx as nx

import random
random.seed(42)


def get_all_split_idx(dataset, num_split=10):
    """
        - Split total number of graphs into 3 (train, val and test) in 80:10:10
        - Stratified split proportionate to original distribution of data
          with respect to classes
        - Using sklearn to perform the split and then save the indexes
        - Preparing 10 such combinations of indexes split to be used
          in Graph NNs
        - As with KFold, each of the 10 fold have unique test set.
    """
    root_idx_dir = './data/citation_node_classification/'
    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
    all_idx = {}

    # If there are no idx files, do the split and store the files
    if not (os.path.exists(root_idx_dir + dataset.name + '_train.index')):
        print("[!] Splitting the data into train/val/test ...")

        # Using num_split-fold cross val to compare with benchmark papers
        cross_val_fold = StratifiedKFold(n_splits=num_split, shuffle=True)

        # this is a temporary index assignment, to be used below
        # for val splitting
        for i in range(len(dataset.graph_lists)):
            dataset[i][0].a = lambda: None
            setattr(dataset[i][0].a, 'index', i)

        for indexes in cross_val_fold.split(dataset.graph_lists,
                                            dataset.labels):
            remain_index, test_index = indexes[0], indexes[1]

            remain_set = format_dataset([dataset[index]
                                         for index in remain_index])

            # Gets final 'train' and 'val'
            train, val, _, __ =\
                train_test_split(remain_set,
                                 range(len(remain_set.graph_lists)),
                                 test_size=1/(num_split-1),
                                 stratify=remain_set.graph_labels)

            train, val = format_dataset(train), format_dataset(val)
            test = format_dataset([dataset[index] for index in test_index])

            # Extracting only idxs
            idx_train = [item[0].a.index for item in train]
            idx_val = [item[0].a.index for item in val]
            idx_test = [item[0].a.index for item in test]

            f_train_w = csv.writer(open(root_idx_dir + dataset.name
                                        + '_train.index', 'a+'))
            f_val_w = csv.writer(open(root_idx_dir + dataset.name
                                      + '_val.index', 'a+'))
            f_test_w = csv.writer(open(root_idx_dir + dataset.name
                                       + '_test.index', 'a+'))

            f_train_w.writerow(idx_train)
            f_val_w.writerow(idx_val)
            f_test_w.writerow(idx_test)

        print("[!] Splitting done!")

    # reading idx from the files
    for section in ['train', 'val', 'test']:
        with open(root_idx_dir + dataset.name
                  + '_' + section + '.index', 'r') as f:
            reader = csv.reader(f)
            all_idx[section] = [list(map(int, idx)) for idx in reader]
    return all_idx


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
            dataset = CoraDataset()
        else:
            dataset = CitationGraphDataset(self.name)

        print("[!] Dataset: ", self.name)

        g = dataset.graph
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
        graph = dgl.DGLGraph(g)

        E = graph.number_of_edges()
        N = graph.number_of_nodes()
        D = dataset.features.shape[1]
        graph.ndata['feat'] = torch.FloatTensor(dataset.features)
        graph.edata['feat'] = torch.zeros((E, D))
        graph.batch_num_nodes = [N]

        self.graph = graph
        self.train_mask = torch.BoolTensor(dataset.train_mask)
        self.val_mask = torch.BoolTensor(dataset.val_mask)
        self.test_mask = torch.BoolTensor(dataset.test_mask)
        self.labels = torch.LongTensor(dataset.labels)
        self.num_classes = dataset.num_labels
        self.num_dims = D

        print("[!] Dataset: ", self.name)
        print("Time taken: {:.4f}s".format(time.time()-t0))

    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True
        self.graph = self_loop(self.graph)