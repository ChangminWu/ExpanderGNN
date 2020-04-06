import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import pickle
import pathlib


def expand_writer(saved_expander, curr_path="./"):
    for key in saved_expander:
        if type(saved_expander[key]) is torch.Tensor:
            bipart_matrix = saved_expander[key].detach().numpy()
            num_output_nodes, num_input_nodes = bipart_matrix.shape
            adj = np.zeros((num_input_nodes+num_output_nodes, num_input_nodes+num_output_nodes))
            # adj[num_input_nodes:, :num_input_nodes] = bipart_matrix
            adj[:num_input_nodes, num_input_nodes:] = bipart_matrix.T
            graph = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
            labels = dict()
            for i in range(num_input_nodes+num_output_nodes):
                if i < num_input_nodes:
                    labels[i] = 0
                else:
                    labels[i] = 1
            nx.set_node_attributes(graph, labels, "bipartite")
            # graph = nx.DiGraph()
            # graph.add_nodes_from(np.arange(num_input_nodes), bipartite=0)
            # graph.add_nodes_from(np.arange(num_input_nodes, num_input_nodes+num_output_nodes, 1), bipartite=1)
            # edges = map(lambda e: (int(e[0]), int(e[1])), zip(*(np.asarray(adj).nonzero())))
            # graph.add_edges_from(edges)
            # with open(curr_path+key+".p") as f:
            #     pickle.dump(graph, f)
            nx.write_gpickle(graph, curr_path+key+".gpickle")
            np.save(curr_path+key+".npy", adj)
        else:
            # if not os.path.exists(curr_path+key):
            #     os.mkdirs(curr_path+key)
            pathlib.Path(curr_path+key).mkdir(parents=True, exist_ok=True)
            expand_writer(saved_expander[key], curr_path=curr_path+key+"/")

