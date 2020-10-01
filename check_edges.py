from data.data import LoadData
import numpy as np
import argparse

def count_edges(dataset_name):
    
    dataset = LoadData(dataset_name)
    
    if dataset in ["ENZYMES", "DD", "PROTEINS_full", "IMDB-BINARY"]:
        num_nodes = np.mean([dataset.all[i][0].number_of_nodes() for i in range(len(dataset.all))])
        num_edges = np.mean([dataset.all[i][0].number_of_edges()/2 for i in range(len(dataset.all))])
        num_graphs = len(num_nodes)
    elif dataset in ["CORA", "CITESEER", "PUBMED"]:
        num_nodes = dataset.graph.number_of_nodes()
        num_edges = dataset.graph.number_of_edges()
        num_graphs = 1
    else:
        num_nodes = []
        num_edges = []
        num_graphs = 0
        for data in [dataset.train, dataset,val, dataset.test]:
            num_nodes += [data[i][0].number_of_nodes() for i in range(len(data))]
            num_edges += [data[i][0].number_of_edges() for i in range(len(data))]
            num_graphs += len(data)
        num_nodes = np.mean(num_nodes)
        num_edges = np.mean(num_edges)

    print("#graphs: {}\n#nodes: {}\n#edges: {}".format(num_graphs, num_nodes, num_edges))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        help="give path of folder that stores result files")
    args = parser.parse_args()
    count_edges(args.dataset)