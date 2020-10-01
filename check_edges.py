from data.data import LoadData
import numpy as np
import argparse

def count_edges(dataset_name):
    dataset = LoadData(dataset_name)
    num_nodes = [dataset.all[i][0].number_of_nodes() for i in range(len(dataset.all))]
    num_edges = [dataset.all[i][0].number_of_edges()/2 for i in range(len(dataset.all))]
    print("#graphs: {}\n#nodes: {}\n#edges: {}".format(len(num_nodes), np.mean(num_nodes), np.mean(num_edges)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        help="give path of folder that stores result files")
    args = parser.parse_args()
    count_edges(args.dataset)