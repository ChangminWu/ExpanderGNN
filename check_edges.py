from data.data import LoadData
import numpy as np

def count_edges(dataset_name):
    dataset = LoadData(dataset_name)
    num_nodes = [dataset.all[i][0].number_of_nodes() for i in range(len(dataset.all))]
    num_edges = [dataset.all[i][0].number_of_edges() for i in range(len(dataset.all))]
    print("#graphs: {}\n#nodes: {}\n#edges: {}".format(len(num_nodes), np.mean(num_nodes), np.mean(num_edges)))