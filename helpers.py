import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx
import pickle
import pathlib

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np

from collections import OrderedDict


########## IO ###########
def expand_writer(saved_expander, curr_path="./"):
    for key in saved_expander:
        if type(saved_expander[key]) is torch.Tensor:
            bipart_matrix = saved_expander[key].cpu().detach().numpy()
            num_output_nodes, num_input_nodes = bipart_matrix.shape
            adj = np.zeros((num_input_nodes+num_output_nodes, num_input_nodes+num_output_nodes))
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
        else:
            pathlib.Path(curr_path+key).mkdir(parents=True, exist_ok=True)
            expand_writer(saved_expander[key], curr_path=curr_path+key+"/")

def weighted_expand_writer(net, saved_expander, saved_layers=None, curr_path="./"):
    num_children = len(list(net.children()))
    if saved_layers is None:
        saved_layers = dict()

    layer_name = str(net.__class__).split(".")[-1].split("'")[0]  # str(net)[:str(net).find('(')]
    if layer_name in saved_layers:
        label = layer_name + "_" + str(len(saved_layers[layer_name]))
        saved_layers[layer_name].append(label)
    else:
        label = layer_name + "_" + str(0)
        saved_layers[layer_name] = [label]

    if num_children == 0:
        if "ExpanderLinear" in label:
            weight = net.weight.cpu().detach().numpy()

            mask = weight.copy()
            mask[mask!=0] = 1
            assert (mask == saved_expander[label].cpu().detach().numpy()).all()
            # with open(curr_path + label + ".pickle") as f:
            #     pickle.dump(weight, f)
            np.save(curr_path + label + ".npy", weight)

    else:
        pathlib.Path(curr_path + label).mkdir(parents=True, exist_ok=True)
        for child in net.children():
            saved_layers = weighted_expand_writer(child, saved_expander[label],
                                                  saved_layers, curr_path=curr_path + label + "/")
    return saved_layers

def register_weight(net, writer, step, saved_layers=None):
    num_children = len(list(net.children()))
    if saved_layers is None:
        saved_layers = dict()
    layer_name = str(net.__class__).split(".")[-1].split("'")[0]  #str(net)[:str(net).find('(')]
    if layer_name in saved_layers:
        label = layer_name + "_" + str(len(saved_layers[layer_name]))
        saved_layers[layer_name].append(label)
    else:
        label = layer_name + "_" + str(0)
        saved_layers[layer_name] = [label]

    if num_children == 0:
        if "ExpanderLinear" in layer_name:
            try:
                index = tuple((net.mask == 0).nonzero()[0])
                writer.add_scalar("train/_weight_{}".format(label), net.weight.data[index].cpu().detach().numpy(), step)
            except:
                if not (net.mask == 1).all():
                    raise ValueError("Stored mask is not in the right format for layer {}".format(layer_name))
                else:
                    pass

            # index_mask = tuple(net.mask.nonzero()[0])
            if step % 30 == 0:
                writer.add_image('train/_gradient_{}'.format(label), net.weight.grad.data.unsqueeze(0).cpu().numpy(), int(step/30))
                writer.add_image('train/_mask_{}'.format(label), net.mask.data.unsqueeze(0).cpu().numpy(), int(step/30))
            # writer.add_scalar('train/_mask', net.mask.data[index_mask].cpu().detach().numpy(), step)
            return writer, saved_layers

        elif "Linear" in layer_name:
            return writer, saved_layers

    elif num_children > 0:
        for child in net.children():
            log_tuple = register_weight(child, writer, step, saved_layers)
            if log_tuple is not None:
                return log_tuple


######### Initialization ###########
def init_expander(net, saved_mask=None, saved_layers=None):
    num_children = len(list(net.children()))
    layer_name = str(net.__class__).split(".")[-1].split("'")[0]  #str(net)[:str(net).find('(')]
    if layer_name in saved_layers:
        label = layer_name + "_" + str(len(saved_layers[layer_name]))
        saved_layers[layer_name].append(label)
    else:
        label = layer_name + "_" + str(0)
        saved_layers[layer_name] = [label]

    if num_children == 0:
        if label in saved_mask:
            net.generate_mask(saved_mask[label])
            # net.mask = saved_mask[label]
        elif "ExpanderLinear" in label:
            net.generate_mask()
            # output_features, input_features = net.output_features, net.input_features  # list(net.parameters())[0].size()
            # mask = torch.zeros(output_features, input_features)
            # if output_features < input_features:
            #     for i in range(output_features):
            #         x = torch.randperm(input_features)
            #         for j in range(expand_size):
            #             mask[i][x[j]] = 1
            # else:
            #     for i in range(input_features):
            #         x = torch.randperm(output_features)
            #         for j in range(expand_size):
            #             mask[x[j]][i] = 1
            # net.mask = mask.cuda()
            saved_mask[label] = net.mask  # ._indices().cpu().detach().numpy()
    else:
        if label not in saved_mask:
            saved_mask[label] = OrderedDict()
        for child in net.children():
            # label_child = str(child)[:str(child).find('(')]
            # if "Expander" in label_child:
            saved_mask[label], saved_layers = init_expander(child, saved_mask[label], saved_layers)
    return saved_mask, saved_layers

def get_model_param(net, num):
    curr_total_num = num
    num_children = len(list(net.children()))
    layer_name = str(net.__class__).split(".")[-1].split("'")[0]  # str(net)[:str(net).find('(')]
    if num_children == 0:
        if "ExpanderLinear" in layer_name:
            curr_total_num += net.n_params
        elif "Linear" in layer_name and "Expander" not in layer_name:
            curr_num = 0
            for param in net.parameters():
                curr_num += np.prod(list(param.data.size()))
            curr_total_num += curr_num
    else:
        for child in net.children():
            curr_total_num = get_model_param(child, curr_total_num)
    return curr_total_num


########## Metrics ###########
def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    return MAE

def accuracy(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores == targets).float().sum().item()
    return acc

def binary_f1_score(scores, targets):
    """Computes the F1 score using scikit-learn for binary class labels.
    Returns the F1 score for the positive class, i.e. labelled '1'.
    """
    y_true = targets.cpu().numpy()
    y_pred = scores.argmax(dim=1).cpu().numpy()
    return f1_score(y_true, y_pred, average='binary')


########## Others ###########
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac
