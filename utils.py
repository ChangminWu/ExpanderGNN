import pathlib
from collections import OrderedDict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from layers.activation_layer import LinearActiveLayer, BiasedRELULayer,\
                                    ConvActivLayer


def activations(activ_name, param=None):
    if activ_name == "sigmoid":
        activation = nn.Sigmoid()
    elif activ_name == "tanh":
        activation = nn.Tanh()
    elif activ_name == "relu":
        activation = nn.ReLU()
    elif activ_name == "prelu":
        activation = nn.PReLU()
    elif activ_name == "rrelu":
        activation = nn.RReLU()
    elif activ_name == "elu":
        activation = nn.ELU()
    elif activ_name == "lelu":
        activation = nn.LeakyReLU()
    elif activ_name == "celu":
        activation = nn.CELU()
    elif activ_name == "selu":
        activation = nn.SELU()
    elif activ_name == "gelu":
        activation = nn.GELU()
    elif activ_name == "softplus":
        activation = nn.Softplus()
    elif activ_name == "softsign":
        activation = nn.Softsign()
    elif activ_name == "softshrink":
        activation = nn.Softshrink()
    elif activ_name == "linear":
        activation = LinearActiveLayer(param)
    elif activ_name == "brelu":
        activation = BiasedRELULayer(intercept=False)
    elif activ_name == "brelu-intercept":
        activation = BiasedRELULayer(intercept=True)
    elif activ_name == "conv":
        activation = ConvActivLayer()
    elif activ_name == "None":
        activation = None
    else:
        raise ValueError("Invalid activation type.")
    return activation


def expander_writer(saved_expander, curr_path="./"):
    for key in saved_expander:
        if type(saved_expander[key]) is torch.Tensor:
            bipart_matrix = saved_expander[key].cpu().detach().numpy()
            num_output_nodes, num_input_nodes = bipart_matrix.shape
            adj = np.zeros((num_input_nodes+num_output_nodes,
                            num_input_nodes+num_output_nodes))
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
            # graph.add_nodes_from(
            #       np.arange(num_input_nodes,
            #                 num_input_nodes+num_output_nodes, 1),
            #       bipartite=1)
            # edges = map(lambda e: (int(e[0]), int(e[1])),
            #             zip(*(np.asarray(adj).nonzero())))
            # graph.add_edges_from(edges)
            # with open(curr_path+key+".p") as f:
            #     pickle.dump(graph, f)
            nx.write_gpickle(graph, curr_path+key+".gpickle")
        else:
            pathlib.Path(curr_path+key).mkdir(parents=True, exist_ok=True)
            expander_writer(saved_expander[key], curr_path=curr_path+key+"/")


def expander_weights_writer(net, saved_expander, saved_layers=None,
                            curr_path="./"):
    num_children = len(list(net.children()))
    if saved_layers is None:
        saved_layers = dict()

    layer_name = str(net.__class__).split(".")[-1].split("'")[0]
    # str(net)[:str(net).find('(')]
    if layer_name in saved_layers:
        label = layer_name + "_" + str(len(saved_layers[layer_name]))
        saved_layers[layer_name].append(label)
    else:
        label = layer_name + "_" + str(0)
        saved_layers[layer_name] = [label]

    if num_children == 0:
        if "Linear" in label:
            weight = net.weight.cpu().detach().numpy()

            mask = weight.copy()
            mask[mask != 0] = 1
            assert (mask == saved_expander[label].cpu().detach().numpy()).all()
            # with open(curr_path + label + ".pickle") as f:
            #     pickle.dump(weight, f)
            np.save(curr_path + label + ".npy", weight)
    else:
        pathlib.Path(curr_path + label).mkdir(parents=True, exist_ok=True)
        for child in net.children():
            saved_layers = expander_weights_writer(child,
                                                   saved_expander[label],
                                                   saved_layers,
                                                   curr_path=curr_path +
                                                   label + "/")
    return saved_layers


def check_tensorboard(net, writer, step, step_size=30, saved_layers=None):
    num_children = len(list(net.children()))
    if saved_layers is None:
        saved_layers = dict()
    layer_name = str(net.__class__).split(".")[-1].split("'")[0]
    # str(net)[:str(net).find('(')]
    if layer_name in saved_layers:
        label = layer_name + "_" + str(len(saved_layers[layer_name]))
        saved_layers[layer_name].append(label)
    else:
        label = layer_name + "_" + str(0)
        saved_layers[layer_name] = [label]

    if num_children == 0:
        if "Expander" in layer_name and "Linear" in layer_name:
            if not (net.mask == 1).all():
                index = tuple((net.mask == 0).nonzero()[0])
                writer.add_scalar(
                    "train/_weight_{}".format(label),
                    net.weight.data[index].cpu().detach().numpy(), step)

            if step % step_size == 0:
                writer.add_image(
                    "train/_gradient_{}".format(label),
                    net.weight.grad.data.unsqueeze(0).cpu().numpy(),
                    int(step/step_size))
                writer.add_image(
                    "train/_mask_{}".format(label),
                    net.mask.data.unsqueeze(0).cpu().numpy(),
                    int(step/step_size))

            return writer, saved_layers

        elif "Linear" in layer_name:
            return writer, saved_layers

    elif num_children > 0:
        for child in net.children():
            log_tuple = check_tensorboard(child, writer, step,
                                          step_size, saved_layers)
            if log_tuple is not None:
                return log_tuple


def init_expander(net, saved_expander=None, saved_layers=None):
    num_children = len(list(net.children()))
    layer_name = str(net.__class__).split(".")[-1].split("'")[0]
    # str(net)[:str(net).find('(')]
    if layer_name in saved_layers:
        label = layer_name + "_" + str(len(saved_layers[layer_name]))
        saved_layers[layer_name].append(label)
    else:
        label = layer_name + "_" + str(0)
        saved_layers[layer_name] = [label]

    if num_children == 0:
        if label in saved_expander and "Expander" in label:
            net.generate_mask(saved_expander[label])
        elif "Expander" in label and "Linear" in label:
            net.generate_mask()
            saved_expander[label] = net.mask
        elif "Linear" in label:
            saved_expander[label] = torch.ones(net.weight.size())
    else:
        if label not in saved_expander:
            saved_expander[label] = OrderedDict()
        for child in net.children():
            saved_expander[label], saved_layers =\
                init_expander(child,
                              saved_expander[label],
                              saved_layers)
    return saved_expander, saved_layers


def get_model_param(net, num):
    curr_total_num = num
    num_children = len(list(net.children()))
    layer_name = str(net.__class__).split(".")[-1].split("'")[0]
    # str(net)[:str(net).find('(')]
    if num_children == 0:
        if "Expander" in layer_name and "Linear" in layer_name:
            curr_total_num += net.n_params
        elif "Linear" in layer_name:
            curr_num = 0
            for param in net.parameters():
                curr_num += np.prod(list(param.data.size()))
            curr_total_num += curr_num
    else:
        for child in net.children():
            curr_total_num = get_model_param(child, curr_total_num)
    return curr_total_num


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
