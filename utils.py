import torch.nn as nn


def activations(activ_name):
    if activ_name == "relu":
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
    elif activ_name is None:
        activation = None
    else:
        raise ValueError("Invalid activation type.")
    return activation
