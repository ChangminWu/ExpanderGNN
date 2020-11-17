import torch
import torch.nn as nn

SUPPORTED_ACTIVATION_MAP = {"ReLU", "Sigmoid", "Tanh", "ELU", "SELU",
                            "GLU", "LeakyReLU", "Softplus", "None"}


def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP
                  if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str),\
        "Unhandled activation function"
    activation = activation[0]
    if activation.lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation]()


class GRULayer(nn.Module):
    """
        Wrapper class for the GRU used by the GNN framework,
        nn.GRU is used for the Gated Recurrent Unit itself
    """

    def __init__(self, indim, hiddim, device):
        super(GRULayer, self).__init__()
        self.indim = indim
        self.hiddim = hiddim
        self.gru = nn.GRU(input_size=indim, hidden_size=hiddim).to(device)

    def forward(self, x, y):
        """
        :param x:   shape: (B, N, Din) where Din <= input_size
                    (difference is padded)
        :param y:   shape: (B, N, Dh) where Dh <= hidden_size
                    (difference is padded)
        :return:    shape: (B, N, Dh)
        """
        assert (x.shape[-1] <= self.indim and y.shape[-1] <= self.hiddim)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        x = self.gru(x, y)[1]
        x = x.squeeze()
        return x