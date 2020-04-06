## code adapted from https://github.com/drimpossible/Deep-Expander-Networks

import torch
import torch.nn as nn
from torch.autograd import Function


class ExpanderLinear(Function):

    @staticmethod
    def forward(ctx, input_, weight, mask):
        ctx.save_for_backward(input_, weight, mask)
        extend_weights = weight.clone()
        extend_weights.mul_(mask.data)
        output = input_.mm(extend_weights.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, mask = ctx.saved_tensors
        grad_input_ = grad_weight = None
        extend_weights = weight.clone()
        extend_weights.mul_(mask.data)

        if ctx.needs_input_grad[0]:
            grad_input_ = grad_output.mm(extend_weights)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.clone().t().mm(input_)
            grad_weight.mul_(mask.data)
        return grad_input_, grad_weight, None

    
class ExpanderLinearLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(ExpanderLinearLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(data=torch.Tensor(output_features, input_features), requires_grad=True)
        nn.init.kaiming_normal_(self.weight.data, mode='fan_in')

        self.mask = torch.zeros(output_features, input_features)
        self.mask = self.mask.cuda()

    def forward(self, input_):
        return ExpanderLinear.apply(input_, self.weight, self.mask)


class ExpanderDoubleLinearLayer(nn.Module):
    def __init__(self, input_features, output_features, hidden_features=16, activation=False):
        super(ExpanderDoubleLinearLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.hidden_features = hidden_features
        self.activation = activation
        self.layer1 = ExpanderLinearLayer(self.input_features, self.hidden_features)
        self.layer2 = ExpanderLinearLayer(self.hidden_features, self.output_features)

    def forward(self, input_):
        input_ = self.layer1(input_)
        if self.activation:
            input_ = F.relu(input_)
        input_ = self.layer2(input_)
        return input_