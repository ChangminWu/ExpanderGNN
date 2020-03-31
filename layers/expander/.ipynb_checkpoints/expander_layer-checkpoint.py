## code adapted from https://github.com/drimpossible/Deep-Expander-Networks

import torch
from torch.autograd import Variable, Function
import torch.nn as nn


class expanderLinear(Function):
#     def __init__(self, mask):
#         super(expanderLinear, self).__init__()
#         self.mask = mask

    @staticmethod
    def forward(self, input, weight, mask):
        self.mask = mask
#         self.mask.requires_grad = False
        self.save_for_backward(input, weight)
        extendWeights = weight.clone()
        extendWeights.mul_(self.mask.data)
        output = input.mm(extendWeights.t())
        return output

    @staticmethod
    def backward(self, grad_output):
        input, weight = self.saved_tensors
        grad_input = grad_weight  = None
        extendWeights = weight.clone()
        extendWeights.mul_(self.mask.data)

        if self.needs_input_grad[0]:
            grad_input = grad_output.mm(extendWeights)
        if self.needs_input_grad[1]:
            grad_weight = grad_output.clone().t().mm(input)
            grad_weight.mul_(self.mask.data)
        return grad_input, grad_weight, None

    
class ExpanderLinear(torch.nn.Module):
    def __init__(self, input_features, output_features, expandSize, mode='random'):
        super(ExpanderLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(data=torch.Tensor(output_features, input_features), requires_grad=True)

        self.mask = torch.zeros(output_features, input_features)
        if output_features < input_features:
            for i in range(output_features):
                x = torch.randperm(input_features)
                for j in range(expandSize):
                    self.mask[i][x[j]] = 1
        else:
            for i in range(input_features):
                x = torch.randperm(output_features)
                for j in range(expandSize):
                    self.mask[x[j]][i] = 1

        self.mask =  self.mask.cuda()
        nn.init.kaiming_normal_(self.weight.data,mode='fan_in')
        self.mask = nn.Parameter(self.mask.cuda())
        self.mask.requires_grad = False

    def forward(self, input):
        return expanderLinear.apply(input, self.weight, self.mask)


class MulExpander(Function):
    def __init__(self,mask):
        super(MulExpander, self).__init__()
        self.mask = mask

    def forward(self, weight):
        extendWeights = weight.clone()
        extendWeights.mul_(self.mask.data)
        return extendWeights

    def backward(self, grad_output):
        grad_weight = grad_output.clone()
        grad_weight.mul_(self.mask.data)
        return grad_weight