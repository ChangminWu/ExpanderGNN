## code adapted from https://github.com/drimpossible/Deep-Expander-Networks

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import torch.sparse as sparse
import numpy as np


class ExpanderLinear(Function):

    @staticmethod
    def forward(ctx, input_, weight, mask):
        ctx.save_for_backward(input_, weight, mask)
        # extend_weights = torch.sparse_coo_tensor(mask, weight[tuple(mask)], weight.size())
        # weight[mask.split(1, dim=1)] = 0
        # weight.data[tuple(mask)] = 0
        # weight.mul_(mask.data)
        # mask method
        # extend_weights = weight.clone()
        # extend_weights.mul_(mask.data)
        weight.data = weight.data * mask 

        # mask.mul_(extend_weights)
        # extend_weights[mask] = 0
        # output = input_.mm(extend_weights.t())
        # output = torch.sparse.mm(extend_weights, input_.t()).t()
        output = input_.mm(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, mask = ctx.saved_tensors
        grad_input_ = grad_weight = None
        # extend_weights = torch.sparse_coo_tensor(mask, weight[tuple(mask)], weight.size())
        # weight.data[tuple(mask)] = 0
        # weight.mul_(mask.data)
        # extend_weights = weight.clone()
        # extend_weights.mul_(mask.data)
        # extend_weights.mul_(mask)
        # extend_weights[mask] = 0
        weight.data = weight.data * mask 
        
        if ctx.needs_input_grad[0]:
            # grad_input_ = grad_output.mm(extend_weights)
            grad_input_ = grad_output.mm(weight)
            # grad_input_ = torch.sparse.mm(extend_weights.t(), grad_output.t()).t()
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.clone().t().mm(input_)
            # grad_weight = torch.zeros_like(gweight)
            # grad_weight[tuple(mask)] = gweight[tuple(mask)]
            
            # grad_weight.mul_(mask.data)
            # grad_weight.mul_(mask)
            # grad_weight.data[tuple(mask)] = 0
            grad_weight.data = grad_weight.data * mask 
        return grad_input_, grad_weight, None

    
class ExpanderLinearLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(ExpanderLinearLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        # self.weight = nn.Parameter(data=torch.Tensor(output_features, input_features), requires_grad=True)
        # nn.init.kaiming_normal_(self.weight.data, mode='fan_in')

        # if output_features < input_features:
        #     for i in range(output_features):
        #         x = torch.randperm(input_features)
        #         for j in range(8):
        #             self.mask[i][x[j]] = 1
        # else:
        #     for i in range(input_features):
        #         x = torch.randperm(output_features)
        #         for j in range(8):
        #             self.mask[x[j]][i] = 1
        #
        # self.mask = self.mask.cuda()

    def forward(self, input_):
        # return ExpanderLinear.apply(input_, self.weight, self.mask)
        return torch.sparse.mm(self.weight, input_.t()).t()

    def _init_mask(self, mask=None):
        weight = torch.Tensor(self.output_features, self.input_features)
        nn.init.kaiming_normal_(weight.data, mode='fan_in') 
        # self.mask = torch.zeros(self.output_features, self.input_features)
        if mask is None:
           
            # mask = torch.zeros(self.output_features, self.input_features)
            indices = [list(), list()]
            if self.output_features < self.input_features:
                for i in range(self.output_features):
                    x = torch.randperm(self.input_features)
                    for j in range(int(self.input_features*(self.expand_size/16))):
                        # self.mask[i][x[j]] = 1
                        indices[0].append(i)
                        indices[1].append(x[j])
            else:
                for i in range(self.input_features):
                    x = torch.randperm(self.output_features)
                    for j in range(int(self.output_features*(self.expand_size/16))):
                        # self.mask[x[j]][i] = 1
                        indices[0].append(x[j])
                        indices[1].append(i)
            indices = np.array(indices)
            self.weight = nn.Parameter(torch.sparse_coo_tensor(indices, weight.data[indices], torch.Size([self.output_features, self.input_features])), requires_grad=True)
            # idx0, idx1 = mask.nonzero(as_tuple=True)
            # self.mask = torch.stack([idx0, idx1])
            # idx1, idx2 = (mask==0).nonzero(as_tuple=True)
            # self.mask = torch.stack([idx1, idx2])
            # self.mask = (mask==0).nonzero()
            # self.mask = torch.sparse_coo_tensor(indices, [1]*indices.shape[1], torch.Size([self.output_features, self.input_features])).coalesce()
            self.mask = indices
        else:
            # self.mask = mask.resize_as_(self.mask)
            #self.mask = torch.sparse_coo_tensor(mask, [1]*mask.shape[1], torch.Size([self.output_features, self.input_features])).coalesce()
            self.mask = mask
            self.weight = nn.Parameter(torch.sparse_coo_tensor(self.mask, weight.data[self.mask], torch.Size([self.output_features, self.input_features])), requires_grad=True)                           
                                       
        # self.mask = (self.mask == 1)
        # self.mask = self.mask.cuda()


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