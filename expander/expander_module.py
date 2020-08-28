import torch
import torch.nn as nn
from expander.samplers import sampler


class ExpanderLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, mask, bias=None):
        ctx.save_for_backward(_input, weight, bias)
        ctx.mask = mask
        weight.mul_(mask)
        if _input.dim() == 2 and bias is not None:
            output = torch.addmm(bias, _input, weight.t())
        else:
            output = _input.matmul(weight.t())
            if bias is not None:
                output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        weight.mul_(ctx.mask)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            if grad_output.dim() == 2:
                grad_weight = grad_output.t().mm(_input)
            else:
                grad_weight = torch.matmul(grad_output.squeeze(-1),
                                           _input.squeeze(-2)).sum(0).sum(0)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, None, grad_bias


class ExpanderLinear(nn.Module):
    def __init__(self, indim, outdim, density=None, bias=True,
                 sampler="regular"):
        super(ExpanderLinear, self).__init__()
        self.indim, self.outdim, self.density, self.sampler = (indim,
                                                               outdim,
                                                               density,
                                                               sampler)

        self.weight = nn.Parameter(data=torch.Tensor(self.outdim, self.indim))

        if bias:
            self.bias = nn.Parameter(data=torch.Tensor(self.outdim))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("mask", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # nn.init.kaiming_normal_(self.weight, mode="fan_ins")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, _input):
        return ExpanderLinearFunction.apply(_input, self.weight,
                                            self.mask, self.bias)

    def generate_mask(self, init=None):
        if init is None:
            self.mask, self.n_params = sampler(self.outdim,
                                               self.indim,
                                               self.density,
                                               method=self.sampler)
        else:
            self.n_params = torch.sum(init)
            self.mask = init

        if self.bias is not None:
            self.n_params += self.outdim
