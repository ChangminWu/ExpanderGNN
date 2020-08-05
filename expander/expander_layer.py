import torch


class ExpanderLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, mask, bias=None):
        ctx.save_for_backward(_input, weight, bias)
        ctx.mask = mask
        weight.mul_(mask)
        output = _input.mm(weight.t())

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        _input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        weight.mul_(ctx.mask)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(_input)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, None, grad_bias