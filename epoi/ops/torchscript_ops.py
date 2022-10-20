"""The fused ops by torchscript."""
from typing import List
import math
import torch
import torch.nn.functional as F


class BiasGeLUFunction(torch.autograd.Function):
    """Bias+GeLU. Copied from Megatron-LM."""

    @torch.jit.script
    def bias_gelu(bias, y):
        x = bias + y
        return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

    # gradient of tanh approximation of gelu
    # gradient of actual gelu is:
    # 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
    @torch.jit.script
    def bias_gelu_back(g, bias, y):
        x = bias + y
        tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
        ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
            1 + tanh_out
        )
        return ff * g

    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return BiasGeLUFunction.bias_gelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = BiasGeLUFunction.bias_gelu_back(grad_output, bias, input)
        return tmp, tmp


class FusedBiasGELU(torch.nn.Module):
    def __init__(self, size, device=None, dtype=None, prev_weight=None, fused=True):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.bias = torch.nn.Parameter(torch.empty(size, **factory_kwargs))
        self.fused = fused
        self.reset_parameters(prev_weight)

    def reset_parameters(self, prev_weight=None):
        range = (0, 1)
        if prev_weight is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(prev_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            range = (-bound, bound)
        torch.nn.init.uniform_(self.bias, *range)

    def forward(self, input):
        if self.fused:
            return BiasGeLUFunction.apply(input, self.bias)
        return F.gelu(input + self.bias, approximate="none")


def fused_dropout_add_layernorm(
    input1,
    input2,
    weight,
    bias,
    dropout_prob: float,
    training: bool,
    normalized_shape: List[int],
    eps: float,
):
    """torchscript tracable fused dropout, add, layernorm.
    (non-tensor arguments must have type annotations)
    """
    dropout_out = F.dropout(input1, dropout_prob, training=training)
    norm_input = dropout_out + input2
    norm_output = F.layer_norm(norm_input, normalized_shape, weight, bias, eps)
    return norm_output


class FusedDropoutAddLayerNorm(torch.nn.Module):
    def __init__(self, size, dropout_prob, eps=1e-5, fused=True):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(size, eps=eps)
        self.dropout_prob = dropout_prob
        self.fused = fused

    def forward(self, input1, input2):
        func = fused_dropout_add_layernorm
        func = torch.jit.script(func) if self.fused else func

        return func(
            input1,
            input2,
            self.layer_norm.weight,
            self.layer_norm.bias,
            self.dropout_prob,
            self.training,
            self.layer_norm.normalized_shape,
            self.layer_norm.eps,
        )
