"""The fused ops by torchscript."""
from typing import List
from functools import partial
import math
import torch
import torch.nn.functional as F
from functorch.compile import memory_efficient_fusion


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


def new_gelu(input):
    """New GELU activation function copied from HuggingFace transformers."""
    return (
        0.5
        * input
        * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    )


def bias_new_gelu(input, bias):
    return new_gelu(input + bias)


class FusedBiasNewGELU(torch.nn.Module):
    def __init__(self, size, device=None, dtype=None, prev_weight=None, fused=True, aot=True):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.bias = torch.nn.Parameter(torch.empty(size, **factory_kwargs))
        self.fused = fused
        self.reset_parameters(prev_weight)
        if self.fused:
            if aot:
                self.func = memory_efficient_fusion(bias_new_gelu)
            else:
                self.func = torch.jit.script(bias_new_gelu)
        else:
            self.func = bias_new_gelu

    def reset_parameters(self, prev_weight=None):
        range = (0, 1)
        if prev_weight is not None and len(prev_weight.shape) > 1:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(prev_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            range = (-bound, bound)
        torch.nn.init.uniform_(self.bias, *range)

    def forward(self, input):
        return self.func(input, self.bias)


class MM(torch.nn.Module):
    """
    Copied from HuggingFace transformers.
    The MM layer defined defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx, bias=True):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        torch.nn.init.normal_(w, std=0.02)
        self.weight = torch.nn.Parameter(w)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(nf))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        if self.bias is not None:
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        else:
            x = torch.mm(x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


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
    """torchscript tracable dropout-add-layernorm.
    (non-tensor arguments must have type annotations)
    """
    dropout_out = F.dropout(input1, dropout_prob, training=training)
    norm_input = dropout_out + input2
    norm_output = F.layer_norm(norm_input, normalized_shape, weight, bias, eps)
    return norm_output


class FusedDropoutAddLayerNorm(torch.nn.Module):
    def __init__(self, size, dropout_prob, eps=1e-5, fused=True, aot=False):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(size, eps=eps)
        self.dropout_prob = dropout_prob
        self.fused = fused
        if fused and aot:
            # FIXME: it works fine in benchmark but failed with HF Trainer with
            # RuntimeError: Trying to backward through the graph a second time
            # (or directly access saved tensors after they have already been freed).
            self.func = partial(
                fused_dropout_add_layernorm,
                dropout_prob=self.dropout_prob,
                training=self.training,
                eps=self.layer_norm.eps,
                normalized_shape=self.layer_norm.normalized_shape,
            )
            self.func = memory_efficient_fusion(self.func)
        else:
            self.func = fused_dropout_add_layernorm
            if fused:
                self.func = torch.jit.script(self.func)
            self.func = partial(
                self.func,
                dropout_prob=self.dropout_prob,
                training=self.training,
                eps=self.layer_norm.eps,
                normalized_shape=self.layer_norm.normalized_shape,
            )

    def forward(self, input1, input2):
        return self.func(
            input1,
            input2,
            self.layer_norm.weight,
            self.layer_norm.bias,
        )
