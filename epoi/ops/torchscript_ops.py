"""The fused ops by torchscript."""
import math
import torch


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


fused_bias_gelu = BiasGeLUFunction.apply


class FusedBiasGELU(torch.nn.Module):
    def __init__(self, size, device=None, dtype=None, prev_weight=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.bias = torch.nn.Parameter(torch.empty(size, **factory_kwargs))
        self.reset_parameters(prev_weight)

    def reset_parameters(self, prev_weight=None):
        if prev_weight is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(prev_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            torch.nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, input):
        return fused_bias_gelu(input, self.bias)
