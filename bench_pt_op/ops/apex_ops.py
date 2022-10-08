"""PyTorch and apex ops."""
import torch
import torch.nn as nn

from ..bench import BenchConfig, bench


def layer_norm(args):
    from apex.normalization import FusedLayerNorm

    def _init(shape, dtype, use_apex):
        ln = nn.LayerNorm(shape[-1]) if not use_apex else FusedLayerNorm(shape[-1])
        if dtype == torch.float16:
            ln = ln.half()
        return ln.cuda()

    def zero_grad(func, _):
        func.weight.grad = None
        func.bias.grad = None

    pt_norm = lambda shape, dtype: _init(shape, dtype, False)
    apex_norm = lambda shape, dtype: _init(shape, dtype, True)

    # (batch, seq, hidden size)
    shapes = [(32, 128, 768), (16, 512, 768), (16, 512, 8192), (4, 2048, 8192)]
    bench(
        shapes,
        [
            BenchConfig(
                pt_norm, torch.float32, "PyTorch (FP32)", not args.forward_only, zero_grad=zero_grad
            ),
            BenchConfig(
                apex_norm, torch.float32, "Apex (FP32)", not args.forward_only, zero_grad=zero_grad
            ),
            BenchConfig(
                pt_norm, torch.float16, "PyTorch (FP16)", not args.forward_only, zero_grad=zero_grad
            ),
            BenchConfig(
                apex_norm, torch.float16, "Apex (FP16)", not args.forward_only, zero_grad=zero_grad
            ),
        ],
        "LayerNorm",
    )
