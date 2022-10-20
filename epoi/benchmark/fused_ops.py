"""Fused Ops."""
import torch

from .bencher import BenchConfig, bench
from ..ops.torchscript_ops import FusedDropoutAddLayerNorm, FusedBiasGELU


def dropout_add_ln(args):
    def _init(shape, dtype, fused):
        dropout_prob = 0.1
        mod = FusedDropoutAddLayerNorm(shape[-1], dropout_prob, fused=fused)
        if dtype == torch.float16:
            mod = mod.half()
        return mod.cuda()

    def zero_grad(_, inputs):
        for inp in inputs:
            inp.grad = None

    def gen_inputs(shape, dtype):
        input1 = torch.randn(*shape, dtype=dtype, device="cuda")
        input2 = torch.rand_like(input1)
        return [input1, input2]

    eager = lambda shape, dtype: _init(shape, dtype, fused=False)
    ts_nvfuser = lambda shape, dtype: _init(shape, dtype, fused=True)

    # (batch, seq, intermediate or hidden size)
    shapes = [
        (32, 128, 768),
        (4, 512, 768),
        (8, 512, 1024),
        (64, 128, 1024),
        (16, 512, 8192),
        (16, 512, 32768),
        (4, 2048, 8192),
        (4, 2048, 32768),
    ]
    bench(
        shapes,
        [
            BenchConfig(
                eager,
                torch.float32,
                "Eager (FP32)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                ts_nvfuser,
                torch.float32,
                "TS+nvFuser (FP32)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                eager,
                torch.float16,
                "Eager (FP16)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                ts_nvfuser,
                torch.float16,
                "TS+nvFuser (FP16)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
        ],
        "Dropout+Add+LayerNorm",
        verbose=args.verbose,
    )


def bias_gelu(args):
    def _init(shape, dtype, fused):
        mod = FusedBiasGELU(shape[-1], fused=fused)
        if dtype == torch.float16:
            mod = mod.half()
        return mod.cuda()

    def gen_inputs(shape, dtype):
        inp = torch.randn(*shape, dtype=dtype, device="cuda")
        return [inp]

    def zero_grad(_, inputs):
        for inp in inputs:
            inp.grad = None

    fused = lambda shape, dtype: _init(shape, dtype, fused=True)
    pt = lambda shape, dtype: _init(shape, dtype, fused=False)

    # (batch, seq, intermediate or hidden size)
    shapes = [
        (8, 512, 1024),
        (8, 512, 768),
        (16, 512, 1024),
        (16, 512, 8192),
        (16, 512, 32768),
        (4, 2048, 8192),
        (4, 2048, 32768),
    ]
    bench(
        shapes,
        [
            BenchConfig(
                pt,
                torch.float32,
                "Eager (FP32)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                fused,
                torch.float32,
                "TS+nvFuser (FP32)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                pt,
                torch.float16,
                "Eager (FP16)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                fused,
                torch.float16,
                "TS+nvFuser (FP16)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
        ],
        "Bias+GeLU",
        verbose=args.verbose,
    )
