"""Fused Ops."""
import torch

from .bencher import BenchConfig, bench
from ..ops.torchscript_ops import FusedDropoutAddLayerNorm, FusedBiasGELU, FusedBiasNewGELU


def dropout_add_ln(args):
    def _init(shape, dtype, fused, aot):
        dropout_prob = 0.1
        mod = FusedDropoutAddLayerNorm(shape[-1], dropout_prob, fused=fused, aot=aot)
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

    eager = lambda shape, dtype: _init(shape, dtype, fused=False, aot=False)
    ts_nvfuser = lambda shape, dtype: _init(shape, dtype, fused=True, aot=False)
    aot_nvfuser = lambda shape, dtype: _init(shape, dtype, fused=True, aot=True)

    # (batch, seq, intermediate or hidden size)
    shapes = [
        (8, 512, 1024), # bert-large
        (8, 512, 4096), # bert-large
        # (16, 512, 8192),
        # (16, 512, 32768),
        # (4, 2048, 8192),
        # (4, 2048, 32768),
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
                aot_nvfuser,
                torch.float32,
                "AOT+nvFuser (FP32)",
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
            BenchConfig(
                aot_nvfuser,
                torch.float16,
                "AOT+nvFuser (FP16)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
        ],
        "Dropout+Add+LayerNorm",
        verbose=args.verbose,
    )


def bias_gelu(args):
    def _init(shape, dtype, fused, aot):
        mod = FusedBiasNewGELU(shape[-1], fused=fused, aot=aot)
        if dtype == torch.float16:
            mod = mod.half()
        return mod.cuda()

    def gen_inputs(shape, dtype):
        inp = torch.randn(*shape, dtype=dtype, device="cuda")
        return [inp]

    def zero_grad(_, inputs):
        for inp in inputs:
            inp.grad = None

    fused_ts = lambda shape, dtype: _init(shape, dtype, fused=True, aot=False)
    fused_aot = lambda shape, dtype: _init(shape, dtype, fused=True, aot=True)
    pt = lambda shape, dtype: _init(shape, dtype, fused=False, aot=False)

    # (batch, seq, intermediate or hidden size)
    shapes = [
        (8, 512, 1024), # bert-large
        (8, 512, 768),  # bert-base
        (2, 1024, 4096), # gpt2-medium
        (16, 512, 8192), # gigantic bert
        (16, 512, 32768), # gigantic bert
        # (4, 2048, 8192), # gigantic bert
        # (4, 2048, 32768), # gigantic bert
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
                fused_ts,
                torch.float32,
                "TS+nvFuser (FP32)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                fused_aot,
                torch.float32,
                "AOT+nvFuser (FP32)",
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
                fused_ts,
                torch.float16,
                "TS+nvFuser (FP16)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                fused_aot,
                torch.float16,
                "AOT+nvFuser (FP16)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
        ],
        "Bias+GeLU",
        verbose=args.verbose,
    )
