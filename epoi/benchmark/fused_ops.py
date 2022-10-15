"""Fused Ops."""
import torch
import torch.nn.functional as F

from .bencher import BenchConfig, bench


def dropout_add_ln(args):
    def compute(input1, input2, weight, bias):
        axis = 2
        dropout_prob = 0.1
        dropout_out = F.dropout(input1, dropout_prob, training=True)
        norm_input = dropout_out + input2
        norm_output = F.layer_norm(norm_input, (input1.size(axis),), weight, bias)
        return norm_output

    def gen_inputs(shape, dtype):
        input1 = torch.randn(*shape, dtype=dtype, device="cuda")
        input2 = torch.rand_like(input1)

        weight = torch.nn.Parameter(torch.randn(shape[2], dtype=dtype, device="cuda"))
        bias = torch.nn.Parameter(torch.randn(shape[2], dtype=dtype, device="cuda"))
        return [input1, input2, weight, bias]

    def zero_grad(_, inputs):
        for inp in inputs:
            inp.grad = None

    eager = lambda shape, dtype: compute
    ts_nvfuser = lambda shape, dtype: torch.jit.script(compute)

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
        verbose=args.verbose
    )


def megatron_bias_gelu(args):
    from megatron.model.fused_bias_gelu import bias_gelu_impl

    def gen_inputs(shape, dtype):
        inp = torch.randn(*shape, dtype=dtype, device="cuda")
        bias = torch.nn.Parameter(torch.randn(shape[2], dtype=dtype, device="cuda"))
        return [inp, bias]

    def zero_grad(_, inputs):
        for inp in inputs:
            inp.grad = None

    def pt_compute(inp, bias):
        return F.gelu(inp + bias, approximate="none")

    mega = lambda shape, dtype: bias_gelu_impl
    pt = lambda shape, dtype: pt_compute

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
                "PyTorch (FP32)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                mega,
                torch.float32,
                "Megatron-LM (FP32)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                pt,
                torch.float16,
                "PyTorch (FP16)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                mega,
                torch.float16,
                "Megatron-LM (FP16)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
        ],
        "Bias+GeLU",
    )
