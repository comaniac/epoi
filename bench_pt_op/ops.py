"""Ops for benchmarking.

For shapes to benchmark, here are common configurations for transformaer models:

                   batch,  seq,  head, hidden, intermediate, vocab
bert-large      :   32,    512,  16,   768,    1024,         30522
gigantic model 1:   16,    512,  64,   8192,   32768,        32008 or 50264 or 256032
gigantic model 2:   4,     2048, 64,   8192,   32768,        32008 or 50264 or 256032
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bench import BenchConfig, bench


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
        (16, 512, 768),
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
    )


def bias_gelu(args):
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


def megatron_softmax(args):
    from megatron import fused_kernels
    from megatron.model.fused_softmax import FusedScaleMaskSoftmax
    from megatron.model.enums import AttnMaskType
    from megatron.model.utils import attention_mask_func

    class FakeArgs:
        def __init__(self):
            self.rank = 0
            self.masked_softmax_fusion = True
            self.gradient_accumulation_fusion = False

    fused_kernels.load(FakeArgs())

    def _init(shape, dtype, use_megatron, softmax_in_fp32):
        softmax = FusedScaleMaskSoftmax(
            input_in_fp16=True,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.padding,
            scaled_masked_softmax_fusion=use_megatron,
            mask_func=attention_mask_func,
            softmax_in_fp32=softmax_in_fp32,
            scale=None,
        )
        return softmax.cuda()

    def gen_inputs(shape, dtype):
        inp = torch.randn(*shape, dtype=dtype, device="cuda")
        # FIXME: How to generate a valid mask?
        # mask = torch.ones(size=(shape[0], 1, *shape[2:]), dtype=torch.bool, device="cuda")
        return [inp, None]

    def zero_grad(_, inputs):
        for inp in inputs:
            if inp is not None and inp.dtype in (torch.float32, torch.float16):
                inp.grad = None

    # (batch, head, seq, seq)
    shapes = [(4, 16, 512, 512), (16, 64, 512, 512), (4, 64, 2048, 2048)]
    bench(
        shapes,
        [
            BenchConfig(
                lambda shape, dtype: _init(shape, dtype, False, False),
                torch.float16,
                "PyTorch (Comp-FP16)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                lambda shape, dtype: _init(shape, dtype, False, True),
                torch.float16,
                "PyTorch (Comp-FP32)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                lambda shape, dtype: _init(shape, dtype, True, False),
                torch.float16,
                "Megatron-LM (Comp-FP16)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                lambda shape, dtype: _init(shape, dtype, True, True),
                torch.float16,
                "Megatron-LM (Comp-FP32)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
        ],
        "Softmax with FP16 input",
    )


def qkv_self_attn(args):
    class UnfusedModule(torch.nn.Module):
        def __init__(self, hidden_size, num_heads):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_size = hidden_size // num_heads
            self.key = torch.nn.Linear(hidden_size, num_heads * self.head_size)
            self.value = torch.nn.Linear(hidden_size, num_heads * self.head_size)
            self.query = torch.nn.Linear(hidden_size, num_heads * self.head_size)

        def transpose_for_scores(self, x):
            new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)

        def forward(self, hidden_states):
            query_layer = self.transpose_for_scores(self.query(hidden_states))
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            return (query_layer, key_layer, value_layer)

    class FusedModule(torch.nn.Module):
        def __init__(self, hidden_size, num_heads):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_size = hidden_size // num_heads
            self.qkv = torch.nn.Linear(hidden_size, num_heads * self.head_size * 3)

        def transpose_for_scores(self, x):
            # (B, S, 3 * H) -> (B, S, D, H/D, 3)
            new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size, 3)
            x = x.view(*new_x_shape)
            # (B, S, D, H/D, 3) -> (B, D, S, H/D, 3)
            return x.permute(0, 2, 1, 3, 4)

        def forward(self, hidden_states):
            combined_layer = self.qkv(hidden_states)
            # (B, S, 3 * H) -> (B, D, S, H/D, 3)
            combined_layer = self.transpose_for_scores(combined_layer)
            # (B, D, S, H/D, 3) -> 3 * (B, D, S, H/D)
            return [torch.squeeze(t) for t in torch.split(combined_layer, 1, dim=-1)]

    def _init(shape, dtype, fused):
        model = UnfusedModule(*shape[2:]) if not fused else FusedModule(*shape[2:])
        if dtype == torch.float16:
            model = model.half()
        return model.cuda()

    def gen_inputs(shape, dtype):
        inp = torch.randn(*shape[:-1], dtype=dtype, device="cuda")
        return [inp]

    def zero_grad(mod, inputs):
        inputs[0].grad = None
        if hasattr(mod, "qkv"):
            mod.qkv.weight.grad = None
            mod.qkv.bias.grad = None
        else:
            mod.key.weight.grad = None
            mod.key.bias.grad = None
            mod.value.weight.grad = None
            mod.value.bias.grad = None
            mod.query.weight.grad = None
            mod.query.bias.grad = None

    no_fuse = lambda shape, dtype: _init(shape, dtype, False)
    fuse = lambda shape, dtype: _init(shape, dtype, True)

    # (batch, seq, hidden size, num head)
    shapes = [
        (4, 512, 1024, 16),  # bert-large w. BS4, seq512
        (8, 512, 1024, 16),  # bert-large w. BS8, seq512
        (16, 512, 1024, 16),  # bert-large w. BS16, seq512
        (16, 512, 8192, 64),  # gigantic model 1 w. BS16, seq512
        (4, 2048, 8192, 64),  # gigantic model 1 w. BS4, seq2048
    ]
    bench(
        shapes,
        [
            BenchConfig(
                no_fuse,
                torch.float16,
                "NoFuse (FP16)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
            BenchConfig(
                fuse,
                torch.float16,
                "Fused (FP16)",
                not args.forward_only,
                gen_inputs=gen_inputs,
                zero_grad=zero_grad,
            ),
        ],
        "QKV in Self-Attention",
    )
