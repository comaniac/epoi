"""Normalization Ops."""
import torch
import torch.nn as nn

from .utils import is_available
from .bencher import BenchConfig, check_correctness, bench


def layer_norm(args):
    try:
        from apex.normalization import FusedLayerNorm as ApexLayerNorm
    except ImportError:
        ApexLayerNorm = lambda k: None

    try:
        from xformers.triton import FusedLayerNorm as xFormerLayerNorm
    except ImportError:
        xFormerLayerNorm = lambda k: None

    try:
        from ..ops.triton_layer_norm import TritonLayerNorm
    except ImportError:
        TritonLayerNorm = lambda k: None

    def _init(shape, dtype, lib="torch"):
        if lib == "torch":
            ln = nn.LayerNorm(shape[-1])
        elif lib == "apex":
            ln = ApexLayerNorm(shape[-1])
        elif lib == "xformers":
            ln = xFormerLayerNorm(shape[-1])
        elif lib == "triton":
            ln = TritonLayerNorm(shape[-1])
        else:
            raise RuntimeError(f"Unknown lib: {lib}")
        if dtype == torch.float16:
            ln = ln.half()
        return ln.cuda()

    def zero_grad(func, _):
        func.weight.grad = None
        func.bias.grad = None

    pt_norm = lambda shape, dtype: _init(shape, dtype, "torch")
    apex_norm = lambda shape, dtype: _init(shape, dtype, "apex")
    triton_norm = lambda shape, dtype: _init(shape, dtype, "triton")
    xformers_norm = lambda shape, dtype: _init(shape, dtype, "xformers")

    # (batch, seq, hidden size)
    shapes = [(32, 128, 768), (8, 512, 1024), (16, 512, 8192), (4, 2048, 8192)]

    configs = [
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
        BenchConfig(
            triton_norm,
            torch.float16,
            "Triton (FP16)",
            not args.forward_only,
            zero_grad=zero_grad,
        ),
        BenchConfig(
            xformers_norm,
            torch.float16,
            "xFormers (FP16)",
            not args.forward_only,
            zero_grad=zero_grad,
        ),
    ]

    # Check correctness. Note that weight and bias are initailized to 1 and 0,
    # so it might not be a comprehensive correctness check.
    fun_pt = configs[2].init_func(shapes[0], configs[2].dtype)
    fun_apex = configs[3].init_func(shapes[0], configs[3].dtype)
    fun_triton = configs[4].init_func(shapes[0], configs[4].dtype)
    fun_xformers = configs[5].init_func(shapes[0], configs[5].dtype)
    check_correctness(shapes[0], fun_pt, fun_apex, configs[2], fwd_tol=1e-3, desc="Apex (FP16)")
    check_correctness(shapes[0], fun_pt, fun_triton, configs[2], fwd_tol=1e-3, desc="Triton (FP16)")
    check_correctness(
        shapes[0], fun_pt, fun_xformers, configs[2], fwd_tol=1e-3, desc="xFormers (FP16)"
    )

    # Benchmark
    return bench(shapes, configs, "LayerNorm", verbose=args.verbose)


def softmax(args):
    if is_available("megatron"):
        from megatron import fused_kernels
        from megatron.model.fused_softmax import FusedScaleMaskSoftmax as MegatronSoftmax
        from megatron.model.enums import AttnMaskType
        from megatron.model.utils import attention_mask_func

        class FakeArgs:
            def __init__(self):
                self.rank = 0
                self.masked_softmax_fusion = True
                self.gradient_accumulation_fusion = False

        fused_kernels.load(FakeArgs())
    else:
        MegatronSoftmax = None

    if is_available("xformers"):
        from xformers.triton.softmax import softmax as xformers_softmax
    else:
        xformers_softmax = None

    def _init(shape, dtype, softmax_in_fp32, lib="torch"):
        if lib == "torch":

            def torch_softmax(input, mask):
                cast_input = input.dtype == torch.float16 and softmax_in_fp32
                if cast_input:
                    input = input.float()
                if mask is not None:
                    input = input + mask
                probs = nn.functional.softmax(input, dim=-1)
                if cast_input:
                    return probs.half()
                return probs

            return torch_softmax
        elif lib == "megatron":
            if MegatronSoftmax is None:
                return None
            return MegatronSoftmax(
                input_in_fp16=True,
                input_in_bf16=False,
                attn_mask_type=AttnMaskType.padding,
                scaled_masked_softmax_fusion=True,
                mask_func=attention_mask_func,
                softmax_in_fp32=softmax_in_fp32,
                scale=None,
            ).cuda()
        elif lib == "xformers":
            if xformers_softmax is None:
                return None
            return lambda input, mask: xformers_softmax(input, mask)
        raise RuntimeError(f"Unknown lib: {lib}")

    def gen_inputs(shape, dtype):
        inp = torch.randn(*shape, dtype=dtype, device="cuda")
        # FIXME: How to generate a valid mask?
        # mask = torch.ones(size=(shape[0], 1, *shape[2:]), dtype=torch.bool, device="cuda")
        return [inp, None]

    def zero_grad(_, inputs):
        for inp in inputs:
            if inp is not None and inp.dtype in (torch.float32, torch.float16):
                inp.grad = None

    # (batch, head, seq, seq). Note that head dimension may be sharded.
    shapes = [(8, 16 // 8, 1024, 1024), (4, 16, 512, 512), (8, 16, 512, 512)]
    configs = [
        BenchConfig(
            lambda shape, dtype: _init(shape, dtype, True, "torch"),
            torch.float16,
            "PyTorch (Comp-FP32)",
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
        ),
        BenchConfig(
            lambda shape, dtype: _init(shape, dtype, False, "torch"),
            torch.float16,
            "PyTorch (Comp-FP16)",
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
        ),
        BenchConfig(
            lambda shape, dtype: _init(shape, dtype, True, "megatron"),
            torch.float16,
            "Megatron-LM (Comp-FP32)",
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
        ),
        BenchConfig(
            lambda shape, dtype: _init(shape, dtype, True, "xformers"),
            torch.float16,
            "xFormers (Comp-FP32)",
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
        ),
    ]

    # Check correctness
    fun_pt = configs[0].init_func(shapes[0], configs[0].dtype)
    fun_megatron = configs[2].init_func(shapes[0], configs[1].dtype)
    fun_xformers = configs[-1].init_func(shapes[0], configs[2].dtype)
    check_correctness(
        shapes[0], fun_pt, fun_megatron, configs[0], fwd_tol=1e-3, desc="Megatron-LM (Comp-FP32)"
    )
    check_correctness(
        shapes[0], fun_pt, fun_xformers, configs[0], fwd_tol=1e-3, desc="xFormers (Comp-FP32)"
    )

    # Benchmark
    return bench(shapes, configs, "Softmax with FP16 input", verbose=args.verbose)
