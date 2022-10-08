"""Megatron-LM custom ops."""
import torch

from ..bench import BenchConfig, bench


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
    shapes = [(4, 16, 512, 512)]
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
