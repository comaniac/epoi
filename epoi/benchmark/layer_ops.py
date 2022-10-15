"""Model structure optimized ops."""
import torch

from .utils import is_available
from .bencher import BenchConfig, bench, check_correctness


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
        verbose=args.verbose,
    )


def bert_attention(args):
    if not is_available("transformers") or not is_available("xformers"):
        print("Skip attention because transformers or xformers is not available")
        return

    from transformers import AutoConfig
    from transformers.models.bert.modeling_bert import BertSelfAttention
    from ..ops.xformers_attn import BertSelfAttention as xFormersSelfAttention

    def _init(shape, dtype, attn_type, no_dropout=False):
        config = AutoConfig.from_pretrained("bert-large-uncased")
        config.hidden_size = shape[2]
        config.num_attention_heads = shape[3]
        config.intermediate_size = shape[4]
        config.vocab_size = shape[5]
        if no_dropout:
            config.attention_probs_dropout_prob = 0.0
        if attn_type is not None:
            attn = xFormersSelfAttention(config, attn_op_name=attn_type)
        else:
            attn = BertSelfAttention(config)
        if dtype == torch.float16:
            attn = attn.half()
        return attn.cuda()

    def gen_inputs(shape, dtype):
        # (batch, seq, hidden size)
        inp_shape = shape[:3]
        hidden_states = torch.randn(*inp_shape, dtype=dtype, device="cuda")
        attn_mask = torch.zeros(inp_shape[0], 1, 1, inp_shape[1], dtype=dtype, device="cuda")
        return [hidden_states, attn_mask]

    def zero_grad(mod, inputs):
        inputs[0].grad = None
        mod.key.weight.grad = None
        mod.key.bias.grad = None
        mod.value.weight.grad = None
        mod.value.bias.grad = None
        mod.query.weight.grad = None
        mod.query.bias.grad = None

    def override_params(this, other):
        this.query.weight = other.query.weight
        this.query.bias = other.query.bias
        this.key.weight = other.key.weight
        this.key.bias = other.key.bias
        this.value.weight = other.value.weight
        this.value.bias = other.value.bias

    # (batch, seq, hidden size, #head, intermediate size, vocab size)
    shapes = [
        (8, 512, 1024, 16, 4096, 30522),
        (16, 512, 8192, 64, 32768, 50264),
        (4, 2048, 8192, 64, 32768, 50264),
    ]
    configs = [
        BenchConfig(
            lambda shape, dtype: _init(shape, dtype, None),
            torch.float16,
            "HF (Attn)",
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
        ),
        BenchConfig(
            lambda shape, dtype: _init(shape, dtype, "base"),
            torch.float16,
            "xFormers (FA)",
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
        ),
        BenchConfig(
            lambda shape, dtype: _init(shape, dtype, "cutlass"),
            torch.float16,
            "xFormers Cutlass (FA)",
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
        ),
        BenchConfig(
            lambda shape, dtype: _init(shape, dtype, "triton"),
            torch.float16,
            "xFormers Triton (FA)",
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
        ),
    ]

    # Check correctness
    fun_attn = _init(shapes[0], configs[0].dtype, None, no_dropout=True)
    fun_xf_base = _init(shapes[0], configs[1].dtype, "base", no_dropout=True)
    fun_xf_cutlass = _init(shapes[0], configs[1].dtype, "cutlass", no_dropout=True)
    fun_xf_triton = _init(shapes[0], configs[1].dtype, "triton", no_dropout=True)
    override_params(fun_xf_base, fun_attn)
    check_correctness(
        shapes[0],
        fun_attn,
        fun_xf_base,
        configs[0],
        tol=1e-3,
        desc="xFormers FlashAttn",
        verbose=args.verbose,
    )
    override_params(fun_xf_cutlass, fun_attn)
    check_correctness(
        shapes[0],
        fun_attn,
        fun_xf_cutlass,
        configs[0],
        tol=1e-3,
        desc="xFormers Cutlass FlashAttn",
        verbose=args.verbose,
    )
    override_params(fun_xf_triton, fun_attn)
    check_correctness(
        shapes[0],
        fun_attn,
        fun_xf_triton,
        configs[0],
        tol=1e-3,
        desc="xFormers Triton FlashAttn",
        verbose=args.verbose,
    )
    bench(shapes, configs, "Attention (Attn) and FlashAttention (FA)", verbose=args.verbose)


def gpt_attention(args):
    """FIXME: This is not working yet"""
    if not is_available("transformers") or not is_available("xformers"):
        print("Skip attention because transformers or xformers is not available")
        return

    from transformers import AutoConfig
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    from ..ops.xformers_attn import GPT2Attention as xFormersSelfAttention

    def _init(shape, dtype, attn_type, no_dropout=False):
        config = AutoConfig.from_pretrained("gpt2")
        # config.hidden_size = shape[2]
        # config.num_attention_heads = shape[3]
        # config.intermediate_size = shape[4]
        # config.vocab_size = shape[5]
        if no_dropout:
            config.attention_probs_dropout_prob = 0.0
        if attn_type is not None:
            attn = xFormersSelfAttention(config, attn_op_name=attn_type)
        else:
            attn = GPT2Attention(config)
        if dtype == torch.float16:
            attn = attn.half()
        return attn.cuda()

    def gen_inputs(shape, dtype):
        # (batch, seq, hidden size)
        inp_shape = shape[:3]
        hidden_states = torch.randn(*inp_shape, dtype=dtype, device="cuda")
        attn_mask = torch.zeros(inp_shape[0], 1, 1, inp_shape[1], dtype=dtype, device="cuda")
        return [hidden_states, attn_mask]

    def zero_grad(mod, inputs):
        inputs[0].grad = None
        mod.c_attn.weight.grad = None
        mod.c_attn.bias.grad = None
        mod.c_proj.weight.grad = None
        mod.c_proj.bias.grad = None

    def override_params(this, other):
        this.query.weight = other.query.weight
        this.query.bias = other.query.bias
        this.key.weight = other.key.weight
        this.key.bias = other.key.bias
        this.value.weight = other.value.weight
        this.value.bias = other.value.bias

    # (batch, seq, hidden size, #head, intermediate size, vocab size)
    shapes = [
        (8, 512, 1024, 16, 4096, 30522),
        # (16, 512, 8192, 64, 32768, 50264),
        # (4, 2048, 8192, 64, 32768, 50264),
    ]
    configs = [
        BenchConfig(
            lambda shape, dtype: _init(shape, dtype, None),
            torch.float16,
            "HF (Attn)",
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
        ),
        BenchConfig(
            lambda shape, dtype: _init(shape, dtype, "native"),
            torch.float16,
            "xFormers (FA)",
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
        ),
        # BenchConfig(
        #     lambda shape, dtype: _init(shape, dtype, "base"),
        #     torch.float16,
        #     "xFormers (FA)",
        #     not args.forward_only,
        #     gen_inputs=gen_inputs,
        #     zero_grad=zero_grad,
        # ),
        # BenchConfig(
        #     lambda shape, dtype: _init(shape, dtype, "cutlass"),
        #     torch.float16,
        #     "xFormers Cutlass (FA)",
        #     not args.forward_only,
        #     gen_inputs=gen_inputs,
        #     zero_grad=zero_grad,
        # ),
        # BenchConfig(
        #     lambda shape, dtype: _init(shape, dtype, "triton"),
        #     torch.float16,
        #     "xFormers Triton (FA)",
        #     not args.forward_only,
        #     gen_inputs=gen_inputs,
        #     zero_grad=zero_grad,
        # ),
    ]

    # Check correctness
    fun_attn = _init(shapes[0], configs[0].dtype, None, no_dropout=True)
    fun_xf_base = _init(shapes[0], configs[1].dtype, "native", no_dropout=True)
    override_params(fun_xf_base, fun_attn)
    check_correctness(
        shapes[0], fun_attn, fun_xf_base, configs[0], tol=1e-3, desc="xFormers FlashAttn"
    )
