"""Model structure optimized ops."""
from functools import partial

import torch

from .utils import is_available
from .bencher import BenchConfig, bench, check_correctness
from .logger import get_logger

logger = get_logger("layer_ops")


def qkv_self_attn(args):
    torch.manual_seed(42)

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

    no_fuse = partial(_init, fused=False)
    fuse = partial(_init, shape, dtype, fused=True)

    # (batch, seq, hidden size, num head)
    shapes = [
        (4, 512, 1024, 16),  # bert-large w. BS4, seq512
        (8, 512, 1024, 16),  # bert-large w. BS8, seq512
        (16, 512, 1024, 16),  # bert-large w. BS16, seq512
        (16, 512, 8192, 64),  # gigantic model 1 w. BS16, seq512
        (4, 2048, 8192, 64),  # gigantic model 1 w. BS4, seq2048
    ]
    return bench(
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
    torch.manual_seed(42)
    if not is_available("transformers") or not is_available("xformers"):
        logger.warning("Skip attention because transformers or xformers is not available")
        return

    from transformers import AutoConfig
    from transformers.models.bert.modeling_bert import BertSelfAttention
    from ..inject.policy.bert import InjectHFBertSelfAttentionPolicy

    def _init(shape, dtype, attn_op_name, no_dropout=False):
        config = AutoConfig.from_pretrained("bert-large-uncased")
        config.hidden_size = shape[2]
        config.num_attention_heads = shape[3]
        config.intermediate_size = shape[4]
        config.vocab_size = shape[5]
        if no_dropout:
            config.attention_probs_dropout_prob = 0.0
        attn = BertSelfAttention(config)
        if attn_op_name is not None:
            attn = InjectHFBertSelfAttentionPolicy.init_from_object(attn, attn_op_name=attn_op_name)

        if dtype == torch.float16:
            attn = attn.half()
        return attn.cuda()

    def gen_inputs(shape, dtype):
        # (batch, seq, hidden size)
        inp_shape = shape[:3]
        hidden_states = torch.randn(*inp_shape, dtype=dtype, device="cuda")
        attn_mask = torch.randn(inp_shape[0], 1, 1, inp_shape[1], dtype=dtype, device="cuda")
        return [hidden_states, attn_mask]

    def zero_grad(mod, inputs):
        inputs[0].grad = None
        mod.key.weight.grad = None
        mod.key.bias.grad = None
        mod.value.weight.grad = None
        mod.value.bias.grad = None
        mod.query.weight.grad = None
        mod.query.bias.grad = None

    # (batch, seq, hidden size, #head, intermediate size, vocab size)
    shapes = [
        (8, 512, 1024, 16, 4096, 30522),
        (16, 512, 8192, 64, 32768, 50264),
        (4, 2048, 8192, 64, 32768, 50264),
    ]
    configs = [
        BenchConfig(
            # FIXME: dropout is not supported in xFormer kernels yet.
            partial(_init, attn_op_name=None, no_dropout=True),
            torch.float16,
            "HF",
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
        )
    ]

    # Check correctness
    fun_attn = _init(shapes[0], configs[0].dtype, None, no_dropout=True)
    for fun_xf_name in ["native", "cutlass", "flshatt"]:
        fun_xf = _init(shapes[0], configs[0].dtype, fun_xf_name, no_dropout=True)
        InjectHFBertSelfAttentionPolicy.assign_params(fun_xf, fun_attn)
        config = BenchConfig(
            # FIXME: dropout is not supported in xFormer kernels yet.
            partial(_init, attn_op_name=fun_xf_name, no_dropout=True),
            torch.float16,
            f"xFormers {fun_xf_name}",
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
        )
        correct = check_correctness(
            shapes[0],
            fun_attn,
            fun_xf,
            config,
            desc=f"xFormers {fun_xf_name}",
            verbose=args.verbose,
        )
        if correct is not None:
            configs.append(config)

    if len(configs) == 1:
        logger.warning(f"Skip benchmark because no xFormers op is valid")
        return None

    return bench(
        shapes,
        configs,
        "HF Bert Attention and xFormers Attention",
        verbose=args.verbose,
    )


def gpt_attention(args):
    torch.manual_seed(42)
    if not is_available("transformers") or not is_available("xformers"):
        logger.warning("Skip attention because transformers or xformers is not available")
        return

    from transformers import AutoConfig
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    from ..inject.policy.gpt import InjectHFGPTAttentionPolicy

    def _init(shape, dtype, attn_op_name, no_dropout=False):
        config = AutoConfig.from_pretrained("gpt2-medium")
        config.max_position_embeddings = shape[1]
        config.n_embed = config.hidden_size = shape[2]  # hidden size
        config.n_head = config.num_attention_heads = shape[3]
        config.vocab_size = shape[4]
        if no_dropout:
            config.attn_pdrop = 0.0
            config.resid_pdrop = 0.0
        attn = GPT2Attention(config)
        if attn_op_name is not None:
            attn = InjectHFGPTAttentionPolicy.init_from_object(attn, attn_op_name=attn_op_name)
        if dtype == torch.float16:
            attn = attn.half()
        return attn.cuda()

    def gen_inputs(shape, dtype):
        # (batch, seq, hidden size * 3)
        inp_shape = shape[:3]
        hidden_states = torch.randn([*inp_shape[:2], inp_shape[2]], dtype=dtype, device="cuda")
        # attn_mask = torch.randn([inp_shape[0], 1, 1, inp_shape[1]], dtype=dtype, device="cuda")
        # FIXME: Enable attn_mask when triton supports it.
        attn_mask = None
        return [hidden_states, None, attn_mask]  # (hidden_states, layer_past, attn_mask)

    def zero_grad(mod, inputs):
        inputs[0].grad = None
        for param_name in ["c_attn", "c_proj", "qkv", "out_proj"]:
            if hasattr(mod, param_name):
                getattr(mod, param_name).weight.grad = None
                getattr(mod, param_name).bias.grad = None

    # (batch, seq, hidden size, #head, vocab size)
    shapes = [
        # (8, 1024, 1024, 16, 50257),  # gpt2-medium
        # (8, 512, 8192, 64, 50264),
        (4, 1024, 2048, 16, 50264),
    ]
    configs = [
        BenchConfig(
            # FIXME: dropout is not supported in xFormer kernels yet.
            partial(_init, attn_op_name=None, no_dropout=True),
            torch.float16,
            "HF",
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
        ),
    ]

    # Check correctness. Note that vanilla FashAttention does not support casual mask.
    fun_attn = _init(shapes[0], configs[0].dtype, None, no_dropout=True)
    for name in ["native", "triton", "cutlass", "flshatt"]:
        fun_xf = _init(shapes[0], configs[0].dtype, name, no_dropout=True)
        InjectHFGPTAttentionPolicy.assign_params(fun_xf, fun_attn)
        config = BenchConfig(
            partial(_init, attn_op_name=name),
            torch.float16,
            name,
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
        )
        correct = check_correctness(
            shapes[0],
            fun_attn,
            fun_xf,
            config,
            desc=name,
            verbose=args.verbose,
        )
        if correct is not None:
            configs.append(config)

    if len(configs) == 1:
        logger.warning(f"Skip benchmark because no xFormers op is valid")
        return None

    return bench(
        shapes,
        configs,
        "HF GPT Attention and xFormer Attention",
        verbose=args.verbose,
    )


def bloom_attention(args):
    torch.manual_seed(42)
    if not is_available("transformers") or not is_available("xformers"):
        logger.warning("Skip attention because transformers or xformers is not available")
        return

    from transformers import AutoConfig
    from transformers.models.bloom.modeling_bloom import BloomAttention
    from ..inject.policy.bloom import InjectHFBloomAttentionPolicy

    def _init(shape, dtype, attn_op_name, no_dropout=False):
        config = AutoConfig.from_pretrained("bigscience/bloom-560m")
        config.hidden_size = shape[2]
        config.n_head = shape[3]
        if no_dropout:
            config.attn_pdrop = 0.0
            config.resid_pdrop = 0.0
        attn = BloomAttention(config)
        if attn_op_name is not None:
            attn = InjectHFBloomAttentionPolicy.init_from_object(attn, attn_op_name=attn_op_name)
        if dtype == torch.float16:
            attn = attn.half()
        return attn.cuda()

    def gen_inputs(shape, dtype):
        bs, seq_len, hidden_size, num_heads = shape[:4]
        hidden_states = torch.randn((bs, seq_len, hidden_size), dtype=dtype, device="cuda")
        residual = torch.randn((bs, seq_len, hidden_size), dtype=dtype, device="cuda")
        alibi = torch.randn((bs * num_heads, 1, seq_len), dtype=dtype, device="cuda")
        attn_mask = torch.triu(
             torch.ones((seq_len, seq_len), device="cuda"),
             diagonal=1,
        )
        attn_mask = attn_mask[None, None, :, :].expand(bs, 1, -1, -1) > 0.5
        # (hidden_states, residual, alibi, attn_mask)
        return [hidden_states, residual, alibi, attn_mask.contiguous()]

    def zero_grad(mod, inputs):
        inputs[0].grad = None
        inputs[1].grad = None
        for param_name in ["query_key_value", "dense", "qkv", "out_proj"]:
            if hasattr(mod, param_name):
                getattr(mod, param_name).weight.grad = None
                getattr(mod, param_name).bias.grad = None

    # (batch, seq, hidden size, #head, vocab size)
    shapes = [
        (4, 2048, 1024, 16, 250880), # bloom-560m
    ]
    configs = [
        BenchConfig(
            partial(_init, attn_op_name=None, no_dropout=True),
            torch.float16,
            "HF",
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
            inputs_requires_grad=(True, True, False, ...),
        ),
    ]

    fun_attn = _init(shapes[0], configs[0].dtype, None, no_dropout=True)
    for name in ["native", "cutlass"]:
        fun_xf = _init(shapes[0], configs[0].dtype, name, no_dropout=True)
        InjectHFBloomAttentionPolicy.assign_params(fun_xf, fun_attn)
        config = BenchConfig(
            partial(_init, attn_op_name=name),
            torch.float16,
            name,
            not args.forward_only,
            gen_inputs=gen_inputs,
            zero_grad=zero_grad,
            inputs_requires_grad=(True, True, False, ...),
        )
        correct = check_correctness(
            shapes[0],
            fun_attn,
            fun_xf,
            config,
            desc=name,
            verbose=args.verbose,
        )
        if correct is not None:
            configs.append(config)

    if len(configs) == 1:
        logger.warning(f"Skip benchmark because no xFormers op is valid")
        return None

    return bench(
        shapes,
        configs,
        "HF Bloom Attention and xFormer Attention",
        verbose=args.verbose,
    )


def t5_attention(args):
    torch.manual_seed(42)
    if not is_available("transformers") or not is_available("xformers"):
        logger.warning("Skip attention because transformers or xformers is not available")
        return

    from transformers import AutoConfig
    from transformers.models.t5.modeling_t5 import T5Attention
    from ..inject.policy.t5 import InjectHFT5AttentionPolicy

    def _init(
        shape, dtype, attn_op_name, is_decoder, has_relative_attention_bias, no_dropout=False
    ):
        config = AutoConfig.from_pretrained("t5-small")
        config.is_encoder_decoder = False
        config.is_decoder = is_decoder
        config.d_model = shape[2]
        config.d_kv = shape[3]
        config.num_heads = shape[4]
        if no_dropout:
            config.dropout_rate = 0.0

        attn = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        if attn_op_name is not None:
            attn = InjectHFT5AttentionPolicy.init_from_object(attn, attn_op_name=attn_op_name)
        if dtype == torch.float16:
            attn = attn.half()
        return attn.cuda()

    def gen_inputs(shape, dtype, cross_attn=True):
        # (batch, seq, hidden size)
        inp_shape = shape[:3]
        hidden_states = torch.randn(inp_shape, dtype=dtype, device="cuda")
        attn_mask = torch.randn(inp_shape[0], 1, 1, inp_shape[1], dtype=dtype, device="cuda")
        if cross_attn:
            kv_states = torch.randn(inp_shape, dtype=dtype, device="cuda")
        else:
            kv_states = None
        return [hidden_states, attn_mask, kv_states]  # (hidden_states, mask, kv_states)

    def zero_grad(mod, inputs):
        inputs[0].grad = None
        for param_name in ["q", "k", "v", "o"]:
            if hasattr(mod, param_name):
                getattr(mod, param_name).weight.grad = None
                if getattr(mod, param_name).bias is not None:
                    getattr(mod, param_name).bias.grad = None

    # (batch, seq, hidden size, d_kv, n_head)
    shapes = [
        (1, 1024, 1024, 64, 16),
        # (1, 512, 1024, 64, 16)
        # (4, 1024, 512, 64, 16),  # t5-small
        # (4, 1024, 1024, 64, 16),  # t5-large
        # (4, 2048, 1024, 64, 16),
    ]

    # Encoder, SelfAttention: has_relative_attention_bias=True since the 2nd layer.
    # Decoder, SelfAttention: has_relative_attention_bias=True since the 2nd layer.
    # Decoder, CrossAttention: has_relative_attention_bias=False. kv_states is not None.
    for is_decoder in [False, True]:
        for cross_attn in [False, True]:
            if not is_decoder and cross_attn:
                # Cross attention only happens in decoder.
                continue

            # Only set to False for cross attention.
            has_relative_attention_bias = not cross_attn

            desc = (
                f"{'Decoder' if is_decoder else 'Encoder'}"
                + f"{'Cross' if cross_attn else 'Self'}Attention "
                + f"(rel bias: {has_relative_attention_bias})"
            )

            configs = [
                BenchConfig(
                    partial(
                        _init,
                        attn_op_name=None,
                        is_decoder=is_decoder,
                        has_relative_attention_bias=has_relative_attention_bias,
                        no_dropout=True,  # FIXME: dropout is not supported in xFormer kernels yet.
                    ),
                    torch.float16,
                    "HF (Attn)",
                    not args.forward_only,
                    gen_inputs=partial(gen_inputs, cross_attn=cross_attn),
                    zero_grad=zero_grad,
                ),
            ]

            # Check correctness.
            fun_attn = _init(
                shapes[0],
                configs[0].dtype,
                None,
                is_decoder,
                has_relative_attention_bias,
                no_dropout=True,
            )
            for name in ["cutlass"]: #"native" , "cutlass", "flshatt"
                fun_xf = _init(
                    shapes[0],
                    configs[0].dtype,
                    name,
                    is_decoder,
                    has_relative_attention_bias,
                    no_dropout=True,
                )
                InjectHFT5AttentionPolicy.assign_params(fun_xf, fun_attn, attn_op_name=name)

                config = BenchConfig(
                    partial(
                        _init,
                        attn_op_name=name,
                        is_decoder=is_decoder,
                        has_relative_attention_bias=has_relative_attention_bias,
                        no_dropout=False,
                    ),
                    torch.float16,
                    f"xFormers {name}",
                    not args.forward_only,
                    gen_inputs=partial(gen_inputs, cross_attn=cross_attn),
                    zero_grad=zero_grad,
                )
                correct = check_correctness(
                    shapes[0],
                    fun_attn,
                    fun_xf,
                    config,
                    desc=f"{desc} by xFormers ({name})",
                    verbose=args.verbose,
                )
                if correct is not None:
                    configs.append(config)

            if len(configs) == 1:
                logger.warning(f"Skip {desc} because no xFormers op is valid")
                break

            bench(
                shapes,
                configs,
                f"{desc}: HF T5Attention and xFormers",
                verbose=args.verbose,
            )
