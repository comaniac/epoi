"""T5 specific injection policies."""

from .base import ModuleInjectPolicy
from ...ops.xformers_attn import T5Attention


class InjectHFT5AttentionPolicy(ModuleInjectPolicy):
    @staticmethod
    def gen_init_config_from_object(orig, **kwargs):
        args = dict(
            is_decoder=orig.is_decoder,
            relative_attention_num_buckets=orig.relative_attention_num_buckets,
            relative_attention_max_distance=orig.relative_attention_max_distance,
            d_model=orig.d_model,
            d_kv=orig.key_value_proj_dim,
            num_heads=orig.n_heads,
            dropout_rate=orig.dropout,
            has_relative_attention_bias=orig.has_relative_attention_bias,
            attn_op_name=kwargs.get("attn_op_name", "cutlass"),
        )
        return args

    @staticmethod
    def gen_init_config_from_config(*args, **kwargs):
        config = args[0]
        new_args = dict(
            is_decoder=config.is_decoder,
            relative_attention_num_buckets=config.relative_attention_num_buckets,
            relative_attention_max_distance=config.relative_attention_max_distance,
            d_model=config.d_model,
            d_kv=config.d_kv,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            has_relative_attention_bias=kwargs.get("has_relative_attention_bias", False),
            attn_op_name=kwargs.get("attn_op_name", "cutlass"),
        )
        return new_args

    @staticmethod
    def assign_params(this, orig, **kwargs):
        this.q.weight = orig.q.weight
        this.q.bias = orig.q.bias
        this.k.weight = orig.k.weight
        this.k.bias = orig.k.bias
        this.v.weight = orig.v.weight
        this.v.bias = orig.v.bias
        this.o.weight = orig.o.weight
        this.o.bias = orig.o.bias
        if hasattr(orig, "relative_attention_bias"):
            this.relative_attention_bias.weight = (
                orig.relative_attention_bias.weight
            )

    @staticmethod
    def target_modules():
        """A list of target modules to be injected."""
        import transformers.models.t5.modeling_t5

        return [(transformers.models.t5.modeling_t5, "T5Attention")]

    @staticmethod
    def inject_module(**kwargs):
        """The custom module to inject."""
        return T5Attention
