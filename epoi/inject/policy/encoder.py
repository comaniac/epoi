"""Encoder specific injection policies."""
from hashlib import new
from .base import ModuleInjectPolicy
from ..utils import get_arg, check_unsupported_arg
from ...ops.xformers_attn import GenericSelfAttention


class InjectHFBertSelfAttentionPolicy(ModuleInjectPolicy):
    @staticmethod
    def gen_init_config_from_object(orig):
        args = {
            "hidden_size": orig.all_head_size,
            "num_attention_heads": orig.num_attention_heads,
            "is_decoder": False,
            "attn_pdrop": orig.dropout.p,
            "resid_pdrop": 0,
            "attn_op_name": "cutlass",
        }
        return args

    @staticmethod
    def assign_params(this, orig):
        this.query.weight = orig.query.weight
        this.query.bias = orig.query.bias
        this.key.weight = orig.key.weight
        this.key.bias = orig.key.bias
        this.value.weight = orig.value.weight
        this.value.bias = orig.value.bias

    @staticmethod
    def target_modules():
        """A list of target modules to be injected."""
        import transformers.models.bert.modeling_bert

        return [(transformers.models.bert.modeling_bert, "BertSelfAttention")]

    @staticmethod
    def inject_module():
        """The custom module to inject."""
        return GenericSelfAttention

    @staticmethod
    def gen_init_config_from_config(*args, **kwargs):
        config = args[0]
        new_args = {
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "is_decoder": False,
            "attn_pdrop": config.attention_probs_dropout_prob,
            "resid_pdrop": 0,
            "attn_op_name": "cutlass",
        }
        return new_args

    @staticmethod
    def gen_wrap_forward(orig_cls, forward):
        """
        Original forward signature:
         (hidden_states, attention_mask, head_mask, encoder_hidden_states,
          encoder_attention_mask, past_key_value, output_attentions)
        New forward signature:
         (hidden_states, attention_mask, layer_past, use_cache)
        """

        def wrapped_forward(*args, **kwargs):
            check_unsupported_arg("head_mask", 2, args, kwargs)
            check_unsupported_arg("encoder_hidden_states", 3, args, kwargs)
            check_unsupported_arg("encoder_attention_mask", 4, args, kwargs)
            check_unsupported_arg("output_attentions", 6, args, kwargs, False)
            new_args = {
                "hidden_states": get_arg("hidden_states", 0, args, kwargs),
                "attention_mask": get_arg("attention_mask", 1, args, kwargs),
                "layer_past": get_arg("past_key_value", 5, args, kwargs),
                "use_cache": False,
            }
            out = forward(**new_args)
            if out[1] is None:
                return (out[0],)
            return out

        return wrapped_forward
