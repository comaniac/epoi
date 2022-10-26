"""Encoder specific injection policies."""
from .base import ModuleInjectPolicy
from ..utils import get_arg, check_unsupported_arg
from ...ops.xformers_attn import GenericSelfAttention


class InjectHFBertSelfAttentionPolicy(ModuleInjectPolicy):
    @staticmethod
    def init_impl(orig, attn_type="cutlass"):
        args = {
            "hidden_size": orig.all_head_size,
            "num_attention_heads": orig.num_attention_heads,
            "is_decoder": False,
            "attn_pdrop": orig.dropout.p,
            "resid_pdrop": 0,
        }
        ret = GenericSelfAttention(**args, attn_op_name=attn_type)
        return ret

    @staticmethod
    def match(other):
        """Check if the other module matches the module that could be replaced."""
        from transformers.models.bert.modeling_bert import BertSelfAttention

        return isinstance(other, BertSelfAttention)

    @staticmethod
    def assign_params(this, orig):
        this.query.weight = orig.query.weight
        this.query.bias = orig.query.bias
        this.key.weight = orig.key.weight
        this.key.bias = orig.key.bias
        this.value.weight = orig.value.weight
        this.value.bias = orig.value.bias

    @staticmethod
    def wrap_forward(this):
        """Original forward signature:
         (hidden_states, attention_mask, head_mask, encoder_hidden_states,
          encoder_attention_mask, past_key_value, output_attentions)
        New forward signature:
         (hidden_states, attention_mask, layer_past, use_cache)
        """
        orig_forward = this.forward

        def forward(*args, **kwargs):
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
            out = orig_forward(**new_args)
            if out[1] is None:
                return (out[0],)
            return out

        this.forward = forward
