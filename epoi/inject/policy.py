"""Model specific injection policies."""
import torch

from .utils import get_arg, check_unsupported_arg
from ..ops.xformers_attn import GenericSelfAttention


class ModuleInjectPolicy:
    @classmethod
    def init(cls, orig, **kwargs):
        """Initialize an instance to inject."""
        ret = cls.init_impl(orig, **kwargs)
        cls.wrap_forward(ret)
        return ret

    @staticmethod
    def init_impl(orig, **kwargs):
        """Initialize an instance to inject."""
        raise NotImplementedError()

    @staticmethod
    def assign_params(this, orig):
        """Assign the parameters in the original module to the injected module."""
        raise NotImplementedError()

    @staticmethod
    def wrap_forward(this):
        """Wrap the original module's forward method to deal with inconsistent I/O layouts."""


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
            check_unsupported_arg("output_attentions", 6, args, kwargs)
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


class InjectHFGPT2SelfAttentionPolicy(ModuleInjectPolicy):
    @staticmethod
    def init_impl(orig, attn_type="cutlass"):
        args = {
            "hidden_size": orig.embed_dim,
            "num_attention_heads": orig.num_heads,
            "is_decoder": True,
            "attn_pdrop": orig.attn_dropout.p,
            "resid_pdrop": orig.resid_dropout.p,
        }
        ret = GenericSelfAttention(**args, attn_op_name=attn_type, fused_qkv=True)
        return ret

    @staticmethod
    def assign_params(this, orig):
        requires_grad = orig.c_attn.weight.requires_grad
        this.qkv.weight = torch.nn.Parameter(
            orig.c_attn.weight.transpose(-1, 0), requires_grad=requires_grad
        )
        this.qkv.bias = orig.c_attn.bias
        this.out_proj.weight = torch.nn.Parameter(
            orig.c_proj.weight.transpose(-1, 0), requires_grad=requires_grad
        )
        this.out_proj.bias = orig.c_proj.bias

    @staticmethod
    def wrap_forward(this):
        """Original forward signature:
         (hidden_states, layer_past, attention_mask, head_mask,
          encoder_hidden_states, encoder_attention_mask, use_cache, output_attentions)
        New forward signature:
         (hidden_states, attention_mask, layer_past, use_cache)
        """
        orig_forward = this.forward

        def forward(*args, **kwargs):
            check_unsupported_arg("head_mask", 3, args, kwargs)
            check_unsupported_arg("encoder_hidden_states", 4, args, kwargs)
            check_unsupported_arg("encoder_attention_mask", 5, args, kwargs)
            check_unsupported_arg("output_attentions", 7, args, kwargs)
            new_args = {
                "hidden_states": get_arg("hidden_states", 0, args, kwargs),
                "layer_past": get_arg("layer_past", 1, args, kwargs),
                "attention_mask": get_arg("attention_mask", 2, args, kwargs),
                "use_cache": get_arg("use_cache", 6, args, kwargs),
            }
            return orig_forward(**new_args)

        this.forward = forward
