"""Encoder specific injection policies."""
import torch
from torch import nn

from .base import ModuleInjectPolicy
from ..utils import get_arg, check_unsupported_arg
from ...ops.torchscript_ops import FusedDropoutAddLayerNorm


class InjectHFBertSelfAttentionPolicy(ModuleInjectPolicy):
    @staticmethod
    def gen_init_config_from_object(orig, **kwargs):
        args = {
            "hidden_size": orig.all_head_size,
            "num_attention_heads": orig.num_attention_heads,
            "is_decoder": False,
            "attn_pdrop": orig.attention_dropout.p if hasattr(orig, "attention_dropout") else orig.dropout.p,
            "resid_pdrop": 0,
            "attn_op_name": kwargs.get("attn_op_name", "cutlass"),
        }
        return args

    @staticmethod
    def gen_init_config_from_config(*args, **kwargs):
        config = args[0]
        new_args = {
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "is_decoder": False,
            "attn_pdrop": config.attention_probs_dropout_prob,
            "resid_pdrop": 0,
            "attn_op_name": kwargs.get("attn_op_name", "cutlass"),
        }
        return new_args

    @staticmethod
    def assign_params(this, orig, **kwargs):
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
    def inject_module(**kwargs):
        """The custom module to inject."""
        from ...ops.xformers_attn import GenericSelfAttention
        return GenericSelfAttention

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
            check_unsupported_arg("output_attentions", 6, args, kwargs, [None, False])
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


class InjectHFBertOutputPolicy(ModuleInjectPolicy):
    @staticmethod
    def gen_init_config_from_object(orig, **kwargs):
        args = {
            "hidden_size": orig.dense.out_features,
            "intermediate_size": orig.dense.in_features,
            "layer_norm_eps": orig.LayerNorm.eps,
            "hidden_dropout_prob": orig.dropout.p,
        }
        return args

    @staticmethod
    def gen_init_config_from_config(*args, **kwargs):
        config = args[0]
        new_args = {
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "layer_norm_eps": config.layer_norm_eps,
            "hidden_dropout_prob": config.hidden_dropout_prob,
        }
        return new_args

    @staticmethod
    def assign_params(this, orig, **kwargs):
        this.dense.weight = orig.dense.weight
        this.dense.bias = orig.dense.bias
        this.fused_op.layer_norm.weight = orig.LayerNorm.weight
        this.fused_op.layer_norm.bias = orig.LayerNorm.bias

    @staticmethod
    def target_modules():
        """A list of target modules to be injected."""
        import transformers.models.bert.modeling_bert

        return [(transformers.models.bert.modeling_bert, "BertOutput")]

    @staticmethod
    def inject_module(**kwargs):
        """The custom module to inject."""

        class FusedBertOutput(nn.Module):
            def __init__(self, intermediate_size, hidden_size, layer_norm_eps, hidden_dropout_prob):
                super().__init__()
                self.dense = nn.Linear(intermediate_size, hidden_size)
                self.fused_op = FusedDropoutAddLayerNorm(
                    hidden_size, hidden_dropout_prob, layer_norm_eps
                )

            def forward(
                self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
            ) -> torch.Tensor:
                hidden_states = self.dense(hidden_states)
                hidden_states = self.fused_op(hidden_states, input_tensor)
                return hidden_states

        return FusedBertOutput

    @staticmethod
    def load_state_dict_post_hook(state_dict):
        name_pairs = []
        replace_rules = [
            ("output.LayerNorm.gamma", "LayerNorm.gamma", "fused_op.layer_norm.weight"),
            ("output.LayerNorm.beta", "LayerNorm.beta", "fused_op.layer_norm.bias"),
        ]
        for name in state_dict.keys():
            new_name = None
            for rule in replace_rules:
                if rule[0] in name and "attention" not in name:
                    new_name = name.replace(rule[1], rule[2])
            if new_name is not None:
                name_pairs.append((name, new_name))

        for old_name, new_name in name_pairs:
            state_dict[new_name] = state_dict.pop(old_name)

        return state_dict
