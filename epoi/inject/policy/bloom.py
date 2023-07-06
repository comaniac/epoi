"""Bloom specific injection policies."""
import torch

from .base import ModuleInjectPolicy
from ...ops.xformers_attn import BloomAttentionWithXF
from ...ops.torchscript_ops import FusedBiasGELU


class InjectHFBloomAttentionPolicy(ModuleInjectPolicy):
    @staticmethod
    def gen_init_config_from_object(orig, **kwargs):
        args = {
            "hidden_size": orig.hidden_size,
            "num_attention_heads": orig.num_heads,
            "attn_pdrop": orig.attention_dropout.p,
            "resid_pdrop": orig.hidden_dropout,
            "attn_op_name": kwargs.get("attn_op_name", "cutlass"),
        }
        return args

    @staticmethod
    def gen_init_config_from_config(*args, **kwargs):
        config = args[0]
        new_args = {
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.n_head,
            "attn_pdrop": config.attention_dropout,
            "resid_pdrop": config.hidden_dropout,
            "attn_op_name": kwargs.get("attn_op_name", "cutlass"),
        }
        return new_args

    @staticmethod
    def assign_params(this, orig, **kwargs):
        this.qkv.weight = orig.query_key_value.weight
        this.qkv.bias = orig.query_key_value.bias
        this.out_proj.weight = orig.dense.weight
        this.out_proj.bias = orig.dense.bias

    @staticmethod
    def target_modules():
        """A list of target modules to be injected."""
        import transformers.models.bloom.modeling_bloom
        return [(transformers.models.bloom.modeling_bloom, "BloomAttention")]

    @staticmethod
    def inject_module(**kwargs):
        """The custom module to inject."""
        return BloomAttentionWithXF

    @staticmethod
    def load_state_dict_post_hook(state_dict):
        name_pairs = []
        replace_rules = [
            ("self_attention.query_key_value", "query_key_value", "qkv"),
            ("self_attention.dense", "dense", "out_proj"),
        ]
        for name in state_dict.keys():
            new_name = None
            for rule in replace_rules:
                if rule[0] in name:
                    new_name = name.replace(rule[1], rule[2])
            if new_name is not None:
                name_pairs.append((name, new_name))

        for old_name, new_name in name_pairs:
            state_dict[new_name] = state_dict.pop(old_name)

        return state_dict


class InjectHFBloomMLPPolicy(ModuleInjectPolicy):
    @staticmethod
    def gen_init_config_from_object(orig):
        hidden_size = orig.dense_h_to_4h.weight.shape[1]
        args = {
            "hidden_size": hidden_size,
            "hidden_dropout": orig.hidden_dropout,
        }
        return args

    @staticmethod
    def assign_params(this, orig, **kwargs):
        this.dense_h_to_4h.weight = orig.dense_h_to_4h.weight
        this.act.bias = orig.dense_h_to_4h.bias
        this.dense_4h_to_h.weight = orig.dense_4h_to_h.weight
        this.dense_4h_to_h.bias = orig.dense_4h_to_h.bias

    @staticmethod
    def target_modules():
        """A list of target modules to be injected."""
        import transformers.models.bloom.modeling_bloom
        return [(transformers.models.bloom.modeling_bloom, "BloomMLP")]

    @staticmethod
    def inject_module(**kwargs):
        """The custom module to inject."""

        class FusedMLP(torch.nn.Module):
            """A wrapper MLP to make use of fused bias+gelu."""

            def __init__(self, hidden_size, hidden_dropout):
                super().__init__()
                self.dense_h_to_4h = torch.nn.Linear(hidden_size, 4 * hidden_size, bias=False)
                self.act = FusedBiasGELU(4 * hidden_size, prev_weight=self.dense_h_to_4h.weight)
                self.dense_4h_to_h = torch.nn.Linear(4 * hidden_size, hidden_size)
                self.dropout = torch.nn.Dropout(hidden_dropout)

            def forward(self, hidden_states, residual):
                hidden_states = self.dense_h_to_4h(hidden_states)
                hidden_states = self.act(hidden_states)
                hidden_states = self.dense_4h_to_h(hidden_states)
                hidden_states = self.dropout(hidden_states) + residual
                return hidden_states

        return FusedMLP

    @staticmethod
    def gen_init_config_from_config(*args, **kwargs):
        config = args[0]
        new_args = {
            "hidden_size": config.hidden_size,
            "hidden_dropout": config.hidden_dropout,
        }
        return new_args

    @staticmethod
    def load_state_dict_post_hook(state_dict):
        name_pairs = []
        replace_rules = [
            ("mlp.dense_h_to_4h.bias", "dense_h_to_4h", "act"),
        ]
        for name in state_dict.keys():
            new_name = None
            for rule in replace_rules:
                if rule[0] in name:
                    new_name = name.replace(rule[1], rule[2])
            if new_name is not None:
                name_pairs.append((name, new_name))

        for old_name, new_name in name_pairs:
            state_dict[new_name] = state_dict.pop(old_name)

        return state_dict
