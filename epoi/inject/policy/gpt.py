"""GPT specific injection policies."""
import torch

from .base import ModuleInjectPolicy
from ..utils import get_arg, check_unsupported_arg
from ...ops.torchscript_ops import FusedBiasGELU, FusedBiasNewGELU


def find_dropout_prob(config_or_mod):
    """A helper function to find the dropout probability
    of GPT models. config_or_mod could either be a model config, or an attention module.
    This function supports GPT-2, GPT-Neo and GPT-J implementations.
    """
    if hasattr(config_or_mod, "attention_dropout"):
        attn_pdrop = config_or_mod.attention_dropout
    elif hasattr(config_or_mod, "attn_pdrop"):
        attn_pdrop = config_or_mod.attn_pdrop
    elif hasattr(config_or_mod, "attn_dropout"):
        attn_pdrop = config_or_mod.attn_dropout
    elif hasattr(config_or_mod, "dropout"):
        attn_pdrop = config_or_mod.dropout
    else:
        raise ValueError("Cannot find attention dropout probability")

    if hasattr(config_or_mod, "resid_pdrop"):
        resid_pdrop = config_or_mod.resid_pdrop
    elif hasattr(config_or_mod, "resid_dropout"):
        resid_pdrop = config_or_mod.resid_dropout
    elif hasattr(config_or_mod, "resid_dropout"):
        resid_pdrop = config_or_mod.resid_dropout
    elif hasattr(config_or_mod, "dropout"):
        resid_pdrop = config_or_mod.dropout
    else:
        raise ValueError("Cannot find resid_pdrop or resid_dropout in config.")

    attn_pdrop = attn_pdrop.p if hasattr(attn_pdrop, "p") else attn_pdrop
    resid_pdrop = resid_pdrop.p if hasattr(resid_pdrop, "p") else resid_pdrop

    return attn_pdrop, resid_pdrop


class InjectHFGPTAttentionPolicy(ModuleInjectPolicy):
    @staticmethod
    def gen_init_config_from_object(orig, **kwargs):
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

        attn_pdrop, resid_pdrop = find_dropout_prob(orig)
        args = {
            "hidden_size": orig.embed_dim,
            "num_attention_heads": orig.num_heads,
            "is_decoder": True,
            "attn_pdrop": attn_pdrop,
            "resid_pdrop": resid_pdrop,
            "attn_op_name": kwargs.get("attn_op_name", "cutlass"),
            "fused_qkv": isinstance(orig, GPT2Attention),
        }
        return args

    @staticmethod
    def gen_init_config_from_config(*args, **kwargs):
        config = args[0]
        attn_pdrop, resid_pdrop = find_dropout_prob(config)
        new_args = {
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "is_decoder": True,
            "attn_pdrop": attn_pdrop,
            "resid_pdrop": resid_pdrop,
            "attn_op_name": kwargs.get("attn_op_name", "cutlass"),
            "fused_qkv": "GPT2" in config.architectures[0],
        }
        return new_args

    @staticmethod
    def assign_params(this, orig, **kwargs):
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

        if isinstance(orig, GPT2Attention):
            # GPT-2 uses fused QKV and Conv1D.
            requires_grad = orig.c_attn.weight.requires_grad
            this.qkv.weight = torch.nn.Parameter(
                orig.c_attn.weight.transpose(-1, 0).contiguous(), requires_grad=requires_grad
            )
            this.qkv.bias = orig.c_attn.bias
            this.out_proj.weight = torch.nn.Parameter(
                orig.c_proj.weight.transpose(-1, 0).contiguous(), requires_grad=requires_grad
            )
            this.out_proj.bias = orig.c_proj.bias
        else:
            this.query.weight = orig.q_proj.weight
            this.query.bias = orig.q_proj.bias
            this.key.weight = orig.k_proj.weight
            this.key.bias = orig.k_proj.bias
            this.value.weight = orig.v_proj.weight
            this.value.bias = orig.v_proj.bias
            this.out_proj.weight = orig.out_proj.weight
            this.out_proj.bias = orig.out_proj.bias

    @staticmethod
    def target_modules():
        """A list of target modules to be injected."""
        import transformers.models.gpt2.modeling_gpt2
        import transformers.models.gpt_neo.modeling_gpt_neo
        import transformers.models.gptj.modeling_gptj

        return [
            (transformers.models.gpt2.modeling_gpt2, "GPT2Attention"),
            (transformers.models.gpt_neo.modeling_gpt_neo, "GPTNeoSelfAttention"),
            (transformers.models.gptj.modeling_gptj, "GPTJAttention"),
        ]

    @staticmethod
    def inject_module(**kwargs):
        """The custom module to inject."""
        attn_op_name = kwargs.get("attn_op_name", "cutlass")
        from ...ops.xformers_attn import GenericSelfAttention
        from ...ops.flash_attention import FlashSelfAttention

        if attn_op_name == "triton":
            return FlashSelfAttention
        return GenericSelfAttention

    @staticmethod
    def gen_wrap_forward(orig_cls, forward):
        """
        Original forward signature of GPT-2:
         (hidden_states, layer_past, attention_mask, head_mask,
          encoder_hidden_states, encoder_attention_mask, use_cache, output_attentions)

        Original forward signature of GPT-Neo and GPT-J:
         (hidden_states, attention_mask, layer_past, head_mask, use_cache, output_attention)

        New forward signature:
         (hidden_states, attention_mask, layer_past, use_cache)
        """
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention as HFGPT2Attention

        if orig_cls is HFGPT2Attention:

            def wrapped_forward(*args, **kwargs):
                check_unsupported_arg("head_mask", 3, args, kwargs)
                check_unsupported_arg("encoder_hidden_states", 4, args, kwargs)
                check_unsupported_arg("encoder_attention_mask", 5, args, kwargs)
                check_unsupported_arg("output_attentions", 7, args, kwargs, [None, False])
                new_args = {
                    "hidden_states": get_arg("hidden_states", 0, args, kwargs),
                    "layer_past": get_arg("layer_past", 1, args, kwargs),
                    "attention_mask": get_arg("attention_mask", 2, args, kwargs),
                    "use_cache": get_arg("use_cache", 6, args, kwargs),
                }
                return forward(**new_args)

        else:

            def wrapped_forward(*args, **kwargs):
                check_unsupported_arg("head_mask", 3, args, kwargs)
                check_unsupported_arg("output_attentions", 5, args, kwargs, [None, False])
                new_args = {
                    "hidden_states": get_arg("hidden_states", 0, args, kwargs),
                    "attention_mask": get_arg("attention_mask", 1, args, kwargs),
                    "layer_past": get_arg("layer_past", 2, args, kwargs),
                    "use_cache": get_arg("use_cache", 4, args, kwargs),
                }
                return forward(**new_args)

        return wrapped_forward

    @staticmethod
    def load_state_dict_post_hook(state_dict):
        name_pairs = []
        replace_rules = [
            # GPT-2
            ("attn.c_attn.weight", "c_attn", "qkv"),
            ("attn.c_attn.bias", "c_attn", "qkv"),
            ("attn.c_proj", "c_proj", "out_proj"),
            # GPTNeo, GPTJ
            ("attn.c_attn.q_proj", "q_proj", "query"),
            ("attn.c_attn.k_proj", "k_proj", "key"),
            ("attn.c_attn.v_proj", "v_proj", "value"),
        ]
        for name in state_dict.keys():
            new_name = None
            for rule in replace_rules:
                if rule[0] in name:
                    new_name = name.replace(rule[1], rule[2])
            if new_name is not None:
                name_pairs.append((name, new_name))

        for old_name, new_name in name_pairs:
            param = state_dict.pop(old_name)
            if "attn.c_attn.weight" in old_name or "attn.c_proj.weight" in old_name:
                state_dict[new_name] = param.transpose(-1, 0).contiguous()
            else:
                state_dict[new_name] = param

        return state_dict


class InjectHFGPTMLPPolicy(ModuleInjectPolicy):
    @staticmethod
    def gen_init_config_from_object(orig):
        from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
        from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoMLP
        from transformers.models.gptj.modeling_gptj import GPTJMLP
        from transformers.activations import ACT2FN

        # Find the name of the activation function.
        act_fn_name = None
        for key, val in ACT2FN.items():
            if isinstance(val, tuple):
                val = val[0]
            if isinstance(orig.act, val):
                act_fn_name = key
                break
        else:
            raise NotImplementedError(f"Unsupported activation: {orig.act}")

        args = {
            "orig_act": act_fn_name,
            "resid_pdrop": orig.dropout.p,
        }

        # Fetch the intermediate size from weight shape.
        if isinstance(orig, GPT2MLP):
            # GPT2 uses legacy Conv1D with transposed weights.
            args["hidden_size"], args["intermediate_size"] = orig.c_fc.weight.shape
        elif isinstance(orig, GPTNeoMLP):
            args["intermediate_size"], args["hidden_size"] = orig.c_fc.weight.shape
        elif isinstance(orig, GPTJMLP):
            args["intermediate_size"], args["hidden_size"] = orig.fc_in.weight.shape

        return args

    @staticmethod
    def assign_params(this, orig, **kwargs):
        from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
        from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoMLP
        from transformers.models.gptj.modeling_gptj import GPTJMLP

        if isinstance(orig, GPT2MLP):
            # GPT2 uses legacy Conv1D with transposed weights.
            fc_names = ["c_fc", "c_proj"]
        elif isinstance(orig, GPTNeoMLP):
            fc_names = ["c_fc", "c_proj"]
        elif isinstance(orig, GPTJMLP):
            fc_names = ["fc_in", "fc_out"]

        requires_grad = getattr(orig, fc_names[0]).weight.requires_grad
        if isinstance(orig, GPT2MLP):
            # GPT2 uses legacy Conv1D with transposed weights.
            this.fc_in.weight = torch.nn.Parameter(
                orig.c_fc.weight.transpose(-1, 0).contiguous(), requires_grad=requires_grad
            )
            this.act.bias = orig.c_fc.bias
            this.fc_out.weight = torch.nn.Parameter(
                orig.c_proj.weight.transpose(-1, 0).contiguous(), requires_grad=requires_grad
            )
            this.fc_out.bias = orig.c_proj.bias
        else:
            this.fc_in.weight = getattr(orig, fc_names[0]).weight
            this.act.bias = getattr(orig, fc_names[0]).bias
            this.fc_out.weight = getattr(orig, fc_names[1]).weight
            this.fc_out.bias = getattr(orig, fc_names[1]).bias

    @staticmethod
    def target_modules():
        """A list of target modules to be injected."""
        import transformers.models.gpt2.modeling_gpt2
        import transformers.models.gpt_neo.modeling_gpt_neo
        import transformers.models.gptj.modeling_gptj

        return [
            (transformers.models.gpt2.modeling_gpt2, "GPT2MLP"),
            (transformers.models.gpt_neo.modeling_gpt_neo, "GPTNeoMLP"),
            (transformers.models.gptj.modeling_gptj, "GPTJMLP"),
        ]

    @staticmethod
    def inject_module(**kwargs):
        """The custom module to inject."""

        class FusedMLP(torch.nn.Module):
            """A wrapper MLP to make use of fused bias+gelu."""

            def __init__(self, hidden_size, intermediate_size, orig_act, resid_pdrop):
                super().__init__()
                if orig_act == "gelu":
                    self.fc_in = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
                    self.act = FusedBiasGELU(intermediate_size, prev_weight=self.fc_in.weight)
                elif orig_act == "gelu_new":
                    self.fc_in = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
                    self.act = FusedBiasNewGELU(intermediate_size, prev_weight=self.fc_in.weight)
                else:
                    raise NotImplementedError(f"Unsupported activation: {orig_act}")

                self.fc_out = torch.nn.Linear(intermediate_size, hidden_size)
                self.dropout = torch.nn.Dropout(resid_pdrop)

            def forward(self, hidden_states):
                hidden_states = self.fc_in(hidden_states)
                hidden_states = self.act(hidden_states)
                hidden_states = self.fc_out(hidden_states)
                hidden_states = self.dropout(hidden_states)
                return hidden_states

        return FusedMLP

    @staticmethod
    def gen_init_config_from_config(*args, **kwargs):
        intermediate_size = args[0]
        config = args[1]

        _, resid_pdrop = find_dropout_prob(config)
        new_args = {
            "hidden_size": config.hidden_size,
            "intermediate_size": intermediate_size,
            "orig_act": config.activation_function,
            "resid_pdrop": resid_pdrop,
        }
        return new_args

    @staticmethod
    def load_state_dict_post_hook(state_dict):
        name_pairs = []
        replace_rules = [
            # GPT-2, GPTNeo
            ("mlp.c_fc.weight", "c_fc", "fc_in"),
            ("mlp.c_fc.bias", "c_fc", "act"),
            ("mlp.c_proj", "c_proj", "fc_out"),
            # GPTJ
            ("mlp.fc_in.bias", "fc_in", "act"),
        ]
        is_gpt2 = False
        for name in state_dict.keys():
            new_name = None
            # a trick to see whether the target model is GPT-2...
            is_gpt2 |= ("attn.c_attn.weight" in name or "attn.qkv" in name)
            for rule in replace_rules:
                if rule[0] in name:
                    new_name = name.replace(rule[1], rule[2])
            if new_name is not None:
                name_pairs.append((name, new_name))

        for old_name, new_name in name_pairs:
            param = state_dict.pop(old_name)
            if is_gpt2 and ("mlp.c_fc.weight" in old_name or "mlp.c_proj.weight" in old_name):
                state_dict[new_name] = param.transpose(-1, 0).contiguous()
            else:
                state_dict[new_name] = param

        return state_dict
