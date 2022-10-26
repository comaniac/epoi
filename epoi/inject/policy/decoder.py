"""GPT specific injection policies."""
import torch

from .base import ModuleInjectPolicy
from ...ops.torchscript_ops import FusedBiasGELU, FusedBiasNewGELU


class InjectHFGPT2AttentionPolicy(ModuleInjectPolicy):
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
    def match(other):
        """Check if the other module matches the module that could be replaced."""
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

        return isinstance(other, GPT2Attention)

    @staticmethod
    def assign_params(this, orig):
        requires_grad = orig.c_attn.weight.requires_grad
        this.qkv.weight = torch.nn.Parameter(
            orig.c_attn.weight.transpose(-1, 0).contiguous(), requires_grad=requires_grad
        )
        this.qkv.bias = orig.c_attn.bias
        this.out_proj.weight = torch.nn.Parameter(
            orig.c_proj.weight.transpose(-1, 0).contiguous(), requires_grad=requires_grad
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
            check_unsupported_arg("output_attentions", 7, args, kwargs, False)
            new_args = {
                "hidden_states": get_arg("hidden_states", 0, args, kwargs),
                "layer_past": get_arg("layer_past", 1, args, kwargs),
                "attention_mask": get_arg("attention_mask", 2, args, kwargs),
                "use_cache": get_arg("use_cache", 6, args, kwargs),
            }
            return orig_forward(**new_args)

        this.forward = forward


class InjectHFGPTMLPPolicy(ModuleInjectPolicy):
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

    @staticmethod
    def init_impl(orig):
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
            args["intermediate_size"], args["hidden_size"] = orig.c_fc.weight.shape
        elif isinstance(orig, GPTNeoMLP):
            args["hidden_size"], args["intermediate_size"] = orig.c_fc.weight.shape
        elif isinstance(orig, GPTJMLP):
            args["hidden_size"], args["intermediate_size"] = orig.fc_in.weight.shape

        ret = InjectHFGPTMLPPolicy.FusedMLP(**args)
        return ret

    @staticmethod
    def match(other):
        """Check if the other module matches the module that could be replaced."""
        from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
        from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoMLP
        from transformers.models.gptj.modeling_gptj import GPTJMLP

        return other.__class__ in [GPT2MLP, GPTNeoMLP, GPTJMLP]

    @staticmethod
    def assign_params(this, orig):
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
