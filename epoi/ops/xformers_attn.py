from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

try:
    import xformers
    import xformers.ops
except ImportError:
    xformers = None


def pt_attention(q, k, v, attn_bias, p=0.0):
    """The native PyTorch implementation of attention with the same signature as the
    FlashAttention implemented in xformers. This is used mainly to check the correctness
    of the xformers implementation, so do not change the functionality of this function.
    """
    assert xformers is not None, "xformers is not installed"

    def attention_bmk(q, k, v, attn_bias=None, p=0.0):
        if isinstance(attn_bias, xformers.ops.AttentionMask):
            attn_bias = attn_bias.to_tensor().to(q.dtype)
        q = q * (1.0 / q.shape[-1] ** 0.5)
        if attn_bias is None:
            attn = q @ k.transpose(-2, -1)
        else:
            # equivalent to (q @ k.transpose(-2, -1) + m).softmax(-1) @ v
            # but faster, and is what is used in PyTorch now
            attn = torch.baddbmm(attn_bias, q, k.transpose(-2, -1))
        attn = attn.softmax(-1)
        if p > 0:
            attn = torch.nn.functional.dropout(attn, p=p)
        return attn @ v

    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape([t.shape[0] * t.shape[2], t.shape[1], t.shape[3]])

    out = attention_bmk(T(q), T(k), T(v), attn_bias, p)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))


class GenericSelfAttention(nn.Module):
    """A generic self attention module to use the xformers attention op.
    Note that this module has limited supports to specialized processing, documetned as follows:
    - Only support absolute positional embeddings.
    - Do not support cross attention.
    - Do not support head mask, encoder_attention_mask, and output attention.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        is_decoder,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        attn_op_name="cutlass",
        fused_qkv=False,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple "
                f"of the number of attention heads ({num_attention_heads})"
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.fused_qkv = fused_qkv
        if fused_qkv:
            self.qkv = nn.Linear(hidden_size, 3 * self.all_head_size)
        else:
            self.query = nn.Linear(hidden_size, self.all_head_size)
            self.key = nn.Linear(hidden_size, self.all_head_size)
            self.value = nn.Linear(hidden_size, self.all_head_size)

        self.is_decoder = is_decoder
        self.attn_pdrop = attn_pdrop

        if self.is_decoder:
            self.out_proj = nn.Linear(hidden_size, hidden_size)
            self.resid_dropout = nn.Dropout(resid_pdrop)

        assert xformers is not None, "xformers is not installed"
        if attn_op_name is None:
            self.attn_op = pt_attention
        else:
            if attn_op_name == "vanilla":
                op = xformers.ops.MemoryEfficientAttentionOp
            elif attn_op_name == "cutlass":
                op = xformers.ops.MemoryEfficientAttentionCutlassOp
            elif attn_op_name == "triton":
                op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
            else:
                raise ValueError(f"Unknown attn_op_name {attn_op_name}")

            self.attn_op = lambda q, k, v, m, p: xformers.ops.memory_efficient_attention(
                q, k, v, m, p, op=op
            )

    def reshape_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Copy from transpose_for_scores but without the transpose"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x

    @staticmethod
    def layout_attention_mask(mask, num_attention_heads):
        # (B, 1, 1, S) -> (B, S)
        mask = mask.squeeze()
        # (B, S) -> (B, 1, S)
        mask = mask.reshape((mask.shape[0], 1, mask.shape[1]))
        # (B, 1, S) -> (B x H, S, S)
        mask = mask.repeat(num_attention_heads, mask.shape[2], 1)
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor]:
        if self.fused_qkv:
            query_layer, key_layer, value_layer = self.qkv(hidden_states).split(
                self.hidden_size, dim=2
            )
        else:
            query_layer = self.query(hidden_states)
            key_layer = self.key(hidden_states)
            value_layer = self.value(hidden_states)
        query_layer = self.reshape_for_scores(query_layer)
        key_layer = self.reshape_for_scores(key_layer)
        value_layer = self.reshape_for_scores(value_layer)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key_layer), dim=-2)
            value = torch.cat((past_value, value_layer), dim=-2)

        if self.is_decoder:
            # Now we always apply casual mask for decoders, but we should also take
            # input attention mask into consideration.
            seq_len = query_layer.shape[1]
            attention_mask = xformers.ops.LowerTriangularMask(
                [1, seq_len, seq_len], dtype=query_layer.dtype, device="cuda"
            )
        else:
            # The required attention mask shape is [batch_size x #heads, seq_length, seq_length];
            # while the input shape is [batch_size, 1, 1, seq_length].
            # In other words, we need to broadcast other dimensions manually.
            attention_mask = self.layout_attention_mask(attention_mask, self.num_attention_heads)

        context_layer = self.attn_op(
            query_layer, key_layer, value_layer, attention_mask, p=self.attn_pdrop
        )
        context_layer = context_layer.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        if self.is_decoder:
            context_layer = self.out_proj(context_layer)
            context_layer = self.resid_dropout(context_layer)

        present = (key, value) if use_cache else None
        outputs = (context_layer, present)
        return outputs
