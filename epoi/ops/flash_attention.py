"""HuggingFace transformers with Flash Attention integration.
https://github.com/jfc4050/flash-attention/commit/f52868287ca9bd3ac1598dad6ce818358c1beafc
"""
from typing import Optional, Tuple
from functools import partial
import math

import torch
from torch import nn

try:
    from flash_attn.flash_attn_triton import flash_attn_func as flash_attn_triton_func
except ImportError:
    flash_attn_triton_func = None

try:
    from einops import rearrange
except ImportError:
    rearrange = None

ATTN_GLOBAL_MSGS = set()


def print_once(msg):
    if msg not in ATTN_GLOBAL_MSGS:
        print(msg, flush=True)
        ATTN_GLOBAL_MSGS.add(msg)


def flash_attn_triton_ref(
    q,
    k,
    v,
    bias=None,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_mask=None,
    upcast=True,
    reorder_ops=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads, head_dim)
        v: (batch_size, seqlen_k, nheads, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        bias: (batch_size, nheads, seqlen_q, seqlen_k)
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    assert softmax_scale is None, "softmax_scale is not supported"
    assert rearrange is not None, "einops is not installed"

    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if bias is not None:
        scores = (scores + bias).to(dtype=scores.dtype)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if causal:
        causal_mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device), 1
        )
        scores.masked_fill_(causal_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    return output.to(dtype=dtype_og)


class FlashAttentionTritonOp(nn.Module):
    """A wrapper module that processes HF attention mask to flash attention mask."""

    def __init__(self, attn_op_name, apply_causal_mask, scale=None):
        super().__init__()
        self.attn_op_name = attn_op_name
        self.apply_causal_mask = apply_causal_mask
        self.scale = scale
        if attn_op_name == "native":
            self.attn_fn = partial(
                flash_attn_triton_ref,
                query_padding_mask=None,
                key_padding_mask=None,
                dropout_mask=None,
                upcast=True,
                reorder_ops=False,
            )
        elif attn_op_name == "triton":
            if flash_attn_triton_func is None:
                raise RuntimeError("flash_attn is not installed")
            self.attn_fn = flash_attn_triton_func
        else:
            raise ValueError(f"Invalid attn_op_name: {attn_op_name}")

    def forward(self, query_layer, key_layer, value_layer, attention_mask, p):
        if attention_mask is not None:
            print_once(
                "WARNING: bias gradient is not supported yet. The given mask will be ignored"
            )
        ret = self.attn_fn(
            query_layer,
            key_layer,
            value_layer,
            None,  # bias
            self.apply_causal_mask,  # causal
            p,  # dropout_p
            self.scale,  # softmax_scale
        )
        ret = ret.to(query_layer.dtype)
        return ret


class FlashSelfAttention(nn.Module):
    """A HuggingFace self attention module with flash attention triton kernel.
    Note that this module has limited supports to specialized processing, documetned as follows:
    - Only support absolute positional embeddings.
    - Do not support cross attention.
    - Do not support head mask, encoder_attention_mask, and attention_output.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        is_decoder,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        attn_op_name="auto",
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

        self.attn_op = FlashAttentionTritonOp(attn_op_name, self.is_decoder)

    def reshape_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Copy from transpose_for_scores but without the transpose"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x

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
            key_layer = torch.cat((past_key, key_layer), dim=-2)
            value_layer = torch.cat((past_value, value_layer), dim=-2)

        context_layer = self.attn_op(
            query_layer.contiguous(),
            key_layer.contiguous(),
            value_layer.contiguous(),
            attention_mask,
            p=self.attn_pdrop,
        )
        context_layer = context_layer.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)

        if self.is_decoder:
            context_layer = self.out_proj(context_layer)
            context_layer = self.resid_dropout(context_layer)

        if use_cache:
            outputs = (context_layer, (key_layer, value_layer))
        else:
            outputs = (context_layer, None)
        return outputs
