from typing import Optional, Tuple, Union
from functools import partial
import math

import torch
import torch.nn as nn

try:
    import xformers
    from xformers import ops as xformers_ops
except ImportError:
    xformers = None

ATTN_GLOBAL_MSGS = set()


def print_once(msg):
    if msg not in ATTN_GLOBAL_MSGS:
        print(msg, flush=True)
        ATTN_GLOBAL_MSGS.add(msg)


def attention_native(q, k, v, attn_bias, p=0.0, scale=None):
    """The native PyTorch implementation of attention with the same signature as the
    FlashAttention implemented in xformers. This is used mainly to check the correctness
    of the xformers implementation, so do not change the functionality of this function.
    """
    assert xformers is not None, "xformers is not installed"
    assert q.ndim == 4

    def attention_bmk(q, k, v, attn_bias=None, p=0.0, scale=None):
        assert q.ndim == 3
        q = q.float()
        k = k.float()
        v = v.float()

        scale = scale if scale is not None else (1 / q.shape[-1] ** 0.5)
        q = q * scale

        attn = q @ k.transpose(-2, -1)
        if attn_bias is not None:
            if attn_bias.ndim == 4:
                assert q.shape[0] == attn_bias.shape[0] * attn_bias.shape[1]
                attn_bias = attn_bias.reshape([-1, *attn_bias.shape[2:]])
            attn = attn + attn_bias.float()
        attn = attn.softmax(-1).to(q.dtype)
        if p > 0:
            attn = torch.nn.functional.dropout(attn, p=p)
        return attn @ v

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape([t.shape[0] * t.shape[2], t.shape[1], t.shape[3]])

    if isinstance(attn_bias, xformers.ops.AttentionBias):
        attn_bias = attn_bias.materialize(
            (q.shape[0], q.shape[2], q.shape[1], k.shape[1]),
            device=q.device,
            dtype=q.dtype,
        ).reshape([q.shape[0] * q.shape[2], q.shape[1], k.shape[1]])
    out = attention_bmk(T(q), T(k), T(v), attn_bias, p, scale=scale)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))


def get_attn_op_by_name(attn_name):
    ops = [
        (xformers_ops.fmha.cutlass.FwOp, xformers_ops.fmha.cutlass.BwOp),
        (xformers_ops.fmha.flash.FwOp, xformers_ops.fmha.flash.BwOp),
        (xformers_ops.fmha.triton.FwOp, xformers_ops.fmha.triton.BwOp),
        (xformers_ops.fmha.small_k.FwOp, xformers_ops.fmha.small_k.BwOp),
    ]
    if attn_name is None or attn_name in {"native", "auto"}:
        return None
    for op in ops:
        if f"{attn_name}F" == op[0].NAME:
            return op
    raise ValueError(f"Unknown attention op name: {attn_name}")


class MemoryEfficientAttentionOp(nn.Module):
    """A wrapper module that processes HF attention mask to xformers attention mask."""

    def __init__(self, attn_op_name, apply_causal_mask, scale=None):
        super().__init__()
        assert xformers is not None, "xformers is not installed"
        self.attn_op_name = attn_op_name
        self.apply_causal_mask = apply_causal_mask

        if self.attn_op_name == "native":
            self.attn_fn = partial(attention_native, scale=scale)
        else:
            # When op=None, the attention op will be automatically selected.
            self.op = get_attn_op_by_name(attn_op_name)
            self.attn_fn = partial(xformers_ops.memory_efficient_attention, op=self.op, scale=scale)

    def forward(self, query_layer, key_layer, value_layer, attention_mask, p):
        if self.apply_causal_mask:
            attn_bias = xformers_ops.fmha.attn_bias.LowerTriangularMask()
            if attention_mask is not None:
                attn_bias = attn_bias.add_bias(attention_mask)
        else:
            attn_bias = attention_mask

        ret = self.attn_fn(query_layer, key_layer, value_layer, attn_bias, p=p)
        ret = ret.to(query_layer.dtype)
        return ret


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

        self.attn_op = MemoryEfficientAttentionOp(attn_op_name, self.is_decoder)

    @staticmethod
    def layout_attention_mask(mask, num_attention_heads):
        # (B, 1, 1, S) -> (B, H, S, S)
        mask = mask.repeat(1, num_attention_heads, mask.shape[-1], 1)
        return mask.contiguous()

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

        if attention_mask is not None:
            # Required attention mask shape: [batch_size, #heads, seq_length, seq_length].
            # The input shape is [batch_size, 1, 1, seq_length].
            # In other words, we need to broadcast other dimensions manually.
            attention_mask = self.layout_attention_mask(attention_mask, self.num_attention_heads)

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


class BertSelfAttentionWithXF(nn.Module):
    """A wrapper of generic attention to align the interface of HF BertSelfAttention
    for injection without policy.
    """

    def __init__(self, config, position_embedding_type=None, attn_op_name="auto"):
        super().__init__()
        from ..inject.policy.bert import InjectHFBertSelfAttentionPolicy

        kwargs = InjectHFBertSelfAttentionPolicy.gen_init_config_from_config(
            config, position_embedding_type=position_embedding_type, attn_op_name=attn_op_name
        )
        self.generic_attn = GenericSelfAttention(**kwargs)
        self.generic_attn.forward = InjectHFBertSelfAttentionPolicy.gen_wrap_forward(
            None, self.generic_attn.forward
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        return self.generic_attn(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )


class GPT2AttentionWithXF(nn.Module):
    """A wrapper of generic attention to align the interface of HF GPT2Attention
    for injection without policy.
    """

    def __init__(self, config, is_cross_attention=False, layer_idx=None, attn_op_name="cutlass"):
        super().__init__()
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention as HFGPT2Attention
        from ..inject.policy.gpt import InjectHFGPTAttentionPolicy

        kwargs = InjectHFGPTAttentionPolicy.gen_init_config_from_config(
            config,
            is_cross_attention=is_cross_attention,
            layer_idx=layer_idx,
            attn_op_name=attn_op_name,
        )
        self.generic_attn = GenericSelfAttention(**kwargs)
        self.generic_attn.forward = InjectHFGPTAttentionPolicy.gen_wrap_forward(
            HFGPT2Attention, self.generic_attn.forward
        )

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if attention_mask is not None:
            print_once(
                "WARNING: Tensor type attention mask is not supported in GPT-2 yet. "
                "The given mask will be ignored and built-in causal mask will be applied"
            )

        return self.generic_attn(
            hidden_states,
            layer_past,
            None,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            use_cache,
            output_attentions,
        )


class RelativeBias(nn.Module):
    def __init__(
        self, relative_attention_num_buckets, relative_attention_max_distance, n_heads, is_decoder
    ):
        super().__init__()
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.n_heads = n_heads
        self.is_decoder = is_decoder
        self.embeddings = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def forward(self, query_length, key_length, device):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.embeddings(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values


class ZeroBiasLike(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.gradient_checkpointing = False  # TODO

    def forward(self, query_length, key_length, ref):
        bias = torch.zeros(
            (1, self.n_heads, query_length, key_length),
            device=ref.device,
            dtype=ref.dtype,
        )
        if self.gradient_checkpointing and self.training:
            bias.requires_grad = True
        return bias


class T5Attention(nn.Module):
    """Modified from HuggingFace's T5Attention to use xformers' attention ops"""

    def __init__(
        self,
        is_decoder,
        relative_attention_num_buckets,
        relative_attention_max_distance,
        d_model,
        d_kv,
        num_heads,
        dropout_rate,
        has_relative_attention_bias=False,
        attn_op_name="auto",
    ):
        super().__init__()
        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.d_model = d_model
        self.key_value_proj_dim = d_kv
        self.n_heads = num_heads
        self.dropout = dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.query = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.key = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.value = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.out = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = RelativeBias(
                self.relative_attention_num_buckets,
                self.relative_attention_max_distance,
                self.n_heads,
                self.is_decoder,
            )
        else:
            self.zero_bias_like = ZeroBiasLike(self.n_heads)
        self.gradient_checkpointing = False

        assert xformers is not None, "xformers is not installed"
        self.attn_op_name = attn_op_name
        self.attn_op = MemoryEfficientAttentionOp(attn_op_name, False, scale=1.0)

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence
        (provided by key_value_states).
        """
        assert layer_head_mask is None, "layer_head_mask is not supported"
        assert not output_attentions, "output_attentions is not supported"
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        seq_length = hidden_states.shape[1]
        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection without transpose"""
            new_x_shape = states.size()[:-1] + (self.n_heads, self.key_value_proj_dim)
            return states.view(new_x_shape)

        def unshape(states):
            """reshape"""
            states = states.contiguous()
            return states.view(states.size()[:-2] + (-1,))

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            self.query(hidden_states)
        )  # (batch_size, seq_length, n_heads, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.key,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.value,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = self.zero_bias_like(real_seq_length, key_length, hidden_states)
            else:
                position_bias = self.relative_attention_bias(
                    real_seq_length, key_length, hidden_states.device
                )

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = (
                    position_bias + mask
                )  # (batch_size, n_heads, seq_length, key_length)
                new_mask = position_bias
            else:
                new_mask = position_bias.repeat(query_states.shape[0], 1, 1, 1)
        else:
            new_mask = position_bias
        new_mask = new_mask.contiguous()

        attn_output = self.attn_op(query_states, key_states, value_states, new_mask, p=self.dropout)
        attn_output = unshape(attn_output)
        attn_output = self.out(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        return outputs
