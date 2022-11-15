from typing import Optional, Tuple, Union
from functools import partial
import math

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

try:
    import xformers
    import xformers.ops
except ImportError:
    xformers = None

ATTN_GLOBAL_MSGS = set()


def print_once(msg):
    if msg not in ATTN_GLOBAL_MSGS:
        print(msg, flush=True)
        ATTN_GLOBAL_MSGS.add(msg)


def pt_attention(q, k, v, attn_bias, p=0.0, weight_scaling=True):
    """The native PyTorch implementation of attention with the same signature as the
    FlashAttention implemented in xformers. This is used mainly to check the correctness
    of the xformers implementation, so do not change the functionality of this function.

    Note that weight_scaling is not supported in xFormers, so it should only be used for
    correctness checking.
    """
    assert xformers is not None, "xformers is not installed"

    def attention_bmk(q, k, v, attn_bias=None, p=0.0):
        if isinstance(attn_bias, xformers.ops.AttentionMask):
            attn_bias = attn_bias.to_tensor().to(q.dtype)
        if weight_scaling:
            q = q * (1.0 / q.shape[-1] ** 0.5)
        if attn_bias is None:
            attn = q @ k.transpose(-2, -1)
        else:
            # equivalent to (q @ k.transpose(-2, -1) + m)
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


class BertSelfAttention(nn.Module):
    """Modified from HuggingFace's BertSelfAttention to use the xformers attention op.
    Used for manual injection.
    """

    def __init__(self, config, position_embedding_type=None, attn_op_name="auto"):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple "
                f"of the number of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            raise NotImplementedError("Not implemented")

        self.is_decoder = config.is_decoder

        assert xformers is not None, "xformers is not installed"
        self.attn_op_name = attn_op_name
        if attn_op_name == "native":
            self.attn_op = pt_attention
        else:
            if attn_op_name == "cuda":
                op = xformers.ops.MemoryEfficientAttentionOp
            elif attn_op_name == "cutlass":
                op = xformers.ops.MemoryEfficientAttentionCutlassOp
            elif attn_op_name == "fctls_bflsh":
                op = xformers.ops.MemoryEfficientAttentionCutlassFwdFlashBwOp
            elif attn_op_name == "triton":
                op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
            else:
                # When op=None, the attention op will be automatically selected.
                op = None

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
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        assert head_mask is None, "head_mask is not supported for now"
        assert not output_attentions, "output_attentions is not supported for now"
        assert past_key_value is None, "past_key_value is not supported for now"

        mixed_query_layer = self.query(hidden_states)
        key_layer = self.reshape_for_scores(self.key(hidden_states))
        value_layer = self.reshape_for_scores(self.value(hidden_states))
        query_layer = self.reshape_for_scores(mixed_query_layer)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        assert not is_cross_attention, "cross attention is not supported for now"

        # The required attention mask shape is [batch_size x #heads, seq_length, seq_length];
        # while the input shape is [batch_size, 1, 1, seq_length].
        # In other words, we need to broadcast other dimensions manually.
        if attention_mask is not None:
            if self.attn_op_name in ["cutlass", "triton"]:
                print_once(
                    f"WARNING: Attention op {self.attn_op_name} does not support attention mask. "
                    "The mask will be ignored"
                )
                attention_mask = None
            else:
                attention_mask = self.layout_attention_mask(
                    attention_mask, self.num_attention_heads
                )

        context_layer = self.attn_op(
            query_layer, key_layer, value_layer, attention_mask, p=self.attention_probs_dropout_prob
        )
        context_layer = context_layer.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class GPT2Attention(nn.Module):
    """Modified from HuggingFace's GPT2SelfAttention to use the xformers attention op.
    Used for manual injection.
    """

    def __init__(self, config, is_cross_attention=False, layer_idx=None, attn_op_name="auto"):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} "
                f"and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention
        assert config.scale_attn_weights, "scale_attn_weights must be True"
        assert not is_cross_attention, "cross attention is not supported for now"

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn
        assert (
            not self.scale_attn_by_inverse_layer_idx
        ), "scale_attn_by_inverse_layer_idx is not supported for now"
        assert not self.reorder_and_upcast_attn, "reorder_and_upcast_attn is not supported for now"

        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_pdrop = config.attn_pdrop
        self.resid_drop = config.resid_pdrop
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        assert xformers is not None, "xformers is not installed"
        if attn_op_name == "native":
            self.attn_op = pt_attention
        else:
            if attn_op_name == "cuda":
                op = xformers.ops.MemoryEfficientAttentionOp
            elif attn_op_name == "cutlass":
                op = xformers.ops.MemoryEfficientAttentionCutlassOp
            elif attn_op_name == "fctls_bflsh":
                op = xformers.ops.MemoryEfficientAttentionCutlassFwdFlashBwOp
            elif attn_op_name == "triton":
                op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
            else:
                # When op=None, the attention op will be automatically selected.
                op = None

            self.attn_op = partial(xformers.ops.memory_efficient_attention, op=op)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor  # (batch, seq_length, head, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

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
                f"WARNING: GPT2Attention only supports builtin casual mask for now. "
                "The given attention mask is ignored."
            )
        assert encoder_hidden_states is None, "Cross attention is not supported yet"
        assert not self.reorder_and_upcast_attn, "reorder_and_upcast_attn is not supported for now"
        assert head_mask is None, "head_mask is not supported for now"

        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        seq_len = query.shape[1]
        attention_mask = xformers.ops.LowerTriangularMask(
            [1, seq_len, seq_len], dtype=query.dtype, device="cuda"
        )
        attn_output = self.attn_op(query, key, value, attention_mask, p=self.attn_pdrop)
        attn_weights = None

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            assert attn_weights is not None, "output attention is not supported for now"
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


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

        assert xformers is not None, "xformers is not installed"
        self.attn_op_name = attn_op_name
        if attn_op_name == "native":
            self.attn_op = pt_attention
        else:
            if attn_op_name == "cuda":
                op = xformers.ops.MemoryEfficientAttentionOp
            elif attn_op_name == "cutlass":
                op = xformers.ops.MemoryEfficientAttentionCutlassOp
            elif attn_op_name == "fctls_bflsh":
                op = xformers.ops.MemoryEfficientAttentionCutlassFwdFlashBwOp
            elif attn_op_name == "triton":
                op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
            else:
                # When op=None, the attention op will be automatically selected.
                op = None

            self.attn_op = partial(xformers.ops.memory_efficient_attention, op=op)

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
            key_layer = torch.cat((past_key, key_layer), dim=-2)
            value_layer = torch.cat((past_value, value_layer), dim=-2)

        if self.is_decoder:
            # Now we always apply casual mask for decoders, but we should also take
            # input attention mask into consideration.
            seq_len = query_layer.shape[1]
            attention_mask = xformers.ops.LowerTriangularMask(
                [1, seq_len, seq_len], dtype=query_layer.dtype, device="cuda"
            )
        else:
            if attention_mask is not None:
                if self.attn_op_name in ["cutlass", "fctls_bflsh", "triton"]:
                    print_once(
                        f"WARNING: Attention op {self.attn_op_name} does not support attention mask. "
                        "The mask will be ignored"
                    )
                    attention_mask = None
                else:
                    # Required attention mask shape: [batch_size x #heads, seq_length, seq_length].
                    # The input shape is [batch_size, 1, 1, seq_length].
                    # In other words, we need to broadcast other dimensions manually.
                    attention_mask = self.layout_attention_mask(
                        attention_mask, self.num_attention_heads
                    )

        context_layer = self.attn_op(
            query_layer, key_layer, value_layer, attention_mask, p=self.attn_pdrop
        )
        context_layer = context_layer.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        if self.is_decoder:
            context_layer = self.out_proj(context_layer)
            context_layer = self.resid_dropout(context_layer)

        if use_cache:
            outputs = (context_layer, (key_layer, value_layer))
        else:
            outputs = (context_layer, None)
        return outputs


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
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
        self.gradient_checkpointing = False

        assert xformers is not None, "xformers is not installed"
        self.attn_op_name = attn_op_name
        if attn_op_name == "native":
            self.attn_op = pt_attention
        else:
            if attn_op_name == "cuda":
                op = xformers.ops.MemoryEfficientAttentionOp
            elif attn_op_name == "cutlass":
                op = xformers.ops.MemoryEfficientAttentionCutlassOp
            elif attn_op_name == "fctls_bflsh":
                op = xformers.ops.MemoryEfficientAttentionCutlassFwdFlashBwOp
            elif attn_op_name == "triton":
                op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
            else:
                # When op=None, the attention op will be automatically selected.
                op = None

            self.attn_op = partial(xformers.ops.memory_efficient_attention, op=op)

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

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values

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
        batch_size, seq_length = hidden_states.shape[:2]
        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection without transpose"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim)

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
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=key_states.device,
                    dtype=key_states.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=key_states.device
                )

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = (
                    position_bias + mask
                )  # (batch_size, n_heads, seq_length, key_length)

                if self.is_decoder:
                    print_once(
                        f"WARNING: Now only supports builtin casual mask in decoder. "
                        "The position bias and given attention mask are ignored."
                    )
                    # FIXME
                    # seq_len = query_states.shape[1]
                    # new_mask = xformers.ops.LowerTriangularMask(
                    #     [1, seq_len, seq_len], dtype=query_states.dtype, device="cuda"
                    # )
                    new_mask = None
                else:
                    if self.attn_op_name in ["cutlass", "fctls_bflsh", "triton"]:
                        print_once(
                            f"WARNING: Attention op {self.attn_op_name} does not support "
                            "attention mask. The position bias and mask are ignored"
                        )
                        new_mask = None
                    else:
                        new_mask = position_bias
            else:
                new_mask = position_bias.repeat(query_states.shape[0], 1, 1, 1)

            if isinstance(new_mask, torch.Tensor):
                new_mask = new_mask.reshape((-1,) + new_mask.shape[2:])

        attn_output = self.attn_op(query_states, key_states, value_states, new_mask, p=self.dropout)
        attn_output = unshape(attn_output)
        attn_output = self.o(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        return outputs
