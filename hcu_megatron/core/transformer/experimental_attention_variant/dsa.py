# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import math
from functools import wraps
from scipy.linalg import hadamard
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import fused_qk_topk_naive


_HADAMARD_CACHE = {}

def get_hadamard_matrix(dim, device=None, dtype=torch.float32):
    """Fetch cached Hadamard matrix."""
    global _HADAMARD_CACHE
    key = (dim, str(device), str(dtype))

    if key not in _HADAMARD_CACHE:
        log_dim = math.ceil(math.log2(dim))
        dim_padded = 2 ** log_dim
        h = hadamard(dim_padded, dtype=float)
        _HADAMARD_CACHE[key] = torch.tensor(h, dtype=dtype, device=device)
    return _HADAMARD_CACHE[key]


def hadamard_transform_optimized(x, scale=1.0):
    """
    x: (..., dim) - dim is set to 128
    """
    x_shape = x.shape
    dim = x.shape[-1]

    h_matrix = get_hadamard_matrix(dim, x.device, x.dtype)

    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim

    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))

    out = F.linear(x, h_matrix)
    if scale != 1.0:
        out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16

    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."

    return hadamard_transform_optimized(x, scale=hidden_size**-0.5)


class DSAIndexer():
    """
    DSA Lightning Indexer for DeepSeek Sparse Attention.

    Computes index scores to identify the top-k most relevant key-value pairs for each query in
    sparse attention.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L431-L480
    """
    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All computations before topk."""
        # =========================================
        # Prepare RoPE params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, x, self.config, packed_seq_params
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)

        # =========================================
        # Gather inputs if sp is enabled
        # =========================================
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
            qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

        if (
            packed_seq_params is not None
            and getattr(packed_seq_params, "qkv_format", None) == "thd"
        ):
            if x.dim() == 3 and x.size(1) == 1:
                x = x.squeeze(1)
            if qr.dim() == 3 and qr.size(1) == 1:
                qr = qr.squeeze(1)

            if x.dim() == 2:
                x, _ = _unpack_thd_to_sbh(
                    x, packed_seq_params.cu_seqlens_q, packed_seq_params.max_seqlen_q
                )
                qr, _ = _unpack_thd_to_sbh(
                    qr, packed_seq_params.cu_seqlens_q, packed_seq_params.max_seqlen_q
                )
                packed_seq_params = None

        # =========================================
        # Get sequence length and batch size
        # =========================================
        seqlen, bsz, _ = x.size()

        # =========================================
        # Prepare RoPE params after unpack
        # =========================================
        rotary_seq_len = seqlen
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)

        # =========================================
        # q linear and apply rope to q
        # =========================================
        # [seqlen, batch, q_lora_rank] -> [seqlen, batch, index_n_heads * index_head_dim]
        q, _ = self.linear_wq_b(qr)
        # [seqlen, batch, index_n_heads * index_head_dim]
        #   -> [seqlen, batch, index_n_heads, index_head_dim]
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
        q = self._apply_rope(q, rotary_pos_emb, mscale)

        # =========================================
        # k linear and apply rope to k
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_head_dim]
        k, _ = self.linear_wk(x)
        k = self.k_norm(k)
        # [seqlen, batch, index_head_dim] -> [seqlen, batch, 1, index_head_dim]
        k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
        k = self._apply_rope(k, rotary_pos_emb, mscale)
        # [seqlen, batch, 1, index_head_dim] -> [seqlen, batch, index_head_dim]
        k = k.reshape(seqlen, bsz, self.index_head_dim)

        # =========================================
        # Rotate activation
        # =========================================
        q = rotate_activation(q)
        k = rotate_activation(k)

        # =========================================
        # Prepare weights for index scores
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_n_heads]
        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale

        return q, k, weights

    def forward_with_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for DSA Indexer that returns both index scores and top-k indices.

        This is used when KL loss is enabled to compare indexer scores with true attention scores.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            index_scores: Index scores [batch, seqlen, seqlen].
            topk_indices: Top-k indices [batch, seqlen, index_topk].
        """
        # [seqlen, batch, index_n_heads * index_head_dim]
        # [seqlen, batch, index_head_dim]
        # [seqlen, batch, index_n_heads]
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)

        # [batch, seqlen, seqlen], [batch, seqlen, index_topk]
        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, self.index_topk, mask)

        return index_scores, topk_indices


def _unpack_thd_to_sbh(tensor: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int):
    """Unpack THD packed tensor to SBHD/SBH style tensor.

    Input shape:
      [total_tokens, ...]
    Output shape:
      [max_seqlen, batch, ...]
    """
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device="cpu", dtype=torch.long)
    batch = int(lengths.numel())
    unpacked = tensor.new_zeros((max_seqlen, batch, *tensor.shape[1:]))
    for batch_idx, length in enumerate(lengths.tolist()):
        if length <= 0:
            continue
        start = int(cu_seqlens[batch_idx].item())
        end = start + length
        unpacked[:length, batch_idx].copy_(tensor[start:end])
    return unpacked, lengths


def _pack_sbh_to_thd(tensor: torch.Tensor, cu_seqlens: torch.Tensor, lengths: torch.Tensor):
    """Pack SBH/SBHD style tensor back to THD layout."""
    packed = tensor.new_empty((int(cu_seqlens[-1].item()), *tensor.shape[2:]))
    for batch_idx, length in enumerate(lengths.tolist()):
        if length <= 0:
            continue
        start = int(cu_seqlens[batch_idx].item())
        end = start + length
        packed[start:end].copy_(tensor[:length, batch_idx])
    return packed


def dsa_attention_forward_wrapper(dsa_attention_forward):
    @wraps(dsa_attention_forward)
    def wrapper(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        packed_3d_input = (
            packed_seq_params is not None
            and getattr(packed_seq_params, "qkv_format", None) == "thd"
            and query.dim() == 3
        )

        packed_lengths = None
        if packed_3d_input:
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv

            query, packed_lengths = _unpack_thd_to_sbh(
                query, cu_seqlens_q, packed_seq_params.max_seqlen_q
            )
            key, _ = _unpack_thd_to_sbh(
                key, cu_seqlens_kv, packed_seq_params.max_seqlen_kv
            )
            value, _ = _unpack_thd_to_sbh(
                value, cu_seqlens_kv, packed_seq_params.max_seqlen_kv
            )

        output = dsa_attention_forward(
            self,
            query,
            key,
            value,
            attention_mask,
            x,
            qr,
            attn_mask_type,
            attention_bias,
            packed_seq_params
        )

        if packed_3d_input:
            output = _pack_sbh_to_thd(output, cu_seqlens_q, packed_lengths)

        return output

    return wrapper
