# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Optional
from functools import lru_cache, wraps

import torch
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import internal_api

from hcu_megatron.core.cached_prefix_utils import CachedPrefixParams
from hcu_megatron.core.models.common.embeddings.rope_utils import get_pos_emb_over_cached_prefix


def rotary_embedding_get_rotary_seq_len_wrapper(fn):
    @wraps(fn)
    def wrapper(
        self,
        inference_context: BaseInferenceContext,
        transformer: TransformerBlock,
        transformer_input: Tensor,
        transformer_config: TransformerConfig,
        packed_seq_params: Optional[PackedSeqParams] = None,
        cached_prefix_params: Optional[CachedPrefixParams] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        if cached_prefix_params is not None:
            if cached_prefix_params.max_total_seqlen is not None:
                return cached_prefix_params.max_total_seqlen
            else:
                return (
                    sum(cached_prefix_params.prefix_seqlens)
                    + cached_prefix_params.this_chunk_seqlen
                )

        return fn(
            self,
            inference_context,
            transformer,
            transformer_input,
            transformer_config,
            packed_seq_params=packed_seq_params,
            inference_params=inference_params,
        )

    return wrapper


class RotaryEmbedding():
    """Rotary Embedding for language model.
    """

    @lru_cache(maxsize=32)
    @internal_api
    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
        cached_prefix_params: Optional[CachedPrefixParams] = None,
    ) -> Tensor:
        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): RoPE offset. Defaults to 0.
            packed_seq (bool, optional): Whether to use packed sequence. Defaults to False.
            cp_group (torch.distributed.ProcessGroup, optional): Context parallel group.
                Defaults to None.
            cached_prefix_params (CachedPrefixParams, optional): Cached prefix params.
                Defaults to None.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        emb = self.get_emb(max_seq_len, offset)

        if cached_prefix_params is not None:
            # Slice rotary_pos_emb along sequence dimension
            # and select the parition of the current span
            emb = get_pos_emb_over_cached_prefix(emb, 0, cached_prefix_params)

        if cp_group is None:
            cp_group = self.cp_group
        if cp_group is not None and cp_group.size() > 1 and not packed_seq:
            # slice rotary_pos_emb along sequence dimension
            # and select the parition of the current CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0, cp_group)

        return emb
