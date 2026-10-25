# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from functools import lru_cache
from typing import Optional

import torch
from torch import Tensor


from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank
from megatron.core.utils import internal_api

from hcu_megatron.core.cached_prefix_utils import CachedPrefixParams
from hcu_megatron.core.models.common.embeddings.rope_utils import get_pos_emb_over_cached_prefix


class YarnRotaryEmbedding():
    """Yarn Rotary Embedding for language model.
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
        """Forward pass of Yarn Rotary Embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): RoPE offset. Defaults to 0.
            packed_seq (bool, optional): Whether to use packed sequence. Defaults to False.
            cp_group (torch.distributed.ProcessGroup, optional): Context parallel group.
                Defaults to None.
            cached_prefix_params (CachedPrefixParams, optional): Cached prefix params.
                Defaults to None.

        Returns:
            Tensor: Embeddings after applying Yarn RoPE.
        """
        emb, _mscale = self.get_emb(max_seq_len, offset)

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
        return emb, _mscale
