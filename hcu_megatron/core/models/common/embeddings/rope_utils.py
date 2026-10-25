from torch import Tensor

from hcu_megatron.core.cached_prefix_utils import CachedPrefixParams


def get_pos_emb_over_cached_prefix(
    pos_emb: Tensor, seq_dim: int, cached_prefix_params: CachedPrefixParams
):
    """Get the position embedding over the cached prefix.

    Args:
        pos_emb (Tensor): Positional embedding tensor
        seq_dim (int): Sequence dimension
        cached_prefix_params (CachedPrefixParams): Cached prefix parameters
    """

    prefix_len = sum(cached_prefix_params.prefix_seqlens)
    this_chunk_len = cached_prefix_params.this_chunk_seqlen
    pos_emb = pos_emb.narrow(seq_dim, prefix_len, this_chunk_len)
    return pos_emb
