# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from functools import wraps
from typing import Optional

import torch
from einops import rearrange

from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.multi_latent_attention import (
    _prepare_mla_core_attention_value,
    _trim_mla_core_attention_output,
)
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params

from hcu_megatron.core.transformer.attention_cached_prefix import ChunkedKVCacheForAttention


def multi_latent_attention_init_wrapper(func):
    @wraps(func)
    def wrapper(
        self,
        config,
        submodules,
        layer_number,
        attn_mask_type,
        attention_type,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
        name: str | None = None,
    ):
        self.cached_prefix_params = None
        func(
            self,
            config,
            submodules,
            layer_number,
            attn_mask_type,
            attention_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )

    return wrapper


def multi_latent_attention_forward_wrapper(func):
    @wraps(func)
    def wrapper(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        cached_prefix_params=None,  # support prefix caching for MLA
        *,
        inference_params=None,
    ):
        self.cached_prefix_params = cached_prefix_params

        return func(
            self,
            hidden_states,
            attention_mask,
            key_value_states=key_value_states,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            position_ids=position_ids,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
        )

    return wrapper


def mla_get_query_key_value_tensors_wrapper(func):
    """Apply chunked-PP prefix offsets to the RoPE that MLA computes internally.

    MLA builds its own rotary embedding inside get_query_key_value_tensors without
    cached_prefix_params, so every chunk would use positions [0, this_chunk_seqlen).
    Bind cached_prefix_params to the rotary module for the duration of the call so
    that later chunks use positions [prefix_len, prefix_len + this_chunk_seqlen).
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        cached_prefix_params = getattr(self, "cached_prefix_params", None)
        if cached_prefix_params is None:
            return func(self, *args, **kwargs)

        rope = self.rotary_pos_emb
        rope_cls = type(rope)

        def get_rotary_seq_len(*fn_args, **fn_kwargs):
            fn_kwargs["cached_prefix_params"] = cached_prefix_params
            return rope_cls.get_rotary_seq_len(rope, *fn_args, **fn_kwargs)

        def forward(*fn_args, **fn_kwargs):
            fn_kwargs["cached_prefix_params"] = cached_prefix_params
            return rope_cls.forward(rope, *fn_args, **fn_kwargs)

        def get_cached_cos_sin(seq_len, offset=0, dtype=torch.get_default_dtype(), packed_seq=False):
            # Same math as YarnRotaryEmbedding._set_cos_sin_cache, but over the
            # current chunk's positions only; the module-level cache stays untouched.
            emb, _mscale = forward(seq_len, offset, packed_seq)
            return (
                (emb.cos() * _mscale).to(dtype).contiguous(),
                (emb.sin() * _mscale).to(dtype).contiguous(),
            )

        overrides = {"get_rotary_seq_len": get_rotary_seq_len, "forward": forward}
        if hasattr(rope_cls, "get_cached_cos_sin"):
            overrides["get_cached_cos_sin"] = get_cached_cos_sin
        assert not any(name in rope.__dict__ for name in overrides), (
            "rotary_pos_emb already has instance-level overrides."
        )
        rope.__dict__.update(overrides)
        try:
            return func(self, *args, **kwargs)
        finally:
            for name in overrides:
                rope.__dict__.pop(name, None)

    return wrapper


class MultiLatentAttention():
    def _run_core_attention(
        self,
        query,
        key,
        value,
        attention_mask,
        packed_seq_params=None,
        attn_mask_type=None,
        **extra_kwargs,
    ):
        """Run MLA core attention with the THD value pad/trim workaround."""
        value, need_v_pad, orig_v_dim, padded_v_dim = _prepare_mla_core_attention_value(
            self, query, value, packed_seq_params
        )
        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type

        orig_v_head_dim = None
        if (
            need_v_pad
            and value is not None
            and hasattr(self.core_attention, 'hidden_size_per_attention_head_v')
        ):
            orig_v_head_dim = self.core_attention.hidden_size_per_attention_head_v
            if value.shape[-1] == orig_v_head_dim:
                orig_v_head_dim = None
            else:
                self.core_attention.hidden_size_per_attention_head_v = value.shape[-1]

        try:
            if self.cached_prefix_params is not None:
                assert not self.offload_core_attention, (
                    "Core attention offloading is not supported with prefix caching, "
                    f"but got {self.config.offload_modules=}."
                )

                kv_cache = self.cached_prefix_params.kv_cache_pool.get_kv_cache(
                    layer_idx=self.layer_number, kv_cache_cls=ChunkedKVCacheForAttention
                )
                core_attn_out = kv_cache.forward_attention_with_kv_cache(
                    module=apply_module(self.core_attention),
                    cached_prefix_params=self.cached_prefix_params,
                    query=query,
                    key=key,
                    value=value,
                    attention_mask=attention_mask,
                    attn_mask_type=attn_mask_type,
                    packed_seq_params=packed_seq_params,
                )
            else:
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    packed_seq_params=packed_seq_params,
                    attn_mask_type=attn_mask_type,
                    **extra_kwargs,
                )
        finally:
            if orig_v_head_dim is not None:
                self.core_attention.hidden_size_per_attention_head_v = orig_v_head_dim

        return _trim_mla_core_attention_output(core_attn_out, need_v_pad, orig_v_dim, padded_v_dim)


class MLASelfAttention():
    """MLA Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """
    def compute_qkv(
        self,
        hidden_states,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,  # pylint: disable=unused-arguments
        rotary_pos_cos=None,  # pylint: disable=unused-arguments
        rotary_pos_sin=None,  # pylint: disable=unused-arguments
        packed_seq_params=None,
        position_ids=None,
        *,
        inference_params=None,
    ):
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context and not inference_context.is_static_batching():
            assert (
                self.config.cache_mla_latents
            ), "currently to use dynamic backend for MLA cache mla latents must be true"

        if self.config.cache_mla_latents:
            self.prepare_for_absorption()

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        qkv_linear_manager = off_interface(self.offload_qkv_linear, hidden_states, "qkv_linear")
        with qkv_linear_manager as hidden_states:
            query, key, value, q_compressed, _ = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
                inference_context=inference_context,
            )
        query = qkv_linear_manager.group_offload(query, forced_released_tensors=[hidden_states])

        self.hidden_states = hidden_states

        if (
            not (self.checkpoint_core_attention and self.training)
            and (inference_context is None or inference_context.is_static_batching())
            and self.config.experimental_attention_variant == "dsa"
        ):
            return query, key, value, q_compressed
        return query, key, value

    def compute_attn(
        self,
        qkv_output,
        attention_mask,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "MLA does not support Flash Decoding"
        assert not rotary_pos_cos_sin, "Flash-infer rope has not been tested with MLA."
        assert not (
            self.training and self.cache_mla_latents
        ), "cache_mla_latents conflicts with training."

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context and not inference_context.is_static_batching():
            assert (
                self.config.cache_mla_latents
            ), "currently to use dynamic backend for MLA cache mla latents must be true"

        query, key, value = qkv_output[:3]
        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        query, key, value, _, attn_mask_type, block_table = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None
        )

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()

        # Value is none during decode for absorption
        if value is not None:
            value = value.contiguous()

        thd_packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"

        core_attention_extra_kwargs = {}
        if getattr(self.core_attention, "requires_dsa_inputs", False):
            core_attention_extra_kwargs = {"x": self.hidden_states, "qr": qkv_output[3]}

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        core_attn_manager = off_interface(
            self.offload_core_attention and self.training, query, "core_attn"
        )
        needs_output_trim = False
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                core_attention_extra_kwargs=core_attention_extra_kwargs,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                with core_attn_manager as query:
                    core_attn_out = self._run_core_attention(
                        query,
                        key,
                        value,
                        attention_mask,
                        packed_seq_params=packed_seq_params,
                        attn_mask_type=attn_mask_type,
                        **core_attention_extra_kwargs,
                    )
            elif self.cache_mla_latents:
                value, need_v_pad, orig_v_dim, padded_v_dim = _prepare_mla_core_attention_value(
                    self, query, value, packed_seq_params
                )
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                )
                # Only rearrange if not in absorption mode (Flash MLA handles format correctly)
                if not inference_context.is_decode_only():
                    core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')
                needs_output_trim = need_v_pad
            core_attn_out = core_attn_manager.group_offload(
                core_attn_out, forced_released_tensors=[query, key, value]
            )
        # We are doing absorption with cache mla latents and decode mode.
        if self.cache_mla_latents and inference_context.is_decode_only():
            # core_attn_out = self.self.up_v_layer(core_attn_out)
            core_attn_out = torch.einsum("sbhc,hdc->sbhd", core_attn_out, self.up_v_weight)
            core_attn_out = core_attn_out.contiguous()

            # Flatten back: [seq, batch, num_heads * v_head_dim]
            core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), -1)

        if needs_output_trim:
            core_attn_out = _trim_mla_core_attention_output(
                core_attn_out, need_v_pad, orig_v_dim, padded_v_dim
            )

        if thd_packed_seq:
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        return core_attn_out

    def compute_proj(self, core_attn_out):
        # =================
        # Output. [sq, b, h]
        # =================
        attn_proj_manager = off_interface(self.offload_attn_proj, core_attn_out, "attn_proj")
        with attn_proj_manager as core_attn_out:
            output, bias = apply_module(self.linear_proj)(core_attn_out)
        output = attn_proj_manager.group_offload(output, forced_released_tensors=[core_attn_out])

        return output, bias
