# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from contextlib import nullcontext
from functools import partial

import torch

from megatron.core import tensor_parallel
from megatron.core.models.common.fine_grained_callables import get_layer_moe_metadata
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    get_mtp_layer_offset,
)
from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor
from megatron.core.transformer.multi_latent_attention import MLASelfAttention

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from hcu_megatron.core.models.gpt.fine_grained_callables import (
    build_transformer_layer_callables,
    build_transformer_layer_callables_with_split_attn
)
from hcu_megatron.training.arguments import get_adaptor_args


def build_mtp_layer_callables_without_split_attn(layer):
    """Callables for multi-token prediction layer nodes.

    This class contains the callable functions for different types of
    multi-token prediction layer nodes (attention, MLP, etc.)
    """

    forward_funcs, backward_dw = build_transformer_layer_callables(layer.mtp_model_layer)
    is_moe, _ = get_layer_moe_metadata(layer.mtp_model_layer)
    pre_dispatch_forward, dispatch_forward, mlp_forward, combine_forward, _ = forward_funcs
    assert is_moe, "MTP layer in a2a overlap only supports MoE layer for now."

    def submodule_mtp_pre_dispatch_forward(node, hidden_states, is_recompute=False,):
        # MTP Block Preprocess
        if node.is_first_layer:
            from megatron.core.models.hybrid.hybrid_model import HybridModel

            model = node.chunk_state.model
            if isinstance(model, HybridModel) and len(model.decoder.layers) == 0:
                final_norm = getattr(model.decoder, "final_norm", None) or getattr(
                    model.decoder, "final_layernorm", None
                )
                if final_norm is not None:
                    hidden_states = final_norm(hidden_states)
                    hidden_states = make_viewless_tensor(
                        inp=hidden_states, requires_grad=True, keep_graph=True
                    )

            offset = get_mtp_layer_offset(layer.config, node.chunk_state.model.vp_stage)
            node.chunk_state.mtp_hidden_states = list(torch.chunk(hidden_states, 1 + offset, dim=0))
            hidden_states = node.chunk_state.mtp_hidden_states[offset]
            if (
                get_adaptor_args().schedule_method == "dualpipev"
                and node.chunk_state.model.embedding.word_embeddings.weight is None
            ):
                from hcu_megatron.core.models.common.language_module.language_module import get_shared_embedding_from_dual_chunk
                node.chunk_state.model.embedding.word_embeddings.weight = get_shared_embedding_from_dual_chunk()

        input_ids, position_ids, padding_mask, decoder_input, hidden_states = layer._get_embeddings(
            input_ids=node.chunk_state.input_ids,
            position_ids=node.chunk_state.position_ids,
            embedding=node.chunk_state.model.embedding,
            hidden_states=hidden_states,
            packed_seq_params=node.chunk_state.packed_seq_params,
            padding_mask=node.chunk_state.padding_mask,
        )
        node.chunk_state.input_ids = input_ids
        node.chunk_state.position_ids = position_ids
        node.chunk_state.padding_mask = padding_mask

        # MTP Layer Preprocess
        # norm, linear projection and transformer
        assert (
            node.chunk_state.context is None
        ), f"multi token prediction + cross attention is not yet supported."
        assert (
            node.chunk_state.packed_seq_params is None
        ), f"multi token prediction + sequence packing is not yet supported."

        if layer.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # fp8 context is added in 1f1b schedule, so we don't need to add it here
        with rng_context:
            hidden_states = layer._concat_embeddings(hidden_states, decoder_input)
            return pre_dispatch_forward(node, hidden_states, is_recompute=is_recompute,)

    def submodule_mtp_postprocess_forward(node, hidden_states, is_recompute=False,):
        hidden_states = layer._postprocess(hidden_states)
        node.chunk_state.mtp_hidden_states.append(hidden_states)
        if node.is_last_layer:
            hidden_states = torch.cat(node.chunk_state.mtp_hidden_states, dim=0)
            node.chunk_state.mtp_hidden_states = None
        return hidden_states

    def rng_context_wrapper(func, *args, **kwargs):
        """
        Wrapper to add rng context to submodule callables
        """
        if layer.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()
        with rng_context:
            return func(*args, **kwargs)

    # Build forward and backward callable functions.
    # pre_dispatch_func already has rng context (rolled into
    # submodule_mtp_pre_dispatch_forward), so it does not need to be wrapped.
    pre_dispatch_func = submodule_mtp_pre_dispatch_forward
    dispatch_func = partial(rng_context_wrapper, dispatch_forward)
    mlp_func = partial(rng_context_wrapper, mlp_forward)
    combine_func = partial(rng_context_wrapper, combine_forward)
    mtp_post_process_func = submodule_mtp_postprocess_forward

    forward_funcs = [
        pre_dispatch_func,
        dispatch_func,
        mlp_func,
        combine_func,
        mtp_post_process_func,
    ]
    pre_dispatch_bwd = backward_dw["pre_dispatch_computation"]
    if isinstance(pre_dispatch_bwd, list):
        pre_dispatch_bwd.append(layer.eh_proj)
    else:
        backward_dw["pre_dispatch_computation"] = [pre_dispatch_bwd, layer.eh_proj]

    return forward_funcs, backward_dw


def build_layer_callables_without_split_attn(layer):
    """
    Builds the callable functions(forward and dw) for the given layer.
    For now, 1f1b overlap only support TransformerLayer and MultiTokenPredictionLayer.

    Args:
        layer: The layer to build callables for.

    Returns:
        forward_funcs: list of callable functions for the layer.
        backward_dw: dict of weight gradient functions for the layer.
    """
    if isinstance(layer, TransformerLayer):
        return build_transformer_layer_callables(layer)
    elif isinstance(layer, MultiTokenPredictionLayer):
        return build_mtp_layer_callables_without_split_attn(layer)

    raise ValueError(f"Unsupported layer type: {type(layer)}")


def build_mtp_layer_callables_with_split_attn(layer):
    """Callables for multi-token prediction layer nodes.

    This class contains the callable functions for different types of
    multi-token prediction layer nodes (attention, MLP, etc.)
    """

    forward_funcs, backward_dw = build_transformer_layer_callables_with_split_attn(layer.mtp_model_layer)
    is_moe, _ = get_layer_moe_metadata(layer.mtp_model_layer)
    attn_qkv_forward, core_attn_forward, attn_proj_forward, dispatch_forward, mlp_forward, combine_forward, _ = (
        forward_funcs
    )
    assert is_moe, "MTP layer in a2a overlap only supports MoE layer for now."

    def submodule_mtp_attn_qkv_forward(node, hidden_states, is_recompute=False):
        # MTP Block Preprocess
        if node.is_first_layer:
            from megatron.core.models.hybrid.hybrid_model import HybridModel

            model = node.chunk_state.model
            if isinstance(model, HybridModel) and len(model.decoder.layers) == 0:
                final_norm = getattr(model.decoder, "final_norm", None) or getattr(
                    model.decoder, "final_layernorm", None
                )
                if final_norm is not None:
                    hidden_states = final_norm(hidden_states)
                    hidden_states = make_viewless_tensor(
                        inp=hidden_states, requires_grad=True, keep_graph=True
                    )

            offset = get_mtp_layer_offset(layer.config, node.chunk_state.model.vp_stage)
            node.chunk_state.mtp_hidden_states = list(torch.chunk(hidden_states, 1 + offset, dim=0))
            hidden_states = node.chunk_state.mtp_hidden_states[offset]
            if (
                get_adaptor_args().schedule_method == "dualpipev"
                and node.chunk_state.model.embedding.word_embeddings.weight is None
            ):
                from hcu_megatron.core.models.common.language_module.language_module import get_shared_embedding_from_dual_chunk
                node.chunk_state.model.embedding.word_embeddings.weight = get_shared_embedding_from_dual_chunk()

        input_ids, position_ids, padding_mask, decoder_input, hidden_states = layer._get_embeddings(
            input_ids=node.chunk_state.input_ids,
            position_ids=node.chunk_state.position_ids,
            embedding=node.chunk_state.model.embedding,
            hidden_states=hidden_states,
            packed_seq_params=node.chunk_state.packed_seq_params,
            padding_mask=node.chunk_state.padding_mask,
        )
        node.chunk_state.input_ids = input_ids
        node.chunk_state.position_ids = position_ids
        node.chunk_state.padding_mask = padding_mask

        # MTP Layer Preprocess
        # norm, linear projection and transformer
        assert (
            node.chunk_state.context is None
        ), f"multi token prediction + cross attention is not yet supported."
        assert (
            node.chunk_state.packed_seq_params is None
        ), f"multi token prediction + sequence packing is not yet supported."

        if layer.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # fp8 context is added in 1f1b schedule, so we don't need to add it here
        with rng_context:
            hidden_states = layer._concat_embeddings(hidden_states, decoder_input)
            return attn_qkv_forward(node, hidden_states, is_recompute=is_recompute)

    def submodule_mtp_postprocess_forward(node, hidden_states, is_recompute=False):
        hidden_states = layer._postprocess(hidden_states)
        node.chunk_state.mtp_hidden_states.append(hidden_states)
        if node.is_last_layer:
            hidden_states = torch.cat(node.chunk_state.mtp_hidden_states, dim=0)
            node.chunk_state.mtp_hidden_states = None
        return hidden_states

    def rng_context_wrapper(func, *args, **kwargs):
        """
        Wrapper to add rng context to submodule callables
        """
        if layer.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()
        with rng_context:
            return func(*args, **kwargs)

    # Build forward and backward callable functions
    # attn_forward already has rng context, no need to wrap
    attn_qkv_func = submodule_mtp_attn_qkv_forward
    core_attn_func = partial(rng_context_wrapper, core_attn_forward)
    attn_proj_func = partial(rng_context_wrapper, attn_proj_forward)
    dispatch_func = partial(rng_context_wrapper, dispatch_forward)
    mlp_func = partial(rng_context_wrapper, mlp_forward)
    combine_func = partial(rng_context_wrapper, combine_forward)
    mtp_post_process_func = submodule_mtp_postprocess_forward

    forward_funcs = [
        attn_qkv_func,
        core_attn_func,
        attn_proj_func,
        dispatch_func,
        mlp_func,
        combine_func,
        mtp_post_process_func,
    ]

    attn_proj_dw_funcs = [layer.mtp_model_layer.self_attention.linear_proj]
    if is_moe and layer.mtp_model_layer.mlp.use_shared_expert and not layer.mtp_model_layer.mlp.shared_expert_overlap:
        attn_proj_dw_funcs.append(layer.mtp_model_layer.mlp.shared_experts)

    if isinstance(layer.mtp_model_layer.self_attention, MLASelfAttention):
        attn_qkv_dw_funcs = [
            layer.mtp_model_layer.self_attention.linear_kv_up_proj,
            layer.mtp_model_layer.self_attention.linear_kv_down_proj,
            layer.eh_proj,
        ]
        if layer.config.q_lora_rank is None:
            attn_qkv_dw_funcs.append(
                layer.mtp_model_layer.self_attention.linear_q_proj
            )
        else:
            attn_qkv_dw_funcs.extend([
                layer.mtp_model_layer.self_attention.linear_q_down_proj,
                layer.mtp_model_layer.self_attention.linear_q_up_proj
            ])
    else:
        attn_qkv_dw_funcs = [layer.mtp_model_layer.self_attention.linear_qkv, layer.eh_proj]

    backward_dw = {
        "attn_qkv": attn_qkv_dw_funcs,
        "attn_proj": attn_proj_dw_funcs,
        "mlp": layer.mtp_model_layer.mlp.experts if is_moe else None
    }
    return forward_funcs, backward_dw


def build_layer_callables_with_split_attn(layer):
    """
    Builds the callable functions(forward and dw) for the given layer.
    For now, 1f1b overlap only support TransformerLayer and MultiTokenPredictionLayer.

    Args:
        layer: The layer to build callables for.

    Returns:
        forward_funcs: list of callable functions for the layer.
        backward_dw: dict of weight gradient functions for the layer.
    """
    if isinstance(layer, TransformerLayer):
        return build_transformer_layer_callables_with_split_attn(layer)
    elif isinstance(layer, MultiTokenPredictionLayer):
        return build_mtp_layer_callables_with_split_attn(layer)

    raise ValueError(f"Unsupported layer type: {type(layer)}")
