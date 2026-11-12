# Some of this code was adopted from https://github.com/AMD-AGI/Primus
###############################################################################
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
###############################################################################
from functools import wraps
from typing import Optional, Tuple

import torch

from megatron.core.transformer.moe.moe_utils import (
    ProcessGroupCollection,
    apply_router_token_dropping,
    compute_routing_scores_for_aux_loss,
    topk_routing_with_score_function,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args

from hcu_megatron.core.cached_prefix_utils import CachedPrefixParams
from hcu_megatron.core.transformer.moe.moe_utils import MoERouterScoreCache


def primus_topk_router_routing(self, logits: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        import primus_turbo.pytorch as pt
    except ImportError as e:
        raise ImportError("Failed to import 'primus_turbo'. Please make sure it is installed.") from e

    seq_length, bsz = logits.shape[:2]
    logits = logits.view(-1, self.config.num_moe_experts)

    # Apply Z-Loss
    logits = self.apply_z_loss(logits)

    scores_for_aux_loss, probs, routing_map = pt.ops.fused_group_topk_routing_with_aux_score(
        logits,
        self.config.moe_router_topk,
        self.config.moe_router_num_groups,
        self.config.moe_router_group_topk,
        self.config.moe_router_score_function,
        self.config.moe_router_topk_scaling_factor,
    )

    routing_map = routing_map.bool()

    # Apply token dropping to probs and routing_map.
    if self.config.moe_expert_capacity_factor is not None:
        probs, routing_map = apply_router_token_dropping(
            probs,
            routing_map,
            router_topk=self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            drop_policy=self.config.moe_token_drop_policy,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        )

    if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
        # todo: fuse the following logic into the turbo fused_group_topk_routing_with_aux_score OP
        # (the routing_map here differs from the one in the OP output is regarding less of group limit)
        if self.config.moe_router_num_groups is None or self.config.moe_router_num_groups <= 1:
            routing_map_for_aux_loss = routing_map
        else:
            _, top_indices_for_aux_loss = torch.topk(scores_for_aux_loss, k=self.topk, dim=1)
            routing_map_for_aux_loss = (
                torch.zeros_like(logits).int().scatter(1, top_indices_for_aux_loss, 1).bool()
            )
        probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
        probs = self._apply_seq_aux_loss(
            probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
        )
        probs = self._apply_global_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)

    # Optionally apply expert bias
    self._apply_expert_bias(routing_map)

    return probs, routing_map


class TopKRouter():
    def routing(
            self,
            logits: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
        ):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.
            padding_mask (torch.Tensor, optional): Boolean mask indicating non-padding tokens.
                                                   Shape [seq_length, bsz]. True for valid tokens,
                                                   False for padding tokens. Defaults to None.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """

        args = get_args()
        if args.sync_free_moe and args.use_turbo_topk_router:
            return primus_topk_router_routing(self, logits)

        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Flatten padding_mask to [num_tokens] if provided
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits, padding_mask=padding_mask)

        # Calculate probs and routing_map for token dispatching
        if self.routing_type == "sinkhorn":
            probs, routing_map = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "quantile_balancing":
            assert (
                padding_mask is None
            ), "Quantile balancing routing does not support padding masks yet."
            probs, routing_map = self.quantile_balancing(logits)
        else:
            probs, routing_map = topk_routing_with_score_function(
                logits,
                self.topk,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
                fused=self.config.moe_router_fusion,
                router_replay=self.router_replay,
            )

        # Apply token dropping to probs and routing_map.
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        # Apply each aux loss type and attach aux loss autograd function to probs
        if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
            # Calculate scores and routing_map for aux loss
            routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
                logits,
                self.topk,
                self.score_function,
                fused=self.config.moe_router_fusion,
                padding_mask=padding_mask,
            )
            if getattr(self, "cached_prefix_params", None) is not None:
                kv_cache = self.cached_prefix_params.kv_cache_pool.get_kv_cache(
                    layer_idx=self.layer_number, kv_cache_cls=MoERouterScoreCache
                )
                probs = kv_cache.apply_aux_loss(
                    router=self,
                    cached_prefix_params=self.cached_prefix_params,
                    probs=probs,
                    scores_for_aux_loss=scores_for_aux_loss,
                    routing_map_for_aux_loss=routing_map_for_aux_loss,
                    seq_length=seq_length,
                    bsz=bsz,
                    with_padding_mask=padding_mask is not None,
                )
            else:
                probs = self._apply_aux_loss(
                    probs,
                    scores_for_aux_loss,
                    routing_map_for_aux_loss,
                    with_padding_mask=padding_mask is not None,
                )
                probs = self._apply_seq_aux_loss(
                    probs,
                    scores_for_aux_loss,
                    routing_map_for_aux_loss,
                    seq_length,
                    bsz,
                    with_padding_mask=padding_mask is not None,
                )
            # TODO: global_aux_loss is still computed per chunk.
            probs = self._apply_global_aux_loss(
                probs,
                scores_for_aux_loss,
                routing_map_for_aux_loss,
                with_padding_mask=padding_mask is not None,
            )

        # Optionally apply expert bias
        self._apply_expert_bias(routing_map, padding_mask=padding_mask)

        return probs, routing_map


def topK_router_init_wrapper(fn):
    @wraps(fn)
    def wrapper(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ):

        fn(
            self,
            config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        self.cached_prefix_params = None

    return wrapper


def topK_router_foward_wrapper(fn):
    @wraps(fn)
    def wrapper(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        cached_prefix_params: Optional[CachedPrefixParams] = None,
    ):
        # cached_prefix_params will be used by _apply_aux_loss
        self.cached_prefix_params = cached_prefix_params

        return fn(
            self,
            input,
            padding_mask=padding_mask,
        )

    return wrapper
