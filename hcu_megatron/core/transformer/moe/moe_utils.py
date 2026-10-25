# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Callable, List

import torch

from megatron.core.transformer.transformer_config import TransformerConfig

from hcu_megatron.core.cached_prefix_utils import CachedPrefixParams, KVCache


class MoERouterScoreCache(KVCache):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        if config.moe_load_balancing_loss_mode_for_chunked_sequence == "all_gather":
            self.scores_cache = []
            self.grad_scores_cache = torch.tensor([], device=torch.cuda.current_device())
            self.prefix_no_grad = []

    def apply_aux_loss(
        self,
        router: Callable,
        cached_prefix_params: CachedPrefixParams,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map_for_aux_loss: torch.Tensor,
        with_padding_mask: bool = False,
    ):
        """Apply aux loss to the probs."""
        if self.config.moe_load_balancing_loss_mode_for_chunked_sequence == "all_gather":
            # Check if there are enough prefixes in the scores cache.
            n_prefix = len(cached_prefix_params.prefix_seqlens)
            assert len(self.scores_cache) >= n_prefix, (
                "Missing enough prefixes: "
                f"expected {n_prefix} scores cache, but got {len(self.scores_cache)=}. "
                "Please check the cached prefix parameters."
            )
            assert all(self.prefix_no_grad[n_prefix:]), (
                f"Already cached {len(self.prefix_no_grad)=} prefixes, "
                f"and the current chunk only needs {n_prefix}, "
                f"so need to drop the last {len(self.prefix_no_grad) - n_prefix}, "
                f"but they are not all no-grad. ({self.prefix_no_grad=})."
            )
            # Drop the prefixes that are not needed.
            self.prefix_no_grad = self.prefix_no_grad[:n_prefix]
            self.scores_cache = self.scores_cache[:n_prefix]

            self.prefix_no_grad.append(not torch.is_grad_enabled())
            probs = AuxLossFunctionAllGather.apply(
                router,
                self.scores_cache,
                self.grad_scores_cache,
                cached_prefix_params.is_terminal,
                probs,
                scores_for_aux_loss,
                routing_map_for_aux_loss,
                with_padding_mask,
            )
        elif self.config.moe_load_balancing_loss_mode_for_chunked_sequence == "reduce":
            probs = router._apply_aux_loss(
                probs,
                scores_for_aux_loss,
                routing_map_for_aux_loss,
                with_padding_mask=with_padding_mask,
            )
        else:
            raise ValueError(
                f"Invalid {self.config.moe_load_balancing_loss_mode_for_chunked_sequence=}, "
                "Only support 'all_gather' or 'reduce'."
            )
        return probs


class AuxLossFunctionAllGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        router: Callable,
        scores_cache: List[torch.Tensor],
        grad_scores_cache: List[torch.Tensor],
        is_terminal: bool,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map_for_aux_loss: torch.Tensor,
        with_padding_mask: bool,
    ):
        # ctx.save_for_backward(scores_for_aux_loss, routing_map_for_aux_loss)
        scores_cache.append(scores_for_aux_loss)
        if is_terminal:
            # Gather the scores along the token dimension
            scores_joined = torch.cat(scores_cache, dim=0).detach().requires_grad_()

            # Compute the aux loss
            with torch.autograd.set_grad_enabled(True):
                probs_out = router._apply_aux_loss(
                    probs,
                    scores_joined,
                    routing_map_for_aux_loss,
                    with_padding_mask=with_padding_mask,
                )

            scores_joined.data = torch.tensor(
                [], device=scores_joined.device, dtype=scores_joined.dtype
            )
            # Save input tensors for backward
            ctx.save_for_backward(probs, scores_joined, probs_out)

        # Save other context variables for backward
        ctx.is_terminal = is_terminal
        ctx.scores_cache = scores_cache
        ctx.grad_scores_cache = grad_scores_cache

        return probs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        is_terminal = ctx.is_terminal
        scores_cache = ctx.scores_cache
        grad_scores_cache = ctx.grad_scores_cache
        current_seq_len = scores_cache[-1].shape[0]
        prefix_seq_len = sum([s.shape[0] for s in scores_cache[:-1]])

        if is_terminal:
            (probs, scores_joined, probs_out) = ctx.saved_tensors
            scores_joined.data = torch.cat(scores_cache, dim=0).data
            grad_probs, grad_scores_joined = torch.autograd.grad(
                outputs=probs_out, inputs=(probs, scores_joined), grad_outputs=grad_output
            )
            if grad_scores_cache.numel() == 0:
                grad_scores_cache.data = grad_scores_joined.data
            else:
                assert (
                    grad_scores_cache.shape == grad_scores_joined.shape
                ), f"{grad_scores_cache.shape=} does not match {grad_scores_joined.shape=}."
                grad_scores_cache.add_(grad_scores_joined.data)
        else:
            grad_probs = grad_output

        grad_scores_prefix, grad_scores_current = torch.split(
            grad_scores_cache, [prefix_seq_len, current_seq_len], dim=0
        )
        grad_scores_cache.data = grad_scores_prefix.data
        scores_cache.pop(-1)

        return (
            None,  # router
            None,  # scores_cache
            None,  # grad_scores_cache
            None,  # is_terminal
            grad_probs,  # probs
            grad_scores_current,  # scores_for_aux_loss
            None,  # routing_map_for_aux_loss
            None,  # with_padding_mask
        )
