# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import contextlib
import warnings
from contextlib import nullcontext
from typing import Callable, Optional
from functools import wraps

import torch
from torch import Tensor

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.transformer.enums import LayerType
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.multi_token_prediction import (
    get_mtp_layer_offset,
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    _compute_mtp_acceptance_counts,
    _roll_tensor_packed_seq,
)
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.core.utils import make_viewless_tensor

from hcu_megatron.core.cached_prefix_utils import CachedPrefixParams
from hcu_megatron.training import get_args


def tie_word_embeddings_state_dict_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if get_args().schedule_method == "dualpipev":
            return

        fn(*args, **kwargs)

    return wrapper


def mtp_on_this_rank(
    layout: PipelineParallelLayerLayout = None,
    mtp_num_layers: Optional[int] = None,
    ignore_virtual: Optional[bool] = True,
    vp_stage: Optional[int] = None,
) -> bool:
    """
    Check if there is MTP on the current rank.

    Behavior:
        - If a custom pipeline model parallel layout is provided:
            - If virtual pipeline parallelism is enabled (and `ignore_virtual` is False), checks
              whether any MTP layers are present on this (pp_rank, vp_stage) pair.
            - Otherwise, checks all virtual pipeline ranks of the current pipeline rank. Returns
              True if any virtual sub-rank includes at least one MTP layer.
        - If no custom layout is provided, assumes all MTP layers (if any) are placed on the last
          pipeline stage. The function returns True only on the last pipeline stage.
    """
    mtp_on_this_rank = False
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    if layout is not None:
        # with custom PP layout, we support put MTP layers on any pipeline stage
        if (
            not ignore_virtual
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"
            num_layers_to_build = layout.layout[pp_rank][vp_stage].count(LayerType.mtp)
            mtp_on_this_rank = num_layers_to_build > 0
        else:
            for vpp_rank in range(len(layout.layout[pp_rank])):
                num_layers_to_build = layout.layout[pp_rank][vpp_rank].count(LayerType.mtp)
                if num_layers_to_build > 0:
                    mtp_on_this_rank = True
                    break
    else:
        # without custom PP layout, we only support put all of MTP layers on the last pipeline stage
        if mtp_num_layers is not None:
            if get_args().schedule_method == 'dualpipev':
                mtp_on_this_rank = parallel_state.is_pipeline_first_stage(
                    ignore_virtual=True
                )
            else:
                mtp_on_this_rank = parallel_state.is_pipeline_last_stage(
                    ignore_virtual=ignore_virtual, vp_stage=vp_stage
                )
        else:
            mtp_on_this_rank = False
    return mtp_on_this_rank


def get_mtp_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None, model=None,
) -> int:
    """Get the number of MTP layers to build."""

    args = get_args()
    dualpipev_first_chunk = getattr(model, "dualpipev_first_chunk", False) if model is not None else getattr(args, "dualpipev_first_chunk", False)
    if args.schedule_method == "dualpipev":
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True) and not dualpipev_first_chunk:
            return config.mtp_num_layers if config.mtp_num_layers else 0
        else:
            return 0

    if config.pipeline_model_parallel_layout is not None:
        # If we have a custom PP layout, get the number of mtp layers in the layout array.
        num_layers_to_build = config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.mtp, vp_stage=vp_stage
        )
        assert num_layers_to_build == config.mtp_num_layers or num_layers_to_build == 0, (
            f"Currently, we only support put all of MTP layers on the last pipeline stage, "
            f"so the number of MTP layers to build ({num_layers_to_build}) must match "
            f"mtp_num_layers ({config.mtp_num_layers}) or be 0."
        )
    else:
        if parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
            num_layers_to_build = config.mtp_num_layers if config.mtp_num_layers else 0
        else:
            num_layers_to_build = 0
    return num_layers_to_build


RecomputeMTPLayerFlag = False

def get_recompute_mtp_layer_flag():
    global RecomputeMTPLayerFlag
    return RecomputeMTPLayerFlag


def set_recompute_mtp_layer_flag(recompute_mtp_layer_flag):
    global RecomputeMTPLayerFlag
    RecomputeMTPLayerFlag = recompute_mtp_layer_flag


@contextlib.contextmanager
def _fork_recompute_mtp_layer_flag():
    # Store the current states.
    current_recompute_mtp_layer_flag = get_recompute_mtp_layer_flag()
    try:
        yield
    finally:
        # Set the states back to what it was at the start of this function.
        set_recompute_mtp_layer_flag(current_recompute_mtp_layer_flag)


def roll_tensor(
    tensor,
    shifts=-1,
    dims=-1,
    cp_group=None,
    packed_seq_params=None,
    boundary_elements=None,
):
    """Roll the tensor input along the sequence dimension with Context Parallelism (CP) support.

    This function extends the original roll_tensor to support Context Parallelism, which allows
    MTP to work with CP > 1. When CP is enabled, the sequence dimension is split across CP ranks,
    and tensor rolling requires communication between adjacent CP ranks to properly handle the
    boundary conditions.

    For CP=1 (default behavior): Uses standard torch.roll with zero padding
    For CP>1: Splits tensor into chunks, performs rolling within each chunk, then exchanges
    boundary elements between adjacent CP ranks to maintain sequence continuity.

    For packed sequences: Respects sequence boundaries when rolling to avoid mixing tokens
    from different sequences.

    Args:
        tensor (Tensor): The input tensor to roll. If None, returns (None, None).
        shifts (int): The shift of the tensor (typically -1 for MTP).
        dims (int): The dimension to roll (typically -1 for sequence dimension).
        cp_group (ProcessGroup): The context parallelism process group. If None or size=1,
                               falls back to standard rolling behavior.
        packed_seq_params (PackedSeqParams): Parameters for packed sequence processing.
                                            If provided, respects sequence boundaries.
        boundary_elements (Tensor, optional): The elements to be filled into the boundary.
    Returns:
        tuple: (rolled_tensor, sum_of_rolled_tensor)
    """
    if tensor is None:
        return None, None

    # Handle packed sequences cases
    if packed_seq_params is not None:
        assert (
            boundary_elements is None
        ), "Manually specifying boundary elements is not supported with sequence packing for now."
        return _roll_tensor_packed_seq(tensor, shifts, dims, packed_seq_params, cp_group)

    # Standard rolling behavior when CP is not enabled (cp_group is None or size=1)
    if cp_group is None or cp_group.size() == 1:
        rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
        if boundary_elements is not None:
            # Fill the specified elements into the boundary.
            rolled_tensor.select(dims, shifts).copy_(boundary_elements)
        else:
            # Fill the boundary elements with 0.
            rolled_tensor.select(dims, shifts).fill_(0)
        return rolled_tensor, rolled_tensor.sum()

    # CP-enabled rolling: Split tensor into chunks and handle boundary communication
    # This matches the batch splitting logic in get_batch_on_this_cp_rank() function
    tensor_list = tensor.chunk(2, dim=dims)
    rolled_tensor_list = []
    for i in range(len(tensor_list)):
        rolled_tensor_list.append(torch.roll(tensor_list[i], shifts=shifts, dims=dims))

    # Prepare tensors for communication between CP ranks
    # Each CP rank needs to send boundary elements to adjacent ranks
    tensor_send_list = []
    tensor_recv_list = []
    for i in range(len(rolled_tensor_list)):
        tensor_send_list.append(rolled_tensor_list[i].select(dims, shifts).contiguous())
        empty_tensor = torch.empty(
            tensor_send_list[i].shape,
            dtype=tensor_send_list[i].dtype,
            device=torch.cuda.current_device(),
        )
        tensor_recv_list.append(empty_tensor)

    # Get the global rank of next and prev process in the cp group
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    local_rank = torch.distributed.get_rank(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % len(global_ranks)]
    prev_rank = global_ranks[(local_rank - 1) % len(global_ranks)]

    # Start send and recv ops
    ops = []
    if local_rank != 0:
        req_send_first_part = torch.distributed.isend(tensor=tensor_send_list[0], dst=prev_rank)
        ops.append(req_send_first_part)
        req_recv_second_part = torch.distributed.irecv(tensor=tensor_recv_list[1], src=prev_rank)
        ops.append(req_recv_second_part)
    else:
        if boundary_elements is not None:
            # Fill the specified elements into the boundary.
            tensor_recv_list[1] = boundary_elements
        else:
            # Fill the boundary elements with 0.
            tensor_recv_list[1] = 0
    if local_rank != len(global_ranks) - 1:
        req_recv_first_part = torch.distributed.irecv(tensor=tensor_recv_list[0], src=next_rank)
        ops.append(req_recv_first_part)
        req_send_second_part = torch.distributed.isend(tensor=tensor_send_list[1], dst=next_rank)
        ops.append(req_send_second_part)
    else:
        # For the last CP rank, the removed elements of second part go into the first part
        tensor_recv_list[0] = tensor_send_list[1]

    # Wait for all communication operations to complete
    for op in ops:
        op.wait()

    # Splicing: Replace boundary elements with received elements from adjacent ranks
    # This ensures proper sequence continuity across CP boundaries
    index = [slice(None)] * rolled_tensor_list[0].dim()
    index[dims] = shifts
    for i in range(len(rolled_tensor_list)):
        rolled_tensor_list[i][tuple(index)] = tensor_recv_list[i]

    # Concatenate the processed chunks back into a single tensor
    rolled_tensor = torch.cat(rolled_tensor_list, dim=dims)

    return rolled_tensor, rolled_tensor.sum()


def process_mtp_loss(
    hidden_states: Tensor,
    labels: Tensor,
    loss_mask: Optional[Tensor],
    output_layer: Callable,
    output_weight: Optional[Tensor],
    runtime_gather_output: Optional[bool],
    is_training: bool,
    compute_language_model_loss: Callable,
    config: TransformerConfig,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    scale_logits_fn: Optional[Callable[[Tensor], Tensor]] = None,
    input_ids: Optional[Tensor] = None,
    cached_prefix_params=None,
) -> Tensor:
    """Process Multi-Token Prediction (MTP) loss computation.

    This is a standalone function that handles MTP loss computation. It's used on the
    post_process rank to split concatenated hidden states and compute MTP losses.

    Args:
        hidden_states (Tensor): Hidden states tensor (concatenated with MTP outputs).
        labels (Tensor): Ground truth labels.
        loss_mask (Optional[Tensor]): Mask for loss computation. If None, uses all ones.
        output_layer (Callable): Output layer method to compute logits.
        output_weight (Optional[Tensor]): Optional output weight for shared embeddings.
        runtime_gather_output (Optional[bool]): Whether to gather output at runtime.
        is_training (bool): Whether the model is in training mode.
        compute_language_model_loss (Callable): Method to compute language model loss.
        config (TransformerConfig): Model configuration containing mtp_num_layers etc.
        cp_group (Optional[ProcessGroup]): Context parallelism process group.
        tp_group (Optional[ProcessGroup]): Tensor parallelism process group.
        packed_seq_params (Optional[PackedSeqParams]): Packed sequence parameters.
        scale_logits_fn (Optional[Callable[[Tensor], Tensor]]): Optional function to
            scale logits before loss computation (e.g., MuP output scaling).
        input_ids (Optional[Tensor]): Input token IDs. Used to derive labels when
            ``labels`` is None (e.g. RL training), by rolling left to match the SFT
            label convention (``label[i] = input_id[i + 1]``). Ignored when ``labels``
            is provided.

    Returns:
        Tensor: Updated hidden states after MTP loss processing (first chunk only).
    """
    hidden_states_list = torch.chunk(hidden_states, 1 + config.mtp_num_layers, dim=0)
    hidden_states = hidden_states_list[0]

    # When labels are not provided (e.g. RL training), derive them from input_ids by
    # rolling left so that label[i] = input_id[i + 1], matching the SFT label format.
    derived_labels_from_input_ids = False
    if labels is None:
        if input_ids is None:
            return hidden_states
        labels, _ = roll_tensor(
            input_ids, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
        )
        derived_labels_from_input_ids = True

    if config.mtp_detach_heads:
        if output_weight is not None:
            output_weight = output_weight.detach()
        else:
            output_weight = output_layer.weight.detach()

    mtp_labels = labels.clone()
    if loss_mask is None:
        loss_mask = torch.ones_like(mtp_labels)
    if derived_labels_from_input_ids:
        # input_ids has no real token beyond the sequence window, so the last rolled-in
        # label is fabricated (zeroed). Roll loss_mask in lockstep with the
        # input_ids -> labels shift so that boundary position is masked.
        loss_mask, _ = roll_tensor(
            loss_mask, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
        )

    # Store the original number of tokens before rolling for proper normalization
    # when calculate_per_token_loss is enabled. This ensures MTP gradients are
    # correctly scaled relative to the main loss gradients in finalize_model_grads.
    original_num_tokens = loss_mask.sum()

    for mtp_layer_number in range(config.mtp_num_layers):
        mtp_logits, _ = output_layer(
            hidden_states_list[mtp_layer_number + 1],
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        )
        if scale_logits_fn is not None:
            mtp_logits = scale_logits_fn(mtp_logits)

        if (
            cached_prefix_params is not None
            and cached_prefix_params.boundary_elements_for_mtp is not None
        ):
            # Boundary elements for MTP are stored in the cached_prefix_params.
            assert (
                mtp_layer_number == 0
            ), "Prefix caching currently only supports at most one MTP layer."
            boundary_mtp_labels = cached_prefix_params.boundary_elements_for_mtp["labels"]
            boundary_loss_mask = cached_prefix_params.boundary_elements_for_mtp["loss_mask"]
        else:
            boundary_mtp_labels = None
            boundary_loss_mask = None
        mtp_labels, _ = roll_tensor(
            mtp_labels, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params,
            boundary_elements=boundary_mtp_labels,
        )
        loss_mask, num_tokens = roll_tensor(
            loss_mask, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params,
            boundary_elements=boundary_loss_mask,
        )

        mtp_loss = compute_language_model_loss(mtp_labels, mtp_logits)

        mtp_loss = loss_mask * mtp_loss

        if is_training:
            mtp_loss_for_log = (
                torch.sum(mtp_loss) * (num_tokens > 0).to(mtp_loss.dtype)
            ) / num_tokens.clamp(min=1)
            correct, total = _compute_mtp_acceptance_counts(
                mtp_logits, mtp_labels, loss_mask, output_layer, runtime_gather_output, tp_group
            )

            MTPLossLoggingHelper.save_metrics_to_tracker(
                mtp_loss_for_log,
                correct,
                total,
                mtp_layer_number,
                config.mtp_num_layers,
                avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )
        mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
        if config.calculate_per_token_loss:
            # When calculate_per_token_loss is enabled, finalize_model_grads will
            # divide all gradients by total_num_tokens (from main loss).
            # However, MTP has fewer valid tokens due to rolling. To ensure correct
            # per-token gradient weighting, we normalize by the rolled token count
            # and re-scale by the original token count.
            # Avoid division by zero
            num_tokens_safe = torch.clamp(num_tokens, min=1)
            mtp_loss_normalized = (
                mtp_loss_scale * mtp_loss * (original_num_tokens / num_tokens_safe)
            )
            hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_normalized)
        else:
            safe_num_tokens = num_tokens.clamp(min=1)
            hidden_states = MTPLossAutoScaler.apply(
                hidden_states, mtp_loss_scale * mtp_loss / safe_num_tokens
            )

    return hidden_states


class MultiTokenPredictionLayer:
    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        decoder_input: Tensor,
        attention_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ):
        """Forward through ``_proj_and_transformer_layer`` with activation
        recomputation.

        Mirrors ``transformer_block._checkpointed_forward``:

        * Non-tensor objects (``attention_bias``, ``inference_params``,
          ``packed_seq_params``) are captured by the ``custom_forward``
          closure; only tensor / ``None`` arguments flow positionally
          through the underlying checkpoint primitive. This is required
          by both backends: ``tensor_parallel.checkpoint`` because its
          ``save_for_backward`` only accepts tensors and ``None``, and
          ``te_checkpoint`` because its reentrant implementation only
          tracks positional tensor inputs as checkpoint inputs (kwarg
          tensors are not represented in the recompute backward path).
        * Quantized recipes (fp8, fp4) route through ``te_checkpoint``;
          everything else uses ``tensor_parallel.checkpoint``.
        * Only ``fp8 + delayed scaling`` needs an outer quantization
          context entered before ``te_checkpoint``; see the
          ``outer_quantization_context`` block below.
        """

        def custom_forward(
            hidden_states,
            decoder_input,
            attention_mask,
            padding_mask,
            context,
            context_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        ):
            return self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )

        # Decide the outer quantization context, matching
        # ``transformer_block._checkpointed_forward``. Only ``fp8 + delayed
        # scaling`` needs an active context at the ``te_checkpoint`` entry
        # point: TE's ``_CheckpointFunction.forward`` samples
        # ``FP8GlobalStateManager.is_fp8_enabled()`` there to gate the
        # phase-1 amax-buffer stash that phase-2 backward looks up via
        # ``global_fp8_buffer_pos_fwd_recompute``. With fp8 only entered
        # *inside* ``_proj_and_transformer_layer``, TE samples fp8 as off,
        # phase-1 skips the stash, and phase-2 raises ``KeyError``.
        # Non-delayed fp8 recipes (MXFP8BlockScaling, Float8CurrentScaling)
        # and fp4 (NVFP4BlockScaling) treat the stash/lookup as a noop, so
        # the inner context entered inside ``_proj_and_transformer_layer``
        # is sufficient.
        if self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed:
            outer_quantization_context = get_fp8_context(self.config)
        else:
            outer_quantization_context = nullcontext()

        def checkpoint_handler():
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            # fp4 quantization is internally implemented via TE's
            # ``fp8_autocast`` (see ``fp4_utils.get_fp4_context``), so
            # quantized recompute on either fp8 or fp4 must go through
            # ``te_checkpoint``. Matches ``transformer_block``'s policy.
            if self.config.fp8 or self.config.fp4:
                from megatron.core.extensions.transformer_engine import te_checkpoint

                return te_checkpoint(
                    custom_forward,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    decoder_input,
                    attention_mask,
                    padding_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    sequence_len_offset,
                )
            else:
                # tensor_parallel.checkpoint stashes args via autograd's
                # ``save_for_backward``, which only accepts tensors and ``None``.
                # Pass tensor / ``None`` args positionally and capture the
                # non-tensor objects (``attention_bias``, ``inference_params``,
                # ``packed_seq_params``) via the ``custom_forward`` closure.
                return tensor_parallel.checkpoint(
                    custom_forward,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    decoder_input,
                    attention_mask,
                    padding_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    sequence_len_offset,
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            assert (
                self.config.recompute_num_layers == 1
            ), "recompute_num_layers must be 1 for MTP recompute"
            with outer_quantization_context:
                outputs = checkpoint_handler()

        elif (
            get_args().recompute_layer_ids is not None
            or get_args().recompute_mtp_layer_ids is not None
        ):
            # layer id is in recompute_mtp_layer_ids
            if get_recompute_mtp_layer_flag():
                outputs = checkpoint_handler()
            else:
                outputs = self._proj_and_transformer_layer(
                    hidden_states=hidden_states,
                    decoder_input=decoder_input,
                    attention_mask=attention_mask,
                    padding_mask=padding_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    attention_bias=attention_bias,
                    inference_params=inference_params,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                )
        elif self.config.recompute_method == 'block':
            # TODO: implement block-based recompute for MTP
            warnings.warn(
                "recompute_method == 'block' is not supported for MTP yet." " Skipping recompute."
            )
            outputs = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        else:
            raise ValueError("Invalid activation recompute method.")

        return outputs

    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        embedding: Callable,
        hidden_states: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask: Optional[torch.Tensor] = None,
        cached_prefix_params: Optional[CachedPrefixParams] = None,
    ):
        """
        Preprocesses input data for the Multi-Token Prediction (MTP) layers.

        This function computes the decoder input and sends updated input_ids and position_ids to
        the next layer.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            position_ids (torch.Tensor): The position IDs corresponding to the input tokens.
            embedding (Callable): The embedding module
                from gpt model to compute the decoder input.
            hidden_states (torch.Tensor): hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            packed_seq_params (PackedSeqParams): Parameters for packed sequence processing.
            cached_prefix_params (CachedPrefixParams, optional): Parameters for cached prefix.
        """
        # Calc logits for the current Multi-Token Prediction (MTP) layers.
        if (
            cached_prefix_params is not None
            and cached_prefix_params.boundary_elements_for_mtp is not None
        ):
            # Boundary elements for MTP are stored in the cached_prefix_params.
            # Currently only support one MTP layer.
            boundary_input_ids = cached_prefix_params.boundary_elements_for_mtp["tokens"]
            boundary_position_ids = cached_prefix_params.boundary_elements_for_mtp["position_ids"]
            boundary_padding_mask = cached_prefix_params.boundary_elements_for_mtp["padding_mask"]
        else:
            boundary_input_ids = None
            boundary_position_ids = None
            boundary_padding_mask = None

        input_ids, _ = roll_tensor(
            input_ids,
            shifts=-1,
            dims=-1,
            cp_group=self.cp_group,
            packed_seq_params=packed_seq_params,
            boundary_elements=boundary_input_ids,
        )
        position_ids, _ = roll_tensor(
            position_ids,
            shifts=-1,
            dims=-1,
            cp_group=self.cp_group,
            packed_seq_params=packed_seq_params,
            boundary_elements=boundary_position_ids,
        )
        if padding_mask is not None:
            padding_mask, _ = roll_tensor(
                padding_mask,
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
                boundary_elements=boundary_padding_mask,
            )
        # embedding
        decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)

        if self.config.mtp_detach_heads:
            decoder_input = decoder_input.detach()

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        # make_viewless_tensor no-ops when hidden_states is not a view (_base is None),
        # which happens after detach() with mtp_detach_heads. Activation
        # checkpointing (CheckpointFunction.apply) requires at least one input tensor
        # with requires_grad=True to produce a differentiable output, so we ensure it
        # here to maintain gradient flow to MTP layer parameters.
        if not hidden_states.requires_grad:
            hidden_states.requires_grad_(True)

        return input_ids, position_ids, padding_mask, decoder_input, hidden_states

    def _proj_and_transformer_layer(
        self,
        hidden_states: Tensor,
        decoder_input: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
        cached_prefix_params: Optional[CachedPrefixParams] = None,
    ) -> torch.Tensor:
        """
        Concatenates embeddings with hidden states and then applies transformer layer forward.
        """
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # Unlike transformer_block.py which needs to support mixed-precision in
        # different layers,currently MTP only use global fp8 context.
        if self.config.fp8:
            fp8_context = get_fp8_context(self.config)
            transformer_layer_fp8_context = get_fp8_context(self.config)
        else:
            fp8_context = nullcontext()
            transformer_layer_fp8_context = nullcontext()

        # TODO: currently ignoring FP4 in MTP layers because we need more numerical validation
        with rng_context:
            with fp8_context:
                hidden_states = self._concat_embeddings(hidden_states, decoder_input)

            # Use a separate fp8 context for the transformer layer. This is to ensure that when the
            # transformer layer is cudagraphed, the FP8GlobalStateManager.is_first_fp8_module() is
            # True so that the fp8 weight caching can be triggered correctly.
            with transformer_layer_fp8_context:
                if self.mtp_layer_pattern is not None:
                    hidden_states = self.mtp_model_layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        padding_mask=padding_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_context=inference_params,
                        packed_seq_params=packed_seq_params,
                        cached_prefix_params=cached_prefix_params,
                    )
                else:
                    # GPT path: single TransformerLayer
                    hidden_states, _ = self.mtp_model_layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        attention_bias=attention_bias,
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset,
                        padding_mask=padding_mask,
                        cached_prefix_params=cached_prefix_params,
                    )

        hidden_states = self._postprocess(hidden_states)

        return hidden_states

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        embedding=None,
        cached_prefix_params: Optional[CachedPrefixParams] = None,
    ):
        """
        Execute the forward pass through the Multi-Token Prediction (MTP) layer.

        Args:
            input_ids (Tensor): Input token IDs .
            position_ids (Tensor): Positional IDs of the input tokens.
            hidden_states (Tensor): Hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention, if applicable.
            context_mask (Tensor, optional): Mask for cross-attention context, if applicable.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Tensor, optional): Cosine component of rotary positional embeddings.
            rotary_pos_sin (Tensor, optional): Sine component of rotary positional embeddings.
            sequence_len_offset (Tensor, optional): Offset for sequence length, if applicable.
            embedding (Callable): The embedding module from gpt model to compute the decoder input.
            cached_prefix_params (CachedPrefixParams, optional): Parameters for cached prefix.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        assert context is None, "multi token prediction + cross attention is not yet supported."
        input_ids, position_ids, padding_mask, decoder_input, hidden_states = self._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            padding_mask=padding_mask,
            embedding=embedding,
            hidden_states=hidden_states,
            packed_seq_params=packed_seq_params,
            cached_prefix_params=cached_prefix_params,
        )

        if self.config.recompute_granularity == 'full' and self.training:
            assert (
                cached_prefix_params is None
            ), "Prefix caching is not supported with MTP recompute for now."
            hidden_states = self._checkpointed_forward(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        else:
            hidden_states = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                cached_prefix_params=cached_prefix_params,
            )

        return hidden_states, input_ids, position_ids, padding_mask

    def backward_dw(self):
        self.eh_proj.backward_dw()
        if (
            hasattr(self, "mtp_model_layer")
            and hasattr(self.mtp_model_layer, "backward_dw")
            and callable(self.mtp_model_layer.backward_dw)
        ):
            self.mtp_model_layer.backward_dw()


class MultiTokenPredictionBlock:
    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        padding_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        extra_block_kwargs: Optional[dict] = None,
        embedding=None,
        cached_prefix_params: Optional[CachedPrefixParams] = None,
    ) -> Tensor:
        """
        Perform the forward pass through all of the MTP modules.

        Args:
            hidden_states (Tensor): Hidden states for input token with the shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.

        Returns:
            (Tensor): The mtp loss tensor of shape [b, s].
        """

        if (
            get_args().schedule_method == "dualpipev"
            and embedding.word_embeddings.weight is None
        ):
            from hcu_megatron.core.models.common.language_module.language_module import get_shared_embedding_from_dual_chunk
            embedding.word_embeddings.weight = get_shared_embedding_from_dual_chunk()

        # get hidden states from previous mtp stages
        offset = get_mtp_layer_offset(self.config, self.vp_stage)
        hidden_states_list = list(torch.chunk(hidden_states, 1 + offset, dim=0))
        hidden_states = hidden_states_list[offset]

        if self.config.mtp_detach_heads:
            hidden_states = hidden_states.detach()

        for iteration in range(self.config.mtp_num_layers):
            layer_idx = 0 if self.mtp_use_repeated_layer else iteration
            global_iteration = iteration + offset
            with _fork_recompute_mtp_layer_flag():
                if get_args().recompute_mtp_layer_ids is not None:
                    set_recompute_mtp_layer_flag(global_iteration in get_args().recompute_mtp_layer_ids)
                hidden_states, input_ids, position_ids, padding_mask = self.layers[layer_idx](
                    input_ids=input_ids,
                    position_ids=position_ids,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    padding_mask=padding_mask,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                    embedding=embedding,
                    cached_prefix_params=cached_prefix_params,
                    **(extra_block_kwargs or {}),
                )

            # append the output hidden states of the current mtp layer
            # to the hidden_states_list
            hidden_states_list.append(hidden_states)

        # concat the hidden states of all mtp layers
        hidden_states = torch.cat(hidden_states_list, dim=0)
        return hidden_states

    def backward_dw(self):
        for layer in self.layers:
            layer.backward_dw()
