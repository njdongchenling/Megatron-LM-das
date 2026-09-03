# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""PR2256-compatible entry points for the HCU Linear CE native backend."""

from __future__ import annotations

import math
import os
from typing import Any, Mapping

from . import extension
from .platform import require_hcu


def _get_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required by the HCU Linear CE backend") from exc
    return torch


def _env_flag(name: str, environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    value = values.get(name, "0").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{name} must be a boolean value, got {values.get(name)!r}")


def _is_contiguous(tensor: Any) -> bool:
    return bool(tensor.is_contiguous())


def _validate_parallelism(tp_group: Any, sequence_parallel: bool) -> None:
    if tp_group is not None or sequence_parallel:
        raise NotImplementedError(
            "HCU Linear CE backend supports DP only: tp_group must be None and "
            "sequence_parallel must be False"
        )


def _validate_and_flatten(hidden: Any, weight: Any, labels: Any, torch: Any) -> tuple[Any, Any]:
    if hidden.dim() not in (2, 3):
        raise ValueError(f"hidden must be 2D or 3D, got dim={hidden.dim()}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got dim={weight.dim()}")
    if labels.dim() != hidden.dim() - 1:
        raise ValueError(
            f"labels dim must be hidden.dim()-1, got hidden.dim={hidden.dim()}, "
            f"labels.dim={labels.dim()}"
        )
    if not all(_is_contiguous(tensor) for tensor in (hidden, weight, labels)):
        raise ValueError("hidden, weight, and labels must be contiguous")
    if not all(bool(getattr(tensor, "is_cuda", False)) for tensor in (hidden, weight, labels)):
        raise ValueError("hidden, weight, and labels must be HCU device tensors")
    if weight.device != hidden.device or labels.device != hidden.device:
        raise ValueError("hidden, weight, and labels must be on the same HCU device")
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("hidden and weight must use BF16")
    if labels.dtype != torch.int64:
        raise ValueError("labels must use int64")
    if hidden.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"hidden D={hidden.shape[-1]} does not match weight D={weight.shape[1]}"
        )
    num_tokens = math.prod(hidden.shape[:-1])
    if labels.numel() != num_tokens:
        raise ValueError(
            f"labels contain {labels.numel()} elements but hidden has N={num_tokens} tokens"
        )
    return hidden.view(-1, hidden.shape[-1]), labels.view(-1)


def forward(
    hidden: Any,
    weight: Any,
    labels: Any,
    tp_group: Any = None,
    reduction: str = "mean",
    ignore_index: int = -100,
    sequence_parallel: bool = False,
) -> tuple[Any, Any, Any, Any, int, int, Any]:
    """Return the seven values consumed by PR2256 LinearCrossEntropy.forward."""
    _validate_parallelism(tp_group, sequence_parallel)
    if reduction != "mean" or ignore_index != -100:
        raise NotImplementedError(
            "HCU Linear CE backend supports reduction='mean' and ignore_index=-100 only"
        )
    torch = _get_torch()
    platform = require_hcu(torch_module=torch)
    hidden_view, labels_view = _validate_and_flatten(hidden, weight, labels, torch)
    loss, maximum, acc, num_valid_tokens = extension.invoke_forward(
        hidden_view,
        weight,
        labels_view,
        ignore_index,
        _env_flag("HCU_LINEAR_CE_LOG_KERNEL"),
        platform.arch,
    )
    return loss, maximum, acc, num_valid_tokens, 0, 1, hidden


def backward(
    dlogprobs: Any,
    global_hidden: Any,
    weight: Any,
    labels: Any,
    maximum: Any,
    acc: Any,
    num_valid_tokens: Any,
    reduction: str,
    ignore_index: int,
    tp_group: Any,
    tp_rank: int,
    tp_world_size: int,
    sequence_parallel: bool,
) -> tuple[Any, Any]:
    """Return ``(d_hidden, d_weight)`` to PR2256 LinearCrossEntropy.backward."""
    _validate_parallelism(tp_group, sequence_parallel)
    if tp_rank != 0 or tp_world_size != 1:
        raise NotImplementedError(
            "HCU Linear CE backend supports tp_rank=0 and tp_world_size=1 only"
        )
    if reduction != "mean" or ignore_index != -100:
        raise NotImplementedError(
            "HCU Linear CE backend supports reduction='mean' and ignore_index=-100 only"
        )
    torch = _get_torch()
    platform = require_hcu(torch_module=torch)
    hidden_view, labels_view = _validate_and_flatten(global_hidden, weight, labels, torch)
    for name, tensor in (
        ("dlogprobs", dlogprobs),
        ("maximum", maximum),
        ("acc", acc),
        ("num_valid_tokens", num_valid_tokens),
    ):
        if not bool(getattr(tensor, "is_cuda", False)) or tensor.device != global_hidden.device:
            raise ValueError(f"{name} must be on the same HCU device as global_hidden")
        if not _is_contiguous(tensor):
            raise ValueError(f"{name} must be contiguous")
    num_tokens = int(hidden_view.shape[0])
    if dlogprobs.dim() != 0:
        raise ValueError("dlogprobs must be a scalar for reduction='mean'")
    if tuple(maximum.shape) != (num_tokens,) or tuple(acc.shape) != (num_tokens,):
        raise ValueError("maximum and acc must each have shape (N,)")
    if num_valid_tokens.dim() != 0 or num_valid_tokens.dtype != torch.int64:
        raise ValueError("num_valid_tokens must be a scalar int64 tensor")
    d_hidden, d_weight = extension.invoke_backward(
        dlogprobs,
        hidden_view,
        weight,
        labels_view,
        maximum,
        acc,
        num_valid_tokens,
        _env_flag("HCU_LINEAR_CE_LOG_KERNEL"),
        platform.arch,
    )
    return d_hidden.view(*global_hidden.shape), d_weight


__all__ = ["backward", "forward"]
