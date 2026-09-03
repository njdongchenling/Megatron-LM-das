# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Load the architecture-specific HCU Linear CE shared library."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping


class HcuLinearCeExtensionError(RuntimeError):
    """Raised when the native runtime is unavailable or incomplete."""


_ARCH_RE = re.compile(r"^gfx[0-9]+$")
_LOADED_PATHS: set[Path] = set()


def _get_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise HcuLinearCeExtensionError(
            "PyTorch is required to load the HCU Linear CE extension"
        ) from exc
    return torch


def _validate_arch(arch: str) -> str:
    normalized = str(arch).lower().split(":", 1)[0]
    if not _ARCH_RE.fullmatch(normalized):
        raise HcuLinearCeExtensionError(f"invalid HCU architecture: {arch!r}")
    return normalized


def default_extension_path(
    arch: str, environ: Mapping[str, str] | None = None
) -> Path:
    values = os.environ if environ is None else environ
    override = values.get("HCU_LINEAR_CE_EXTENSION_PATH")
    if override:
        return Path(override).expanduser()

    normalized_arch = _validate_arch(arch)
    library_name = f"libhcu_linear_ce_{normalized_arch}.so"
    try:
        from hcu_linear_ce_artifacts import paths as artifact_paths
    except ImportError:
        return Path(__file__).with_name("lib") / library_name

    try:
        return Path(artifact_paths.extension_path(normalized_arch))
    except TypeError as exc:
        raise HcuLinearCeExtensionError(
            "the installed HCU Linear CE artifact package is obsolete; "
            "install an architecture-aware artifact wheel"
        ) from exc


def _registered_ops(torch_module: Any) -> Any | None:
    namespace = getattr(getattr(torch_module, "ops", None), "hcu_linear_ce", None)
    if namespace is None:
        return None
    if getattr(namespace, "forward", None) is None or getattr(namespace, "backward", None) is None:
        return None
    return namespace


def require_ops(
    arch: str,
    torch_module: Any | None = None,
    environ: Mapping[str, str] | None = None,
) -> Any:
    torch = _get_torch() if torch_module is None else torch_module
    registered = _registered_ops(torch)
    if registered is not None:
        return registered

    library_path = default_extension_path(arch, environ=environ).resolve()
    if not library_path.is_file():
        raise HcuLinearCeExtensionError(
            "HCU Linear CE extension is not installed or loaded. Build the native bridge, "
            "set HCU_LINEAR_CE_EXTENSION_PATH to the architecture-specific shared library, "
            f"and load it with torch.ops.load_library(). Expected path: {library_path}"
        )
    try:
        if library_path not in _LOADED_PATHS:
            torch.ops.load_library(str(library_path))
            _LOADED_PATHS.add(library_path)
    except (OSError, RuntimeError) as exc:
        raise HcuLinearCeExtensionError(
            f"failed to load HCU Linear CE extension {library_path}: {exc}"
        ) from exc

    registered = _registered_ops(torch)
    if registered is None:
        raise HcuLinearCeExtensionError(
            "loaded HCU Linear CE extension does not register both "
            "torch.ops.hcu_linear_ce.forward and backward"
        )
    return registered


def invoke_forward(
    hidden: Any,
    weight: Any,
    labels: Any,
    ignore_index: int,
    log_kernel: bool,
    arch: str,
) -> tuple[Any, Any, Any, Any]:
    outputs = require_ops(arch).forward(hidden, weight, labels, ignore_index, log_kernel)
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
        raise HcuLinearCeExtensionError(
            "torch.ops.hcu_linear_ce.forward must return "
            "(loss, maximum, acc, num_valid_tokens)"
        )
    return tuple(outputs)  # type: ignore[return-value]


def invoke_backward(
    dlogprobs: Any,
    global_hidden: Any,
    weight: Any,
    labels: Any,
    maximum: Any,
    acc: Any,
    num_valid_tokens: Any,
    log_kernel: bool,
    arch: str,
) -> tuple[Any, Any]:
    outputs = require_ops(arch).backward(
        dlogprobs,
        global_hidden,
        weight,
        labels,
        maximum,
        acc,
        num_valid_tokens,
        log_kernel,
    )
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise HcuLinearCeExtensionError(
            "torch.ops.hcu_linear_ce.backward must return (d_hidden, d_weight)"
        )
    return tuple(outputs)  # type: ignore[return-value]


def reset_loader_state() -> None:
    _LOADED_PATHS.clear()


__all__ = [
    "HcuLinearCeExtensionError",
    "default_extension_path",
    "invoke_backward",
    "invoke_forward",
    "require_ops",
    "reset_loader_state",
]
