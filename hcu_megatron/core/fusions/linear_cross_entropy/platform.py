# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HCU runtime detection kept outside the PR2256 top-level dispatcher."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping


_HCU_NAME_MARKERS = ("hygon", "hcu", "bw1000")
_HCU_ARCH_RE = re.compile(r"^gfx[0-9]+$")


@dataclass(frozen=True)
class HcuPlatformInfo:
    available: bool
    device_index: int | None
    device_name: str
    arch: str
    hip_runtime: str | None
    reason: str


def _get_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to detect the HCU runtime") from exc
    return torch


def _normalize_arch(value: Any) -> str:
    return str(value or "").lower().split(":", 1)[0]


def _get_property(properties: Any, *names: str) -> Any:
    for name in names:
        if hasattr(properties, name):
            return getattr(properties, name)
    return None


def detect_hcu(
    torch_module: Any | None = None, environ: Mapping[str, str] | None = None
) -> HcuPlatformInfo:
    torch = _get_torch() if torch_module is None else torch_module
    values = os.environ if environ is None else environ
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return HcuPlatformInfo(False, None, "", "", None, "torch.cuda runtime unavailable")

    try:
        device_index = int(cuda.current_device())
        device_name = str(cuda.get_device_name(device_index))
        properties = cuda.get_device_properties(device_index)
    except Exception as exc:
        return HcuPlatformInfo(
            False, None, "", "", None, f"unable to query accelerator properties: {exc}"
        )

    arch = _normalize_arch(
        _get_property(properties, "gcnArchName", "gcn_arch_name", "arch", "architecture")
    )
    if not arch:
        try:
            arch_list = list(cuda.get_arch_list())
        except (AttributeError, TypeError, RuntimeError):
            arch_list = []
        gfx_arches = [_normalize_arch(item) for item in arch_list if str(item).startswith("gfx")]
        if len(gfx_arches) == 1:
            arch = gfx_arches[0]

    version = getattr(torch, "version", None)
    hip_runtime = getattr(version, "hip", None) if version is not None else None
    normalized_name = device_name.lower()
    name_is_hcu = any(marker in normalized_name for marker in _HCU_NAME_MARKERS)
    arch_is_hcu = bool(_HCU_ARCH_RE.fullmatch(arch))
    runtime_hints = bool(
        hip_runtime
        or values.get("HCCL_HOME")
        or values.get("HCU_VISIBLE_DEVICES")
        or values.get("HIP_VISIBLE_DEVICES")
    )

    if runtime_hints and (name_is_hcu or arch_is_hcu):
        return HcuPlatformInfo(
            True,
            device_index,
            device_name,
            arch,
            str(hip_runtime) if hip_runtime else None,
            "HCU identity matched runtime and device/architecture evidence",
        )
    return HcuPlatformInfo(
        False,
        device_index,
        device_name,
        arch,
        str(hip_runtime) if hip_runtime else None,
        "device does not match HCU runtime identity",
    )


def is_hcu_available(torch_module: Any | None = None) -> bool:
    try:
        return detect_hcu(torch_module=torch_module).available
    except RuntimeError:
        return False


def require_hcu(torch_module: Any | None = None) -> HcuPlatformInfo:
    info = detect_hcu(torch_module=torch_module)
    if not info.available:
        raise RuntimeError(f"HCU Linear CE backend unavailable: {info.reason}")

    return info


def get_hcu_arch(torch_module: Any | None = None) -> str:
    return require_hcu(torch_module=torch_module).arch


__all__ = [
    "HcuPlatformInfo",
    "detect_hcu",
    "get_hcu_arch",
    "is_hcu_available",
    "require_hcu",
]
