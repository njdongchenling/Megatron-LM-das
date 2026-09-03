# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for the top-level PR2256-style Linear CE API.

New callers should import from
``hcu_megatron.core.fusions.fused_linear_cross_entropy``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    if name in {"LinearCrossEntropy", "linear_cross_entropy"}:
        module = import_module("..fused_linear_cross_entropy", package=__name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LinearCrossEntropy", "linear_cross_entropy"]
