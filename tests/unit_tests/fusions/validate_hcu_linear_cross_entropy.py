#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run an HCU Linear CE forward/backward smoke test through Megatron."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch


IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--d", type=int, default=4096)
    parser.add_argument("--v", type=int, default=32000)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--all-ignore", action="store_true")
    return parser.parse_args()


def run_once(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor):
    from hcu_megatron.core.fusions.fused_linear_cross_entropy import (
        linear_cross_entropy,
    )

    hidden_input = hidden.detach().requires_grad_(True)
    weight_input = weight.detach().requires_grad_(True)
    loss = linear_cross_entropy(
        hidden_input,
        weight_input,
        labels,
        tp_group=None,
        reduction="mean",
        ignore_index=IGNORE_INDEX,
        sequence_parallel=False,
    )
    loss.backward()
    torch.cuda.synchronize(hidden.device)
    return loss.detach(), hidden_input.grad, weight_input.grad


def main() -> None:
    args = parse_args()
    if args.library is not None:
        library = args.library.expanduser().resolve()
        if not library.is_file():
            raise FileNotFoundError(library)
        os.environ["HCU_LINEAR_CE_EXTENSION_PATH"] = str(library)
    if not torch.cuda.is_available():
        raise RuntimeError("HCU runtime is unavailable through torch.cuda")
    torch.cuda.set_device(args.device)
    torch.manual_seed(20260901)
    device = torch.device("cuda", args.device)
    hidden = (torch.randn(args.n, args.d, device=device) * 0.02).to(torch.bfloat16)
    weight = (torch.randn(args.v, args.d, device=device) * 0.02).to(torch.bfloat16)
    if args.all_ignore:
        labels = torch.full(
            (args.n,), IGNORE_INDEX, device=device, dtype=torch.long
        )
    else:
        labels = torch.arange(args.n, device=device, dtype=torch.long) % args.v
        labels[::17] = IGNORE_INDEX

    first = run_once(hidden, weight, labels)
    second = run_once(hidden, weight, labels)
    if args.all_ignore:
        if not math.isnan(first[0].item()) or not math.isnan(second[0].item()):
            raise AssertionError("all-ignore loss must be NaN")
        for execution, outputs in (("first", first), ("second", second)):
            if outputs[1].count_nonzero().item() or outputs[2].count_nonzero().item():
                raise AssertionError(
                    f"all-ignore gradients must be zero in {execution} execution"
                )
    else:
        for name, tensor in (("loss", first[0]), ("dHidden", first[1]), ("dWeight", first[2])):
            if not torch.isfinite(tensor.float()).all().item():
                raise AssertionError(f"{name} contains non-finite values")
        torch.testing.assert_close(second[0].float(), first[0].float(), rtol=0, atol=0)
        torch.testing.assert_close(second[1].float(), first[1].float(), rtol=0, atol=0)
        torch.testing.assert_close(second[2].float(), first[2].float(), rtol=0, atol=0)

    properties = torch.cuda.get_device_properties(args.device)
    print(
        "MEGATRON_LINEAR_CE_SMOKE_PASS",
        f"device={torch.cuda.get_device_name(args.device)!r}",
        f"arch={getattr(properties, 'gcnArchName', '')}",
        f"n={args.n}",
        f"d={args.d}",
        f"v={args.v}",
        f"all_ignore={int(args.all_ignore)}",
        f"loss={first[0].item()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
