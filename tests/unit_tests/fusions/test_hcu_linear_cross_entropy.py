# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""No-launch tests for the Megatron HCU Linear CE integration."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[3]
OPERATOR_ROOT = REPO_ROOT / "hcu_megatron" / "core" / "fusions" / "linear_cross_entropy"
PUBLIC_API_PATH = REPO_ROOT / "hcu_megatron" / "core" / "fusions" / "fused_linear_cross_entropy.py"
PACKAGE_NAME = "_hcu_linear_ce_under_test"


def _load_module(name: str):
    qualified_name = f"{PACKAGE_NAME}.{name}"
    spec = importlib.util.spec_from_file_location(
        qualified_name, OPERATOR_ROOT / f"{name}.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {qualified_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(OPERATOR_ROOT)]
sys.modules[PACKAGE_NAME] = package
platform_module = _load_module("platform")
extension_module = _load_module("extension")
entry_module = _load_module("entry")


class FakeTensor:
    def __init__(
        self,
        shape,
        dtype,
        *,
        device: str = "hcu:0",
        contiguous: bool = True,
        is_cuda: bool = True,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.is_cuda = is_cuda
        self._contiguous = contiguous

    def dim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return self._contiguous

    def numel(self) -> int:
        return math.prod(self.shape)

    def view(self, *shape):
        resolved = list(shape)
        if -1 in resolved:
            known = math.prod(value for value in resolved if value != -1)
            resolved[resolved.index(-1)] = self.numel() // known
        return FakeTensor(
            resolved,
            self.dtype,
            device=self.device,
            contiguous=self._contiguous,
            is_cuda=self.is_cuda,
        )


class FakeTorch:
    bfloat16 = "bfloat16"
    int64 = "int64"


def _inputs(hidden_shape=(2048, 4096), label_shape=(2048,)):
    return (
        FakeTensor(hidden_shape, "bfloat16"),
        FakeTensor((32000, 4096), "bfloat16"),
        FakeTensor(label_shape, "int64"),
    )


class TestPlatform(unittest.TestCase):
    def test_detects_current_and_future_gfx_architectures(self) -> None:
        for arch in ("gfx936", "gfx938"):
            properties = SimpleNamespace(gcnArchName=f"{arch}:sramecc+:xnack-")
            cuda = SimpleNamespace(
                is_available=lambda: True,
                current_device=lambda: 1,
                get_device_name=lambda _device: "Hygon HCU",
                get_device_properties=lambda _device: properties,
            )
            torch_module = SimpleNamespace(
                cuda=cuda, version=SimpleNamespace(hip="6.3")
            )
            info = platform_module.detect_hcu(torch_module=torch_module, environ={})
            self.assertTrue(info.available)
            self.assertEqual(info.arch, arch)


class TestExtension(unittest.TestCase):
    def tearDown(self) -> None:
        extension_module.reset_loader_state()

    def test_default_path_uses_detected_architecture(self) -> None:
        with mock.patch.dict(
            sys.modules, {"hcu_linear_ce_artifacts": None}
        ):
            self.assertEqual(
                extension_module.default_extension_path("gfx938").name,
                "libhcu_linear_ce_gfx938.so",
            )
        override = {"HCU_LINEAR_CE_EXTENSION_PATH": "~/custom-linear-ce.so"}
        self.assertEqual(
            extension_module.default_extension_path("gfx936", override),
            Path("~/custom-linear-ce.so").expanduser(),
        )
        with self.assertRaisesRegex(
            extension_module.HcuLinearCeExtensionError, "invalid HCU architecture"
        ):
            extension_module.default_extension_path("not-an-arch")

    def test_loads_one_shared_library_and_reuses_registered_ops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            library = Path(temporary_directory) / "libhcu_linear_ce_gfx936.so"
            library.touch()
            namespace = SimpleNamespace(forward=mock.Mock(), backward=mock.Mock())
            ops = SimpleNamespace()

            def load_library(_path: str) -> None:
                ops.hcu_linear_ce = namespace

            torch_module = SimpleNamespace(ops=ops)
            torch_module.ops.load_library = mock.Mock(side_effect=load_library)
            environ = {"HCU_LINEAR_CE_EXTENSION_PATH": str(library)}
            first = extension_module.require_ops(
                "gfx936", torch_module=torch_module, environ=environ
            )
            second = extension_module.require_ops(
                "gfx936", torch_module=torch_module, environ=environ
            )
            self.assertIs(first, namespace)
            self.assertIs(second, namespace)
            torch_module.ops.load_library.assert_called_once_with(str(library.resolve()))


class TestEntry(unittest.TestCase):
    def test_forward_passes_detected_architecture_to_native_loader(self) -> None:
        hidden, weight, labels = _inputs()
        native_outputs = ("loss", "maximum", "acc", "valid")
        with (
            mock.patch.object(entry_module, "_get_torch", return_value=FakeTorch()),
            mock.patch.object(
                entry_module,
                "require_hcu",
                return_value=SimpleNamespace(arch="gfx938"),
            ),
            mock.patch.object(
                entry_module.extension,
                "invoke_forward",
                return_value=native_outputs,
            ) as invoke,
        ):
            outputs = entry_module.forward(hidden, weight, labels)
        self.assertEqual(outputs[:4], native_outputs)
        self.assertEqual(outputs[4:6], (0, 1))
        self.assertIs(outputs[6], hidden)
        self.assertEqual(invoke.call_args.args[-1], "gfx938")

    def test_backward_restores_three_dimensional_hidden_shape(self) -> None:
        hidden, weight, labels = _inputs((32, 64, 4096), (32, 64))
        dloss = FakeTensor((), "float32")
        maximum = FakeTensor((2048,), "float32")
        saved_lse = FakeTensor((2048,), "float32")
        valid_count = FakeTensor((), "int64")
        flat_dhidden = FakeTensor((2048, 4096), "bfloat16")
        dweight = FakeTensor((32000, 4096), "bfloat16")
        with (
            mock.patch.object(entry_module, "_get_torch", return_value=FakeTorch()),
            mock.patch.object(
                entry_module,
                "require_hcu",
                return_value=SimpleNamespace(arch="gfx936"),
            ),
            mock.patch.object(
                entry_module.extension,
                "invoke_backward",
                return_value=(flat_dhidden, dweight),
            ) as invoke,
        ):
            dhidden, observed_dweight = entry_module.backward(
                dloss,
                hidden,
                weight,
                labels,
                maximum,
                saved_lse,
                valid_count,
                "mean",
                -100,
                None,
                0,
                1,
                False,
            )
        self.assertEqual(dhidden.shape, hidden.shape)
        self.assertIs(observed_dweight, dweight)
        self.assertEqual(invoke.call_args.args[-1], "gfx936")

    def test_rejects_unsupported_parallelism_and_reduction(self) -> None:
        hidden, weight, labels = _inputs()
        with self.assertRaisesRegex(NotImplementedError, "DP only"):
            entry_module.forward(hidden, weight, labels, tp_group=object())
        with self.assertRaisesRegex(NotImplementedError, "reduction='mean'"):
            entry_module.forward(hidden, weight, labels, reduction="sum")


class TestRepositoryContract(unittest.TestCase):
    def test_public_signature_and_autograd_wrapper_are_preserved(self) -> None:
        tree = ast.parse(PUBLIC_API_PATH.read_text(encoding="utf-8"))
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        public = functions["linear_cross_entropy"]
        self.assertEqual(
            [argument.arg for argument in public.args.args],
            [
                "hidden",
                "weight",
                "labels",
                "tp_group",
                "reduction",
                "ignore_index",
                "sequence_parallel",
            ],
        )
        text = PUBLIC_API_PATH.read_text(encoding="utf-8")
        self.assertIn("class LinearCrossEntropy(torch.autograd.Function)", text)
        self.assertIn("from .linear_cross_entropy import entry as gpu_entry", text)

    def test_megatron_contains_no_kernel_or_manifest_implementation(self) -> None:
        self.assertFalse((OPERATOR_ROOT / "manifest.py").exists())
        self.assertFalse((OPERATOR_ROOT / "registry.py").exists())
        self.assertFalse((OPERATOR_ROOT / "kernels").exists())
        self.assertFalse((OPERATOR_ROOT / "native").exists())
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(OPERATOR_ROOT.glob("*.py"))
        )
        for forbidden in (
            "hipModuleLoadData",
            "hipModuleGetFunction",
            "hipModuleLaunchKernel",
            "ExactShapeSpec",
            "shape_plans",
        ):
            self.assertNotIn(forbidden, source)

    def test_sources_compile_and_have_submission_headers(self) -> None:
        paths = sorted(OPERATOR_ROOT.glob("*.py")) + [PUBLIC_API_PATH, Path(__file__)]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            compile(text, str(path), "exec")
            head = "\n".join(text.splitlines()[:4])
            self.assertIn("Hygon Information Technology Co., Ltd.", head)
            self.assertIn("SPDX-License-Identifier: Apache-2.0", head)


if __name__ == "__main__":
    unittest.main()
