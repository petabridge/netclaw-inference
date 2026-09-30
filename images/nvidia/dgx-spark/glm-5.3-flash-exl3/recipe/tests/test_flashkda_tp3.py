#!/usr/bin/env python3
"""Pinned-source composition, recurrence preservation and prelaunch guards."""
import ast
import importlib.util
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


flash = load("flash", ROOT / "overlay/patch_flashkda_tp3.py")
fp8 = load("fp8", ROOT / "overlay/patch_dense_fp8.py")
source = (ROOT / "tests/fixtures/kda-487ecf187.py.txt").read_text()


class FlashPatchTests(unittest.TestCase):
    def test_composition_and_idempotence(self):
        actual = flash.apply(source, "fixture")
        ast.parse(actual)
        self.assertEqual(flash.apply(actual, "fixture"), actual)
        self.assertEqual(flash.apply(source.replace(fp8.KDA_OLD, fp8.KDA_NEW), "fp8-first"), actual.replace(fp8.KDA_OLD, fp8.KDA_NEW))
        for fn in ("fused_recurrent_kda",):
            def calls(text):
                return [ast.dump(n) for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == fn]
            self.assertEqual(calls(source), calls(actual))
            self.assertEqual(len(calls(actual)), 2)

    def test_anchor_or_partial_patch_refused(self):
        for name, old, _ in flash.ANCHORS:
            with self.subTest(name=name), self.assertRaises(SystemExit):
                flash.apply(source.replace(old, "", 1), name)
        with self.assertRaises(RuntimeError): flash.apply(source + "\n# HAREM-FLASHKDA\n", "partial")

    def test_resolver_rejects_unbounded_gate(self):
        text = flash.apply(source, "fixture")
        nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in ("_harem_log_kda_backend", "_harem_kda_prefill_backend")]
        bf16 = object()
        ns = {"os": os, "_HAREM_KDA_BACKEND_LOGGED": set(), "torch": types.SimpleNamespace(bfloat16=bf16),
              "current_platform": types.SimpleNamespace(is_cuda=lambda: True, get_device_capability=lambda: types.SimpleNamespace(major=12))}
        exec(compile(ast.Module(nodes, type_ignores=[]), "resolver", "exec"), ns)
        with patch.dict(os.environ, {"HAREM_KDA_FLASHKDA": "1"}):
            f = ns["_harem_kda_prefill_backend"]
            self.assertEqual(f({}, 128, bf16, -5.0), "flashkda")
            with self.assertRaises(RuntimeError): f({}, 128, bf16, None)
            self.assertEqual(f({"kda_prefill_backend": "triton"}, 128, bf16, None), "triton")

    def test_capacity_guard_precedes_workspace(self):
        text = flash.apply(source, "fixture")
        node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "_harem_flashkda_prefill")
        def forbidden_workspace(): raise AssertionError("workspace reached")
        ns = {"torch": types.SimpleNamespace(Tensor=object), "current_workspace_manager": forbidden_workspace}
        exec(compile(ast.Module([node], type_ignores=[]), "wrapper", "exec"), ns)
        f = ns[node.name]
        obj = types.SimpleNamespace(_flashkda_buffer_specs=(), _flashkda_max_tokens=4096, _flashkda_max_seqs=8)
        for tokens, sequences, starts in ((4097, 1, 2), (4096, 9, 10), (4096, 8, 8)):
            q = types.SimpleNamespace(ndim=4, shape=(1, tokens, 22, 128))
            state = types.SimpleNamespace(shape=(sequences,22,128,128))
            with self.subTest(tokens=tokens,sequences=sequences), self.assertRaises(RuntimeError):
                f(obj, q, q, q, q, q, state, types.SimpleNamespace(numel=lambda: starts), None)


if __name__ == "__main__": unittest.main()
