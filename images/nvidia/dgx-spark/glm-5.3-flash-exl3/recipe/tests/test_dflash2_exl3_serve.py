#!/usr/bin/env python3
"""Pinned-image drift guard and BF16/EXL3 fused context-KV behavior."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "overlay/patch_dflash2_exl3.py"
FIXTURE = ROOT / "tests/fixtures/qwen3_dflash-927d6521.py.txt"
FIXTURE_SHA256 = "40b3a4c7b8893fe92b9e291b566d763a2c6e29712f3a4d56d1a5b246d1815745"


def patch_module():
    spec = importlib.util.spec_from_file_location("draft_patch_test", PATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_runtime_patch_idempotence_and_drift(tmp_path):
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    site, opt = tmp_path / "site", tmp_path / "opt"
    models = site / "model_executor/models"
    models.mkdir(parents=True)
    opt.mkdir()
    target = models / "qwen3_dflash.py"
    target.write_bytes(FIXTURE.read_bytes())
    (opt / "qwen3_dflash2.py").write_bytes((ROOT / "overlay/qwen3_dflash2.py").read_bytes())
    cache = models / "__pycache__"
    cache.mkdir()
    stale = cache / "qwen3_dflash.cpython-312.pyc"
    stale.write_bytes(b"stale")
    env = {**os.environ, "GLM53_SITE": str(site), "GLM53_OPT": str(opt)}
    result = subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    compile(target.read_text(), str(target), "exec")
    first = target.read_bytes()
    assert not stale.exists()
    result = subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == first
    mod = patch_module()
    drifted = FIXTURE.read_text().replace(mod.FUSED_KV_OLD, "")
    target.write_text(drifted)
    result = subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert target.read_text() == drifted


def test_fused_kv_preserves_bf16_and_mixed_staging():
    mod = patch_module()

    def run(weight):
        attn = SimpleNamespace(q_size=32, kv_size=16,
                               qkv_proj=SimpleNamespace(weight=weight))
        ns = {"layers_attn": [attn]}
        exec(textwrap.dedent(mod.FUSED_KV_NEW), ns)
        return ns["kv_weights"][0]

    full = torch.arange(64 * 16, dtype=torch.float32).reshape(64, 16).to(torch.bfloat16)
    bf16_kv = run(full)
    assert torch.equal(bf16_kv, full[32:])
    assert bf16_kv.data_ptr() == full[32:].data_ptr()
    staged = full[32:].clone()
    exl3_kv = run(staged)
    assert torch.equal(exl3_kv, staged)
    assert exl3_kv.data_ptr() == staged.data_ptr()
    with pytest.raises(AssertionError):
        run(full[:16])
