# SPDX-License-Identifier: Apache-2.0
"""Focused tests for DeepSeek-V4 default torch.compile fail-closed policy."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

from vllm.config.compilation import CompilationMode, CUDAGraphImplementation
from vllm.config.vllm import VllmConfig


def _cfg(mode: CompilationMode, arches: list[str]) -> VllmConfig:
    cfg = object.__new__(VllmConfig)
    cfg.model_config = SimpleNamespace(architectures=arches)
    cfg.compilation_config = SimpleNamespace(
        mode=mode,
        cudagraph_strict=True,
        cudagraph_implementation=CUDAGraphImplementation.REGULAR,
    )
    return cfg


def test_dsv4_forces_none_without_opt_in() -> None:
    cfg = _cfg(CompilationMode.VLLM_COMPILE, ["DeepseekV4ForCausalLM"])
    with mock.patch.dict(os.environ, {"VLLM_DSV4_ENABLE_TORCH_COMPILE": "0"}, clear=False):
        cfg._maybe_force_dsv4_torch_compile_off()
    assert cfg.compilation_config.mode is CompilationMode.NONE


def test_dsv4_allows_compile_with_opt_in() -> None:
    cfg = _cfg(CompilationMode.VLLM_COMPILE, ["DeepseekV4ForCausalLM"])
    with mock.patch.dict(os.environ, {"VLLM_DSV4_ENABLE_TORCH_COMPILE": "1"}, clear=False):
        cfg._maybe_force_dsv4_torch_compile_off()
    assert cfg.compilation_config.mode is CompilationMode.VLLM_COMPILE


def test_non_dsv4_unchanged() -> None:
    cfg = _cfg(CompilationMode.VLLM_COMPILE, ["LlamaForCausalLM"])
    with mock.patch.dict(os.environ, {"VLLM_DSV4_ENABLE_TORCH_COMPILE": "0"}, clear=False):
        cfg._maybe_force_dsv4_torch_compile_off()
    assert cfg.compilation_config.mode is CompilationMode.VLLM_COMPILE
