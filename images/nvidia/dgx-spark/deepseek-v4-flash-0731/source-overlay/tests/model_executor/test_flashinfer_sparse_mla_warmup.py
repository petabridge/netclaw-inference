# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
    _DSV4_PORTABLE_PREFILL_WARM_SHAPES,
    _resolve_dsv4_portable_prefill_num_heads,
)


class _FakeModel:
    def __init__(self, modules: list[SimpleNamespace]) -> None:
        self._modules = modules

    def modules(self):
        return iter(self._modules)


class _Pad32Attention:
    n_local_heads = 32
    head_dim = 512
    padded_heads = 32

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return 32 if num_heads <= 32 else 64


class _Pad64Attention:
    n_local_heads = 32
    head_dim = 512
    padded_heads = 64

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return 64 if num_heads <= 64 else 128


def _runner_from_modules(modules: list[Any]) -> Any:
    return SimpleNamespace(
        model=_FakeModel(modules),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    num_attention_heads=64,
                    head_dim=512,
                )
            ),
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
        ),
    )


def _runner(*padded_heads: int) -> Any:
    modules = [
        SimpleNamespace(
            n_local_heads=32,
            padded_heads=value,
            head_dim=512,
        )
        for value in padded_heads
    ]
    return _runner_from_modules(modules)


def test_portable_prefill_warmup_uses_runtime_padded_heads() -> None:
    assert _resolve_dsv4_portable_prefill_num_heads(_runner(32, 32)) == (32, 64)


def test_portable_prefill_warmup_accepts_multiple_head_layouts() -> None:
    # FlashMLA pads local 32 -> 64 while SM120 FlashInfer keeps 32.
    assert _resolve_dsv4_portable_prefill_num_heads(_runner(32, 64)) == (32, 64)


def test_portable_prefill_warmup_includes_classmethod_padding() -> None:
    modules = [_Pad32Attention(), _Pad64Attention()]
    assert _resolve_dsv4_portable_prefill_num_heads(
        _runner_from_modules(modules)
    ) == (32, 64)


def test_portable_prefill_warmup_rejects_missing_attention_layout() -> None:
    with pytest.raises(RuntimeError, match="padded Q-head count"):
        _resolve_dsv4_portable_prefill_num_heads(_runner())


def test_portable_prefill_warmup_rejects_padding_below_tp_local_heads() -> None:
    with pytest.raises(RuntimeError, match="below local heads 32"):
        _resolve_dsv4_portable_prefill_num_heads(_runner(16))


def test_portable_prefill_warmup_rejects_nondivisible_global_heads() -> None:
    runner = _runner(32)
    runner.vllm_config.model_config.hf_config.num_attention_heads = 65
    with pytest.raises(RuntimeError, match="not divisible by TP=2"):
        _resolve_dsv4_portable_prefill_num_heads(runner)


def _triton_scalar_specialization_class(value: int) -> str:
    if value == 1:
        return "equal_one"
    if value % 16 == 0:
        return "divisible_16"
    return "unaligned"


def test_portable_prefill_warmup_covers_runtime_scalar_specializations() -> None:
    classes = {
        (
            _triton_scalar_specialization_class(num_candidates),
            _triton_scalar_specialization_class(num_kv_rows),
        )
        for _, num_candidates, num_kv_rows in _DSV4_PORTABLE_PREFILL_WARM_SHAPES
    }
    assert {
        ("divisible_16", "divisible_16"),
        ("unaligned", "unaligned"),
        ("unaligned", "divisible_16"),
        ("equal_one", "equal_one"),
        ("equal_one", "unaligned"),
    } <= classes
