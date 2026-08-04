# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

from vllm.config import (
    CompilationConfig,
    CUDAGraphImplementation,
    CUDAGraphMode,
    VllmConfig,
)


def _selector_config(
    requested: CUDAGraphImplementation | str = CUDAGraphImplementation.AUTO,
    *,
    architecture: str = "LlamaForCausalLM",
    strict: bool = False,
    enforce_eager: bool = False,
) -> VllmConfig:
    config = object.__new__(VllmConfig)
    config.compilation_config = CompilationConfig(
        cudagraph_implementation=requested,
        cudagraph_strict=strict,
    )
    config.model_config = SimpleNamespace(
        architectures=[architecture],
        enforce_eager=enforce_eager,
    )
    return config


def test_case_insensitive_native_value() -> None:
    config = CompilationConfig(cudagraph_implementation="ReGuLaR")
    assert config.cudagraph_implementation is CUDAGraphImplementation.REGULAR


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (CUDAGraphImplementation.REGULAR, CUDAGraphImplementation.REGULAR),
        (CUDAGraphImplementation.BREAKABLE, CUDAGraphImplementation.BREAKABLE),
    ],
)
def test_explicit_native_value_is_authoritative(
    monkeypatch: pytest.MonkeyPatch,
    requested: CUDAGraphImplementation,
    expected: CUDAGraphImplementation,
) -> None:
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    config = _selector_config(requested)
    assert config.resolve_cudagraph_implementation() is expected
    assert config.compilation_config.cudagraph_implementation is expected


def test_auto_preserves_deepseek_v4_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    config = _selector_config(architecture="DeepseekV4ForCausalLM")
    assert (
        config.resolve_cudagraph_implementation()
        is CUDAGraphImplementation.BREAKABLE
    )


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ("0", CUDAGraphImplementation.REGULAR),
        ("1", CUDAGraphImplementation.BREAKABLE),
    ],
)
def test_auto_legacy_compatibility_warns(
    monkeypatch: pytest.MonkeyPatch,
    legacy: str,
    expected: CUDAGraphImplementation,
) -> None:
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", legacy)
    config = _selector_config()
    with pytest.warns(DeprecationWarning, match="VLLM_USE_BREAKABLE_CUDAGRAPH"):
        resolved = config.resolve_cudagraph_implementation()
    assert resolved is expected


@pytest.mark.parametrize(
    ("requested", "legacy"),
    [
        (CUDAGraphImplementation.REGULAR, "1"),
        (CUDAGraphImplementation.BREAKABLE, "0"),
    ],
)
def test_conflicting_native_and_legacy_fail(
    monkeypatch: pytest.MonkeyPatch,
    requested: CUDAGraphImplementation,
    legacy: str,
) -> None:
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", legacy)
    config = _selector_config(requested)
    with pytest.raises(ValueError, match="Conflicting native and legacy"):
        config.resolve_cudagraph_implementation()


@pytest.mark.parametrize("legacy", ["", "true", "false", "2", "01", " yes "])
def test_invalid_legacy_value_fails(
    monkeypatch: pytest.MonkeyPatch,
    legacy: str,
) -> None:
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", legacy)
    config = _selector_config()
    with pytest.raises(ValueError, match="must be exactly 0 or 1"):
        config.resolve_cudagraph_implementation()


def test_matching_native_and_legacy_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    config = _selector_config(CUDAGraphImplementation.REGULAR)
    with pytest.warns(DeprecationWarning, match="VLLM_USE_BREAKABLE_CUDAGRAPH"):
        resolved = config.resolve_cudagraph_implementation()
    assert resolved is CUDAGraphImplementation.REGULAR


def test_native_json_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    config = _selector_config()
    with pytest.warns(DeprecationWarning):
        config.resolve_cudagraph_implementation()

    adapter = TypeAdapter(CompilationConfig)
    payload = adapter.dump_json(
        config.compilation_config,
        include={"cudagraph_implementation", "cudagraph_strict"},
    )
    restored = adapter.validate_json(payload)
    assert restored.cudagraph_implementation is CUDAGraphImplementation.REGULAR


def test_worker_pickle_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    config = _selector_config(CUDAGraphImplementation.REGULAR)
    config.resolve_cudagraph_implementation()
    worker = pickle.loads(pickle.dumps(config))
    assert (
        worker.compilation_config.cudagraph_implementation
        is CUDAGraphImplementation.REGULAR
    )
    assert (
        worker.compilation_config.cudagraph_implementation_requested
        is CUDAGraphImplementation.REGULAR
    )


def test_native_value_changes_compilation_hash() -> None:
    regular = CompilationConfig(cudagraph_implementation="regular")
    breakable = CompilationConfig(cudagraph_implementation="breakable")
    assert regular.compute_hash() != breakable.compute_hash()


def test_strict_regular_rejects_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    config = _selector_config(
        CUDAGraphImplementation.REGULAR,
        strict=True,
        enforce_eager=True,
    )
    config.resolve_cudagraph_implementation()
    with pytest.raises(ValueError, match="regular.*eager"):
        config.validate_cudagraph_runtime_config()


def test_strict_regular_accepts_decode_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    config = _selector_config(CUDAGraphImplementation.REGULAR, strict=True)
    config.resolve_cudagraph_implementation()
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    config.validate_cudagraph_runtime_config()


def test_strict_regular_rejects_no_decode_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    config = _selector_config(CUDAGraphImplementation.REGULAR, strict=True)
    config.resolve_cudagraph_implementation()
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    with pytest.raises(ValueError, match="regular.*decode.*NONE"):
        config.validate_cudagraph_runtime_config()


def test_runtime_legacy_reads_are_confined_to_resolver() -> None:
    root = Path(__file__).parents[2]
    allowed = {
        root / "vllm" / "config" / "vllm.py",
        root / "vllm" / "envs.py",
    }
    offenders = []
    for path in (root / "vllm").rglob("*.py"):
        if path in allowed:
            continue
        if "VLLM_USE_BREAKABLE_CUDAGRAPH" in path.read_text():
            offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_runtime_sources_emit_capture_and_replay_evidence() -> None:
    root = Path(__file__).parents[2]
    regular = (
        root / "vllm" / "v1" / "worker" / "gpu" / "cudagraph_utils.py"
    ).read_text()
    breakable = (
        root / "vllm" / "compilation" / "breakable_cudagraph.py"
    ).read_text()
    runner = (
        root / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
    ).read_text()

    assert "CUDA_GRAPH_CAPTURE implementation=regular status=complete" in runner
    assert "CUDA_GRAPH_REPLAY implementation=regular status=observed" in regular
    assert "CUDA_GRAPH_CAPTURE implementation=breakable status=complete" in runner
    assert "CUDA_GRAPH_REPLAY implementation=breakable status=observed" in breakable
