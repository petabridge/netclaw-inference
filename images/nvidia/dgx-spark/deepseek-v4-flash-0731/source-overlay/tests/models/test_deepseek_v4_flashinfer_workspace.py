from __future__ import annotations

import inspect

import torch

from vllm.models.deepseek_v4.nvidia import flashinfer_sparse


def _uninitialized_attention() -> flashinfer_sparse.DeepseekV4FlashInferSM120Attention:
    attention = object.__new__(
        flashinfer_sparse.DeepseekV4FlashInferSM120Attention
    )
    torch.nn.Module.__init__(attention)
    attention._flashinfer_workspace = None
    attention._flashinfer_decode_lse = None
    return attention


def test_sm120_workspace_is_bound_before_compiled_forward(monkeypatch) -> None:
    attention = _uninitialized_attention()
    workspace = torch.empty(16, dtype=torch.uint8)
    decode_lse = torch.empty((4, 8), dtype=torch.float32)
    monkeypatch.setattr(
        type(attention),
        "_get_workspace",
        staticmethod(lambda _device: workspace),
    )
    monkeypatch.setattr(
        flashinfer_sparse,
        "_get_flashinfer_dsv4_decode_lse",
        lambda _device: decode_lse,
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

    attention._initialize_workspace_bindings()

    assert attention._reserved_workspace() is workspace
    assert attention._reserved_decode_lse() is decode_lse


def test_sm120_compiled_paths_do_not_read_mutable_workspace_dicts() -> None:
    source = inspect.getsource(
        flashinfer_sparse.DeepseekV4FlashInferSM120Attention
    )
    compiled_region = source[source.index("def _forward_prefill") :]
    assert "_get_flashinfer_dsv4_workspace" not in compiled_region
    assert "_get_flashinfer_dsv4_decode_lse" not in compiled_region
    assert "self._get_workspace" not in compiled_region
    assert "self._initialize_workspace_bindings" not in compiled_region
    assert "self._reserved_workspace()" in compiled_region
    assert "self._reserved_decode_lse()" in compiled_region