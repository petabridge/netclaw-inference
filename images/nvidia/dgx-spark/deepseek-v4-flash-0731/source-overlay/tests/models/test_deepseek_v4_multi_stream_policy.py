from unittest.mock import patch

from vllm.models.deepseek_v4.attention import make_dsv4_aux_stream_list


def test_dsv4_multi_stream_defaults_to_compile_safe_sequential(monkeypatch):
    monkeypatch.delenv("VLLM_DSV4_ENABLE_MULTI_STREAM", raising=False)
    assert make_dsv4_aux_stream_list() is None


def test_dsv4_multi_stream_explicit_zero_is_sequential(monkeypatch):
    monkeypatch.setenv("VLLM_DSV4_ENABLE_MULTI_STREAM", "0")
    assert make_dsv4_aux_stream_list() is None


def test_dsv4_multi_stream_explicit_one_allocates_three_streams(monkeypatch):
    monkeypatch.setenv("VLLM_DSV4_ENABLE_MULTI_STREAM", "1")
    with patch("torch.cuda.Stream", side_effect=["s0", "s1", "s2"]):
        assert make_dsv4_aux_stream_list() == ["s0", "s1", "s2"]