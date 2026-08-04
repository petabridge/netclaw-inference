# SPDX-License-Identifier: Apache-2.0
import inspect

import torch

import vllm.model_executor.kernels.mhc.tilelang as mhc_tilelang
import vllm.utils.deep_gemm as deep_gemm


def test_explicit_mhc_backend_bypasses_platform_introspection(monkeypatch):
    def fail_if_called():
        raise AssertionError("platform introspection entered compile-safe path")

    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", fail_if_called)
    assert mhc_tilelang._resolve_use_deep_gemm(True) is True
    assert mhc_tilelang._resolve_use_deep_gemm(False) is False


def test_automatic_mhc_backend_preserves_legacy_resolution(monkeypatch):
    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: True)
    assert mhc_tilelang._resolve_use_deep_gemm(None) is True
    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: False)
    assert mhc_tilelang._resolve_use_deep_gemm(None) is False


def test_explicit_mhc_backend_is_dynamo_fullgraph_safe(monkeypatch):
    def fail_if_called():
        raise AssertionError("platform introspection entered compiled explicit path")

    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", fail_if_called)

    @torch.compile(fullgraph=True, backend="eager")
    def explicit_false(value):
        if mhc_tilelang._resolve_use_deep_gemm(False):
            return value + 1
        return value - 1

    result = explicit_false(torch.tensor([3.0]))
    assert torch.equal(result, torch.tensor([2.0]))


def test_registered_mhc_schemas_accept_optional_backend_flag():
    pre_schema = str(torch.ops.vllm.mhc_pre_tilelang.default._schema)
    fused_schema = str(torch.ops.vllm.mhc_fused_post_pre_tilelang.default._schema)
    assert "bool? use_deep_gemm=None" in pre_schema
    assert "bool? use_deep_gemm=None" in fused_schema


def test_nvidia_deepseek_v4_routes_mhc_through_registered_ops():
    from vllm.models.deepseek_v4.nvidia import dspark, model, mtp

    assert model._MHC_PRE_OP is torch.ops.vllm.mhc_pre_tilelang.default
    assert (
        model._MHC_FUSED_POST_PRE_OP
        is torch.ops.vllm.mhc_fused_post_pre_tilelang.default
    )
    assert model._MHC_POST_OP is torch.ops.vllm.mhc_post_tilelang.default
    assert model._HC_HEAD_OP is torch.ops.vllm.hc_head_fused_kernel_tilelang.default
    assert mtp._MHC_POST_OP is torch.ops.vllm.mhc_post_tilelang.default
    assert mtp._HC_HEAD_OP is torch.ops.vllm.hc_head_fused_kernel_tilelang.default
    assert dspark._MHC_POST_OP is torch.ops.vllm.mhc_post_tilelang.default
    assert dspark._HC_HEAD_OP is torch.ops.vllm.hc_head_fused_kernel_tilelang.default


def test_nvidia_deepseek_v4_o_proj_uses_registered_deep_gemm_op():
    from vllm.models.deepseek_v4.nvidia.ops import o_proj

    assert (
        o_proj._DEEP_GEMM_FP8_O_PROJ_EINSUM_OP
        is torch.ops.vllm.deep_gemm_fp8_o_proj_einsum.default
    )


def test_fused_indexer_q_cutedsl_schemas_are_functional():
    from vllm.models.deepseek_v4.common.ops import fused_indexer_q as indexer_q

    assert (
        indexer_q._FUSED_INDEXER_Q_ROPE_QUANT_FP8_CUTEDSL_OP
        is torch.ops.vllm.fused_indexer_q_rope_quant_fp8_cutedsl.default
    )
    assert (
        indexer_q._FUSED_INDEXER_Q_ROPE_QUANT_MXFP4_CUTEDSL_OP
        is torch.ops.vllm.fused_indexer_q_rope_quant_mxfp4_cutedsl.default
    )
    fp8_schema = str(
        torch.ops.vllm.fused_indexer_q_rope_quant_fp8_cutedsl.default._schema
    )
    mxfp4_schema = str(
        torch.ops.vllm.fused_indexer_q_rope_quant_mxfp4_cutedsl.default._schema
    )
    assert "bool use_fnuz" in fp8_schema
    assert "-> (Tensor, Tensor)" in fp8_schema
    assert "-> (Tensor, Tensor, Tensor)" in mxfp4_schema


def test_explicit_fused_indexer_q_backend_is_fullgraph_safe(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm.models.deepseek_v4.common.ops import fused_indexer_q as indexer_q

    def fail_if_called():
        raise AssertionError(
            "CuTeDSL capability probing entered explicit compiled path"
        )

    monkeypatch.setattr(indexer_q, "has_cutedsl", fail_if_called)

    class ExplicitIndexerQ(torch.nn.Module):
        def forward(self, positions, q, cos_sin_cache, weights):
            return indexer_q.fused_indexer_q_rope_quant(
                positions,
                q,
                cos_sin_cache,
                weights,
                0.125,
                0.125,
                use_fp4=False,
                cutedsl_available=True,
                fp8_use_fnuz=False,
            )

    with FakeTensorMode():
        args = (
            torch.zeros((1,), dtype=torch.int64),
            torch.empty((1, 2, 128), dtype=torch.bfloat16),
            torch.empty((4096, 64), dtype=torch.bfloat16),
            torch.empty((1, 2), dtype=torch.bfloat16),
        )
        q_fp8, weights_out = ExplicitIndexerQ()(*args)
        assert q_fp8.shape == args[1].shape
        assert q_fp8.dtype == torch.float8_e4m3fn
        assert weights_out.shape == args[3].shape
        assert weights_out.dtype == torch.float32
        exported = torch.export.export(ExplicitIndexerQ(), args)
        assert "vllm.fused_indexer_q_rope_quant_fp8_cutedsl.default" in str(
            exported.graph
        )


def test_deepseek_v4_indexer_hoists_cutedsl_resolution():
    from vllm.models.deepseek_v4.attention import DeepseekV4Indexer

    init_source = inspect.getsource(DeepseekV4Indexer.__init__)
    forward_source = inspect.getsource(DeepseekV4Indexer.forward)
    assert "self.indexer_q_use_cutedsl = has_cutedsl()" in init_source
    assert "self.indexer_q_use_fnuz" in init_source
    assert "cutedsl_available=self.indexer_q_use_cutedsl" in forward_source
    assert "fp8_use_fnuz=self.indexer_q_use_fnuz" in forward_source
