#!/usr/bin/env python3
"""CPU checks for overlay/exl3.py [dense-exl3]: config parsing, module
matching, per-shard TP geometry for TP2/TP4 (incl. replicated and bf16
shards), conflict refusals, codebook-marker validation, and the ABLIT
EXL3-o_proj refusal. Runs with real torch and stubbed vllm modules — it never
imports exllamav3 (marker checks raise before LinearEXL3 construction).

."""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import types
from pathlib import Path

import torch
from torch.nn.parameter import Parameter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXL3_SRC = ROOT / "overlay" / "exl3.py"
ABLIT_SRC = ROOT / "overlay" / "ablit_runtime.py"

MLA_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
LAYER_TYPES = [
    "linear_attention" if i not in MLA_LAYERS else "deepseek_sparse_attention"
    for i in range(45)
]

# GLM-5.3-Flash dims (config.json + image model code): hidden 4096, dense MLP
# intermediate 12288 (layers 0-2), shared experts 2048 (= moe_intermediate x 1),
# 64 attention heads x 256 head dims (MLA), KDA 64 heads x 128.
H = 4096
DENSE_I = 12288
SHARED_I = 2048
KDA_HEADS, KDA_HD = 64, 128
MLA_HEAD_DIMS = 64 * 256


class _Cfg:
    def __init__(self, layer_types):
        class _HF:
            pass

        class _MC:
            hf_text_config = _HF()

        self.model_config = _MC()
        self.model_config.hf_text_config.layer_types = layer_types


def _install_vllm_stubs(layer_types=None):
    """Real classes for isinstance dispatch; everything else is inert."""
    linear = types.ModuleType("vllm.model_executor.layers.linear")

    class LinearBase(torch.nn.Module):
        pass

    class RowParallelLinear(LinearBase):
        pass

    class ReplicatedLinear(LinearBase):
        pass

    class MergedColumnParallelLinear(LinearBase):
        pass

    class QKVParallelLinear(LinearBase):
        pass

    class UnquantizedLinearMethod:
        def apply(self, layer, x, bias=None):
            raise AssertionError("stock apply must not run in these tests")

    class LinearMethodBase:
        pass

    for name, obj in vars().items():
        setattr(linear, name, obj)

    fm_base = types.ModuleType("vllm.model_executor.layers.fused_moe.fused_moe_method_base")

    class FusedMoEMethodBase:
        pass

    fm_base.FusedMoEMethodBase = FusedMoEMethodBase

    class RoutedExperts(torch.nn.Module):
        pass

    routed = types.ModuleType("vllm.model_executor.layers.fused_moe.routed_experts")
    routed.RoutedExperts = RoutedExperts

    qc = types.ModuleType("vllm.model_executor.layers.quantization.base_config")

    class QuantizationConfig:
        def is_layer_free(self, layer):  # pragma: no cover - unused here
            return True

    qc.QuantizationConfig = QuantizationConfig

    vllm = types.ModuleType("vllm")
    vllm_logger = types.ModuleType("vllm.logger")
    vllm_logger.init_logger = lambda name: logging.getLogger(name)
    vllm_config_mod = types.ModuleType("vllm.config")
    vllm_config_mod.get_current_vllm_config = lambda: _Cfg(layer_types or LAYER_TYPES)
    registered = {}

    def register(name):
        def deco(cls):
            registered[name] = cls
            return cls

        return deco

    vllm_quant = types.ModuleType("vllm.model_executor.layers.quantization")
    vllm_quant.register_quantization_config = register
    vllm_utils = types.ModuleType("vllm.model_executor.utils")
    vllm_utils.set_weight_attrs = lambda weight, attrs: [
        setattr(weight, k, v) for k, v in (attrs or {}).items()
    ]
    fm_cfg = types.ModuleType("vllm.model_executor.layers.fused_moe.config")
    fm_cfg.FusedMoEQuantConfig = object

    vpe = types.ModuleType("vllm.model_executor.layers.vocab_parallel_embedding")

    class VocabParallelEmbedding(torch.nn.Module):
        pass

    class ParallelLMHead(VocabParallelEmbedding):
        pass

    vpe.VocabParallelEmbedding = VocabParallelEmbedding
    vpe.ParallelLMHead = ParallelLMHead

    for mod in (vllm, vllm_logger, vllm_config_mod, vllm_quant, vllm_utils, fm_cfg):
        sys.modules.setdefault(mod.__name__, mod)
    for name, mod in (
        ("vllm.logger", vllm_logger),
        ("vllm.config", vllm_config_mod),
        ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
        ("vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers")),
        ("vllm.model_executor.layers.linear", linear),
        ("vllm.model_executor.layers.fused_moe", types.ModuleType("vllm.model_executor.layers.fused_moe")),
        ("vllm.model_executor.layers.fused_moe.fused_moe_method_base", fm_base),
        ("vllm.model_executor.layers.fused_moe.routed_experts", routed),
        ("vllm.model_executor.layers.quantization", vllm_quant),
        ("vllm.model_executor.layers.quantization.base_config", qc),
        ("vllm.model_executor.utils", vllm_utils),
        ("vllm.model_executor.layers.fused_moe.config", fm_cfg),
        ("vllm.model_executor.layers.vocab_parallel_embedding", vpe),
    ):
        sys.modules[name] = mod
    return dict(
        LinearBase=LinearBase,
        RowParallelLinear=RowParallelLinear,
        ReplicatedLinear=ReplicatedLinear,
        MergedColumnParallelLinear=MergedColumnParallelLinear,
        QKVParallelLinear=QKVParallelLinear,
        UnquantizedLinearMethod=UnquantizedLinearMethod,
        VocabParallelEmbedding=VocabParallelEmbedding,
        ParallelLMHead=ParallelLMHead,
        registered=registered,
    )


def _load_exl3():
    spec = importlib.util.spec_from_file_location("exl3_dense_test", EXL3_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _nr_config(mod, layers=None, base_bits=4):
    return mod.Exl3Config(bits=base_bits, non_routed_exl3={"layers": layers or {}})


def config_tests(mod):
    C = mod.Exl3Config
    # from_config round-trip: the pack's non_routed_exl3 block flows through
    # the kwargs passthrough exactly as dense_overlay.py writes it.
    cfg = C.from_config({
        "quant_method": "exl3", "bits": 4, "codebook": "mcg",
        "scope": "glm53_routed_experts_only",
        "tensor_storage": {"big": "ledger"},
        "non_routed_exl3": {
            "codebook": "mul1",
            "layers": {
                "language_model.model.layers.0.self_attn.in_proj_qkvbfg_a": {
                    "bits": 6, "bf16_shards": [3, 4, 5]},
                "language_model.model.layers.0.mlp.gate_up_proj": {"bits": 5},
            },
        },
    })
    assert "tensor_storage" not in cfg.raw_config
    m = cfg._matches_non_routed_exl3
    assert m("language_model.model.layers.0.self_attn.in_proj_qkvbfg_a")
    assert m("language_model.model.layers.0.mlp.gate_up_proj")
    assert not m("language_model.model.layers.0.self_attn.o_proj")
    assert not m("language_model.model.layers.0.self_attn.f_b_proj")
    assert not m("language_model.model.layers.3.self_attn.kv_b_proj")
    assert cfg._bits_for_non_routed("language_model.model.layers.0.self_attn.in_proj_qkvbfg_a") == 6
    assert cfg._bits_for_non_routed("language_model.model.layers.0.mlp.gate_up_proj") == 5
    assert cfg._bf16_shards_for("language_model.model.layers.0.self_attn.in_proj_qkvbfg_a") == [3, 4, 5]
    assert cfg._bf16_shards_for("language_model.model.layers.0.mlp.gate_up_proj") == []

    # A config without the block matches nothing and stays stock.
    stock = C.from_config({"quant_method": "exl3", "bits": 4})
    assert not stock.non_routed_exl3
    assert not stock._matches_non_routed_exl3("language_model.model.layers.0.mlp.down_proj")

    # bf16_shards must arrive sorted and contiguous (tail checked at create).
    for bad in ({"layers": {"a.b": {"bits": 6, "bf16_shards": [4, 3, 5]}}},
                {"layers": {"a.b": {"bits": 6, "bf16_shards": [3, 5]}}}):
        try:
            _nr_config(mod, **bad)
        except ValueError as exc:
            assert "bf16_shards" in str(exc), exc
        else:
            raise AssertionError(f"bad bf16_shards accepted: {bad}")

    # B2: declared-but-never-constructed modules refuse loudly
    declared = {"x.self_attn.o_proj": {"bits": 6}, "y.mlp.down_proj": {"bits": 5}}
    cfg_b2 = _nr_config(mod, layers=declared)
    mod.Exl3LinearMethod(cfg_b2, "x.self_attn.o_proj", bits=6)
    try:
        cfg_b2._assert_non_routed_built()
    except RuntimeError as exc:
        assert "y.mlp.down_proj" in str(exc) and "1/2" in str(exc), exc
    else:
        raise AssertionError("unbuilt declared module must refuse")
    empty = _nr_config(mod, layers={})
    empty._assert_non_routed_built()  # no declarations -> no-op

    for bad in ({"layers": {"a.b": {"bits": 7}}}, {"layers": {"a.b": {"bits": 1}}},
                {"layers": {"a.b": {}}}):
        try:
            _nr_config(mod, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad non_routed bits accepted: {bad}")
    # kv_b_proj is refused wherever it is declared (MLA reads the weight).
    try:
        _nr_config(mod, layers={"x.self_attn.kv_b_proj": {"bits": 6}})
    except ValueError as exc:
        assert "kv_b_proj" in str(exc)
    else:
        raise AssertionError("kv_b_proj accepted")
    print("config parsing OK")


def _fake_layer(cls, prefix, tp_rank, tp_size, out_sizes, in_size, replicated=()):
    layer = cls()
    layer.prefix = prefix
    layer.tp_rank = tp_rank
    layer.tp_size = tp_size
    layer.replicated_shard_ids = tuple(replicated)
    layer.output_partition_sizes = list(out_sizes)
    layer.input_size_per_partition = in_size
    return layer


def _pack(mod, k, out_rows, in_cols, marker="mul1", seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(
        -30000, 30000, (in_cols // 16, out_rows // 16, 16 * k),
        dtype=torch.int16, generator=g)
    suh = torch.randn(in_cols, dtype=torch.float16, generator=g)
    svh = torch.randn(out_rows, dtype=torch.float16, generator=g)
    mk = torch.tensor([mod.MUL1_MARKER_SIGNED_INT32 if marker == "mul1"
                       else mod.MCG_MARKER_SIGNED_INT32], dtype=torch.int32)
    return t, suh, svh, mk, ("mcg" if marker == "mcg" else "mul1")


def _load_all(mod, method, layer, shard_tensor_map, shard_ids):
    """Feed the loaders exactly what the pack ships: packed parts only for
    EXL3 shards, `weight` staging for every shard (EXL3 ones get discarded)."""
    for suffix in ("trellis", "suh", "svh"):
        p = getattr(layer, suffix)
        for idx, sid in enumerate(shard_ids):
            if (idx, suffix) not in shard_tensor_map:
                continue
            p.weight_loader(p, shard_tensor_map[(idx, suffix)], sid)
    for idx, sid in enumerate(shard_ids):
        if (idx, "marker") not in shard_tensor_map:
            continue
        # the pack names the marker tensor *.mul1 or *.mcg; only that param
        # receives it (the other stays zero)
        p = getattr(layer, shard_tensor_map[(idx, "marker_suffix")])
        p.weight_loader(p, shard_tensor_map[(idx, "marker")], sid)
    if method.quant_config._bf16_shards_for(layer.prefix):
        w = layer.weight
        for idx, sid in enumerate(shard_ids):
            if (idx, "weight") not in shard_tensor_map:
                continue
            w.weight_loader(w, shard_tensor_map[(idx, "weight")], sid)


def geometry_tests(mod, stubs):
    Exl3LinearMethod = mod.Exl3LinearMethod

    def method_for(prefix, bits=6, bf16=None):
        c = _nr_config(mod, layers={prefix: {"bits": bits, **({"bf16_shards": bf16} if bf16 else {})}})
        return Exl3LinearMethod(c, prefix, bits=bits)

    # ---- MLA o_proj (RowParallel, input sharded) at TP2 and TP4 -----------
    for tp in (2, 4):
        methods = []
        layers = []
        for rank in range(tp):
            m = method_for("language_model.model.layers.3.self_attn.o_proj")
            layer = _fake_layer(
                stubs["RowParallelLinear"],
                "language_model.model.layers.3.self_attn.o_proj",
                rank, tp, [H], MLA_HEAD_DIMS // tp)
            m.create_weights(layer, MLA_HEAD_DIMS // tp, [H], MLA_HEAD_DIMS, H, torch.bfloat16)
            methods.append(m)
            layers.append(layer)
            assert layer.trellis.shape == (MLA_HEAD_DIMS // tp // 16, H // 16, 96)
            assert layer.suh.shape == (1, MLA_HEAD_DIMS // tp)
            assert layer.svh.shape == (H,)
        full_t, full_suh, full_svh, mk, mkind = _pack(mod, 6, H, MLA_HEAD_DIMS, seed=1)
        for rank in range(tp):
            _load_all(mod, methods[rank], layers[rank],
                      {(0, "trellis"): full_t, (0, "suh"): full_suh,
                       (0, "svh"): full_svh, (0, "marker"): mk,
                       (0, "marker_suffix"): mkind},
                      [None])
            assert layers[rank].svh.tolist() == full_svh.tolist()
        # Row-parallel trellis/suh narrow on dim 0: ranks tile the rows.
        joined_t = torch.cat([l.trellis for l in layers], dim=0)
        assert torch.equal(joined_t, full_t)
        joined_suh = torch.cat([l.suh[:, :].squeeze(0) for l in layers], dim=0)
        assert torch.equal(joined_suh, full_suh)
    print("MLA o_proj row-parallel TP2/TP4 OK")

    # ---- MLA q_b_proj (ColumnParallel, output sharded) at TP2/TP4 ---------
    for tp in (2, 4):
        out_local = MLA_HEAD_DIMS // tp
        layers = []
        methods = []
        for rank in range(tp):
            m = method_for("language_model.model.layers.3.self_attn.q_b_proj")
            layer = _fake_layer(
                stubs["LinearBase"],  # plain ColumnParallel path
                "language_model.model.layers.3.self_attn.q_b_proj",
                rank, tp, [out_local], 1536)
            m.create_weights(layer, 1536, [out_local], MLA_HEAD_DIMS, MLA_HEAD_DIMS, torch.bfloat16)
            layers.append(layer)
            methods.append(m)
        full_t, full_suh, full_svh, mk, mkind = _pack(mod, 6, MLA_HEAD_DIMS, 1536, seed=2)
        for rank in range(tp):
            _load_all(mod, methods[rank], layers[rank],
                      {(0, "trellis"): full_t, (0, "suh"): full_suh,
                       (0, "svh"): full_svh, (0, "marker"): mk,
                       (0, "marker_suffix"): mkind}, [None])
            assert torch.equal(layers[rank].suh[0], full_suh)  # suh not sharded
        joined_t = torch.cat([l.trellis for l in layers], dim=1)
        assert torch.equal(joined_t, full_t)
        joined_svh = torch.cat([l.svh for l in layers], dim=0)
        assert torch.equal(joined_svh, full_svh)
    print("MLA q_b_proj column-parallel TP2/TP4 OK")

    # ---- KDA in_proj_qkvbfg_a: merged shards, q/k/v EXL3, b/f_a/g_a bf16,
    # f_a/g_a replicated. Shard ids are ints 0..5 from stacked_params_mapping.
    for tp in (2, 4):
        P = KDA_HEADS * KDA_HD
        sizes = [P // tp, P // tp, P // tp, KDA_HEADS // tp, KDA_HD, KDA_HD]
        full_sizes = [P, P, P, KDA_HEADS // 1, KDA_HD, KDA_HD]
        # full b shard rows: KDA_HEADS (not divided by tp in the checkpoint
        # module contract: b is divided; the loader slices per-rank rows).
        methods, layer_sets = [], []
        for rank in range(tp):
            m = method_for(
                "language_model.model.layers.0.self_attn.in_proj_qkvbfg_a",
                bf16=[3, 4, 5])
            layer = _fake_layer(
                stubs["MergedColumnParallelLinear"],
                "language_model.model.layers.0.self_attn.in_proj_qkvbfg_a",
                rank, tp, sizes, H, replicated=(4, 5))
            m.create_weights(layer, H, sizes, H, sum(full_sizes), torch.bfloat16)
            methods.append(m)
            layer_sets.append(layer)
            exl_tiles = sum(s // 16 for i, s in enumerate(sizes) if i not in (3, 4, 5))
            assert layer.trellis.shape == (H // 16, exl_tiles, 96)
            assert layer.weight.shape == (
                KDA_HEADS // tp + 2 * KDA_HD, H)
        for rank in range(tp):
            layer = layer_sets[rank]
            tensors = {}
            for i in range(6):
                tensors[(i, "weight")] = torch.randn(full_sizes[i], H,
                                                     dtype=torch.bfloat16,
                                                     generator=torch.Generator().manual_seed(20 + i))
                if i in (3, 4, 5):
                    continue  # b/f_a/g_a are bf16 shards: no packed parts ship
                t, suh, svh, mk, mkind = _pack(mod, 6, full_sizes[i], H, seed=10 + i)
                tensors[(i, "trellis")] = t
                tensors[(i, "suh")] = suh
                tensors[(i, "svh")] = svh
                tensors[(i, "marker")] = mk
                tensors[(i, "marker_suffix")] = mkind
            _load_all(mod, methods[rank], layer, tensors, [0, 1, 2, 3, 4, 5])
            # markers: q/k/v carry mul1, bf16 shards carry nothing
            for i in range(3):
                assert layer.mul1[i, 0].item() == mod.MUL1_MARKER_SIGNED_INT32
                assert layer.mcg[i, 0].item() == 0
            for i in (3, 4, 5):
                assert layer.mul1[i, 0].item() == 0
                assert layer.mcg[i, 0].item() == 0
        # EXL3 shards tile the full q/k/v tensors across ranks.
        for i in range(3):
            joined_t = torch.cat([l.trellis[:, i * (P // tp // 16):(i + 1) * (P // tp // 16), :]
                                  for l in layer_sets], dim=1)
            full_t = tensors[(i, "trellis")]
            assert joined_t.shape == full_t.shape, (joined_t.shape, full_t.shape)
            # per-rank trellis columns == the rank's slice of the full tensor
            for rank in range(tp):
                cols = slice(rank * (P // tp // 16), (rank + 1) * (P // tp // 16))
                assert torch.equal(
                    layer_sets[rank].trellis[:, i * (P // tp // 16):(i + 1) * (P // tp // 16), :],
                    full_t[:, cols])
        # bf16 staging: q/k/v stale weights discarded; b/f_a/g_a rows staged.
        for rank in range(tp):
            w = layer_sets[rank].weight
            # b shard is column-parallel: rank rows are a contiguous slice
            # of the full KDA_HEADS rows.
            b_full = tensors[(3, "weight")]
            assert torch.equal(
                w[: KDA_HEADS // tp].to(torch.float32),
                b_full[rank * (KDA_HEADS // tp):(rank + 1) * (KDA_HEADS // tp)].to(torch.float32))
            # f_a/g_a replicated: full tensors on every rank.
            assert torch.equal(w[KDA_HEADS // tp: KDA_HEADS // tp + KDA_HD].to(torch.float32),
                               tensors[(4, "weight")].to(torch.float32))
            assert torch.equal(w[-KDA_HD:].to(torch.float32),
                               tensors[(5, "weight")].to(torch.float32))
    print("KDA in_proj merged shards + replicated + bf16 staging TP2/TP4 OK")

    # ---- dense MLP gate_up_proj at TP2/TP4 (merged, no bf16 shards) -------
    for tp in (2, 4):
        sizes = [DENSE_I // tp, DENSE_I // tp]
        layers = []
        methods = []
        for rank in range(tp):
            m = method_for("language_model.model.layers.1.mlp.gate_up_proj", bits=5)
            layer = _fake_layer(
                stubs["MergedColumnParallelLinear"],
                "language_model.model.layers.1.mlp.gate_up_proj",
                rank, tp, sizes, H)
            m.create_weights(layer, H, sizes, H, 2 * DENSE_I, torch.bfloat16)
            layers.append(layer)
            methods.append(m)
        gate_t, gate_suh, gate_svh, gate_mk, _gkind = _pack(mod, 5, DENSE_I, H, seed=30)
        up_t, up_suh, up_svh, up_mk, _ukind = _pack(mod, 5, DENSE_I, H, seed=31)
        for rank in range(tp):
            _load_all(mod, methods[rank], layers[rank],
                      {(0, "trellis"): gate_t, (0, "suh"): gate_suh,
                       (0, "svh"): gate_svh, (0, "marker"): gate_mk,
                       (0, "marker_suffix"): _gkind,
                       (1, "trellis"): up_t, (1, "suh"): up_suh,
                       (1, "svh"): up_svh, (1, "marker"): up_mk,
                       (1, "marker_suffix"): _ukind},
                      [0, 1])
            for i in range(2):
                assert layers[rank].mul1[i, 0].item() == mod.MUL1_MARKER_SIGNED_INT32
        for i, full_t in ((0, gate_t), (1, up_t)):
            cols = DENSE_I // tp // 16
            for rank in range(tp):
                got = layers[rank].trellis[:, i * cols:(i + 1) * cols, :]
                assert torch.equal(got, full_t[:, rank * cols:(rank + 1) * cols])
    print("dense MLP gate_up merged shards TP2/TP4 OK")

    # ---- shared experts: TP2 packed output columns ----------------------
    m = method_for("language_model.model.layers.5.mlp.shared_experts.gate_up_proj", bits=6)
    layer = _fake_layer(
        stubs["MergedColumnParallelLinear"],
        "language_model.model.layers.5.mlp.shared_experts.gate_up_proj",
        0, 2, [SHARED_I // 2, SHARED_I // 2], H)
    m.create_weights(layer, H, [SHARED_I // 2, SHARED_I // 2], H, 2 * SHARED_I, torch.bfloat16)
    full_t, _, _, mk, _ = _pack(mod, 6, SHARED_I, H, seed=40)
    p = layer.trellis
    p.weight_loader(p, full_t, 0)
    assert p.data.shape[1] == SHARED_I // 16

    # ---- trellis tile-width refusal (16-multiple) -------------------------
    m = method_for("x.self_attn.o_proj")
    layer = _fake_layer(
        stubs["RowParallelLinear"], "x.self_attn.o_proj", 0, 1, [4096], 4096 + 8)
    try:
        m.create_weights(layer, 4096 + 8, [4096], 4104, 4096, torch.bfloat16)
    except ValueError as exc:
        assert "16-wide" in str(exc), exc
    else:
        raise AssertionError("non-16-multiple input must refuse")
    # bf16 shards are exempt from the 16-multiple check (b=32/16 rows at TP2,
    # 16 at TP4; f_a/g_a 128) but a packed tensor for them still refuses.
    m = method_for("language_model.model.layers.0.self_attn.in_proj_qkvbfg_a",
                   bf16=[3, 4, 5])
    layer = _fake_layer(
        stubs["MergedColumnParallelLinear"],
        "language_model.model.layers.0.self_attn.in_proj_qkvbfg_a",
        0, 2, [4096, 4096, 4096, 32, 128, 128], H, replicated=(4, 5))
    m.create_weights(layer, H, [4096, 4096, 4096, 32, 128, 128], H, 12444,
                     torch.bfloat16)
    t3, s3, v3, mk3, _ = _pack(mod, 6, 32, H, seed=50)
    try:
        layer.trellis.weight_loader(layer.trellis, t3, 3)
    except RuntimeError as exc:
        assert "declared bf16" in str(exc), exc
    else:
        raise AssertionError("packed tensor for a bf16 shard must refuse")
    print("tile-width refusals OK")

    # ---- bf16 shards must be the shard tail (forward runs one tail GEMM) ---
    m = method_for("z.self_attn.in_proj_qkvbfg_a", bf16=[1, 2])
    layer = _fake_layer(
        stubs["MergedColumnParallelLinear"], "z.self_attn.in_proj_qkvbfg_a",
        0, 1, [1024, 1024, 1024, 64, 128, 128], H)
    try:
        m.create_weights(layer, H, [1024, 1024, 1024, 64, 128, 128], H, 3328,
                         torch.bfloat16)
    except ValueError as exc:
        assert "shard tail" in str(exc), exc
    else:
        raise AssertionError("non-tail bf16_shards must refuse")

    # ---- marker validation before any LinearEXL3 construction -------------
    m = method_for("language_model.model.layers.3.self_attn.o_proj")
    layer = _fake_layer(
        stubs["RowParallelLinear"], "language_model.model.layers.3.self_attn.o_proj",
        0, 1, [H], MLA_HEAD_DIMS)
    m.create_weights(layer, MLA_HEAD_DIMS, [H], MLA_HEAD_DIMS, H, torch.bfloat16)
    layer.mul1.data[0, 0] = mod.MUL1_MARKER_SIGNED_INT32
    layer.mcg.data[0, 0] = mod.MCG_MARKER_SIGNED_INT32  # both set -> refuse
    try:
        m.process_weights_after_loading(layer)
    except RuntimeError as exc:
        assert "exactly one codebook marker" in str(exc), exc
    else:
        raise AssertionError("both markers set must refuse")
    layer.mcg.data[0, 0] = 0
    layer.mul1.data[0, 0] = 12345  # set but wrong value -> refuse
    try:
        m.process_weights_after_loading(layer)
    except RuntimeError as exc:
        assert "bad mul1 marker" in str(exc), exc
    else:
        raise AssertionError("wrong marker value must refuse")
    print("codebook marker validation OK")


def draft_qkv_tests(mod, stubs):
    """DFlash2 draft qkv_proj (the first QKVParallelLinear on the EXL3 path):
    string shard ids "q"/"k"/"v" load exactly like ints 0/1/2, the bf16 k/v
    tail stages exactly the rank's [k; v] rows (the tensor the image's fused
    context-KV build consumes whole), and an unknown string id refuses."""
    Exl3LinearMethod = mod.Exl3LinearMethod
    H_D = 4096  # draft hidden
    Q, KV = 4096, 1024  # full-width q rows / k rows (32 q / 8 kv heads x 128)
    prefix = "model.layers.45.self_attn.qkv_proj"  # runtime prefix (offset)

    def build(rank, tp, bf16=((1, 2))):
        cfg = _nr_config(
            mod, layers={prefix: {"bits": 5, "bf16_shards": list(bf16)}})
        m = Exl3LinearMethod(cfg, prefix, bits=5)
        layer = _fake_layer(
            stubs["QKVParallelLinear"], prefix, rank, tp,
            [Q // tp, KV // tp, KV // tp], H_D)
        m.create_weights(layer, H_D, [Q // tp, KV // tp, KV // tp], H_D,
                         Q + 2 * KV, torch.bfloat16)
        return m, layer

    qt, qsuh, qsvh, qmk, qkind = _pack(mod, 5, Q, H_D, seed=60)
    g = torch.Generator().manual_seed(61)
    k_full = torch.randn(KV, H_D, dtype=torch.bfloat16, generator=g)
    v_full = torch.randn(KV, H_D, dtype=torch.bfloat16, generator=g)

    for tp in (1, 2):
        q, kv = Q // tp, KV // tp
        staged = []
        for sids in (["q", "k", "v"], [0, 1, 2]):
            rank_layers = []
            for rank in range(tp):
                m, layer = build(rank, tp)
                # q EXL3 + k/v bf16 tail: trellis covers only the q tiles,
                # the staging weight holds exactly the k+v rows.
                assert layer.trellis.shape == (H_D // 16, q // 16, 80)
                assert layer.weight.shape == (2 * kv, H_D)
                _load_all(mod, m, layer,
                          {(0, "trellis"): qt, (0, "suh"): qsuh,
                           (0, "svh"): qsvh, (0, "marker"): qmk,
                           (0, "marker_suffix"): qkind,
                           (1, "weight"): k_full, (2, "weight"): v_full},
                          sids)
                w = layer.weight
                # The staged weight IS the rank's fused KV block: the exact
                # bf16 rows the context-KV precompute must see (D6 prefill).
                assert torch.equal(w[:kv], k_full[rank * kv:(rank + 1) * kv])
                assert torch.equal(w[kv:], v_full[rank * kv:(rank + 1) * kv])
                assert int(layer.mul1[0, 0]) == mod.MUL1_MARKER_SIGNED_INT32
                assert int(layer.mul1[1, 0]) == 0
                assert int(layer.mul1[2, 0]) == 0
                rank_layers.append(layer)
            staged.append(rank_layers)
        # string ids and int ids stage byte-identical tensors. Only the q
        # shard's suh/svh rows are live — the bf16 shards' suh/svh rows are
        # dead staging (process_weights_after_loading never reads them).
        for rank in range(tp):
            for suffix in ("trellis", "weight", "mul1", "mcg"):
                assert torch.equal(getattr(staged[0][rank], suffix),
                                   getattr(staged[1][rank], suffix)), suffix
            assert torch.equal(staged[0][rank].suh[0], staged[1][rank].suh[0])
            assert torch.equal(staged[0][rank].svh[:q], staged[1][rank].svh[:q])
        # trellis column tiles partition the full q trellis across ranks
        for rank in range(tp):
            cols = slice(rank * q // 16, (rank + 1) * q // 16)
            assert torch.equal(staged[0][rank].trellis, qt[:, cols])

    # all-EXL3 QKV (no bf16 tail) with string ids, loaded out of order
    cfg = _nr_config(mod, layers={prefix: {"bits": 5}})
    m = Exl3LinearMethod(cfg, prefix, bits=5)
    layer = _fake_layer(stubs["QKVParallelLinear"], prefix, 0, 1,
                        [Q, KV, KV], H_D)
    m.create_weights(layer, H_D, [Q, KV, KV], H_D, Q + 2 * KV, torch.bfloat16)
    assert layer.trellis.shape == (H_D // 16, (Q + 2 * KV) // 16, 80)
    parts = {}
    for i, rows in ((0, Q), (1, KV), (2, KV)):
        t, suh, svh, mk, mkind = _pack(mod, 5, rows, H_D, seed=70 + i)
        parts[i] = (t, suh, svh, mk, mkind)
    for sid, i in (("v", 2), ("q", 0), ("k", 1)):  # deliberate disorder
        t, suh, svh, mk, mkind = parts[i]
        layer.trellis.weight_loader(layer.trellis, t, sid)
        layer.suh.weight_loader(layer.suh, suh, sid)
        layer.svh.weight_loader(layer.svh, svh, sid)
        getattr(layer, mkind).weight_loader(getattr(layer, mkind), mk, sid)
    assert torch.equal(layer.trellis[:, : Q // 16], parts[0][0])
    assert torch.equal(layer.trellis[:, Q // 16:(Q + KV) // 16], parts[1][0])
    assert torch.equal(layer.trellis[:, (Q + KV) // 16:], parts[2][0])
    assert torch.equal(layer.svh[:Q], parts[0][2])
    assert torch.equal(layer.svh[Q:Q + KV], parts[1][2])
    assert torch.equal(layer.svh[Q + KV:], parts[2][2])

    # unknown string ids refuse loudly
    m, layer = build(0, 1)
    try:
        layer.trellis.weight_loader(layer.trellis, qt, "w")
    except ValueError as exc:
        assert "unknown EXL3 shard id" in str(exc) and "'w'" in str(exc), exc
    else:
        raise AssertionError("unknown string shard id must refuse")
    print("draft qkv string shard ids + bf16 k/v staging (TP1/TP2) OK")



def draft_replicated_tests(mod, stubs):
    """The draft's fc and conv kernel_projection are ReplicatedLinear: vLLM's
    LinearBase stamps the GLOBAL tp_rank/tp_size on them (disable_tp "take[s]
    no effect for replicated linear layers") while the weight is duplicated
    per rank — output_partition_sizes reports the FULL output. At draft TP=2
    the loader must not column-shard the full pack tensors against a
    full-size destination. Generate 6-bit tensors at the DFlash2 FC/conv
    geometry; no model-derived header fixture or weights are needed."""
    Exl3LinearMethod = mod.Exl3LinearMethod

    for name, out_f, in_f in (
        ("model.fc", 4096, 20480),
        ("model.layers.45.attention_conv.kernel_projection", 1024, 4096),
        ("model.layers.45.mlp_conv.kernel_projection", 1024, 4096),
    ):
        cfg = _nr_config(mod, layers={name: {"bits": 6}})
        t, suh, svh, mk, mkind = _pack(mod, 6, out_f, in_f, seed=90)
        for rank in (0, 1):
            m = Exl3LinearMethod(cfg, name, bits=6)
            # vLLM ReplicatedLinear at world TP2: global tp attrs, FULL
            # output partition — exactly what create_weights must reconcile.
            layer = _fake_layer(
                stubs["ReplicatedLinear"], name, rank, 2, [out_f], in_f
            )
            m.create_weights(
                layer, in_f, [out_f], in_f, out_f, torch.bfloat16
            )
            assert (m.tp_rank, m.tp_size) == (0, 1), name  # replicated
            _load_all(mod, m, layer,
                      {(0, "trellis"): t, (0, "suh"): suh, (0, "svh"): svh,
                       (0, "marker"): mk, (0, "marker_suffix"): mkind},
                      [None])
            # the full pack tensors land whole on every rank, unsharded
            assert torch.equal(layer.trellis, t), name
            assert torch.equal(layer.svh, svh), name
            assert torch.equal(layer.suh[0], suh), name

    # A genuinely column-sharded layer keeps the world geometry: gate_up at
    # TP2 partitions each proj to 6144/rank (sum 12288 < output_size 24576),
    # and the pack ships the full-width per-proj tensors.
    name = "model.layers.45.mlp.gate_up_proj"
    cfg = _nr_config(mod, layers={name: {"bits": 6}})
    gate = _pack(mod, 6, 12288, 4096, seed=91)
    up = _pack(mod, 6, 12288, 4096, seed=92)
    for rank in (0, 1):
        m = Exl3LinearMethod(cfg, name, bits=6)
        layer = _fake_layer(
            stubs["MergedColumnParallelLinear"], name, rank, 2,
            [6144, 6144], 4096)
        m.create_weights(layer, 4096, [6144, 6144], 4096, 24576,
                         torch.bfloat16)
        assert (m.tp_rank, m.tp_size) == (rank, 2), name
        _load_all(mod, m, layer,
                  {(0, "trellis"): gate[0], (0, "suh"): gate[1],
                   (0, "svh"): gate[2], (0, "marker"): gate[3],
                   (0, "marker_suffix"): gate[4],
                   (1, "trellis"): up[0], (1, "suh"): up[1],
                   (1, "svh"): up[2], (1, "marker"): up[3],
                   (1, "marker_suffix"): up[4]},
                  [0, 1])
        gcols = slice(rank * 6144 // 16, (rank + 1) * 6144 // 16)
        assert torch.equal(layer.trellis[:, :6144 // 16], gate[0][:, gcols])
        assert torch.equal(layer.trellis[:, 6144 // 16:], up[0][:, gcols])
        rows = slice(rank * 6144, (rank + 1) * 6144)
        assert torch.equal(layer.svh[:6144], gate[2][rows])
        assert torch.equal(layer.svh[6144:], up[2][rows])
    print("draft replicated layers (fc + conv kernel_projection, TP2) OK")


def draft_prefix_offset_tests(mod, stubs):
    """EXL3 draft packs declare checkpoint-relative "model.layers.N" keys;
    the runtime builds draft layers offset by the target's layer count (45).
    offset_draft_layer_prefixes shifts the declarations once, leaves
    non-layer keys (model.fc) alone, and makes dispatch land on the runtime
    prefixes."""
    declared = {}
    for i in range(5):
        p = f"model.layers.{i}"
        declared[f"{p}.self_attn.qkv_proj"] = {"bits": 5, "bf16_shards": [1, 2]}
        declared[f"{p}.self_attn.o_proj"] = {"bits": 5}
        declared[f"{p}.mlp.gate_up_proj"] = {"bits": 5}
        declared[f"{p}.mlp.down_proj"] = {"bits": 5}
        declared[f"{p}.attention_conv.kernel_projection"] = {"bits": 5}
        declared[f"{p}.mlp_conv.kernel_projection"] = {"bits": 5}
    declared["model.fc"] = {"bits": 5}
    cfg = _nr_config(mod, layers=declared)
    assert len(declared) == 31  # the served module count (gate/up merged)

    cfg.offset_draft_layer_prefixes(45)
    keys = set(cfg.non_routed_exl3["layers"])
    assert len(keys) == 31
    assert "model.fc" in keys  # non-layer key untouched
    assert "model.layers.45.self_attn.qkv_proj" in keys
    assert "model.layers.49.mlp_conv.kernel_projection" in keys
    assert not any("model.layers.0." in k for k in keys)
    # bf16_shards survive the shift
    assert cfg._bf16_shards_for("model.layers.45.self_attn.qkv_proj") == [1, 2]
    # dispatch lands on the shifted runtime prefix, misses the unshifted one
    m = cfg.get_quant_method(stubs["QKVParallelLinear"](),
                             "model.layers.45.self_attn.qkv_proj")
    assert type(m).__name__ == "Exl3LinearMethod" and m.bits == 5
    m = cfg.get_quant_method(stubs["QKVParallelLinear"](),
                             "model.layers.0.self_attn.qkv_proj")
    assert type(m).__name__ == "UnquantizedLinearMethod"
    # idempotent on the same instance; a fresh instance shifts once
    cfg.offset_draft_layer_prefixes(45)
    assert set(cfg.non_routed_exl3["layers"]) == keys
    cfg2 = _nr_config(mod, layers=dict(declared))
    cfg2.offset_draft_layer_prefixes(0)  # no offset -> no shift
    assert set(cfg2.non_routed_exl3["layers"]) == set(declared)
    empty = _nr_config(mod, layers={})
    empty.offset_draft_layer_prefixes(45)  # nothing declared -> no-op
    # the declared-vs-built assert reads the shifted keys
    mod._DENSE_EXL3_MODULES.clear()
    try:
        cfg._assert_non_routed_built()
    except RuntimeError as exc:
        assert "31/31" in str(exc) and "model.fc" in str(exc), exc
    else:
        raise AssertionError("unbuilt shifted declarations must refuse")
    print("draft prefix offset (checkpoint-relative -> runtime) OK")


def dispatch_tests(mod, stubs):
    Exl3Config = mod.Exl3Config
    def set_env(**kw):
        for k in ("GLM53_DENSE_FP8", "GLM53_KDA_BF16_LARGE_M",
                  "GLM53_DENSE_EXL3_PREFILL_BF16"):
            os.environ.pop(k, None)
        os.environ.update(kw)

    try:
        prefix = "language_model.model.layers.1.mlp.gate_up_proj"
        cfg = _nr_config(mod, layers={prefix: {"bits": 5}})
        set_env(GLM53_DENSE_FP8="off")
        method = cfg.get_quant_method(stubs["LinearBase"](), prefix)
        assert type(method).__name__ == "Exl3LinearMethod", method
        assert method.bits == 5

        set_env(GLM53_DENSE_FP8="dense")
        try:
            cfg.get_quant_method(stubs["LinearBase"](), prefix)
        except RuntimeError as exc:
            assert "GLM53_DENSE_FP8" in str(exc) and prefix in str(exc), exc
        else:
            raise AssertionError("DENSE_FP8 overlap must refuse")

        # LARGE_M is now supported on the EXL3 in_proj (load-time BF16 copy);
        # dispatch must return the EXL3 method, not refuse.
        kda_prefix = "language_model.model.layers.0.self_attn.in_proj_qkvbfg_a"
        kda_cfg = _nr_config(mod, layers={kda_prefix: {"bits": 6, "bf16_shards": [3, 4, 5]}})
        set_env(GLM53_DENSE_FP8="off", GLM53_KDA_BF16_LARGE_M="1")
        method = kda_cfg.get_quant_method(stubs["LinearBase"](), kda_prefix)
        assert type(method).__name__ == "Exl3LinearMethod", method
        assert method.bits == 6

        # No pack block: DENSE_FP8 still dispatches its own groups, and a
        # non-matching prefix stays Unquantized.
        set_env(GLM53_DENSE_FP8="dense")
        stock = Exl3Config(bits=4)
        m = stock.get_quant_method(stubs["LinearBase"](),
                                   "language_model.model.layers.1.mlp.gate_up_proj")
        assert type(m).__name__ == "Glm53DenseFp8Method", m
        m = stock.get_quant_method(stubs["LinearBase"](),
                                   "language_model.model.layers.1.self_attn.q_conv1d")
        assert type(m).__name__ == "UnquantizedLinearMethod", m

        # Pack present, prefix not declared: stays stock even with groups off.
        set_env(GLM53_DENSE_FP8="off")
        m = cfg.get_quant_method(stubs["LinearBase"](),
                                 "language_model.model.layers.1.self_attn.o_proj")
        assert type(m).__name__ == "UnquantizedLinearMethod", m
    finally:
        for k in ("GLM53_DENSE_FP8", "GLM53_KDA_BF16_LARGE_M",
                  "GLM53_DENSE_EXL3_PREFILL_BF16"):
            os.environ.pop(k, None)
    print("get_quant_method dispatch + conflict refusals OK")



def draft_layer_fp8_bound_tests(mod, stubs):
    """GLM53_DENSE_FP8 stays on the target's layers, and an EXL3 DFlash2 draft
    loads as EXL3.

    The image builds a DFlash2 draft with start_layer_id = the target's layer
    count, so a 5-layer draft on the 45-layer target is model.layers.45..49
    and an EXL3 draft pack declares its dense modules under that shape. The
    classifier must not claim them (campaign fd6: GLM53_DENSE_FP8=all + the
    pinned 6bpw EXL3 draft refused at load on both ranks with "is EXL3-dense
    (pack non_routed_exl3) and also targets GLM53_DENSE_FP8 group 'dense'").
    A BF16 draft's layers stay unquantized too, target layers keep their
    groups, and a real same-module overlap on a target layer still refuses."""
    Exl3Config = mod.Exl3Config

    def set_env(**kw):
        for k in ("GLM53_DENSE_FP8", "GLM53_KDA_BF16_LARGE_M",
                  "GLM53_DENSE_EXL3_PREFILL_BF16"):
            os.environ.pop(k, None)
        os.environ.update(kw)

    try:
        # Selector boundary: the target's last layer is claimed, the draft's
        # first and last are not — for every group.
        groups = {"shared", "dense", "kda", "mla"}
        last = len(LAYER_TYPES) - 1
        assert mod._glm53_dense_fp8_group(
            f"model.layers.{last}.mlp.gate_up_proj", groups, LAYER_TYPES
        ) == "dense"
        for idx in (last + 1, last + 5):
            for suffix in ("mlp.gate_up_proj", "mlp.down_proj",
                           "mlp.shared_experts.down_proj", "self_attn.o_proj",
                           "self_attn.in_proj_qkvbfg_a"):
                assert mod._glm53_dense_fp8_group(
                    f"model.layers.{idx}.{suffix}", groups, LAYER_TYPES
                ) is None, (idx, suffix)
        assert all(
            mod._glm53_dense_fp8_group(f"model.layers.{i}.mlp.down_proj", groups, LAYER_TYPES)
            == "dense"
            for i in range(len(LAYER_TYPES))
        ), "target layers keep their groups"

        # Dispatch with the real draft shift: checkpoint-relative pack keys
        # offset by the target's layer count, GLM53_DENSE_FP8=all.
        declared = {}
        for i in range(5):
            declared[f"model.layers.{i}.self_attn.qkv_proj"] = {"bits": 6, "bf16_shards": [1, 2]}
            declared[f"model.layers.{i}.mlp.gate_up_proj"] = {"bits": 6}
            declared[f"model.layers.{i}.mlp.down_proj"] = {"bits": 6}
        cfg = _nr_config(mod, layers=dict(declared))
        cfg.offset_draft_layer_prefixes(len(LAYER_TYPES))
        set_env(GLM53_DENSE_FP8="all")
        for idx in range(len(LAYER_TYPES), len(LAYER_TYPES) + 5):
            m = cfg.get_quant_method(
                stubs["LinearBase"](), f"model.layers.{idx}.mlp.gate_up_proj"
            )
            assert type(m).__name__ == "Exl3LinearMethod" and m.bits == 6, (idx, m)
            m = cfg.get_quant_method(
                stubs["LinearBase"](), f"model.layers.{idx}.self_attn.qkv_proj"
            )
            assert type(m).__name__ == "Exl3LinearMethod" and m.bits == 6, (idx, m)
        m = cfg.get_quant_method(
            stubs["LinearBase"](), f"model.layers.{last}.mlp.gate_up_proj"
        )
        assert type(m).__name__ == "Glm53DenseFp8Method" and m.group == "dense", m

        # A target-layer overlap is not bypassed by the bound.
        overlap = _nr_config(
            mod, layers={"model.layers.1.mlp.gate_up_proj": {"bits": 5}}
        )
        set_env(GLM53_DENSE_FP8="dense")
        try:
            overlap.get_quant_method(stubs["LinearBase"](), "model.layers.1.mlp.gate_up_proj")
        except RuntimeError as exc:
            assert "GLM53_DENSE_FP8" in str(exc) and "layers.1" in str(exc), exc
        else:
            raise AssertionError("target-layer overlap must refuse")

        # BF16 draft (no pack block): its layers are not pulled into FP8.
        set_env(GLM53_DENSE_FP8="all")
        stock = Exl3Config(bits=4)
        assert type(stock.get_quant_method(
            stubs["LinearBase"](), "model.layers.45.mlp.gate_up_proj"
        )).__name__ == "UnquantizedLinearMethod"
        for prefix, group in (
            ("language_model.model.layers.1.mlp.gate_up_proj", "dense"),
            ("language_model.model.layers.3.self_attn.q_b_proj", "mla"),
            ("language_model.model.layers.0.self_attn.in_proj_qkvbfg_a", "kda"),
            ("language_model.model.layers.5.mlp.shared_experts.down_proj", "shared"),
        ):
            m = stock.get_quant_method(stubs["LinearBase"](), prefix)
            assert type(m).__name__ == "Glm53DenseFp8Method" and m.group == group, (prefix, m)
    finally:
        for k in ("GLM53_DENSE_FP8", "GLM53_KDA_BF16_LARGE_M",
                  "GLM53_DENSE_EXL3_PREFILL_BF16"):
            os.environ.pop(k, None)
    print("draft-layer FP8 bound (selector boundary + EXL3/BF16 draft dispatch) OK")


def prefill_bf16_tests(mod):
    """GLM53_DENSE_EXL3_PREFILL_BF16 on dense-EXL3 modules: selection
    parsing/validation, per-type module selection (kda/mla o_proj split by
    layer_types), the retained BF16 weight equals the concat of
    get_weight_tensor().t() + the bf16 tail bitwise on stub shards, and
    apply routes rows > 144 to the copy (one F.linear in x.dtype) and
    rows <= 144 to the custom op."""

    # ---- selection parsing/validation ------------------------------------
    def types_for(**env):
        for k in ("GLM53_DENSE_EXL3_PREFILL_BF16", "GLM53_KDA_BF16_LARGE_M"):
            os.environ.pop(k, None)
        os.environ.update(env)
        try:
            return mod._dense_exl3_prefill_bf16_types()
        finally:
            for k in ("GLM53_DENSE_EXL3_PREFILL_BF16", "GLM53_KDA_BF16_LARGE_M"):
                os.environ.pop(k, None)

    assert types_for() == set()
    assert types_for(GLM53_DENSE_EXL3_PREFILL_BF16="off") == set()
    assert types_for(GLM53_DENSE_EXL3_PREFILL_BF16="all") == \
        set(mod._DENSE_EXL3_PREFILL_BF16_SUFFIXES)
    assert types_for(GLM53_DENSE_EXL3_PREFILL_BF16="kda_in,shared_down,mla_qkv_a") == \
        {"kda_in", "shared_down", "mla_qkv_a"}
    assert types_for(GLM53_DENSE_EXL3_PREFILL_BF16=" KDA_IN , mla_o ") == {"kda_in", "mla_o"}
    # backward compatible: the phase-1 in_proj knob selects kda_in
    assert types_for(GLM53_KDA_BF16_LARGE_M="1") == {"kda_in"}
    assert types_for(GLM53_DENSE_EXL3_PREFILL_BF16="mla_o",
                     GLM53_KDA_BF16_LARGE_M="1") == {"mla_o", "kda_in"}
    for bad in ("kda", "kda-in", "all,kda_in", "foo"):
        try:
            types_for(GLM53_DENSE_EXL3_PREFILL_BF16=bad)
        except ValueError as exc:
            assert "GLM53_DENSE_EXL3_PREFILL_BF16" in str(exc), exc
        else:
            raise AssertionError(f"{bad!r} must be rejected")

    # ---- per-type module selection ----------------------------------------
    f = mod._dense_exl3_prefill_bf16_type
    lt = LAYER_TYPES  # layer 0 KDA (linear_attention), layer 3 MLA
    kda = "language_model.model.layers.0.self_attn"
    mla = "language_model.model.layers.3.self_attn"
    assert f(f"{kda}.in_proj_qkvbfg_a", {"kda_in"}, lt) == "kda_in"
    assert f(f"{kda}.o_proj", {"kda_o"}, lt) == "kda_o"
    assert f(f"{mla}.o_proj", {"kda_o"}, lt) is None
    assert f(f"{mla}.o_proj", {"mla_o"}, lt) == "mla_o"
    assert f(f"{kda}.o_proj", {"mla_o"}, lt) is None
    assert f(f"{mla}.fused_qkv_a_proj", {"mla_qkv_a"}, lt) == "mla_qkv_a"
    assert f(f"{mla}.q_b_proj", {"mla_q_b"}, lt) == "mla_q_b"
    p = "language_model.model.layers.0.mlp"
    assert f(f"{p}.shared_experts.gate_up_proj", {"shared_gate_up"}, lt) == "shared_gate_up"
    assert f(f"{p}.gate_up_proj", {"shared_gate_up"}, lt) is None
    assert f(f"{p}.shared_experts.gate_up_proj", {"dense_gate_up"}, lt) is None
    assert f(f"{p}.down_proj", {"dense_down"}, lt) == "dense_down"
    assert f("language_model.model.layers.45.mtp.self_attn.o_proj",
             {"kda_o", "mla_o"}, lt) is None
    assert f(f"{kda}.o_proj", set(), lt) is None

    # ---- retention: one bf16 pre-concat weight, bitwise on stub shards ----
    class ReconLinear:
        """Stub LinearEXL3: get_weight_tensor returns a known [k, out] fp16."""

        def __init__(self, k, out, seed):
            g = torch.Generator().manual_seed(seed)
            self.w = torch.randn(k, out, dtype=torch.float16, generator=g)
            self.out_features = out

        def get_weight_tensor(self):
            return self.w

        def forward(self, x, params, out_dtype=torch.float16):
            # the custom op computes each shard as one fp16 GEMM
            assert out_dtype == torch.float16
            return torch.nn.functional.linear(x, self.w.t())

    # real TP2 shape: 3 x 4096 EXL3 shards + b(32)/f_a(128)/g_a(128) tail
    sizes = [4096, 4096, 4096, 32, 128, 128]
    n_exl3 = sum(sizes[:3])
    n_local = sum(sizes)
    cfg = _nr_config(mod, layers={
        "language_model.model.layers.0.self_attn.in_proj_qkvbfg_a": {
            "bits": 6, "bf16_shards": [3, 4, 5]}})
    m = mod.Exl3LinearMethod(
        cfg, "language_model.model.layers.0.self_attn.in_proj_qkvbfg_a", 6)
    m.n_shards = len(sizes)
    m.output_sizes = list(sizes)
    m.bf16_shards = [3, 4, 5]
    m.is_row_parallel = False
    m.tp_rank = 0
    m.tp_size = 2
    m.replicated_shards = frozenset((4, 5))
    m.in_per_partition = H

    linears = [ReconLinear(H, 4096, 1), ReconLinear(H, 4096, 2),
               ReconLinear(H, 4096, 3), None, None, None]
    tail = torch.randn(sum(sizes[3:]), H, dtype=torch.bfloat16,
                       generator=torch.Generator().manual_seed(4))

    mod._DENSE_EXL3_PREFILL_BF16_RETAINED.clear()
    layer = torch.nn.Module()
    try:
        os.environ["GLM53_DENSE_EXL3_PREFILL_BF16"] = "kda_in"
        m._retain_bf16_large_m(layer, linears, tail)
    finally:
        os.environ.pop("GLM53_DENSE_EXL3_PREFILL_BF16", None)
    w = layer.glm53_bf16_lm_w
    assert w.dtype == torch.bfloat16 and w.shape == (n_local, H)
    assert layer.glm53_bf16_lm_n == n_local and layer.glm53_bf16_lm_k == H
    assert layer.glm53_bf16_lm_min_m == mod.DENSE_EXL3_PREFILL_BF16_MIN_M == 144
    # EXL3 rows are the recon transposes, cast once to bf16; the tail rows
    # follow in shard order, unmodified.
    off = 0
    for i in range(3):
        assert torch.equal(w[off:off + sizes[i]],
                           linears[i].w.t().to(torch.bfloat16))
        off += sizes[i]
    assert torch.equal(w[n_exl3:], tail)
    assert mod._DENSE_EXL3_PREFILL_BF16_RETAINED == [
        (m.prefix, "kda_in", n_local * H * 2)]
    # not selected -> nothing retained (default off is unchanged behaviour)
    layer_off = torch.nn.Module()
    m._retain_bf16_large_m(layer_off, linears, tail)
    assert not hasattr(layer_off, "glm53_bf16_lm_w")
    # ---- retention never selects the lm_head or draft modules -----------
    mod._DENSE_EXL3_PREFILL_BF16_RETAINED.clear()
    try:
        os.environ["GLM53_DENSE_EXL3_PREFILL_BF16"] = "all"
        # EXL3 lm_head (vocab head): guarded out before any suffix match.
        hm = mod.Exl3LinearMethod(cfg, "language_model.lm_head", 6)
        hm.is_vocab_head = True
        hl = torch.nn.Module()
        hm._retain_bf16_large_m(hl, linears, tail)
        assert not hasattr(hl, "glm53_bf16_lm_w")
        # EXL3 DFlash2 draft mlp: the runtime prefix has no "draft" marker
        # and gate_up suffix-matches dense_gate_up (load-bearing: the type
        # function alone would select it) — only the draft quant config's
        # prefix-offset marker excludes it.
        dpre = "model.layers.90.mlp.gate_up_proj"
        assert mod._dense_exl3_prefill_bf16_type(
            dpre, mod._dense_exl3_prefill_bf16_types()) == "dense_gate_up"
        dcfg = _nr_config(mod, layers={"model.layers.45.mlp.gate_up_proj": {"bits": 5}})
        dcfg.offset_draft_layer_prefixes(45)  # declarations shift to dpre
        dm = mod.Exl3LinearMethod(dcfg, dpre, 5)
        dl = torch.nn.Module()
        dm._retain_bf16_large_m(dl, linears, tail)
        assert not hasattr(dl, "glm53_bf16_lm_w")
    finally:
        os.environ.pop("GLM53_DENSE_EXL3_PREFILL_BF16", None)
    assert mod._DENSE_EXL3_PREFILL_BF16_RETAINED == []


    # ---- routing: rows > 144 hit the copy, rows <= 144 the custom op ------
    mod._DENSE_EXL3_LAYERS.append(
        {"linears": list(linears), "bf16_shards": [3, 4, 5],
         "output_sizes": list(sizes), "bf16_weight": tail})
    layer._exl3_dense_handle = len(mod._DENSE_EXL3_LAYERS) - 1
    x = torch.randn(145, H, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(5))
    y = m.apply(layer, x)
    assert y.dtype == torch.bfloat16 and y.shape == (145, n_local)
    assert torch.equal(y, torch.nn.functional.linear(x, w)), \
        "rows>144 must be one F.linear against the retained bf16 copy"
    # rows <= 144 falls through to the custom op: unregisterable here, so
    # prove the branch was not taken via the stats counter.
    stats0 = mod.kda_large_m_dispatch_stats()["bf16_calls"]
    x_lo = torch.randn(144, H, dtype=torch.bfloat16)
    try:
        m.apply(layer, x_lo)
    except Exception as exc:  # noqa: BLE001  (torch.ops lookup on CPU)
        assert "dense_exl3_forward" in repr(exc), exc
    else:
        raise AssertionError("rows<=144 must fall through to the custom op")
    assert mod.kda_large_m_dispatch_stats()["bf16_calls"] == stats0, \
        "rows<=144 must not take the retained-copy branch"
    print("dense-exl3 prefill-bf16 (parsing, selection, bitwise copy, M routing) OK")


def ablit_tests():
    spec = importlib.util.spec_from_file_location("ablit_dense_test", ABLIT_SRC)
    ablit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ablit)

    # End to end through apply_ablit: any EXL3 o_proj in range fails loud
    # even before a weight edit could be attempted.
    class Exl3OProj(torch.nn.Module):
        _exl3_linear_n_shards = 1

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleDict()
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.self_attn.o_proj = Exl3OProj()
            self.layers["15"] = layer

    model = Model()
    r = torch.ones(4)
    try:
        ablit.apply_ablit(model, r, [15], alpha=3.0, include_mtp=False)
    except ablit.AblitError as exc:
        assert "EXL3-dense" in str(exc), exc
    else:
        raise AssertionError("apply_ablit over an EXL3 o_proj must refuse")
    print("ablit EXL3 o_proj refusal OK")



def forward_op_tests(mod):
    """The custom-op impl on CPU with stub LinearEXL3 handles: bf16 tail
    output must be exactly F.linear(x, w) in x.dtype (never through fp16 —
    a value past fp16 range would become inf), and the no-bf16 paths return
    x.dtype with the EXL3 outputs cast once."""

    class StubLinear:
        out_features = 64

        def forward(self, x, params, out_dtype=torch.float16):
            assert out_dtype == torch.float16
            return torch.zeros(x.shape[0], self.out_features, dtype=torch.float16)

    def entry(linears, output_sizes, bf16_weight):
        mod._DENSE_EXL3_LAYERS.append(
            {"linears": linears, "bf16_shards": [i for i, l in enumerate(linears) if l is None],
             "output_sizes": output_sizes, "bf16_weight": bf16_weight})
        return len(mod._DENSE_EXL3_LAYERS) - 1

    x = torch.randn(4, 128, dtype=torch.bfloat16)
    # bf16 tail whose true outputs exceed fp16 range: a fp16 round-trip
    # would flush them to inf and fail the equality.
    w = torch.zeros(32, 128, dtype=torch.bfloat16)
    w[:, :] = 1e5  # 128 * 1e5 = 1.28e7 > fp16 max 65504
    h = entry([StubLinear(), None], [64, 32], w)
    y = mod._dense_exl3_forward_impl(x, h)
    assert y.dtype == x.dtype, y.dtype
    assert y.shape == (4, 96), y.shape
    tail = torch.nn.functional.linear(x, w)
    assert torch.equal(y[:, 64:], tail), "bf16 tail must be the exact BF16 GEMM"
    assert torch.isfinite(y).all() and float(y[:, 64:].abs().max()) > 65504
    assert float(y[:, :64].abs().max()) == 0.0

    h = entry([StubLinear(), StubLinear()], [64, 64], None)
    y = mod._dense_exl3_forward_impl(x, h)
    assert y.dtype == x.dtype and y.shape == (4, 128)

    h = entry([StubLinear()], [64], None)
    y = mod._dense_exl3_forward_impl(x, h)
    assert y.dtype == x.dtype and y.shape == (4, 64)
    print("dense_exl3_forward impl (bf16 tail exactness, dtypes) OK")


def warmup_tests(mod):
    """The pre-capture autotune warmup must run every unique dense-EXL3 GEMM
    shape at every row bucket derivable from the capture sizes <= 144. The
    autotune hash keys on MIN(roundup_pow2(rows), 16) + dims (image
    coop_autotune.cu / exl3_gemm.cu:gemm_autotune_hash), so bucket coverage
    is exact coverage; dedup across modules sharing a shape is required
    (tune once per shape x bucket)."""

    calls = []

    class T:  # bare tensor carrier for .device
        device = torch.device("cpu")

    class StubLin:
        """mcg/mul1 as bools, exactly like the image's LinearEXL3."""

        def __init__(self, in_f, out_f, K, cb):
            self.in_features, self.out_features, self.K = in_f, out_f, K
            self.mcg = cb == "mcg"
            self.mul1 = cb == "mul1"
            self.trellis = T()

        def forward(self, x, params, out_dtype=torch.float16):
            assert out_dtype == torch.float16
            calls.append((self.in_features, self.out_features, self.K,
                          "mcg" if self.mcg else "mul1", x.shape[0]))
            return torch.zeros(x.shape[0], self.out_features, dtype=torch.float16)

    a1 = StubLin(4096, 8192, 6, "mul1")
    a2 = StubLin(4096, 8192, 6, "mul1")  # same shape as a1 -> dedup
    b = StubLin(1536, 8192, 6, "mcg")    # distinct shape
    mod._DENSE_EXL3_LAYERS.clear()
    mod._DENSE_EXL3_LAYERS.append(
        {"linears": [a1, a2, b], "bf16_shards": [], "output_sizes": [8192, 8192, 8192],
         "bf16_weight": None})

    orig_avail = torch.cuda.is_available
    orig_sync = torch.cuda.synchronize
    torch.cuda.is_available = lambda: True
    torch.cuda.synchronize = lambda *a, **k: None
    try:
        # capture sizes 1,2,4,8 bucket to themselves; 24, 96 bucket to 16;
        # 512 exceeds the 144 reconstruct threshold and must NOT be warmed.
        n = mod._dense_exl3_warmup_autotune([1, 2, 4, 8, 24, 96, 512])
    finally:
        torch.cuda.is_available = orig_avail
        torch.cuda.synchronize = orig_sync

    buckets = {1, 2, 4, 8, 16}
    assert n == 2 * len(buckets), n  # two unique shapes x five buckets
    per_shape = {}
    for in_f, out_f, K, cb, rows in calls:
        per_shape.setdefault((in_f, out_f, K, cb), []).append(rows)
        assert rows in buckets, rows  # nothing > 144, nothing odd
    assert set(per_shape) == {(4096, 8192, 6, "mul1"), (1536, 8192, 6, "mcg")}
    for shape, rows in per_shape.items():
        assert sorted(rows) == sorted(buckets), (shape, rows)

    # unreadable capture sizes -> all five buckets warmed (fail-safe)
    calls.clear()
    torch.cuda.is_available = lambda: True
    torch.cuda.synchronize = lambda *a, **k: None
    try:
        n = mod._dense_exl3_warmup_autotune(None)
    finally:
        torch.cuda.is_available = orig_avail
        torch.cuda.synchronize = orig_sync
    assert n == 2 * 5 and {r for *_, r in calls} == buckets
    print("coop-autotune warmup (shape dedup, bucket coverage) OK")


def lm_head_tests(mod, stubs):
    """EXL3 lm_head (ParallelLMHead): exact-prefix dispatch (embed_tokens
    unmatchable), contiguous vocab-column shard geometry at TP1/TP2, padded
    vocab refusal, process-time registry/log/count contracts, no LARGE_M
    retention on the head, head_dtype refusal."""
    HEAD = "language_model.lm_head"
    ORG, IN = 512, 64

    # ---- dispatch ------------------------------------------------------
    cfg = _nr_config(mod, layers={HEAD: {"bits": 6}})
    m = cfg.get_quant_method(stubs["ParallelLMHead"](), HEAD)
    assert type(m).__name__ == "Exl3LinearMethod" and m.bits == 6, m
    assert mod._DENSE_EXL3_MODULES[-1] == (HEAD, 6), "boot count must include the head"
    # pack without the key: the head falls back to stock (no behaviour change)
    stock = mod.Exl3Config(bits=4)
    assert stock.get_quant_method(stubs["ParallelLMHead"](), HEAD) is None
    # embed_tokens can never match: not a ParallelLMHead, even at the
    # declared prefix
    assert cfg.get_quant_method(stubs["VocabParallelEmbedding"](), HEAD) is None
    assert cfg.get_quant_method(stubs["VocabParallelEmbedding"](),
                                "language_model.model.embed_tokens") is None

    # ---- vocab-parallel shard geometry (contiguous column block) -------
    t, suh, svh, mk, msuffix = _pack(mod, 6, ORG, IN)
    for tp in (1, 2):
        vp = ORG // tp
        layers = []
        for rank in range(tp):
            mm = mod.Exl3LinearMethod(cfg, HEAD, 6)
            layer = stubs["ParallelLMHead"]()
            layer.prefix = HEAD
            layer.tp_rank = rank
            layer.tp_size = tp
            layer.org_vocab_size = ORG
            mm.create_weights(layer, IN, [vp], IN, ORG,
                              params_dtype=torch.bfloat16)
            _load_all(mod, mm, layer,
                      {(0, "trellis"): t, (0, "suh"): suh, (0, "svh"): svh,
                       (0, "marker"): mk, (0, "marker_suffix"): msuffix}, [0])
            layers.append(layer)
        for rank, layer in enumerate(layers):
            assert layer.trellis.shape == (IN // 16, vp // 16, 96)
            assert torch.equal(
                layer.trellis,
                t[:, rank * vp // 16:(rank + 1) * vp // 16, :]), "trellis tile columns"
            assert torch.equal(layer.svh, svh[rank * vp:(rank + 1) * vp]), "svh rows"
            assert torch.equal(layer.suh[0], suh), "suh is a full copy on every rank"
            assert int(layer.mul1[0, 0]) == mod.MUL1_MARKER_SIGNED_INT32
            assert int(layer.mcg[0, 0]) == 0
        assert torch.equal(torch.cat([l.trellis for l in layers], dim=1), t)
        assert torch.equal(torch.cat([l.svh for l in layers]), svh)

    # ---- padded vocab refuses (vLLM pads to 64; 154880 is aligned, so the
    # served head has padded == org; anything else is a pack mismatch) -----
    mm = mod.Exl3LinearMethod(cfg, HEAD, 6)
    layer = stubs["ParallelLMHead"]()
    layer.prefix = HEAD
    layer.tp_rank = 0
    layer.tp_size = 2
    layer.org_vocab_size = ORG - 64
    try:
        mm.create_weights(layer, IN, [ORG // 2], IN, ORG,
                          params_dtype=torch.bfloat16)
    except ValueError as exc:
        assert "padded vocab" in str(exc), exc
    else:
        raise AssertionError("padded != org vocab must refuse")

    # ---- process: registry entry, per-module + summary lines, count, no
    # LARGE_M retention on the head ---------------------------------------
    class StubLin:
        def __init__(self, suh, svh):
            self.in_features, self.out_features = suh.numel(), svh.numel()
            self.mcg, self.mul1 = False, True

        def forward(self, x, params, out_dtype=torch.float16):
            return torch.zeros(x.shape[0], self.out_features, dtype=torch.float16)

    orig_make, orig_reg = mod.make_linear_exl3, mod._register_dense_exl3_op
    mod.make_linear_exl3 = lambda trellis, suh, svh, mcg, mul1, out_dtype: StubLin(suh, svh)
    mod._register_dense_exl3_op = lambda: None
    mod._DENSE_EXL3_MODULES.clear()
    mod._DENSE_EXL3_LAYERS.clear()
    records = []

    class _Cap(logging.Handler):
        def emit(self, r):
            records.append(r.getMessage())

    log = logging.getLogger("exl3_dense_test")
    cap = _Cap()
    log.addHandler(cap)
    log.setLevel(logging.INFO)
    try:
        # one dense module + the head: the summary must count both
        dpre = "language_model.model.layers.1.mlp.down_proj"
        cfg2 = _nr_config(mod, layers={dpre: {"bits": 6}, HEAD: {"bits": 6}})
        dm = cfg2.get_quant_method(stubs["RowParallelLinear"](), dpre)
        dl = _fake_layer(stubs["RowParallelLinear"], dpre, 0, 1, [64], 128)
        dm.create_weights(dl, 128, [64], 128, 64, params_dtype=torch.bfloat16)
        t2, suh2, svh2, mk2, msuf2 = _pack(mod, 6, 64, 128)
        _load_all(mod, dm, dl,
                  {(0, "trellis"): t2, (0, "suh"): suh2, (0, "svh"): svh2,
                   (0, "marker"): mk2, (0, "marker_suffix"): msuf2}, [0])
        hm = cfg2.get_quant_method(stubs["ParallelLMHead"](), HEAD)
        hl = stubs["ParallelLMHead"]()
        hl.prefix = HEAD
        hl.tp_rank = 0
        hl.tp_size = 1
        hl.org_vocab_size = ORG
        hm.create_weights(hl, IN, [ORG], IN, ORG, params_dtype=torch.bfloat16)
        _load_all(mod, hm, hl,
                  {(0, "trellis"): t, (0, "suh"): suh, (0, "svh"): svh,
                   (0, "marker"): mk, (0, "marker_suffix"): msuffix}, [0])
        os.environ["GLM53_KDA_BF16_LARGE_M"] = "1"
        dm.process_weights_after_loading(dl)
        hm.process_weights_after_loading(hl)
    finally:
        os.environ.pop("GLM53_KDA_BF16_LARGE_M", None)
        log.removeHandler(cap)
        mod.make_linear_exl3 = orig_make
        mod._register_dense_exl3_op = orig_reg

    assert not hasattr(hl, "glm53_bf16_lm_w"), "retention must never select the head"
    assert not hasattr(dl, "glm53_bf16_lm_w")
    assert not hasattr(hl, "trellis"), "staging params must be deleted"
    entry = mod._DENSE_EXL3_LAYERS[hl._exl3_dense_handle]
    assert entry["output_sizes"] == [ORG] and len(entry["linears"]) == 1
    assert entry["bf16_weight"] is None and entry["bf16_shards"] == []

    # ---- head_dtype != model dtype refuses at load, not first request ---
    vc = sys.modules["vllm.config"]
    orig_gc = vc.get_current_vllm_config
    bad = orig_gc()
    bad.model_config.head_dtype = torch.float32
    bad.model_config.dtype = torch.bfloat16
    vc.get_current_vllm_config = lambda: bad
    mod._DENSE_EXL3_MODULES.clear()
    mod._DENSE_EXL3_LAYERS.clear()
    try:
        hm2 = cfg.get_quant_method(stubs["ParallelLMHead"](), HEAD)
        hl2 = stubs["ParallelLMHead"]()
        hl2.prefix = HEAD
        hl2.tp_rank = 0
        hl2.tp_size = 1
        hl2.org_vocab_size = ORG
        hm2.create_weights(hl2, IN, [ORG], IN, ORG, params_dtype=torch.bfloat16)
        _load_all(mod, hm2, hl2,
                  {(0, "trellis"): t, (0, "suh"): suh, (0, "svh"): svh,
                   (0, "marker"): mk, (0, "marker_suffix"): msuffix}, [0])
        try:
            hm2.process_weights_after_loading(hl2)
        except RuntimeError as exc:
            assert "head_dtype" in str(exc), exc
        else:
            raise AssertionError("fp32 head_dtype must refuse the EXL3 head")
    finally:
        vc.get_current_vllm_config = orig_gc
    print("lm_head dispatch/shard-geometry/padding/process/count/retention/head_dtype OK")


def missing_parts_tests(mod, stubs):
    prefix = "language_model.model.layers.0.mlp.down_proj"
    cfg = _nr_config(mod, {prefix: {"bits": 6}})
    method = mod.Exl3LinearMethod(cfg, prefix, 6)
    layer = _fake_layer(stubs["RowParallelLinear"], prefix, 0, 2, [32], 16)
    method.create_weights(layer, 16, [32], 32, 32, torch.bfloat16)
    marker = torch.tensor([mod.MUL1_MARKER_SIGNED_INT32], dtype=torch.int32)
    layer.mul1.weight_loader(layer.mul1, marker)
    try:
        method.process_weights_after_loading(layer)
    except RuntimeError as exc:
        assert "missing weight parts" in str(exc)
    else:
        raise AssertionError("partial checkpoint must not serve uninitialized weights")
    try:
        layer.mul1.weight_loader(layer.mul1, marker, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative shard index accepted")
    layer.tp_size = 3
    try:
        method.create_weights(layer, 16, [32], 48, 32, torch.bfloat16)
    except ValueError as exc:
        assert "TP=3" in str(exc)
    else:
        raise AssertionError("TP3 must not use unpadded trellises")


def test_cpu_behavior_in_isolated_process():
    # The runtime double replaces vllm modules; never leak it into other tests.
    import subprocess
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def main() -> int:
    stubs = _install_vllm_stubs()
    mod = _load_exl3()
    config_tests(mod)
    geometry_tests(mod, stubs)
    draft_qkv_tests(mod, stubs)
    draft_replicated_tests(mod, stubs)
    draft_prefix_offset_tests(mod, stubs)
    forward_op_tests(mod)
    dispatch_tests(mod, stubs)
    draft_layer_fp8_bound_tests(mod, stubs)
    prefill_bf16_tests(mod)
    warmup_tests(mod)
    lm_head_tests(mod, stubs)
    missing_parts_tests(mod, stubs)
    ablit_tests()
    print("dense-exl3 overlay CPU checks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
