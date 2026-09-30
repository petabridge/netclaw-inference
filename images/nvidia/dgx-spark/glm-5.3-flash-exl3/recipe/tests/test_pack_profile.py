"""Asset-backed profile selection, precedence, and pre-stop refusal."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pack_profile", ROOT / "tools/pack_profile.py")
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


def packed(tensors, base, bits=6):
    for suffix, dtype, shape in (("trellis", "I16", [1, 1, bits * 16]),
                                 ("suh", "F16", [16]), ("svh", "F16", [16]),
                                 ("mul1", "I32", [1])):
        tensors[base + "." + suffix] = (dtype, shape)


def save_tensors(path, tensors):
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = 4 if dtype == "I32" else 2
        for dim in shape:
            size *= dim
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


def stage(tmp_path):
    hub = tmp_path / "hub"
    target = hub / "models--fixture--target" / "snapshots" / "target-rev"
    target.mkdir(parents=True)
    (target.parent.parent / "refs").mkdir()
    (target.parent.parent / "refs/main").write_text(target.name)
    draft = hub / "models--fixture--draft" / "snapshots" / ("a" * 40)
    draft.mkdir(parents=True)
    tensors, layers = {}, {"model.fc": {"bits": 6}}
    packed(tensors, "fc")
    for i in range(5):
        for suffix in ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                       "mlp.down_proj", "attention_conv.kernel_projection", "mlp_conv.kernel_projection"):
            packed(tensors, f"layers.{i}.{suffix}")
            served = suffix.replace("q_proj", "qkv_proj") if suffix == "self_attn.q_proj" else suffix
            served = re.sub(r"(gate|up)_proj$", "gate_up_proj", served)
            layers[f"model.layers.{i}.{served}"] = {"bits": 6}
        layers[f"model.layers.{i}.self_attn.qkv_proj"]["bf16_shards"] = [1, 2]
        for suffix in ("k_proj", "v_proj"):
            tensors[f"layers.{i}.self_attn.{suffix}.weight"] = ("BF16", [16, 16])
    save_tensors(draft / "model.safetensors", tensors)
    dc = {"architectures": ["DFlash2DraftModel"], "hidden_size": 4096, "num_hidden_layers": 5,
          "quantization_config": {"quant_method": "exl3", "bits": 6, "scope": "dflash2_draft",
                                  "non_routed_exl3": {"layers": layers}}}
    raw = json.dumps(dc).encode()
    (draft / "config.json").write_bytes(raw)
    target_layers, target_tensors = {}, {}
    for suffix, shards in {
        "self_attn.in_proj_qkvbfg_a": ("q_proj", "k_proj", "v_proj"),
        "mlp.shared_experts.down_proj": ("down_proj",),
        "self_attn.fused_qkv_a_proj": ("q_a_proj", "kv_a_proj_with_mqa"),
        "mlp.shared_experts.gate_up_proj": ("gate_proj", "up_proj"),
        "self_attn.o_proj": ("o_proj",), "self_attn.q_b_proj": ("q_b_proj",),
    }.items():
        target_layers[f"language_model.model.layers.0.{suffix}"] = {"bits": 6}
        for shard in shards:
            packed(target_tensors, "model.language_model.layers.0." + suffix.rsplit(".", 1)[0] + "." + shard)
    save_tensors(target / "model.safetensors", target_tensors)
    (target / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: "model.safetensors" for key in target_tensors}}))
    tc = {"hidden_size": 4096, "num_hidden_layers": 45,
          "quantization_config": {"quant_method": "exl3", "non_routed_exl3": {"layers": target_layers}},
          "glm53_profile": {"name": profile.PROFILE, "draft": {"model": "fixture/draft", "revision": "a" * 40,
                            "config_sha256": hashlib.sha256(raw).hexdigest()}}}
    (target / "config.json").write_text(json.dumps(tc))
    return hub, target, draft


def test_asset_backed_profile_and_normal_pack(tmp_path):
    hub, target, draft = stage(tmp_path)
    result = profile.resolve(target, hub, {"GLM53_DENSE_FP8": "all"}, set())
    assert result["GLM53_DENSE_FP8"] == "off"
    assert result["GLM53_DENSE_EXL3_PREFILL_BF16"].split(",") == [
        "kda_in", "shared_down", "mla_qkv_a", "shared_gate_up", "kda_o", "mla_q_b"]
    assert result["DFLASH_MODEL"] == "fixture/draft"
    assert result["DFLASH_REVISION"] == draft.name
    assert result["EXPECTED_SHARDS"] == "1"
    (target / "config.json").write_text('{"quantization_config":{"quant_method":"exl3"}}')
    assert profile.resolve(target, hub, {"GLM53_DENSE_FP8": "all"}, set()) == {}


@pytest.mark.parametrize("env,explicit", [
    ({"TP": "3"}, set()), ({"ABLIT": "1", "ABLIT_LAYERS": "0-3"}, set()),
    ({"GLM53_DENSE_FP8": "all"}, {"GLM53_DENSE_FP8"}),
    ({"GLM53_DENSE_EXL3": "0"}, {"GLM53_DENSE_EXL3"}),
    ({"GLM53_DENSE_EXL3_PREFILL_BF16": "dense_gate_up"}, set()),
    ({"DFLASH_MODEL": "different/draft"}, set()), ({"SPEC_METHOD": "mtp"}, set()),
])
def test_conflicts_refuse(tmp_path, env, explicit):
    hub, target, _ = stage(tmp_path)
    with pytest.raises(ValueError):
        profile.resolve(target, hub, env, explicit)


def test_ablit_selects_only_layers_whose_o_proj_stays_bf16(tmp_path):
    hub, target, _ = stage(tmp_path)  # the fixture declares EXL3 o_proj on layer 0 only
    assert profile.resolve(target, hub, {"ABLIT": "1"}, set())["ABLIT"] == "1"  # default 15-45
    assert profile.resolve(target, hub, {"ABLIT": "0", "ABLIT_LAYERS": "0"}, set())["ABLIT"] == "0"
    tc = json.loads((target / "config.json").read_text())
    layers = tc["quantization_config"]["non_routed_exl3"]["layers"]
    layers["language_model.model.layers.15.self_attn.o_proj"] = {"bits": 6}
    (target / "config.json").write_text(json.dumps(tc))
    with pytest.raises(ValueError, match=r"layers \[15\]"):
        profile.resolve(target, hub, {"ABLIT": "1"}, set())
    profile.check_ablit(tc, "16-45")  # layer 45 is the MTP block, never a target o_proj
    with pytest.raises(ValueError):
        profile.check_ablit(tc, "15")


def _runtime_parse_layers():
    """overlay/ablit_runtime.parse_layers without importing torch."""
    import ast
    tree = ast.parse((ROOT / "overlay/ablit_runtime.py").read_text())
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and n.name in ("AblitError", "parse_layers")]
    ns: dict = {}
    exec(compile(ast.Module(keep, type_ignores=[]), "ablit_runtime", "exec"), ns)
    return ns["parse_layers"]


@pytest.mark.parametrize("spec", ["15-45", "15,17-19", " 15 - 20 ", "+15", "15-", "-5",
                                  "20-15", "x", "", ",", "1 5"])
def test_ablit_layer_spec_matches_the_runtime_parser(spec):
    """The pre-stop check must accept exactly what the in-container hook accepts:
    a spec it passes but the runtime refuses would stop a healthy serve first."""
    try:
        want = set(_runtime_parse_layers()(spec))
    except Exception:  # AblitError
        with pytest.raises(ValueError):
            profile.parse_layers(spec)
    else:
        assert profile.parse_layers(spec) == want


@pytest.mark.parametrize("damage", ["missing", "truncated", "hash", "bitrate", "missing_part"])
def test_absent_or_incompatible_assets_refuse(tmp_path, damage):
    hub, target, draft = stage(tmp_path)
    weights = draft / "model.safetensors"
    if damage == "missing":
        weights.unlink()
    elif damage == "truncated":
        weights.write_bytes(weights.read_bytes()[:-1])
    elif damage == "missing_part":
        raw = weights.read_bytes()
        size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + size])
        del header["fc.suh"]
        blob = json.dumps(header).encode()
        weights.write_bytes(struct.pack("<Q", len(blob)) + blob + raw[8 + size:])
    else:
        dc = json.loads((draft / "config.json").read_text())
        dc["quantization_config"]["bits"] = 5
        raw = json.dumps(dc).encode()
        (draft / "config.json").write_bytes(raw)
        if damage == "bitrate":
            tc = json.loads((target / "config.json").read_text())
            tc["glm53_profile"]["draft"]["config_sha256"] = hashlib.sha256(raw).hexdigest()
            (target / "config.json").write_text(json.dumps(tc))
    with pytest.raises((ValueError, OSError)):
        profile.resolve(target, hub, {}, set())


def test_launcher_resolves_before_restart_stops(tmp_path):
    hub, target, _ = stage(tmp_path)
    launcher = (ROOT / "start.sh").read_text()
    functions = []
    for name in ("resolve_pack_profile", "main"):
        functions.append(re.search(rf"(?ms)^{name}\(\) \{{\n.*?^\}}", launcher).group())
    script = "\n".join(functions) + '''
set -eu
log() { :; }; banner() { :; }; validate_numeric_config() { :; }; select_dense_h3() { :; }
configure_capture_sizes() { :; }; validate_overlay_artifacts() { :; }
with_cluster_lock() { :; }
stop_containers() { printf 'STOP\n'; }
start_unlocked() {
    [ "$MODEL_REVISION" = "$MODEL_SNAPSHOT" ]
    [ "$FALLBACK_MODEL_PATH" = "$MODEL_PATH" ]
    [ "$MODEL_FALLBACK_SNAPSHOT" = "$MODEL_SNAPSHOT" ]
    [ "$EXPECTED_SHARDS" = 1 ]
    printf '%s|%s|%s\\n' "$GLM53_DENSE_EXL3" "$GLM53_DENSE_EXL3_PREFILL_BF16" "$DFLASH_MODEL"
}
main restart
'''
    env = {"PATH": os.environ["PATH"], "SCRIPT_DIR": str(ROOT), "MODEL_PATH": str(target.parent.parent),
           "HF_CACHE_DIR": str(hub.parent), "_GLM53_PROFILE_EXPLICIT": "", "GLM53_DENSE_FP8": "all",
           "MODEL": "fixture/target", "MODEL_CACHE_NAME": "models--fixture--target"}
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["STOP", f"1|{profile.H3}|fixture/draft"]
    env["_GLM53_PROFILE_EXPLICIT"] = "GLM53_DENSE_FP8"
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "STOP" not in result.stdout
    env["_GLM53_PROFILE_EXPLICIT"] = "MODEL_REVISION"
    env["MODEL_REVISION"] = "different-revision"
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "STOP" not in result.stdout
