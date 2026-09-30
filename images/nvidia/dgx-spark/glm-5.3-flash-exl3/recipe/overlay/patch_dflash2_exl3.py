#!/usr/bin/env python3
"""[dense-exl3] EXL3 DFlash2 draft support — runtime overlay.

Runs at every container start (GLM53_OVERLAY_ORDER) on both ranks and once
at image build (right after patch_dflash2.py), so a fresh build and a
patched-at-boot 927d6521 land the same bytes. Two actions:

1. Install the mounted overlay qwen3_dflash2.py (quant_config threading
   into DFlashGroupedConv, draft EXL3 boot line) over the image's baked
   copy — the same install-if-differs convention as patch_dense_fp8.py's
   exl3.py install.
2. Apply two anchors to the image's qwen3_dflash.py:
   - run the draft quant config's offset_draft_layer_prefixes before the
     draft layers are constructed: the image offsets draft layer prefixes
     by the target's layer count (start_layer_id keeps draft KV-cache
     layer names distinct from the target's), while an EXL3 draft pack
     declares checkpoint-relative model.layers.N prefixes;
   - shape-guard the fused context-KV weight build: with q EXL3 + k/v BF16
     (bf16_shards=[1,2]) the Exl3LinearMethod loader stages exactly the
     K/V rows into qkv_proj.weight — no q rows are ever materialized — so
     the staging tensor IS the KV block and must be used whole. The
     full-BF16 layout still slices off the q rows: identical bytes,
     identical GEMM (cold-TTFT path untouched).

Idempotent per anchor, fail-closed on drift; both edits are inert for BF16
drafts (None quant config / no non_routed_exl3 declarations). Marker:
[dense-exl3-dflash2].
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SITE = Path(os.environ.get("GLM53_SITE", "/usr/local/lib/python3.12/dist-packages/vllm"))
OPT = Path(os.environ.get("GLM53_OPT", "/opt/glm53"))
MARK = "[dense-exl3-dflash2]"

QWEN = SITE / "model_executor/models/qwen3_dflash.py"
MODEL_SRC = OPT / "qwen3_dflash2.py"
MODEL_DST = SITE / "model_executor/models/qwen3_dflash2.py"

# Anchor 1: shift the draft pack's declared prefixes by start_layer_id
# before any draft layer is constructed. hasattr-guarded: BF16 drafts
# (quant_config None) and non-EXL3 quant configs pass through untouched.
QUANT_PREFIX_SHIFT_OLD = (
    "        self.quant_config = get_draft_quant_config(vllm_config)\n"
    "\n"
    '        drafter_config = getattr(self.config, "eagle_config", {})\n'
)
QUANT_PREFIX_SHIFT_NEW = (
    "        self.quant_config = get_draft_quant_config(vllm_config)\n"
    f"        # {MARK} an EXL3 draft pack declares checkpoint-relative\n"
    "        # layers.N prefixes; the runtime offsets draft layers by the\n"
    "        # target's layer count (KV-cache layer names stay distinct).\n"
    "        if start_layer_id and hasattr(\n"
    '            self.quant_config, "offset_draft_layer_prefixes"\n'
    "        ):\n"
    "            self.quant_config.offset_draft_layer_prefixes(start_layer_id)\n"
    "\n"
    '        drafter_config = getattr(self.config, "eagle_config", {})\n'
)

# Anchor 2: the fused context-KV weight build.
FUSED_KV_OLD = (
    "        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]\n"
    "        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]\n"
)
FUSED_KV_NEW = (
    "        # KV projection weights: [num_layers * 2 * kv_size, hidden_size].\n"
    f"        # {MARK} q EXL3 + k/v BF16: qkv_proj.weight stages exactly\n"
    "        # the K/V rows (q is never materialized) — use it whole.\n"
    "        kv_weights = []\n"
    "        for a in layers_attn:\n"
    "            w = a.qkv_proj.weight\n"
    "            if w.shape[0] == a.q_size + 2 * a.kv_size:  # full BF16 QKV\n"
    "                w = w[a.q_size :]\n"
    "            else:\n"
    "                assert w.shape[0] == 2 * a.kv_size, (\n"
    '                    f"qkv_proj.weight rows {w.shape[0]}: neither the full "\n'
    '                    f"QKV ({a.q_size + 2 * a.kv_size}) nor the EXL3 KV "\n'
    '                    f"staging ({2 * a.kv_size})"\n'
    "                )\n"
    "            kv_weights.append(w)\n"
)


def _apply(text: str, old: str, new: str, label: str) -> str:
    """Idempotent single replacement; fail closed when neither state matches."""
    if new in text:
        return text
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{QWEN}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob(f"{target.stem}*.pyc"):
        pyc.unlink(missing_ok=True)


def main() -> int:
    if not MODEL_SRC.is_file():
        raise SystemExit(f"missing {MODEL_SRC}")
    if not QWEN.is_file():
        raise SystemExit(f"missing {QWEN}")

    src_text = MODEL_SRC.read_text()
    compile(src_text, str(MODEL_SRC), "exec")
    if not MODEL_DST.is_file() or MODEL_DST.read_text() != src_text:
        MODEL_DST.write_text(src_text)
        clear_pyc(MODEL_DST)
        print(f"installed {MODEL_SRC.name} -> {MODEL_DST}")

    text = QWEN.read_text()
    patched = _apply(text, QUANT_PREFIX_SHIFT_OLD, QUANT_PREFIX_SHIFT_NEW, "quant prefix shift")
    patched = _apply(patched, FUSED_KV_OLD, FUSED_KV_NEW, "fused context-KV")
    if patched != text:
        compile(patched, str(QWEN), "exec")
        QWEN.write_text(patched)
        clear_pyc(QWEN)
        print(f"patched {QWEN.name} ({MARK})")
    else:
        print(f"{QWEN.name}: {MARK} anchors already present — skipping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
