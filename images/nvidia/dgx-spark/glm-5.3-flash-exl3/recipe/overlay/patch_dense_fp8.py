#!/usr/bin/env python3
"""Install the overlay exl3.py (dense-FP8 Marlin + dense-EXL3 paths) and, when
GLM53_DENSE_FP8 or GLM53_DENSE_EXL3 is on, let the KDA and MLA constructors
keep the quant config so their projections reach Exl3Config.get_quant_method
(idempotent, fail closed).

GLM53_DENSE_FP8=off (default) and GLM53_DENSE_EXL3=0: only the module file is
refreshed — its new code is unreachable, exactly as before. GLM53_DENSE_EXL3=1
additionally serves packs whose config carries non_routed_exl3; the pack must
not be served with the flag off (the overlay index drops the replaced BF16
tensors, so those modules would load empty) — refused here. Which modules
actually quantize is decided per module by the pack config (EXL3) and
GLM53_DENSE_FP8 (Marlin); overlay/exl3.py refuses to mix them on one module.
The indexer wk_weights_proj and the vision tower stay quant_config=None: the
turboderp dense pack keeps them BF16."""
from __future__ import annotations

import os
import sys
from pathlib import Path

SITE = Path(os.environ.get("GLM53_SITE", "/usr/local/lib/python3.12/dist-packages/vllm"))
OPT = Path(os.environ.get("GLM53_OPT", "/opt/glm53"))
MARK = "# [glm53-dense-fp8]"

KDA_OLD = """        saved_quant_config = vllm_config.quant_config
        vllm_config.quant_config = None
        super().__init__(config, vllm_config, prefix)
        vllm_config.quant_config = saved_quant_config
"""
KDA_NEW = """        saved_quant_config = vllm_config.quant_config
        if getattr(saved_quant_config, "get_name", lambda: "")() != "exl3":  # [glm53-dense-fp8]
            vllm_config.quant_config = None
        super().__init__(config, vllm_config, prefix)
        vllm_config.quant_config = saved_quant_config
"""
MLA_OLD = """                quant_config=None,  # MLA projections are BF16 in checkpoint
                prefix=f"{prefix}.self_attn",
"""
MLA_NEW = """                quant_config=(quant_config if getattr(quant_config, "get_name", lambda: "")() == "exl3" else None),  # [glm53-dense-fp8]
                prefix=f"{prefix}.self_attn",
"""


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if MARK in text:
        print(f"{path.name}: {MARK} already present — skipping")
        return
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected one {label} target, found {n}")
    path.write_text(text.replace(old, new, 1))
    print(f"patched {path.name} ({label})")


def _pack_has_non_routed() -> bool:
    """True when the served pack's config carries a non_routed_exl3 block."""
    import json

    model_dir = os.environ.get("MODEL_DIR", "")
    cfg_path = Path(model_dir) / "config.json" if model_dir else None
    if cfg_path is None or not cfg_path.is_file():
        return False
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        return False
    return bool(((cfg.get("quantization_config") or {}).get("non_routed_exl3") or {}))


def main() -> int:
    src = OPT / "exl3.py"
    dst = SITE / "model_executor/layers/quantization/exl3.py"
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    if not dst.is_file():
        raise SystemExit(f"missing {dst}")
    if dst.read_text() != src.read_text():
        dst.write_text(src.read_text())
        print(f"installed {src} -> {dst}")
    else:
        print(f"{dst.name}: already current")
    fp8 = os.environ.get("GLM53_DENSE_FP8", "off").strip().lower()
    exl3 = os.environ.get("GLM53_DENSE_EXL3", "0")
    if exl3 not in ("0", "1"):
        raise SystemExit(f"GLM53_DENSE_EXL3 must be 0 or 1 (got {exl3!r})")
    if exl3 == "0" and _pack_has_non_routed():
        raise SystemExit(
            "served pack carries non_routed_exl3 but GLM53_DENSE_EXL3=0 "
            "(the replaced BF16 tensors are absent from the pack index, so "
            "the dense modules would load empty) — set GLM53_DENSE_EXL3=1 "
            "or serve a non-dense pack"
        )
    if fp8 in ("", "off", "0", "no", "none") and exl3 == "0":
        print("GLM53_DENSE_FP8=off GLM53_DENSE_EXL3=0 — constructors untouched")
        return 0
    kda = SITE / "models/glm5next/nvidia/kda.py"
    if not kda.is_file():
        kda = SITE / "model_executor/models/glm5next/nvidia/kda.py"
    model = kda.parent / "model.py"
    replace_once(kda, KDA_OLD, KDA_NEW, "kda quant_config")
    replace_once(model, MLA_OLD, MLA_NEW, "mla quant_config")
    print(f"dense fp8 groups: {fp8}; dense exl3: {exl3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
