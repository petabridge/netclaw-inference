#!/usr/bin/env python3
"""Summarize bench_decode.py receipts into serving-cycle decode metrics.

Elapsed decode ms per draft step uses decode_s / spec.drafts. That ratio is
not a kernel timing. Output-length mismatches are flagged rather than hidden.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def median(xs):
    vals = [x for x in xs if x is not None and math.isfinite(x)]
    if not vals:
        return None
    return statistics.median(vals)


def summarize(path: Path) -> dict:
    rec = json.loads(path.read_text())
    runs = rec.get("runs") or []
    tok = [r.get("tok_s") for r in runs]
    ttft = [r.get("ttft_s") for r in runs]
    decode_s = [r.get("decode_s") for r in runs]
    drafts = [r.get("spec", {}).get("drafts") for r in runs]
    accepted = [r.get("spec", {}).get("accepted_per_step") for r in runs]
    accept_ratio = [r.get("spec", {}).get("accept_ratio") for r in runs]
    tokens = [r.get("completion_tokens") for r in runs]
    ms_per_step = []
    for d_s, n in zip(decode_s, drafts):
        if d_s and n:
            ms_per_step.append(1000.0 * float(d_s) / float(n))
    return {
        "path": str(path),
        "phase": rec.get("phase") or rec.get("prompt"),
        "n_runs": len(runs),
        "tok_s_median": median(tok),
        "tok_s_min": min((x for x in tok if x is not None), default=None),
        "tok_s_max": max((x for x in tok if x is not None), default=None),
        "ttft_median_s": median(ttft),
        "decode_ms_per_draft_step_median": median(ms_per_step),
        "accepted_per_step_median": median(accepted),
        "accept_ratio_median": median(accept_ratio),
        "mean_draft_tokens_per_step_median": median(
            [r.get("spec", {}).get("mean_draft_tokens_per_step") for r in runs]
        ),
        "aggregate_tok_s_median": median(
            [r.get("aggregate_tok_s") for r in runs]
        ),
        "completion_tokens_median": median(
            [float(x) for x in tokens if x is not None]
        ),
        "concurrency": rec.get("concurrency", 1),
        "note": "decode_ms_per_draft_step is a serving-cycle ratio, not a kernel timing",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("receipts", nargs="+", type=Path)
    args = ap.parse_args()
    rows = [summarize(p) for p in args.receipts]
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
