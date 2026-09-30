#!/usr/bin/env python3
"""Assistant-output quality probes for stock vs cooperative serving.

Temperature 0, thinking off. Scores generated assistant text. This is not a
user-turn logprob test and is not a substitute for the GPU numerical gate.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("GLM53_API", "http://127.0.0.1:8888")
MODEL = "GLM-5.3-Flash-EXL3"
NAN_RE = re.compile(r"\bnan\b|locklock", re.I)

PROBES = {
    "structured": (
        "Count from 1 to 30. Output only the numbers, separated by spaces. No other text.",
        80,
    ),
    "prose": (
        "Write a detailed step-by-step explanation of how a hash map works, "
        "including collision handling, resizing, and time complexity. Be thorough.",
        200,
    ),
    "coding": (
        "Write a Python function named clamp_range that takes a list of ints and "
        "returns a new list with each value clamped to [0, 50]. Include a short docstring.",
        200,
    ),
    "paris": ("What is the capital of France? Answer with one word.", 32),
    "cmp": ("Is 9.9 greater than 9.11? Answer yes or no, then one sentence.", 48),
    "sky": ("Write one sentence about a sky-blue color.", 48),
}


def _headers():
    key = os.environ.get("API_KEY") or os.environ.get("VLLM_API_KEY")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def chat(prompt: str, max_tokens: int) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=_headers(),
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = json.loads(resp.read().decode())
        http = resp.status
    wall = time.perf_counter() - t0
    msg = ((raw.get("choices") or [{}])[0].get("message") or {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    return {
        "http": http,
        "content": content,
        "reasoning": reasoning,
        "text": content,
        "wall_s": wall,
        "usage": raw.get("usage"),
        "finish_reason": ((raw.get("choices") or [{}])[0].get("finish_reason")),
    }


def score(name: str, rec: dict) -> dict:
    text = rec["content"] or ""
    lower = text.lower()
    nan = bool(NAN_RE.search(text) or NAN_RE.search(rec.get("reasoning") or ""))
    ok = bool(text.strip()) and not nan and rec["http"] == 200
    detail = ""
    if name == "structured":
        nums = [int(x) for x in re.findall(r"\b\d+\b", text)]
        expected = list(range(1, 31))
        ok = ok and nums[:30] == expected
        detail = f"ints={nums[:35]}"
    elif name == "prose":
        ok = ok and ("hash" in lower or "map" in lower) and len(text) > 80
        detail = f"len={len(text)}"
    elif name == "coding":
        ok = ok and "def clamp_range" in text and "50" in text
        detail = f"len={len(text)}"
    elif name == "paris":
        ok = ok and "paris" in lower
    elif name == "cmp":
        ok = ok and ("yes" in lower or ("9.9" in text and ">" in text))
    elif name == "sky":
        ok = ok and "blue" in lower
    return {"ok": ok, "nan": nan, "detail": detail, "head": text[:400]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="unspecified")
    args = ap.parse_args()
    rec = {"label": args.label, "ts": time.time(), "probes": {}}
    all_ok = True
    for name, (prompt, max_tokens) in PROBES.items():
        result = chat(prompt, max_tokens)
        judged = score(name, result)
        rec["probes"][name] = {**result, **judged}
        all_ok = all_ok and judged["ok"] and not judged["nan"]
        print(json.dumps({"probe": name, **judged}), flush=True)
    rec["pass"] = all_ok
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=2))
    print(json.dumps({"wrote": args.out, "pass": all_ok, "label": args.label}))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
