#!/usr/bin/env python3
"""Calibrated numerical panel for quantization/kernel A/B (GLM-5.3-Flash class).

Teacher-forced prompt_logprobs on fixed texts; judge a candidate against the
model's OWN stock-vs-stock variation instead of fixed absolute thresholds.

    GLM53_BENCH_BASE=http://127.0.0.1:8888 python3 tests/bench_numerical_calibrated.py capture --out A.json
    python3 tests/bench_numerical_calibrated.py calibrate --out C.json A1.json A2.json A3.json A4.json [A5.json ...]
    python3 tests/bench_numerical_calibrated.py compare --calibration C.json --stock A1.json A2.json --cand B1.json B2.json B3.json
    python3 tests/bench_numerical_calibrated.py selftest

Why: a fixed "argmax agreement >= 0.99 / mean KL <= 0.01" gate fails on
unchanged stock repeats of this model family (near-tie tokens flip between
identical runs). Calibration measures that wobble first (>= 4 stock captures),
classifies every position, and the comparison only judges the positions stock
itself is stable on. Screening panel, not full qualification.

Position classes are local and calibration-owned: CONF = stock never changed
argmax and every stock margin is at least 0.25 nats; TIE = a smaller margin;
UNSTABLE = stock changed argmax despite every margin being at least 0.25.
The tie cut is fixed, not raised by a noisy position elsewhere in the text.
TIE positions are annotated; UNSTABLE positions are excluded from every gate.

Gates on CONF positions: consensus flips; reference-token range separation
beyond max(0.25, that position's stock spread); and mean standardized KL.
Each position's A/B mean KL is divided by its own maximum stock-pair KL
(floored at 1e-12 for roundoff), then those ratios are averaged and compared
with 2. Separation flags at two positions (one if only one is eligible).
Standardizing BEFORE averaging prevents noisy positions inflating a global
KL allowance. The 1e-12 floor is numerical slack, not a quality tolerance.
These are screening heuristics, not significance tests or evidence of equivalence.
Missing-token bounds respect vLLM's top-k PLUS at most one observed-token
mapping. New captures record k. Without k, the second-smallest logprob is a
conservative upper bound (the smallest may be the extra observed token).
Consensus vote ties use mean logprob only with complete evidence; otherwise
they break lexicographically, without imputing a censored value as an observation.

In particular the selected CONF population has zero observed flips by
construction: its rule-of-three-style count allowance has no claimed coverage.
The fuller PR #182 study was formally INCONCLUSIVE for both candidate arms.

Exit codes: 0 within calibrated variation, 1 FLAG, 2 unqualified inputs
(malformed, mismatched or empty receipts are unqualified, never FLAG).
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path
from decimal import Decimal, localcontext

BASE = os.environ.get("GLM53_BENCH_BASE", "http://127.0.0.1:8888")
MODEL = os.environ.get("GLM53_BENCH_MODEL", "GLM-5.3-Flash-EXL3")
MIN_STOCK_CAPTURES = 4
MARGIN_FLOOR = 0.25
# Position classes.
CONF, TIE, UNSTABLE = 0, 1, 2
CLASS_NAMES = {CONF: "conf", TIE: "tie", UNSTABLE: "unstable"}
# Heuristic count allowance; CONF selection invalidates a coverage claim.
FLIP_NULL_FACTOR = 2.0
# Position-local empirical KL envelope; no population-level confidence claim.
KL_NULL_FACTOR = 2.0
KL_ROUNDOFF = 1e-12
SCHEMA = "calibrated-numerical-panel/v3"


TEXTS = {
    "prose": (
        "Explain how a hash map handles collisions, covering separate chaining "
        "with linked lists, open addressing with linear probing, load factor "
        "thresholds, and amortized resizing cost. Be thorough and precise."
    ),
    "code": (
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
        "    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n"
        "    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n"
        "    return quicksort(left) + middle + quicksort(right)\n"
        "Explain the average-case time complexity of this implementation."
    ),
    "arithmetic": (
        "A train travels 120 km in 2 hours, then 180 km in the next 3 hours. "
        "What is its average speed over the whole journey? Show each step."
    ),
    "structured": "Count from 1 to 40. Output only the numbers, separated by spaces.",
    "dense_tokens": (
        " ".join(
            f"Entry {i}: node NODE{i % 7} reported checksum CK-{i:06d} after "
            f"the maintenance window; temperature {40 + (i * 7) % 23} C, "
            f"fan duty {30 + (i * 13) % 60} percent, no faults."
            for i in range(40)
        )
    ),
}


def _post(path: str, body: dict, timeout: float = 600.0):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())


def parse_position(pos) -> tuple[dict, dict, str, float]:
    """Return probabilities, logprobs, deterministic argmax and top-two margin."""
    if not isinstance(pos, dict) or len(pos) < 2:
        raise ValueError("position requires at least two token probabilities")
    probs, logs = {}, {}
    for token, value in pos.items():
        logprob = value.get("logprob") if isinstance(value, dict) else value
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)) \
                or not math.isfinite(logprob) or logprob > 0:
            raise ValueError("invalid log probability")
        logs[token] = logprob
        probs[token] = math.exp(logprob)
    if math.fsum(probs.values()) > 1.00001:
        raise ValueError("position probabilities exceed total mass one")
    argmax = max(sorted(logs), key=logs.get)
    ordered = sorted(logs.values(), reverse=True)
    margin = ordered[0] - ordered[1]
    return probs, logs, argmax, margin


def _missing_logprob_bound(logs: dict, top_k: int | None) -> float:
    """Upper bound, not an imputed logprob, for an omitted token.

    vLLM append_logprobs_for_next_position inserts the observed token and then
    all top-k tokens, deduplicating overlap. At most ONE entry is outside top-k.
    Thus without recorded k, dropping the lowest entry gives a safe (possibly
    loose) bound for either layout. Never infer k from the mapping's width.
    """
    ordered = sorted(logs.values(), reverse=True)
    if top_k is None:
        if len(ordered) < 2:
            raise ValueError("no justified missing-token boundary")
        return ordered[-2]
    if type(top_k) is not int or top_k < 1 or len(ordered) not in (top_k, top_k + 1):
        raise ValueError("position width does not establish the declared top-k boundary")
    return ordered[top_k - 1]


def kl_shared(ax: dict, by: dict) -> float:
    keys = sorted(ax.keys() & by.keys())
    if not keys or any(ax[k] <= 0 or by[k] <= 0 for k in keys):
        raise ValueError("unusable shared top-k probability support")
    za, zb = math.fsum(ax[k] for k in keys), math.fsum(by[k] for k in keys)
    return max(0.0, math.fsum(
        (ax[k] / za) * (math.log(ax[k] / za) - math.log(by[k] / zb)) for k in keys))

def poisson_critical(lam: float, alpha: float = 0.05) -> int:
    """Smallest c with P(X >= c | lam) <= alpha, without a normal fallback.

    Decimal avoids exp(-lam) underflow in the large-count case. The screening
    caller's lambda is at most six; larger values are useful for boundary checks.
    """
    if not math.isfinite(lam) or lam < 0 or not 0 < alpha < 1:
        raise ValueError("invalid Poisson parameters")
    with localcontext() as ctx:
        ctx.prec = 50
        rate = Decimal(str(lam))
        term = (-rate).exp()
        tail = Decimal(1)
        cutoff = Decimal(str(alpha))
        c = 0
        while tail > cutoff:
            tail -= term  # P(X >= c + 1), including the off-by-one boundary
            c += 1
            term *= rate / c
        return c


def _load_object(path: str) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _validate_texts(receipt: dict) -> dict:
    texts = receipt.get("texts")
    if (not isinstance(receipt.get("model"), str) or not receipt["model"]
            or not isinstance(texts, dict) or not texts):
        raise ValueError("receipt requires a model and nonempty texts")
    for name, rec in texts.items():
        if not isinstance(rec, dict):
            raise ValueError(f"{name}: expected a text object")
        fingerprint = rec.get("prompt_sha256")
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(c not in "0123456789abcdef" for c in fingerprint)):
            raise ValueError(f"{name}: missing or invalid prompt_sha256")
    return texts


def _validate_positions(positions, top_k: int | None = None) -> None:
    if (not isinstance(positions, list) or len(positions) < 2
            or positions[0] is not None):
        raise ValueError("positions require a leading null and scored tokens")
    for position in positions[1:]:
        _, logs, _, _ = parse_position(position)
        _missing_logprob_bound(logs, top_k)


def load_receipt(path: str) -> dict:
    receipt = _load_object(path)
    for name, rec in _validate_texts(receipt).items():
        if rec.get("mode") != "prompt_logprobs":
            raise ValueError(f"{path}/{name}: unqualified capture mode")
        _validate_positions(rec.get("positions"), rec.get("prompt_logprobs_k"))
    return receipt


def load_calibration(path: str) -> dict:
    cal = _load_object(path)
    if cal.get("schema") != SCHEMA:
        raise ValueError("calibration schema differs; recalibrate with this version")
    count, hashes = cal.get("stock_captures"), cal.get("inputs_sha256")
    if (type(count) is not int or count < MIN_STOCK_CAPTURES
            or not isinstance(hashes, list) or len(hashes) != count
            or any(not isinstance(h, str) or len(h) != 64
                   or any(c not in "0123456789abcdef" for c in h) for h in hashes)):
        raise ValueError("invalid calibration input provenance")
    # This validates metadata consistency, not independence or source authenticity.
    for name, rec in _validate_texts(cal).items():
        n = rec.get("positions")
        classes = rec.get("classes")
        if (type(n) is not int or n < 1 or not isinstance(classes, list)
                or len(classes) != n
                or any(type(c) is not int or c not in CLASS_NAMES for c in classes)):
            raise ValueError(f"{name}: invalid position classes")
        for klass, label in CLASS_NAMES.items():
            count = rec.get("n_" + ("confident" if klass == CONF else label))
            if type(count) is not int or count != classes.count(klass):
                raise ValueError(f"{name}: inconsistent class counts")
        if rec["n_confident"] == 0 or rec.get("m_star") != MARGIN_FLOOR:
            raise ValueError(f"{name}: unqualified confidence population")
        for key in ("r_up", "null_kl_mean", "tie_disagreement_rate"):
            value = rec.get(key)
            if (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name}: invalid {key}")
        for key in ("stock_spread_by_position", "null_kl_by_position"):
            values = rec.get(key)
            if not isinstance(values, list) or len(values) != n:
                raise ValueError(f"{name}: invalid {key} layout")
            for klass, value in zip(classes, values):
                if klass != CONF:
                    if value is not None:
                        raise ValueError(f"{name}: excluded position has a threshold")
                elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name}: invalid {key} threshold")
        if (rec["r_up"] != min(1.0, 3.0 / rec["n_confident"])
                or rec["tie_disagreement_rate"] > 1):
            raise ValueError(f"{name}: inconsistent screening allowances")
    return cal


def parsed_texts(receipt: dict) -> dict:
    out = {}
    for name, rec in receipt["texts"].items():
        rows = []
        for pos in rec["positions"][1:]:
            probs, logs, argmax, margin = parse_position(pos)
            bound = _missing_logprob_bound(logs, rec.get("prompt_logprobs_k"))
            rows.append((probs, logs, argmax, margin, bound))
        out[name] = rows
    return out


# ---------------------------------------------------------------- capture

def capture(out: str) -> int:
    res = {"base": BASE, "model": MODEL, "texts": {}}
    for name, text in TEXTS.items():
        rec = {"prompt_chars": len(text), "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "prompt_logprobs_k": 20}
        try:
            status, d = _post("/v1/completions", {
                "model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0,
                "echo": True, "logprobs": 1, "prompt_logprobs": rec["prompt_logprobs_k"],
            })
            if status != 200 or not isinstance(d, dict):
                raise ValueError("invalid completion response")
            choices = d.get("choices")
            if (not isinstance(choices, list) or not choices
                    or not isinstance(choices[0], dict)):
                raise ValueError("missing completion choice")
            pl = choices[0].get("prompt_logprobs")
            _validate_positions(pl, rec["prompt_logprobs_k"])
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
            print(f"{name}: capture request failed ({exc}); capture is unqualified",
                  file=sys.stderr)
            return 2
        rec.update(mode="prompt_logprobs", positions=pl)
        print(f"{name}: positions={len(pl)}", flush=True)
        res["texts"][name] = rec
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res))
    print("wrote", out)
    return 0


# ------------------------------------------------------------- calibrate

def calibrate(out: str, inputs: list[str]) -> int:
    if len(inputs) < MIN_STOCK_CAPTURES:
        print(f"calibrate needs >= {MIN_STOCK_CAPTURES} stock receipts", file=sys.stderr)
        return 2
    try:
        receipts = [load_receipt(p) for p in inputs]
        parsed = [parsed_texts(r) for r in receipts]
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return _unqualified(exc)
    if not receipts:
        return _unqualified(ValueError("no stock receipts"))
    model = receipts[0]["model"]
    if any(r["model"] != model for r in receipts):
        print("UNQUALIFIED: mixed models in calibration set")
        return 2
    names = receipts[0]["texts"].keys()
    if any(r["texts"].keys() != receipts[0]["texts"].keys() for r in receipts):
        print("UNQUALIFIED: calibration texts differ")
        return 2
    cal = {"schema": SCHEMA, "model": model,
           "stock_captures": len(receipts), "inputs_sha256":
           [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs], "texts": {}}
    for name in names:
        if any(r["texts"][name]["prompt_sha256"] != receipts[0]["texts"][name]["prompt_sha256"]
               for r in receipts):
            print(f"UNQUALIFIED: {name} prompt changed between calibration captures")
            return 2
        caps = [r[name] for r in parsed]
        n = len(caps[0])
        if any(len(c) != n for c in caps):
            print(f"UNQUALIFIED: {name} position count differs")
            return 2
        # Local classification: another position can never raise the tie cut.
        min_margin = [min(c[i][3] for c in caps) for i in range(n)]
        disagree = [len({c[i][2] for c in caps}) > 1 for i in range(n)]
        m_star = MARGIN_FLOOR
        # Calibration owns the classes: CONF positions are the only gated ones,
        # TIE positions are annotated, and UNSTABLE positions (stock itself
        # changed argmax at a confident margin) are unusable material. A single
        # bimodal position therefore cannot widen tau, raise the flip bar or
        # deaden the KL gate -- it is excluded and reported instead.
        classes = []
        for i in range(n):
            if min_margin[i] < m_star:
                classes.append(TIE)
            elif disagree[i]:
                classes.append(UNSTABLE)
            else:
                classes.append(CONF)
        conf = [i for i in range(n) if classes[i] == CONF]
        ties = [i for i in range(n) if classes[i] == TIE]
        unstable = [i for i in range(n) if classes[i] == UNSTABLE]
        n_conf = len(conf)
        if n_conf == 0:
            print(f"UNQUALIFIED: {name} has no confident positions at m*={m_star}")
            return 2
        # Selected CONF positions have zero observed disagreements, so this
        # rule-of-three-style allowance is a heuristic, NOT a confidence bound.
        # lambda = n_conf * 2 * r_up = 6 (critical count 11) for n_conf >= 3.
        f = sum(disagree[i] for i in conf)  # 0 by construction; kept in the receipt
        r_up = min(1.0, 3.0 / n_conf)
        cons = [_candidate_argmax(caps, i) for i in range(n)]
        # CONF consensus is present in every calibration capture by definition.
        spreads, null_kl = [None] * n, [None] * n
        kls, tie_dis = [], []
        for i in conf:
            vals = [c[i][1][cons[i]] for c in caps]
            spreads[i] = max(vals) - min(vals)
            try:
                pair = [kl_shared(a[i][0], b[i][0])
                        for ai, a in enumerate(caps) for bi, b in enumerate(caps) if ai != bi]
            except ValueError as exc:
                return _unqualified(exc)
            null_kl[i] = max(pair)
            kls.extend(pair)
        for ai, a in enumerate(caps):
            for bi, b in enumerate(caps):
                if ai != bi:
                    tie_dis.extend(a[i][2] != b[i][2] for i in ties)
        cal["texts"][name] = {
            "positions": n, "m_star": m_star, "classes": classes,
            "n_confident": n_conf, "n_tie": len(ties), "n_unstable": len(unstable),
            "prompt_sha256": receipts[0]["texts"][name]["prompt_sha256"],
            "stock_disagreements_confident": f, "r_up": r_up,
            "null_kl_mean": math.fsum(kls) / len(kls),
            "null_kl_by_position": null_kl,
            "stock_spread_by_position": spreads,
            "tie_disagreement_rate": (sum(tie_dis) / len(tie_dis)) if tie_dis else 0.0,
        }
        print(f"{name:14s} pos={n} m*={m_star:.2f} conf={n_conf} tie={len(ties)} "
              f"unstable={len(unstable)} flips={f} r_up={r_up:.2e} "
              f"nullKLmean={cal['texts'][name]['null_kl_mean']:.2e}")
    Path(out).write_text(json.dumps(cal, indent=1))
    print("wrote", out)
    return 0


# --------------------------------------------------------------- compare

def _candidate_argmax(rows: list, i: int) -> str:
    """Deterministic candidate-consensus argmax at position i.

    Vote ties use mean logprob only when ALL tied tokens occur in ALL captures.
    Otherwise use lexicographic order: censored bounds are not observations.
    """
    counts: dict[str, int] = {}
    # Rank tied tokens by their mean over ALL captures, not only winning votes.
    for r in rows:
        tok = r[i][2]
        counts[tok] = counts.get(tok, 0) + 1
    best = max(counts.values())
    tied = sorted(tok for tok, count in counts.items() if count == best)
    if len(tied) == 1:
        return tied[0]
    if any(tok not in r[i][1] for tok in tied for r in rows):
        return tied[0]
    means = {tok: math.fsum(r[i][1][tok] for r in rows) / len(rows) for tok in tied}
    return max(tied, key=means.get)


def compare(cal_path: str, a_paths: list[str], b_paths: list[str]) -> int:
    """Consensus compare: stock (A) and candidate (B) receipts, explicit sides.

    A position counts as a flip only when the CANDIDATE CONSENSUS argmax
    (majority across B captures) differs from the STOCK CONSENSUS argmax
    (majority across A captures), and only at positions the CALIBRATION
    classified CONF. Systematic damage flips whole consensus groups; random
    near-tie wobble does not. TIE positions are annotated only; UNSTABLE
    positions (stock itself moved there) are excluded and reported."""
    try:
        cal = load_calibration(cal_path)
        ra = [load_receipt(p) for p in a_paths]
        rb = [load_receipt(p) for p in b_paths]
        pa = [parsed_texts(r) for r in ra]
        pb = [parsed_texts(r) for r in rb]
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return _unqualified(exc)
    if not ra or not rb:
        return _unqualified(ValueError("a comparison side is empty"))
    if any(r["texts"].keys() != cal["texts"].keys() for r in ra + rb):
        return _unqualified(ValueError("capture text set differs from calibration"))
    if any(r["model"] != cal["model"] for r in ra + rb):
        print("UNQUALIFIED: model differs from calibration")
        return 2
    flagged, invalid = 0, False
    print(f"{'text':14s} {'pos':>4s} {'conf':>5s} {'unstd':>5s} {'consFlips':>9s} {'crit':>4s} "
          f"{'sep':>5s} {'tieDis':>6s} {'stockTie':>8s} {'meanKL':>9s} {'KLratio':>9s}")
    for name in cal["texts"]:
        if any(name not in x for x in pa + pb):
            print(f"{name}: UNQUALIFIED missing from a capture")
            invalid = True
            continue
        tc = cal["texts"][name]
        n = tc["positions"]
        classes = tc.get("classes")
        rows_a = [x[name] for x in pa]
        rows_b = [x[name] for x in pb]
        fingerprint = tc.get("prompt_sha256")
        if (not isinstance(classes, list) or len(classes) != n
                or any(len(r) != n for r in rows_a + rows_b)):
            print(f"{name}: UNQUALIFIED position set differs from calibration")
            invalid = True
            continue
        if any(r["texts"][name]["prompt_sha256"] != fingerprint for r in ra + rb):
            print(f"{name}: UNQUALIFIED prompt differs from calibration")
            invalid = True
            continue
        flips = tie_dis = sep_exceed = unstable = 0
        kls, kl_ratios = [], []
        cons = [_candidate_argmax(rows_a, i) for i in range(n)]
        for i in range(n):
            klass = classes[i]
            if klass == UNSTABLE:
                unstable += 1
                continue
            amax = _candidate_argmax(rows_b, i)
            if klass == TIE:
                tie_dis += amax != cons[i]
                continue
            flips += amax != cons[i]
            try:
                position_kls = [kl_shared(x[i][0], y[i][0]) for x in rows_a for y in rows_b]
            except ValueError as exc:
                return _unqualified(exc)
            kls.extend(position_kls)
            kl_mean = math.fsum(position_kls) / len(position_kls)
            kl_scale = max(KL_ROUNDOFF, tc["null_kl_by_position"][i])
            kl_ratios.append(kl_mean / kl_scale)
            # separation: every candidate capture's logprob for the stock
            # consensus token strictly outside every stock capture's, by > tau.
            # Incomplete stock ranges cannot support a separation judgment.
            svals = [r[i][1].get(cons[i]) for r in rows_a]
            if any(v is None for v in svals):
                return _unqualified(ValueError(f"{name}/{i}: incomplete stock reference"))
            # The kth boundary excludes any extra low-probability observed
            # prompt token. All candidate captures, including absences, contribute.
            c_hi = [r[i][1].get(cons[i], r[i][4]) for r in rows_b]
            c_lo = [r[i][1].get(cons[i], -math.inf) for r in rows_b]
            tau_gate = max(MARGIN_FLOOR, tc["stock_spread_by_position"][i])
            if (min(svals) - max(c_hi) > tau_gate
                    or min(c_lo) - max(svals) > tau_gate):
                sep_exceed += 1
        n_conf = max(1, tc["n_confident"])
        lam = n_conf * FLIP_NULL_FACTOR * tc["r_up"]
        crit = poisson_critical(lam)
        n_tie = max(1, tc["n_tie"])
        tie_rate = tie_dis / n_tie
        mean_kl = math.fsum(kls) / len(kls)
        mean_ratio = math.fsum(kl_ratios) / n_conf
        text_flag = flips >= crit or mean_ratio > KL_NULL_FACTOR or sep_exceed >= min(2, n_conf)
        flagged += text_flag
        annotation = ""
        if tie_rate > 2 * tc["tie_disagreement_rate"] + 0.05:
            annotation += " tieRate!"
        if text_flag:
            annotation += " FLAG"
        print(f"{name:14s} {n:4d} {tc['n_confident']:5d} {unstable:5d} {flips:9d} {crit:4d} "
              f"{sep_exceed:5d} {tie_rate:6.3f} {tc['tie_disagreement_rate']:8.3f} "
              f"{mean_kl:9.2e} {mean_ratio:9.2e}{annotation}")
    if invalid:
        print("UNQUALIFIED")
        return 2
    print("FLAG (exceeds calibrated stock variation)" if flagged
          else "WITHIN CALIBRATED STOCK VARIATION (shared top-k screening)")
    return 1 if flagged else 0


# --------------------------------------------------------------- selftest

def _base_rows() -> list[tuple[int, int, float]]:
    """Fixed row skeleton shared by every synthetic capture (deterministic)."""
    rng = random.Random(7)
    rows = []
    for i in range(200):
        top = rng.randrange(100000, 999999)
        gap = 0.02 if i % 10 == 0 else 4.0  # near-tie rows wobble; confident rows don't
        rows.append((top, top + 1, gap))
    return rows


_BASE = _base_rows()


def _synthetic(seed: int, flips: dict[str, int], kl_boost: float,
               jitter: float = 0.0, unstable_at: int | None = None,
               vanish: int = 0) -> dict:
    """Deterministic synthetic capture.

    Near-tie rows wobble with `seed`; `flips` counts extra CONFIDENT-row flips;
    `kl_boost` shifts the runner-up probability mass on confident rows (a
    distribution shift); `jitter` adds capture-specific logprob noise so the
    stock null is not exactly zero; `unstable_at` swaps one high-margin row's
    winner in THIS capture (stock wobble the calibration must classify UNSTABLE);
    `vanish` confident rows in THIS capture drop the stock's top token from the
    candidate's top-k entirely (a censored probability, not negative infinity).
    """
    rng = random.Random(seed)
    texts = {}
    for name in TEXTS:
        n_flips = flips.get(name, 0)
        rows, done, vanished = [], 0, 0
        for i, (top, second, gap) in enumerate(_BASE):
            eff_gap, eff_top, eff_second = gap, top, second
            if gap >= 0.25 and vanished < vanish:
                vanished += 1
                rows.append({str(second): {"logprob": 0.0},
                             str(top + 13): {"logprob": -3.0},
                             str(top + 29): {"logprob": -6.0}})
                continue
            if gap < 0.25:  # near-tie: wobble the winner per capture
                if rng.random() < 0.5:
                    eff_top, eff_second = second, top
            elif unstable_at is not None and i == unstable_at:
                eff_top, eff_second = second, top
            elif done < n_flips:  # damage: flip a confident row
                eff_top, eff_second = second, top
                done += 1
            if gap >= 0.25 and kl_boost:
                eff_gap = max(0.3, gap - kl_boost)
            # Relative jitter: it varies the runner-up mass between captures
            # without ever making a non-argmax logprob positive or reordering
            # the top-1.
            scale = rng.uniform(1.0 - jitter, 1.0 + jitter) if jitter else 1.0
            rows.append({str(eff_top): {"logprob": 0.0},
                         str(eff_second): {"logprob": -eff_gap * scale},
                         str(top + 13): {"logprob": -(eff_gap + 3.0) * scale}})
        for row in rows:
            normalizer = math.log(math.fsum(math.exp(v["logprob"]) for v in row.values()))
            for value in row.values():
                value["logprob"] -= normalizer
        texts[name] = {"prompt_sha256": hashlib.sha256(TEXTS[name].encode()).hexdigest(),
                       "mode": "prompt_logprobs", "prompt_logprobs_k": 3,
                       "positions": [None] + rows}
    return {"model": MODEL, "texts": texts}


def selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        paths = []
        for k in range(5):
            # Stock wobbles at near-ties; capture 4 also swaps ONE high-margin
            # row, the bimodal-stock case that used to deaden every gate.
            p = tmpd / f"s{k}.json"
            p.write_text(json.dumps(_synthetic(1000 + k, {}, 0.0, jitter=0.05,
                                               unstable_at=5 if k == 4 else None)))
            paths.append(str(p))
        calp = tmpd / "c.json"
        assert calibrate(str(calp), paths) == 0, "calibrate failed"
        clean = tmpd / "clean.json"
        clean.write_text(json.dumps(_synthetic(999, {}, 0.0, jitter=0.05)))
        damaged = tmpd / "damaged.json"
        damaged.write_text(json.dumps(_synthetic(999, {"prose": 15, "dense_tokens": 12}, 0.0, jitter=0.05)))
        klshift = tmpd / "kl.json"
        klshift.write_text(json.dumps(_synthetic(999, {}, 2.0)))
        shifted = tmpd / "shift.json"
        shifted.write_text(json.dumps(_synthetic(999, {"prose": 6}, 0.0, jitter=0.05)))
        assert compare(str(calp), [str(paths[0])], [str(clean)]) == 0, \
            "clean candidate should be WITHIN"
        assert compare(str(calp), [str(paths[0])], [str(damaged)]) == 1, \
            "systematic confident flips should FLAG"
        assert compare(str(calp), [str(paths[0])], [str(klshift)]) == 1, \
            "KL shift should FLAG"
        assert compare(str(calp), [str(paths[0])], [str(shifted)]) == 1, \
            "a moderate shift must still FLAG despite the unstable stock row"

        def tail_at(start: int, lam: float) -> float:
            bound = int(lam + 12 * math.sqrt(lam)) + 40
            return sum(math.exp(-lam) * lam ** k / math.factorial(k)
                       for k in range(max(0, start), bound + 1))

        # The critical count is the documented tail definition: the smallest c
        # with P(X >= c) <= alpha, not alpha one step early.
        for lam in (0.5, 5.0, 30.0):
            crit = poisson_critical(lam)
            assert tail_at(crit, lam) <= 0.05 < tail_at(crit - 1, lam), (lam, crit)

        # A capture whose prompt does not match the calibration is unqualified
        # instead of being compared position-by-position.
        tampered = json.loads(json.dumps(_synthetic(999, {}, 0.0, jitter=0.05)))
        tampered["texts"]["prose"]["prompt_sha256"] = "0" * 64
        tampered_path = tmpd / "tampered.json"
        tampered_path.write_text(json.dumps(tampered))
        assert compare(str(calp), [str(paths[0])], [str(tampered_path)]) == 2, \
            "prompt mismatch should be UNQUALIFIED"

        # Missing candidate tokens are bounded by the captured top-k floor.
        gone = tmpd / "vanish.json"
        gone.write_text(json.dumps(_synthetic(999, {}, 0.0, jitter=0.05, vanish=2)))
        assert compare(str(calp), [str(paths[0])], [str(gone)]) == 1, \
            "large censored-token shift should FLAG"

        # Malformed receipts are unqualified (2), never the FLAG code (1).
        broken = tmpd / "broken.json"
        broken.write_text("{not json")
        assert compare(str(calp), [str(paths[0])], [str(broken)]) == 2, \
            "malformed receipt should be UNQUALIFIED"

        # The CLI must keep the sides apart with several receipts per side.
        saved = sys.argv
        try:
            sys.argv = ["x", "compare", "--calibration", str(calp), "--stock", str(paths[0]),
                        "--cand", str(damaged), str(klshift)]
            assert main() == 1, "CLI compare with two candidate receipts"
            sys.argv = ["x", "compare", "--calibration", str(calp), "--stock", str(paths[0]),
                        "--cand", str(clean)]
            assert main() == 0, "CLI compare with one candidate receipt"
        finally:
            sys.argv = saved

        print("selftest: clean WITHIN, flips FLAG, kl-shift FLAG, unstable-row "
              "regression, censored shift FLAG, poisson tail, prompt pin, malformed=2, "
              "CLI sides -> OK")
        return 0


def _unqualified(exc: Exception) -> int:
    """Malformed / missing / mismatched inputs are unqualified (2), never FLAG."""
    print(f"UNQUALIFIED: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture"); c.add_argument("--out", required=True)
    k = sub.add_parser("calibrate"); k.add_argument("--out", required=True); k.add_argument("inputs", nargs="+")
    p = sub.add_parser("compare", help="judge candidate receipt(s) against stock receipt(s)")
    p.add_argument("--calibration", required=True)
    p.add_argument("--stock", nargs="+", required=True, help="stock receipt(s): the A side")
    p.add_argument("--cand", nargs="+", required=True, help="candidate receipt(s): the B side")
    sub.add_parser("selftest")
    args = ap.parse_args()
    try:
        if args.cmd == "capture":
            return capture(args.out)
        if args.cmd == "calibrate":
            return calibrate(args.out, args.inputs)
        if args.cmd == "compare":
            return compare(args.calibration, list(args.stock), list(args.cand))
        return selftest()
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        return _unqualified(exc)


if __name__ == "__main__":
    raise SystemExit(main())
