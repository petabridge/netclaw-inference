#!/usr/bin/env python3
"""0010 exactness: an image conversation's turn resumed from a kept prompt state must equal the same turn prefilled
fresh, bit for bit.

Usage: petabridge-tools/media_exact.py [--size 1280x720] [--filler 6000] [--max-tokens 64]
                                       [--log-cmd 'docker logs tensorfold-qwen38fn'] [--scenario NAME ...]
  API_URL / PORT / MODEL as in the recipe's tools/client.py (imported from ../recipe/tools); thinking off, greedy,
  "draft": false. Needs the 0010 image (the "prompt_cache": false request field gives the fresh pass on the same
  running server; no restart between the two).

Each scenario sends turn 1 (kept at its history), then turn 2 twice: resumed (the server's prompt cache) and fresh
("prompt_cache": false). Checks per scenario:
  - the resumed turn 2 reports cached tokens in the range the scenario expects, the fresh one 0;
  - resumed and fresh replies are identical: text, finish reason and the server's token sha;
  - with --log-cmd and the server started with TF_FLASHNEXT_PREFILL_DIGEST=1: the final prompt chunk's logits digest
    (rank 0's "digest rank 0 chunk N: ... logits <hex>" line) is the same for the resumed and the fresh pass. That is
    the first-token logits, compared before sampling. Without it the logits check is UNCHECKED (replies still are).
Scenarios (all over a --filler token text block, so prompts pass the cache's 4,096-token floor):
  after-images    turn 1: text, pictures A and B, a question; turn 2 adds picture C (a new image in the suffix)
  ends-on-image   turn 1's user message ends with a picture (the kept history ends right after the image span)
  same-grid-swap  turn 2 replaces picture A with A' (same size and grid, other pixels): nothing past A may resume
  many-images     turn 1 with 12 pictures, turn 2 adds a 13th (the fixed per-image cap keeps the 12 unchanged)
Keep points inside an image span are not reachable through the API (marks land at the history, the stable prefix
and shared cuts); the engine side of that case is covered by the host tests and by every long fresh pass, whose
prompt chunks end inside image rows.
Exit 0 pass, 1 a check failed, 2 could not run, 3 a check could not be verified.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "recipe", "tools"))
import client  # noqa: E402
import media_images  # noqa: E402

DIGEST = re.compile(r"digest rank 0 chunk \d+: \d+ rows, streams [0-9a-f]+ logits ([0-9a-f]+)")


class Logs:
    """Rank 0's log through --log-cmd; each `since()` returns the lines written after the previous call."""

    def __init__(self, cmd):
        self.cmd = cmd
        self.seen = len(self._read()) if cmd else 0

    def _read(self):
        out = subprocess.run(self.cmd, shell=True, capture_output=True, text=True, errors="replace")
        return (out.stdout + out.stderr).splitlines()

    def since(self):
        if not self.cmd:
            return []
        lines = self._read()
        new, self.seen = lines[self.seen:], len(lines)
        return new

    def last_logits(self):
        found = [m.group(1) for line in self.since() for m in [DIGEST.search(line)] if m]
        return found[-1] if found else None


def ask(convo, a, **extra):
    return client.chat(convo, max_tokens=a.max_tokens, draft=False, seed=7, **extra)


def user(*parts):
    return {"role": "user", "content": [p if isinstance(p, dict) else {"type": "text", "text": p} for p in parts]}


def run_scenario(name, turn1, turn2, a, logs, check_cached):
    """turn1 (kept at its history), then turn2 resumed and fresh; returns (status, line)."""
    r1 = ask(turn1, a)
    return compare(name, turn1 + [{"role": "assistant", "content": r1.content}] + turn2, a, logs, check_cached)


def compare(name, convo2, a, logs, check_cached):
    """convo2 resumed (whatever the server kept), then fresh ("prompt_cache": false); returns (status, line)."""
    logs.since()
    resumed = ask(convo2, a)
    d_resumed = logs.last_logits()
    fresh = ask(convo2, a, prompt_cache=False)
    d_fresh = logs.last_logits()
    problems, unchecked = [], []
    if resumed.cached_tokens is None:
        unchecked.append("no cached-token count reported")
    else:
        why = check_cached(resumed.cached_tokens, resumed.prompt_tokens)
        if why:
            problems.append(why)
    if fresh.cached_tokens not in (None, 0):
        problems.append(f"the fresh pass resumed {fresh.cached_tokens} tokens")
    if (resumed.content, resumed.finish_reason) != (fresh.content, fresh.finish_reason):
        problems.append("resumed and fresh replies differ")
    if resumed.token_sha and fresh.token_sha and resumed.token_sha != fresh.token_sha:
        problems.append(f"token sha {resumed.token_sha[:12]} != {fresh.token_sha[:12]}")
    if a.log_cmd:
        if d_resumed is None or d_fresh is None:
            unchecked.append("no logits digest in the log (TF_FLASHNEXT_PREFILL_DIGEST=1?)")
        elif d_resumed != d_fresh:
            problems.append(f"first-token logits digest {d_resumed} != {d_fresh}")
    else:
        unchecked.append("logits not compared (no --log-cmd)")
    line = (f"{name}: prompt {resumed.prompt_tokens}, resumed {resumed.cached_tokens} cached, prefill "
            f"{resumed.prefill_seconds} s resumed vs {fresh.prefill_seconds} s fresh; logits {d_resumed} / {d_fresh}")
    if problems:
        return client.FAIL, line + " -- FAIL: " + "; ".join(problems)
    if unchecked:
        return client.UNCHECKED, line + " -- UNCHECKED: " + "; ".join(unchecked)
    return client.PASS, line + " -- PASS"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--size", default="1280x720", help="picture size WxH (default 1280x720)")
    p.add_argument("--filler", type=int, default=6000, help="filler text tokens in turn 1 (default 6000)")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--log-cmd", default=os.environ.get("MEDIA_EXACT_LOG_CMD", ""),
                   help="shell command printing rank 0's log (e.g. docker logs <container>)")
    p.add_argument("--scenario", action="append", choices=["after-images", "ends-on-image", "same-grid-swap",
                                                           "many-images"])
    a = p.parse_args()
    w, h = media_images.size(a.size)
    cpt, _ = client.calibrate(client.sentence, 11)
    nonce = random.Random().randrange(1 << 30)          # a new conversation each run: no state of an earlier run
    filler = " ".join(client.filler_units(client.sentence, a.filler, nonce, cpt))
    system = {"role": "system", "content": f"Session {nonce}. You read screenshots and answer in one sentence."}
    img = lambda seed: media_images.part(w, h, nonce * 100 + seed)   # noqa: E731
    logs = Logs(a.log_cmd)
    chosen = a.scenario or ["after-images", "ends-on-image", "same-grid-swap", "many-images"]
    results = []
    base1 = [system, user("Field notes:\n" + filler, img(1), "and", img(2), "What do the two screens show?")]
    if "after-images" in chosen:
        results.append(run_scenario("after-images", base1, [user("A new screen:", img(3), "What changed?")], a, logs,
                                    lambda c, n: None if c and c >= n // 2 else f"resumed only {c} of {n}"))
    if "ends-on-image" in chosen:
        t1 = [system, user("Field notes (ends-on-image):\n" + filler, "The screen:", img(4))]
        results.append(run_scenario("ends-on-image", t1, [user("Describe the top left panel.")], a, logs,
                                    lambda c, n: None if c and c >= n // 2 else f"resumed only {c} of {n}"))
    if "same-grid-swap" in chosen:
        # base1's state is kept at its history (sent again: a cache hit or a new keep); the next turn of the same
        # conversation, but with picture A' (same size and grid, other pixels) where A was, must resume nothing past
        # the filler before A, and must still equal its fresh pass
        r1 = ask(base1, a)
        swapped = [system, user("Field notes:\n" + filler, img(91), "and", img(2), "What do the two screens show?"),
                   {"role": "assistant", "content": r1.content}, user("And now?")]
        limit = a.filler + 1024                          # the system message, the filler and the template around it
        results.append(compare("same-grid-swap", swapped, a, logs,
                               lambda c, n: None if c is not None and c <= limit
                               else f"resumed {c} tokens: past the swapped picture"))
    if "many-images" in chosen:
        parts = ["Field notes (many):\n" + filler] + [img(10 + i) for i in range(12)] + ["Summarize the screens."]
        t1 = [system, user(*parts)]
        results.append(run_scenario("many-images", t1, [user("One more:", img(30), "What is new?")], a, logs,
                                    lambda c, n: None if c and c >= n // 2 else
                                    f"resumed only {c} of {n} (fixed per-image cap off, or past its budget?)"))
    for _, line in results:
        print(line, flush=True)
    codes = [c for c, _ in results]
    if client.FAIL in codes:
        print("media_exact: FAIL")
        return client.FAIL
    if client.UNCHECKED in codes:
        print("media_exact: UNCHECKED")
        return client.UNCHECKED
    print("media_exact: PASS")
    return client.PASS


if __name__ == "__main__":
    client.run(main)
