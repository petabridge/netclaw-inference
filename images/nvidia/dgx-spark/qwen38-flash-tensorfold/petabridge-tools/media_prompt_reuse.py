#!/usr/bin/env python3
"""Prompt reuse with screenshots: an agent-style conversation adds a screenshot every turn, and each turn must resume
most of its prompt from the server's cache (the recipe's tools/prompt_reuse.py, with image turns; 0010).

Usage: petabridge-tools/media_prompt_reuse.py [size] [turns] [--first-images N] [--per-turn N] [--size 1920x1080]
                                              [--min-reuse 0.9] [--other]
  size  the first turn's text in tokens (default 30000; '128k' accepted), turns the follow-up turns (default 6).
  --first-images  screenshots in the first turn (default 4); --per-turn  screenshots each follow-up adds (default 1).
  --other  before each follow-up, one image request of another conversation with the same system prompt.
  API_URL / PORT / MODEL as in the recipe's tools/client.py (imported from ../recipe/tools); thinking off, greedy.
Each turn prints prompt tokens, cached tokens, the share resumed, the server's prefill seconds and the wall time.
Before 0010 every image turn reports 0 cached (a full prefill). Pass: every follow-up resumed >= min-reuse of its
prompt. A conversation whose screenshots outgrow TENSORFOLD_IMAGE_TOKENS at TENSORFOLD_IMAGE_TOKENS_EACH falls back
to the shared split and is expected to FAIL from that turn on (the tool says so). With no cached-token count the
result is UNCHECKED (exit 3), never a pass. Exit 0 pass, 1 fail, 2 could not run, 3 unchecked.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "recipe", "tools"))
import client  # noqa: E402
import media_images  # noqa: E402


def text(tokens: int, seed: int, cpt: float) -> str:
    return " ".join(client.filler_units(client.sentence, tokens, seed, cpt))


def fmt(v, spec="{:.2f} s"):
    return spec.format(v) if v is not None else "n/a"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("size", nargs="?", default="30000")
    p.add_argument("turns", nargs="?", type=int, default=6)
    p.add_argument("--first-images", type=int, default=4)
    p.add_argument("--per-turn", type=int, default=1)
    p.add_argument("--size", dest="picture", default="1920x1080", help="screenshot size WxH (default 1920x1080)")
    p.add_argument("--other", action="store_true", help="interleave another conversation with the same system prompt")
    p.add_argument("--min-reuse", type=float, default=0.9)
    p.add_argument("--image-tokens", type=int, default=int(os.environ.get("TENSORFOLD_IMAGE_TOKENS") or 16384),
                   help="the server's TENSORFOLD_IMAGE_TOKENS, to say when a turn passes the fixed cap's budget")
    p.add_argument("--image-tokens-each", type=int,
                   default=int(os.environ.get("TENSORFOLD_IMAGE_TOKENS_EACH") or 1024))
    a = p.parse_args()
    if a.turns < 1:
        p.error("turns is at least 1")
    size = client.parse_size(a.size)
    w, h = media_images.size(a.picture)
    nonce = random.Random().randrange(1 << 30)
    seeds = iter(range(nonce * 1000, nonce * 1000 + 100000))
    shot = lambda: media_images.part(w, h, next(seeds))   # noqa: E731
    cpt, _ = client.calibrate(client.sentence, 11)
    system = {"role": "system", "content": f"Session {nonce}. You are a careful agent. Look at each screenshot, then "
                                           "answer in one short sentence. " + text(1500, 1, cpt)}
    images = a.first_images
    first = [{"type": "text", "text": "Here is the task log and the screens so far.\n\n" + text(size, 2, cpt)}]
    first += [shot() for _ in range(a.first_images)]
    convo = [system, {"role": "user", "content": first}]
    r = client.chat(convo, max_tokens=48)
    cold_prefill = r.prefill_seconds
    print(f"turn 0 (cold, {images} screenshots): prompt {r.prompt_tokens} tokens, cached {r.cached_tokens}, prefill "
          f"{fmt(cold_prefill)}, wall {r.seconds:.1f} s", flush=True)
    shares, unreported, past_budget = [], False, None
    rng = random.Random(5)
    for turn in range(1, a.turns + 1):
        if a.other:
            o = client.chat([system, {"role": "user", "content": [
                {"type": "text", "text": f"Side question {turn}: what is on this screen?"}, shot()]}], max_tokens=32)
            print(f"  other conversation: prompt {o.prompt_tokens} tokens, cached {o.cached_tokens}")
        images += a.per_turn
        if past_budget is None and a.image_tokens_each and images * a.image_tokens_each > a.image_tokens:
            past_budget = turn
        content = [{"type": "text", "text": f"Step {turn}: {text(150, 100 + turn, cpt)} I clicked "
                                            f"{rng.choice(client._NOUNS)}. New screen:"}]
        content += [shot() for _ in range(a.per_turn)]
        convo += [{"role": "assistant", "content": r.content}, {"role": "user", "content": content}]
        r = client.chat(convo, max_tokens=48)
        cached, prompt = r.cached_tokens, r.prompt_tokens
        if cached is None or not prompt:
            unreported = True
            share = None
        else:
            share = cached / prompt
            shares.append(share)
        note = " (past the fixed cap's budget: shared split)" if past_budget is not None and turn >= past_budget else ""
        print(f"turn {turn} ({images} screenshots{note}): prompt {prompt} tokens, cached {cached}"
              + (f" ({share:.1%})" if share is not None else " (not reported)")
              + f", prefill {fmt(r.prefill_seconds)}, wall {r.seconds:.1f} s", flush=True)
    if unreported:
        print("media_prompt_reuse: UNCHECKED: the server reports no cached-token count; compare the prefill times above "
              f"with the cold turn's ({fmt(cold_prefill)})")
        return client.UNCHECKED
    worst = min(shares)
    if worst >= a.min_reuse:
        print(f"media_prompt_reuse: PASS: every follow-up resumed >= {a.min_reuse:.0%} of its prompt "
              f"(worst {worst:.1%})")
        return client.PASS
    hint = (f"; turns from {past_budget} on are past TENSORFOLD_IMAGE_TOKENS={a.image_tokens} at "
            f"{a.image_tokens_each} an image, expected to resume little" if past_budget is not None else "")
    print(f"media_prompt_reuse: FAIL: a follow-up resumed only {worst:.1%} of its prompt (want >= {a.min_reuse:.0%})"
          + hint)
    return client.FAIL


if __name__ == "__main__":
    client.run(main)
