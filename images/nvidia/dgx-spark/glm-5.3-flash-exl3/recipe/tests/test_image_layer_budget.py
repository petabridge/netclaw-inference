#!/usr/bin/env python3
"""The serve image must stay under overlay2's runnable layer depth.

Every image layer adds a ``lowerdir=`` path to one PAGE_SIZE-limited mount
option string (moby/moby#46740). A build past that depth still succeeds on
the head, and ``docker save`` still works, but the worker's ``docker load``
fails with ``max depth exceeded`` and two-node bring-up stops. This happened
at 128 layers (#280, fixed in 582fac8) and at 126 (#301, fixed in #302).

Observed on 2x GB10 hosts: a 123-layer image loaded on the worker; 126 did
not. The cap is the largest depth known to work, not a guess at the limit.
Past it, fold new steps into existing layers (several files per COPY, patch
RUNs chained with &&, in order), as 582fac8 and #302 did.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"

# Layers in the pinned base image (docker image inspect -f
# '{{len .RootFS.Layers}}' on the digest below). Changing the base changes
# this count: re-measure it and update both constants together.
BASE_DIGEST = "sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce"
BASE_LAYERS = 32
MAX_LAYERS = 123

LAYER_INSTRUCTIONS = {"COPY", "RUN", "ADD"}
HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def layer_steps(text: str) -> int:
    """COPY/RUN/ADD instructions, skipping continuation lines and heredoc bodies."""
    lines = text.splitlines()
    count = i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        instruction = stripped.split(None, 1)[0].upper()
        # Gather the full instruction across backslash continuations.
        full = line
        while full.rstrip().endswith("\\") and i < len(lines):
            full = full.rstrip()[:-1] + " " + lines[i]
            i += 1
        # Skip every heredoc body the instruction opens, in order.
        for match in HEREDOC.finditer(full):
            terminator = match.group(2)
            while i < len(lines) and lines[i].strip() != terminator:
                i += 1
            i += 1
        if instruction in LAYER_INSTRUCTIONS:
            count += 1
    return count


def test_base_image_is_the_measured_one() -> None:
    text = DOCKERFILE.read_text()
    base = re.search(r"^ARG BASE=(\S+)$", text, re.M)
    assert base, "Dockerfile has no ARG BASE line"
    assert base.group(1).endswith("@" + BASE_DIGEST), (
        f"base image changed to {base.group(1)}: re-measure its layer count and "
        "update BASE_DIGEST and BASE_LAYERS in this test")


def test_image_stays_within_runnable_layer_depth() -> None:
    steps = layer_steps(DOCKERFILE.read_text())
    total = BASE_LAYERS + steps
    assert total <= MAX_LAYERS, (
        f"image would have {total} layers ({BASE_LAYERS} base + {steps} COPY/RUN/ADD); "
        f"workers fail docker load past {MAX_LAYERS}. Fold new steps into existing "
        "layers: several files per COPY, patch RUNs chained with && in order.")


def test_counter_skips_heredoc_bodies_and_continuations() -> None:
    sample = (
        "FROM x\n"
        "RUN python3 - <<'PY'\n"
        "RUN = 1  # a heredoc line that looks like an instruction\n"
        "COPY this is data\n"
        "PY\n"
        "COPY a \\\n"
        "  b /dst/\n"
        "# RUN commented out\n"
        "ENV K=v\n"
        "run lower-case instructions count too\n"
    )
    assert layer_steps(sample) == 3


if __name__ == "__main__":
    test_base_image_is_the_measured_one()
    test_image_stays_within_runnable_layer_depth()
    test_counter_skips_heredoc_bodies_and_continuations()
    print(f"image layer budget OK ({BASE_LAYERS} + {layer_steps(DOCKERFILE.read_text())} <= {MAX_LAYERS})")
