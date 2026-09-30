#!/usr/bin/env python3
"""GLM53_EXTRA_ENV: extra container env for diagnostics, on both ranks.

CPU-only: drives the shipped launcher with docker/ssh/scp stubbed (the harness of
tests/test_launcher_rank_parity.py), so the guard is fed the same nccl_common and
serve_env arguments a real launch builds -- no second copy of the launcher's env
surface to drift -- and both rank command lines are inspected.

Run:  python3 tests/test_launcher_extra_env.py   (or pytest)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_launcher_rank_parity import Harness, rank_runs  # noqa: E402
from test_start_overrides import _run_preamble  # noqa: E402

FAILURES: list[str] = []
MODEL_DIR = "/root/.cache/huggingface/x"


def check(cond: bool, label: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILURES.append(label)


def container_starts(h: Harness) -> list[list[str]]:
    """Recorded container starts: head (`docker run`) and worker (`ssh ... docker run`)."""
    return [
        c
        for c in h.calls()
        if c[:2] == ["docker", "run"] or (c and c[0] == "ssh" and "docker run" in c[-1])
    ]


def test_extra_env_guard() -> None:
    with tempfile.TemporaryDirectory() as raw:
        h = Harness(Path(raw))

        # Non-owned diagnostics reach BOTH ranks as -e pairs.
        ranks = rank_runs(h, GLM53_EXTRA_ENV="VLLM_LOGGING_LEVEL=DEBUG VLLM_DEBUG_WORKSPACE=1")
        check(ranks is not None, "F1 both rank launches are captured")
        if ranks is not None:
            head, worker, _scp = ranks
            check(
                all(
                    r.env.get("VLLM_LOGGING_LEVEL") == "DEBUG"
                    and r.env.get("VLLM_DEBUG_WORKSPACE") == "1"
                    for r in (head, worker)
                ),
                "F1 both ranks receive the diagnostics as -e pairs",
            )

        # A name the launcher already forwards -- or reserves -- aborts the launch
        # instead of adding a duplicate -e that docker would take last.
        for name, value in (
            ("GPU_MEM_UTIL", "1.5"),  # serve_env: forwarded and range-checked
            ("VLLM_CACHE_ROOT", "/tmp/x"),  # nccl_common
            ("VLLM_HOST_IP", "10.0.0.9"),  # per-rank block
            ("NCCL_SOCKET_IFNAME", "lo"),  # launcher namespace
            ("GLM53_ADAPTIVE_K", "4"),  # launcher namespace
            ("PATH", "/tmp/bin"),  # reserved
        ):
            r = h.run(
                "launch_cluster",
                entry="start.fn.sh",
                MODEL_DIR=MODEL_DIR,
                GLM53_EXTRA_ENV=f"{name}={value}",
            )
            check(
                r.returncode != 0 and not container_starts(h) and name in r.stderr,
                f"F2 {name} is rejected before any container starts (rc={r.returncode})",
            )

        # One owned name anywhere in the list fails the whole launch: no partial set.
        r = h.run(
            "launch_cluster",
            entry="start.fn.sh",
            MODEL_DIR=MODEL_DIR,
            GLM53_EXTRA_ENV="VLLM_LOGGING_LEVEL=DEBUG GPU_MEM_UTIL=1.5",
        )
        check(
            r.returncode != 0 and not container_starts(h),
            "F3 one owned name aborts the whole list",
        )

        # ... and the diagnostics name the entry without echoing its value.
        r = h.run(
            "launch_cluster",
            entry="start.fn.sh",
            MODEL_DIR=MODEL_DIR,
            GLM53_EXTRA_ENV="VLLM_LOGGING_LEVEL=s3cr3t-value",
        )
        check(
            r.returncode == 0
            and "VLLM_LOGGING_LEVEL" in r.stdout
            and "s3cr3t-value" not in r.stdout + r.stderr,
            "F4 the launch log names the entry and redacts its value",
        )

        # Malformed entries fail closed before any container starts.
        for bad in ("novalue", "1ABC=2", "A=x;id", "A=$(id)", "A=a b", "A=*", 'A="q"'):
            r = h.run(
                "launch_cluster",
                entry="start.fn.sh",
                MODEL_DIR=MODEL_DIR,
                GLM53_EXTRA_ENV=bad,
            )
            check(
                r.returncode != 0 and not container_starts(h),
                f"F5 malformed entry fails closed: {bad!r}",
            )


def test_malformed_input_redaction() -> None:
    """Malformed entries fail closed without echoing any fragment of the list.

    The guard word-splits GLM53_EXTRA_ENV, so a whitespace-containing value
    arrives as separate fragments: the raw token and the not-yet-validated name
    both carry value text. A rejection may report the entry position only.
    """
    with tempfile.TemporaryDirectory() as raw:
        h = Harness(Path(raw))
        cases = (
            ("bare fragment after a whitespace split", "VLLM_LOGGING_LEVEL=top s3cr3t-fragment"),
            ("fragment shaped like NAME=VALUE", "VLLM_LOGGING_LEVEL=top s3cr3t-fragment=1"),
            ("lowercase name", "s3cr3t-fragment=1"),
            ("no '=' at all", "s3cr3t-fragment"),
        )
        for label, value in cases:
            r = h.run(
                "launch_cluster",
                entry="start.fn.sh",
                MODEL_DIR=MODEL_DIR,
                GLM53_EXTRA_ENV=value,
            )
            check(
                r.returncode != 0
                and not container_starts(h)
                and "s3cr3t-fragment" not in r.stdout + r.stderr,
                f"F7 {label} is rejected without echoing it (rc={r.returncode})",
            )


def test_caller_precedence() -> None:
    """GLM53_EXTRA_ENV rides the generic caller-export replay, with no per-knob capture."""
    dotenv = "GLM53_EXTRA_ENV=VLLM_LOGGING_LEVEL=DEBUG\n"
    probe = '\nprintf "[%s]\\n" "${GLM53_EXTRA_ENV-UNSET}"\n'
    check(
        _run_preamble(dotenv, {}, probe) == "[VLLM_LOGGING_LEVEL=DEBUG]",
        "F6 .env value survives an unset caller",
    )
    check(
        _run_preamble(dotenv, {"GLM53_EXTRA_ENV": "VLLM_DEBUG_WORKSPACE=1"}, probe)
        == "[VLLM_DEBUG_WORKSPACE=1]",
        "F6 caller export wins over .env",
    )
    check(
        _run_preamble(dotenv, {"GLM53_EXTRA_ENV": ""}, probe) == "[]",
        "F6 explicitly empty caller value wins over .env",
    )


if __name__ == "__main__":
    test_extra_env_guard()
    test_malformed_input_redaction()
    test_caller_precedence()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        raise SystemExit(1)
    print("GLM53_EXTRA_ENV guard OK")
