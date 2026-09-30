#!/usr/bin/env python3
"""start-tp4.sh forwards adaptive verification length (GLM53_ADAPTIVE_K*) to every rank.

CPU-only static + generator check. The two-node launcher has carried this since
2026-09-08; the four-node launcher never did, so a TP=4 kit setting
GLM53_ADAPTIVE_K=ema got stock k on all ranks with no error. Mirrors the spinwait
overlay's wiring shape (host path, existence check, scp to ranks 1-3, read-only
mount, container-start application, env on head and workers) plus the
capture-size generator start.sh uses.

Run:  python3 tests/test_tp4_adaptive_k_wiring.py   (or pytest)
"""
from __future__ import annotations

import re
import ast
import os
import tempfile
from unittest.mock import patch
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start-tp4.sh"
KNOBS = (
    "GLM53_ADAPTIVE_K", "GLM53_ADAPTIVE_K_SET", "GLM53_ADAPTIVE_K_ALPHA", "GLM53_ADAPTIVE_K_MARGIN",
    "GLM53_ADAPTIVE_K_MIN_STEPS", "GLM53_ADAPTIVE_K_SATURATE", "GLM53_ADAPTIVE_K_HIST",
)


def test_recipe_wiring() -> None:
    s = START.read_text()
    assert 'ADAPTIVE_K_PATCH_HOST="${ADAPTIVE_K_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_adaptive_k.py}"' in s
    assert '[ -f "$ADAPTIVE_K_PATCH_HOST" ] || die "$ADAPTIVE_K_PATCH_HOST missing"' in s
    assert s.count("python3 /opt/glm53/patch_adaptive_k.py") == 2, "head and worker inner scripts"
    assert s.count('"${ssh_t}:/tmp/patch_adaptive_k.py"') == 1
    assert "-v '/tmp/patch_adaptive_k.py:/opt/glm53/patch_adaptive_k.py:ro'" in s
    assert '-v "$ADAPTIVE_K_PATCH_HOST:/opt/glm53/patch_adaptive_k.py:ro"' in s
    loop = s[s.index('local serve_env=""'):s.index('serve_env+=" -e VLLM_API_KEY')]
    for k in KNOBS:
        assert f'{k}="${{{k}-' in s, f"default for {k}"
        assert f'-e "{k}=${k}"' in s, f"head env for {k}"
        assert k in loop, f"worker env for {k}"
    # defaults match start.sh exactly
    two = (ROOT / "start.sh").read_text()
    for k in KNOBS:
        d4 = re.search(rf'^{k}="\$\{{{k}-([^}}]*)\}}"', s, re.M)
        d2 = re.search(rf'^{k}="\$\{{{k}:-([^}}]*)\}}"', two, re.M)
        assert d4 and d2 and d4.group(1) == d2.group(1), (k, d4 and d4.group(1), d2 and d2.group(1))
    assert (ROOT / "overlay" / "patch_adaptive_k.py").is_file()
    dead = re.search(r": <<'TP4_SKIP_OLD_SCP'\n(.*?)\nTP4_SKIP_OLD_SCP", s, re.S)
    assert dead and "patch_adaptive_k.py" not in dead.group(1)


def _generator() -> str:
    s = START.read_text()
    m = re.search(r"capture_sizes=\"\$\(python3 -S -c '(.*?)'", s, re.S)
    assert m, "capture-size generator not found in start-tp4.sh"
    return m.group(1)


def _sizes(mode: str, kset: str, tokens: str, seqs: str) -> list[int]:
    out = subprocess.run([sys.executable, "-S", "-c", _generator(), mode, kset, tokens, seqs],
                         text=True, capture_output=True, check=True).stdout.split()
    return [int(x) for x in out]


def test_capture_sizes_off_is_stock() -> None:
    assert _sizes("off", "2,4,7", "7", "8") == [1, 2, 4, 8, 16, 24, 32]


def test_capture_sizes_ema_covers_every_k_at_every_batch() -> None:
    got = set(_sizes("ema", "2,4,7", "7", "8"))
    for n in range(1, 9):
        for q in (3, 5, 8):
            assert n * q in got, (n, q)
    assert {1, 2, 4, 8, 16, 24, 32} <= got
    assert got == set(_sizes("ON", " 2, 4 ,7 ", "7", "8"))


def test_launcher_syntax() -> None:
    subprocess.run(["bash", "-n", str(START)], check=True)


def test_rank_scripts_syntax() -> None:
    """bash -n on the launcher cannot see inside the quoted heredocs that become
    each rank's /start.sh; check those bodies too (a dropped `fi` there only
    shows up as `/start.sh: syntax error: unexpected end of file` at boot)."""
    import tempfile, os
    s = START.read_text()
    bodies = re.findall(r"<<\s*'([A-Z_]+)'\n(.*?)\n\1\n", s, re.S)
    assert len(bodies) >= 2, "expected head and worker heredocs"
    for tag, body in bodies:
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write(body)
            name = f.name
        try:
            r = subprocess.run(["bash", "-n", name], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
        finally:
            os.unlink(name)



def _launch(caller=None, shared="", topology="", probe="main restart"):
    """Execute the real preamble/guards/dispatch with host functions replaced."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = START.read_text().rsplit('main "$@"', 1)[0]
        source = "python3() { echo PYTHON_CALLED >&2; return 99; }\n" + source
        source += """
validate_loadclone_artifacts() { :; }
banner() { :; }
start() { echo HOST_START; }
stop() { echo HOST_STOP; }
status() { echo HOST_STATUS; }
logs() { echo HOST_LOGS; }
""" + probe + "\n"
        (root / "start-tp4.sh").write_text(source)
        (root / ".env").write_text(shared)
        (root / ".env.tp4").write_text(topology)
        env = {"PATH": os.environ["PATH"], "HOME": directory, "USER": "test"}
        env.update(caller or {})
        return subprocess.run(["bash", str(root / "start-tp4.sh")], env=env,
                              capture_output=True, text=True)


def test_caller_setness_wins_all_knobs():
    dotenv = "".join(f"{k}=file\n" for k in KNOBS)
    probe = 'printf "[%s]\\n" ' + " ".join(f'"${{{k}}}"' for k in KNOBS)
    for value in ("caller", ""):
        r = _launch(dict.fromkeys(KNOBS, value), dotenv, dotenv, probe)
        assert r.returncode == 0, r.stderr
        assert r.stdout.splitlines() == [f"[{value}]"] * len(KNOBS), r.stdout


def test_tp4_opt_in_and_source():
    for topology, caller, expected in (("", {}, "off"),
            ("GLM53_ADAPTIVE_K=on\n", {}, "on"),
            ("GLM53_ADAPTIVE_K=off\n", {"GLM53_ADAPTIVE_K": "ema"}, "ema")):
        r = _launch(caller, "GLM53_ADAPTIVE_K=ema\n", topology,
                    'printf "%s\\n" "$GLM53_ADAPTIVE_K"')
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == expected
    for caller, topology, origin in (({}, "GLM53_ADAPTIVE_K=on\n", ".env.tp4"),
            ({"GLM53_ADAPTIVE_K": "ema"}, "", "caller environment")):
        r = _launch(caller, topology=topology,
                    probe='validate_numeric_config; configure_capture_sizes')
        assert origin in r.stderr, r.stderr
    example = (ROOT / ".env.tp4.example").read_text()
    assert "\nGLM53_ADAPTIVE_K=off\n" in example
    tokens = int(re.search(r"^DFLASH_TOKENS=(\d+)$", example, re.M).group(1))
    kset = re.search(r"^# GLM53_ADAPTIVE_K_SET=([0-9,]+)$", example, re.M).group(1)
    assert max(map(int, kset.split(","))) == tokens
    assert "prose +13-21" not in example


def test_invalid_knobs_before_host_actions():
    invalid = {"": ("bogus", ""), "_ALPHA": ("0,25", "0", "1.1", "nan", ""),
               "_MARGIN": ("-1", "inf", ""), "_MIN_STEPS": ("-1", "1.2", ""),
               "_HIST": ("-1", "x", ""), "_SATURATE": ("bogus", ""),
               "_SET": ("x", "-1,3", "", "2,,3")}
    for suffix, values in invalid.items():
        key = "GLM53_ADAPTIVE_K" + suffix
        for value in values:
            for cmd in ("start", "restart"):
                r = _launch({"GLM53_ADAPTIVE_K": "ema", key: value}, probe=f"main {cmd}")
                assert r.returncode == 2, (key, value, r.returncode, r.stderr)
                assert key in r.stderr, r.stderr
                assert "HOST_" not in r.stdout and "PYTHON_CALLED" not in r.stderr


def test_off_and_management_commands_skip_generator():
    for cmd in ("stop", "status", "logs"):
        r = _launch(dict.fromkeys(KNOBS, "invalid"), probe=f"main {cmd}")
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "HOST_" + cmd.upper()
        assert "PYTHON_CALLED" not in r.stderr
    r = _launch({"SPEC_METHOD": "dflash"}, probe="main restart")
    assert r.returncode == 0, r.stderr
    assert "PYTHON_CALLED" not in r.stderr
    assert r.stdout.splitlines() == ["HOST_STOP", "HOST_START"]


def test_enabled_capture_configuration_and_valid_boundaries():
    for mode in ("ema", " ON ", "\t1\n"):
        r = _launch({"GLM53_ADAPTIVE_K": mode, "GLM53_ADAPTIVE_K_SET": "0,1,2,3",
                     "GLM53_ADAPTIVE_K_ALPHA": "1", "GLM53_ADAPTIVE_K_MARGIN": "0",
                     "GLM53_ADAPTIVE_K_MIN_STEPS": "0", "GLM53_ADAPTIVE_K_HIST": "0",
                     "GLM53_ADAPTIVE_K_SATURATE": " N ", "SPEC_METHOD": "dflash"},
                    topology="DFLASH_TOKENS=3\nMAX_NUM_SEQS=4\n",
                    probe='unset -f python3; validate_numeric_config; configure_capture_sizes; printf "%s" "$EXTRA_ARGS"')
        assert r.returncode == 0, r.stderr
        assert r.stdout == "--cudagraph-capture-sizes " + " ".join(map(str, _sizes(mode, "0,1,2,3", "3", "4")))
    r = _launch({"GLM53_ADAPTIVE_K": "ema", "SPEC_METHOD": "dflash"})
    assert r.returncode == 2 and "HOST_" not in r.stdout, (r.returncode, r.stdout)
    # Unused policy values cannot affect the disabled start path.
    r = _launch({**dict.fromkeys(KNOBS, "invalid"), "GLM53_ADAPTIVE_K": " OFF "})
    assert r.returncode == 0 and "PYTHON_CALLED" not in r.stderr, r.stderr


def test_capture_overrides_and_eager_skip_python():
    for arg in ("--cudagraph-capture-sizes 3 4", "--cudagraph-capture-sizes=3,4",
                "cudagraph-capture-sizes 3 4", "cudagraph-capture-sizes=3,4",
                "--cudagraph-capture-sizes\t3 4"):
        r = _launch({"GLM53_ADAPTIVE_K": "ema", "EXTRA_ARGS": arg},
                    probe='validate_numeric_config; configure_capture_sizes; printf "%s" "$EXTRA_ARGS"')
        assert r.returncode == 0, r.stderr
        assert r.stdout == arg and "PYTHON_CALLED" not in r.stderr
    r = _launch({"GLM53_ADAPTIVE_K": "ema", "ENFORCE_EAGER": "1"})
    assert r.returncode == 0 and "PYTHON_CALLED" not in r.stderr, r.stderr


def test_rank_patch_gate_normalization():
    bodies = re.findall(r"<<'EOF'\n(.*?)\nEOF", START.read_text(), re.S)
    assert len(bodies) == 2
    for body in bodies:
        # Run the actual gate, replacing only its filesystem probe and patch command.
        gate = re.search(r'if \[\[.*?\nfi', body[body.index('# Adaptive verification'):], re.S)
        assert gate, "missing adaptive patch gate"
        script = gate.group().replace('[ -f /opt/glm53/patch_adaptive_k.py ]', 'true')
        script = script.replace('python3 /opt/glm53/patch_adaptive_k.py', 'echo PATCH')
        for mode in ("off", "", "ema", " ON ", "\t1\n", "bogus"):
            r = subprocess.run(["bash", "-c", script], env={"GLM53_ADAPTIVE_K": mode},
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            assert (r.stdout.strip() == "PATCH") == (mode.strip().lower() in ("ema", "on", "1"))


def test_generators_agree_with_runtime_query_lengths():
    two = re.search(r"capture_sizes=\"\$\(python3 -S -c '(.*?)'",
                    (ROOT / "start.sh").read_text(), re.S).group(1)
    tree = ast.parse((ROOT / "overlay/patch_adaptive_k.py").read_text())
    helper = next(ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CG_HELPER" for t in n.targets))
    namespace = {}
    exec(helper, namespace)
    for mode in ("off", "ema", " ON ", "1"):
        for kset in ("2,4,7", "1,2,3", " 2, 2, 9 ", "0,1,3"):
            for tokens in ("3", "7"):
                for seqs in ("1", "4", "8"):
                    args = [mode, kset, tokens, seqs]
                    expected = subprocess.check_output([sys.executable, "-S", "-c", two, *args], text=True)
                    got = _sizes(*args)
                    assert got == list(map(int, expected.split())), args
                    with patch.dict(os.environ, {"GLM53_ADAPTIVE_K": mode, "GLM53_ADAPTIVE_K_SET": kset}):
                        import contextlib, io
                        with contextlib.redirect_stdout(io.StringIO()):
                            lens = namespace["_glm53_adaptive_k_query_lens"]([int(tokens) + 1], int(tokens) + 1)
                    sizes = {1, 2, 4, 8, 16, 24, 32}
                    if mode.strip().lower() in ("ema", "on", "1"):
                        sizes.update(n * q for n in range(1, int(seqs) + 1) for q in lens)
                    assert got == sorted(sizes), (args, lens, got)

if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1; print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
