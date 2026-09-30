#!/usr/bin/env python3
"""Regression tests for caller overrides that must win over ``.env``."""

from __future__ import annotations

import re
import shlex
import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

# start.sh clears these right after sourcing .env, so a `.env` value does NOT
# reach the launcher when the caller stays silent. Only ABLIT is cleared today:
# the abliterated preset ships pre-edited o_proj weights, so a stale
# `.env` ABLIT=1 must not silently edit them again.
DOTENV_CLEARED_KEYS = {"ABLIT"}


def test_max_num_seqs_inline_override_wins() -> None:
    source = (ROOT / "start.sh").read_text()
    marker = "# ----------------------------- configuration -------------------------------"
    preamble, separator, _rest = source.partition(marker)
    assert separator, "start.sh configuration marker is missing"

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(
            preamble
            + '\nprintf "MAX_NUM_SEQS=%s\\n" "${MAX_NUM_SEQS:-unset}"\n'
        )
        script.chmod(0o755)
        (tmp / ".env").write_text("MAX_NUM_SEQS=2\n")

        # isolated: no BASH_ENV / PATH surprises from the developer shell
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp), "USER": "glm53", "MAX_NUM_SEQS": "4"}
        result = subprocess.run(
            ["bash", str(script)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

    assert result.stdout.strip() == "MAX_NUM_SEQS=4"


def _run_preamble_proc(
    env_file: str, caller: dict[str, str], probe: str
) -> subprocess.CompletedProcess[str]:
    """Run start.sh's pre-configuration preamble with a synthetic .env."""
    source = (ROOT / "start.sh").read_text()
    marker = "# ----------------------------- configuration -------------------------------"
    preamble, separator, _rest = source.partition(marker)
    assert separator, "start.sh configuration marker is missing"

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(preamble + probe)
        script.chmod(0o755)
        (tmp / ".env").write_text(env_file)

        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp), "USER": "glm53"}
        env.update(caller)
        return subprocess.run(
            ["bash", str(script)], check=True, capture_output=True, text=True, env=env
        )


def _run_preamble(env_file: str, caller: dict[str, str], probe: str) -> str:
    """Stdout of the preamble run, stripped."""
    return _run_preamble_proc(env_file, caller, probe).stdout.strip()


def test_default_reasoning_effort_caller_override_is_setness_aware() -> None:
    """An explicitly EMPTY caller value must beat .env, not be swallowed by it.

    The knob's own default is empty, so ``[ -n "$_cli_x" ]`` cannot tell
    ``GLM53_DEFAULT_REASONING_EFFORT= ./start.sh`` (deliberately back to the
    template default) apart from an unset var. Only ``${VAR+1}`` can.
    """
    probe = '\nprintf "EFFORT=[%s]\\n" "${GLM53_DEFAULT_REASONING_EFFORT-unset}"\n'
    env_file = "GLM53_DEFAULT_REASONING_EFFORT=high\n"

    # caller unset -> .env wins
    assert _run_preamble(env_file, {}, probe) == "EFFORT=[high]"

    # caller sets a value -> caller wins
    assert _run_preamble(
        env_file, {"GLM53_DEFAULT_REASONING_EFFORT": "low"}, probe
    ) == "EFFORT=[low]"

    # caller sets it EMPTY -> caller still wins (the setness-aware case)
    assert _run_preamble(
        env_file, {"GLM53_DEFAULT_REASONING_EFFORT": ""}, probe
    ) == "EFFORT=[]"


@pytest.mark.parametrize("launcher,topology_env", [("start-tp3.sh", ".env.tp3"),
                                                    ("start-tp4.sh", ".env.tp4")])
def test_default_reasoning_effort_precedence_on_tp3_tp4(launcher: str, topology_env: str) -> None:
    """caller export > .env.tpX > shared .env, setness-aware, as on start.sh.

    A .env copied from .env.example always carries the knob (empty), so a
    launcher that does not capture the caller loses its export silently."""
    source = (ROOT / launcher).read_text()
    marker = "# ----------------------------- configuration -------------------------------"
    preamble, separator, _rest = source.partition(marker)
    assert separator, f"{launcher} configuration marker is missing"
    probe = '\nprintf "EFFORT=[%s]\\n" "${GLM53_DEFAULT_REASONING_EFFORT-unset}"\n'

    def run(shared: str, topology: str, caller: dict[str, str]) -> str:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            script = tmp / launcher
            script.write_text(preamble + probe)
            (tmp / ".env").write_text(shared)
            (tmp / topology_env).write_text(topology)
            env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp), "USER": "glm53", **caller}
            return subprocess.run(["bash", str(script)], check=True, capture_output=True,
                                  text=True, env=env).stdout.strip().splitlines()[-1]

    example_line = "GLM53_DEFAULT_REASONING_EFFORT=\n"  # as shipped in .env.example
    assert run(example_line, "", {}) == "EFFORT=[]"
    assert run("GLM53_DEFAULT_REASONING_EFFORT=high\n", "", {}) == "EFFORT=[high]"
    assert run("GLM53_DEFAULT_REASONING_EFFORT=high\n",
               "GLM53_DEFAULT_REASONING_EFFORT=max\n", {}) == "EFFORT=[max]"
    effort = "GLM53_DEFAULT_REASONING_EFFORT"
    assert run(example_line, "", {effort: "low"}) == "EFFORT=[low]"
    assert run("GLM53_DEFAULT_REASONING_EFFORT=high\n",
               "GLM53_DEFAULT_REASONING_EFFORT=max\n", {effort: "low"}) == "EFFORT=[low]"
    assert run("GLM53_DEFAULT_REASONING_EFFORT=high\n", "", {effort: ""}) == "EFFORT=[]"


def test_indexer_workspace_caller_capture_is_setness_aware() -> None:
    """An explicitly EMPTY caller value must not be swallowed by ``.env``.

    ``GLM53_INDEXER_WORKSPACE=`` is an operator error; the enum guard has to see
    it. A ``[ -n "$_cli_..." ]`` restore would silently hand back the ``.env``
    value instead, so the capture uses the ``${VAR+1}`` setness probe.
    """
    probe = '\nprintf "V=[%s]\\n" "${GLM53_INDEXER_WORKSPACE-UNSET}"\n'
    env_file = "GLM53_INDEXER_WORKSPACE=rightsize\n"

    # Caller silent: .env wins.
    assert _run_preamble(env_file, {}, probe) == "V=[rightsize]"
    # Caller sets a real value: caller wins (the pre-existing contract).
    assert _run_preamble(
        env_file, {"GLM53_INDEXER_WORKSPACE": "stock"}, probe
    ) == "V=[stock]"
    # Caller sets it EMPTY: the empty value survives to the guard.
    assert _run_preamble(
        env_file, {"GLM53_INDEXER_WORKSPACE": ""}, probe
    ) == "V=[]"
    # ... and with no .env value either.
    assert _run_preamble("", {"GLM53_INDEXER_WORKSPACE": ""}, probe) == "V=[]"
    # Unset on both sides stays unset until the configuration default.
    assert _run_preamble("", {}, probe) == "V=[UNSET]"


def test_spinwait_caller_capture_is_setness_aware() -> None:
    probe = '\nprintf "V=[%s]\\n" "${GLM53_SPINWAIT_MS-UNSET}"\n'
    env_file = "GLM53_SPINWAIT_MS=16\n"

    assert _run_preamble(env_file, {}, probe) == "V=[16]"
    assert _run_preamble(
        env_file, {"GLM53_SPINWAIT_MS": "stock"}, probe
    ) == "V=[stock]"
    assert _run_preamble(
        env_file, {"GLM53_SPINWAIT_MS": ""}, probe
    ) == "V=[]"
    assert _run_preamble("", {"GLM53_SPINWAIT_MS": ""}, probe) == "V=[]"
    assert _run_preamble("", {}, probe) == "V=[UNSET]"


def test_every_env_example_key_preserves_caller_setness() -> None:
    keys = re.findall(
        r"^(?:# )?([A-Za-z_][A-Za-z0-9_]*)=", (ROOT / ".env.example").read_text(), re.M
    )
    # start.sh deliberately clears a `.env` ABLIT after sourcing it; only a
    # caller export opts back in (see the test below).
    keys = [key for key in keys if key not in DOTENV_CLEARED_KEYS]
    keys.append("FUTURE_LAUNCHER_KNOB")
    dotenv = "".join(f"{key}=dotenv\n" for key in keys)
    child_probe = 'printf "[%s]\\n" ' + " ".join(
        f'"${{{key}-UNSET}}"' for key in keys
    )
    probe = '\n"$BASH" -c ' + shlex.quote(child_probe) + "\n"
    for value in ("caller", ""):
        caller = dict.fromkeys(keys, value)
        assert _run_preamble(dotenv, caller, probe) == "\n".join(
            f"[{value}]" for key in keys
        )
    assert _run_preamble(dotenv, {}, probe) == "\n".join(
        "[dotenv]" for key in keys
    )


def test_shell_assignments_preserve_caller_values() -> None:
    dotenv = (
        "export FUTURE_A=dotenv; FUTURE_B=dotenv\n"
        "declare -x FUTURE_C=dotenv\n"
        "unset FUTURE_D\n"
        "FUTURE_E+=suffix\n"
        "SCRIPT_DIR=/dotenv-internal\n"
    )
    caller = {
        "FUTURE_A": 'quoted \"value\" = with spaces',
        "FUTURE_B": "first line\nsecond line",
        "FUTURE_C": "",
        "FUTURE_D": "caller",
        "FUTURE_E": "prefix",
        "SHELLOPTS": "braceexpand:hashall",
    }
    probe = (
        '\nprintf "[%s]\\n" "$FUTURE_A" "$FUTURE_B" "$FUTURE_C" '
        '"$FUTURE_D" "$FUTURE_E" "$SCRIPT_DIR"\n'
    )
    assert _run_preamble(dotenv, caller, probe) == (
        '[quoted "value" = with spaces]\n[first line\nsecond line]\n'
        "[]\n[caller]\n[prefix]\n[/dotenv-internal]"
    )


def test_ablit_env_value_is_cleared_unless_the_caller_exported_it() -> None:
    """A `.env` ABLIT never opts in; an exported one always wins.

    start.sh forces ``ABLIT=0`` immediately after sourcing ``.env`` and then
    restores the caller's exports, which is exactly what makes the documented
    ``ABLIT=1 ./start.sh`` work while a stale ``.env`` ABLIT=1 does not.
    """
    probe = '\nprintf "ABLIT=[%s]\\n" "${ABLIT-unset}"\n'

    # Caller silent: the .env opt-in is cleared.
    assert _run_preamble("ABLIT=1\n", {}, probe) == "ABLIT=[0]"
    # Documented caller opt-in survives, whatever .env says.
    assert _run_preamble("ABLIT=1\n", {"ABLIT": "1"}, probe) == "ABLIT=[1]"
    assert _run_preamble("ABLIT=0\n", {"ABLIT": "1"}, probe) == "ABLIT=[1]"
    # A caller 0 turns it off even when .env opted in.
    assert _run_preamble("ABLIT=1\n", {"ABLIT": "0"}, probe) == "ABLIT=[0]"


def _run_preamble_stderr(env_file: str, caller: dict[str, str]) -> str:
    """Stderr of the preamble run with no probe appended."""
    return _run_preamble_proc(env_file, caller, "\n").stderr


def test_ambient_override_of_model_affecting_key_is_announced() -> None:
    """#168: an inherited env value that displaces .env for a model-affecting key is
    named on stderr, with both values. Silent when nothing diverges."""
    dotenv = "HF_HOME=/from/dotenv\nMODEL=from/dotenv\nMAX_NUM_SEQS=2\n"
    err = _run_preamble_stderr(dotenv, {"HF_HOME": "/from/ambient"})
    assert "NOTE: HF_HOME=/from/ambient from the environment overrides .env value /from/dotenv" in err
    assert err.count("NOTE:") == 1
    # ambient value equal to .env: nothing to report
    assert "NOTE:" not in _run_preamble_stderr(dotenv, {"HF_HOME": "/from/dotenv"})
    # clean environment: nothing to report
    assert "NOTE:" not in _run_preamble_stderr(dotenv, {})
    # an unwatched key still wins (PR #161) and is not announced
    assert "NOTE:" not in _run_preamble_stderr(dotenv, {"MAX_NUM_SEQS": "4"})


if __name__ == "__main__":
    test_ambient_override_of_model_affecting_key_is_announced()
    test_ablit_env_value_is_cleared_unless_the_caller_exported_it()
    test_every_env_example_key_preserves_caller_setness()
    test_shell_assignments_preserve_caller_values()
    test_max_num_seqs_inline_override_wins()
    test_default_reasoning_effort_caller_override_is_setness_aware()
    test_indexer_workspace_caller_capture_is_setness_aware()
    test_spinwait_caller_capture_is_setness_aware()
    print("start.sh caller override regression OK")
