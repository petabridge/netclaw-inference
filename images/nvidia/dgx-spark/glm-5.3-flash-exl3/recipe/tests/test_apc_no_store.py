#!/usr/bin/env python3
"""Host test for overlay/patch_apc_no_store.py (no GPU).

Part A  patch mechanics on COPIES of the three target files: the result
        parses, byte-identical idempotent re-apply, every anchor individually
        drifted -> non-zero exit AND no file written (transactional), and
        partially-marked or altered files are refused.
Part B  resolver semantics: the injected helper block is exec'd in a bare
        namespace with a capturing logger and driven over the full
        accept/reject matrix, the typed > extra_args precedence, and the
        GLM53_APC_NO_STORE kill-switch matrix (parse/reject happens BEFORE the
        switch).
Part C  behaviour on a real vLLM (CPU is enough): patched COPIES of the three
        files are injected into ``sys.modules`` before ``vllm.v1.core`` is
        imported, so real ``BlockPool`` / ``KVCacheManager`` /
        ``HybridKVCacheCoordinator`` / managers run on top of the patched code.
        Skipped with a loud line when ``import vllm`` fails; set
        ``GLM53_REQUIRE_VLLM=1`` to make that a failure (the Dockerfile does).
        The in-image gate runs this file AFTER patch_hybrid_prefix_hit.py and
        patch_apc_per_group_retention.py, so Part C also composes the real
        hybrid + per-group retention + no-store stack. The live-shape legs run
        once per retention mode (``VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA``
        unset = inherit the global policy, ``"0"`` = boundary-only retention
        for the EAGLE-exempt drafter group) and prove that hash suppression,
        per-group num_cached_block progression, follow-up hits and recycling
        behaviour remain correct on that stack. The fixture pins the global
        retention to dense (``None``, the
        pinned runtime's default) so this image-level gate does not depend on
        the deployment's launcher environment. Group layouts are asserted and
        printed: a runtime that exposes the fork's ``KpoolTailSpec`` must
        construct it, otherwise the seven-group composition fails instead of
        silently reporting a six-group upstream shape under the fork's name.
        Set ``GLM53_REQUIRE_COMPOSITION=1`` (the Dockerfile does) to make a
        missing per-group retention overlay a failure rather than a skip.
Part D  launcher (host-only: the image copy carries no start.sh): the
        "GLM53 numeric config guard" block accepts GLM53_APC_NO_STORE only as
        exactly 0 or 1 (unset -> 1), the knob is forwarded to both ranks, and
        the generic caller-override block (caller exports beat .env, explicit
        empties included) decides what the guard then validates.

Sources: ``GLM53_VLLM_SRC_ROOT`` = a vLLM package directory holding
``sampling_params.py``, ``v1/request.py``, ``v1/core/block_pool.py`` (default:
the in-image path). Files that already carry the overlay MARK are accepted
(Part A's apply/drift legs then run on the vendored anchor-context replicas
below; Part C injects them as they are). With no source at all, Part A runs on
the replicas only and Part C is skipped.

Run:  python3 tests/test_apc_no_store.py
      GLM53_VLLM_SRC_ROOT=/path/to/vllm python3 tests/test_apc_no_store.py
      (Part C on the Mac: PYTHONPATH=<upstream clone> <cpu venv python> ...)
      In-image composition gate (hybrid + per-group retention already applied):
      GLM53_VLLM_SRC_ROOT=<pinned fork package> GLM53_REQUIRE_VLLM=1 \
          GLM53_REQUIRE_COMPOSITION=1 python3 tests/test_apc_no_store.py
"""

from __future__ import annotations

import ast
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = next(
    (
        p
        for p in (HERE / "patch_apc_no_store.py", HERE.parent / "overlay" / "patch_apc_no_store.py")
        if p.is_file()
    ),
    None,
)
START = HERE.parent / "start.sh"
# start.sh's generic caller-override block: exports win over .env, explicit
# empties included (#161/#92). Lifted by its own sentinels.
CALLER_BEGIN = "_caller_overrides=()"
CALLER_END = "unset _k _kv _flags _caller_overrides"
DEFAULT_SRC_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARK = "# [glm53-apc-no-store]"
REL = {
    "GLM53_SAMPLING_PARAMS_PY": "sampling_params.py",
    "GLM53_REQUEST_PY": "v1/request.py",
    "GLM53_BLOCK_POOL_PY": "v1/core/block_pool.py",
}
FAILURES: list[str] = []
CHECKS = 0


def check(cond: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILURES.append(label)


def load_patcher():
    spec = importlib.util.spec_from_file_location("glm53_patch_apc_no_store", PATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- replicas -----
# Vendored anchor context: the exact lines the overlay anchors on, embedded in
# enough surrounding code to parse. Used when no pristine source is available
# (or the available one is already patched) so Parts A/B run anywhere.

REPLICA_SAMPLING = '''# replica of vllm/sampling_params.py (anchor context only)
from vllm.logger import init_logger

logger = init_logger(__name__)


class SamplingParams(
    object,
):
    """replica"""

    n: int = 1
    extra_args: dict | None = None
    prompt_logprobs: int | None = None
    skip_reading_prefix_cache: bool | None = None
    thinking_token_budget: int | None = None

    def __post_init__(self) -> None:
        self._verify_args()

        if self.skip_reading_prefix_cache is None:
            # If prefix caching is enabled,
            # the output of prompt logprobs may less than n_prompt_tokens,
            # we need to skip reading cache at this request.
            self.skip_reading_prefix_cache = self.prompt_logprobs is not None

    def _verify_args(self) -> None:
        pass
'''

REPLICA_REQUEST = '''# replica of vllm/v1/request.py (anchor context only)
from vllm.sampling_params import SamplingParams


class Request:
    def __init__(self, request_id, sampling_params=None, pooling_params=None):
        self.request_id = request_id
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.status = None

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return False
'''

REPLICA_BLOCK_POOL = '''# replica of vllm/v1/core/block_pool.py (anchor context only)
from vllm.logger import init_logger

logger = init_logger(__name__)


class BlockPool:
    def __init__(self, hash_block_size):
        self.hash_block_size = hash_block_size
        self.enable_kv_cache_events = False

    def cache_full_blocks(
        self,
        request,
        blocks,
        num_cached_blocks,
        num_full_blocks,
        block_size,
        kv_cache_group_id,
        block_mask=None,
    ) -> None:
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        for i, blk in enumerate(new_full_blocks):
            self._insert_block_hash(
                (kv_cache_group_id, i),
                blk,
                num_tokens=(num_cached_blocks + i + 1) * block_size,
            )

    def cache_partial_block(
        self,
        request,
        block,
        num_tokens,
        kv_cache_group_id,
        block_size,
    ):
        if block.is_null:
            return None

        assert block_size > self.hash_block_size
        block_hash_with_group_id = (kv_cache_group_id, num_tokens)
        self._insert_block_hash(
            block_hash_with_group_id,
            block,
            num_tokens=num_tokens,
        )
        return block_hash_with_group_id

    def _insert_block_hash(self, block_hash_with_group_id, block, num_tokens):
        block.block_hash = block_hash_with_group_id

    def move_block_hashes(self, src_block, dst_block) -> None:
        assert dst_block.block_hash is None
        for block_hash in [src_block.block_hash]:
            self._insert_block_hash(block_hash, dst_block, num_tokens=None)
'''

REPLICAS = {
    "GLM53_SAMPLING_PARAMS_PY": REPLICA_SAMPLING,
    "GLM53_REQUEST_PY": REPLICA_REQUEST,
    "GLM53_BLOCK_POOL_PY": REPLICA_BLOCK_POOL,
}


def source_root() -> Path | None:
    raw = os.environ.get("GLM53_VLLM_SRC_ROOT", "").strip()
    root = Path(raw) if raw else DEFAULT_SRC_ROOT
    if all((root / rel).is_file() for rel in REL.values()):
        return root
    if raw:
        raise SystemExit(f"GLM53_VLLM_SRC_ROOT={root} lacks one of {sorted(REL.values())}")
    return None


def stage(into: Path, root: Path | None, pristine_only: bool) -> dict[str, Path]:
    """Copy the three targets (real sources, or replicas) into ``into``."""
    staged = {}
    for env, rel in REL.items():
        dst = into / Path(rel).name
        src = (root / rel) if root else None
        if src is not None and (not pristine_only or MARK not in src.read_text()):
            shutil.copyfile(src, dst)
        else:
            dst.write_text(REPLICAS[env])
        staged[env] = dst
    return staged


def run_patcher(staged: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GLM53_")}
    env.update({k: str(v) for k, v in staged.items()})
    return subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True, text=True)




# Other overlays a staged source may already carry: the in-image gate runs this
# file after patch_hybrid_prefix_hit.py and patch_apc_per_group_retention.py.
# Reported (not asserted) so a Part A pass is not misread as pristine-upstream
# application.
OVERLAY_MARKS = {
    "# [glm53-hybrid-apc]": "hybrid-apc",
    "# [glm53-apc-per-group]": "apc-per-group",
    "# [glm53-apc-drafter-priority]": "drafter-priority",
}


def overlay_provenance(root: Path | None) -> str:
    if root is None:
        return "vendored replicas"
    parts = []
    for rel in REL.values():
        text = (root / rel).read_text()
        marks = [name for mark, name in OVERLAY_MARKS.items() if mark in text]
        parts.append(f"{Path(rel).name}={'+'.join(marks) or 'clean'}")
    return f"{root} [{', '.join(parts)}]"


# ---------------------------------------------------------------- part A ----


def part_a(root: Path | None) -> None:
    print("Part A: patch mechanics")
    patcher = load_patcher()
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        staged = stage(tmp, root, pristine_only=True)
        using_real = root is not None and all(MARK not in (root / r).read_text() for r in REL.values())
        print(f"  sources: {'pre-no-store ' + str(root) if using_real else 'vendored replicas'}")
        if using_real:
            print(f"  staged-source overlays: {overlay_provenance(root)}")

        r = run_patcher(staged)
        check(r.returncode == 0, f"A1 first application exits 0 ({r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[:120]})")
        texts = {k: p.read_text() for k, p in staged.items()}
        for k, p in staged.items():
            try:
                ast.parse(texts[k])
                ok = True
            except SyntaxError as exc:
                ok = False
                print(f"       {exc}")
            check(ok, f"A1 {p.name} parses")


        r2 = run_patcher(staged)
        check(r2.returncode == 0 and all(p.read_text() == texts[k] for k, p in staged.items()), "A3 re-apply exits 0 and is byte-identical (idempotent)")

    # A4 drift: every anchor, one at a time, and no file may be written.
    for fname, (_, edits, _requires) in patcher.PLAN.items():
        env_key = next(k for k, rel in REL.items() if Path(rel).name == fname)
        for label, old, _new in edits:
            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                staged = stage(tmp, root, pristine_only=True)
                victim = staged[env_key]
                text = victim.read_text()
                # duplicate the anchor -> count 2 -> replace_once must refuse
                text = text.replace(old, old + old, 1)
                victim.write_text(text)
                before = {k: p.read_text() for k, p in staged.items()}
                r = run_patcher(staged)
                untouched = all(p.read_text() == before[k] for k, p in staged.items())
                check(r.returncode != 0 and label in (r.stderr + r.stdout) and untouched, f"A4 drifted {fname}:{label} -> exit {r.returncode}, named in the error, nothing written")
    # A5 partial marker: a file with SOME of its marks is refused, nothing written.
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        staged = stage(tmp, root, pristine_only=True)
        run_patcher(staged)
        bp = staged["GLM53_BLOCK_POOL_PY"]
        text = bp.read_text()
        text = text.replace(BLOCK_POOL_PARTIAL_LINE, "        if getattr(request, \"skip_writing_prefix_cache\", False):\n", 1)
        bp.write_text(text)
        # and make the other two pristine again so they WOULD be written
        pristine = stage(tmp, root, pristine_only=True)
        bp.write_text(text)
        before = {k: p.read_text() for k, p in pristine.items()}
        r = run_patcher(pristine)
        check(r.returncode != 0 and all(p.read_text() == before[k] for k, p in pristine.items()), f"A5 partially-marked block_pool.py refused (rc={r.returncode}) and the other two files stay untouched")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        staged = stage(tmp, root, pristine_only=True)
        run_patcher(staged)
        bp = staged["GLM53_BLOCK_POOL_PY"]
        text = bp.read_text()
        # every MARK still present, but one guard body edited -> not "complete"
        edited = text.replace('_glm53_log_nostore(request, "full")', 'pass  # edited', 1)
        check(edited != text and edited.count(MARK) == text.count(MARK), "A5 fixture: marks intact, one snippet altered")
        bp.write_text(edited)
        r = run_patcher(staged)
        check(r.returncode != 0 and bp.read_text() == edited, f"A5 fully-marked file with an altered snippet is refused (rc={r.returncode}), not skipped as applied")
        bp.write_text(text[: len(text) // 2])
        r = run_patcher(staged)
        check(r.returncode != 0 and bp.read_text() == text[: len(text) // 2], f"A5 truncated (already-marked) file is refused (rc={r.returncode})")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        staged = stage(tmp, root, pristine_only=True)
        missing = staged["GLM53_REQUEST_PY"]
        missing.unlink()
        before = {k: p.read_bytes() for k, p in staged.items() if p.exists()}
        r = run_patcher(staged)
        check(r.returncode != 0 and all(staged[k].read_bytes() == data for k, data in before.items()), "A6 missing target -> refused, nothing written")
        check(not [p for p in tmp.iterdir() if p.suffix == ".glm53"], "A6 no temp-file litter left behind")


BLOCK_POOL_PARTIAL_LINE = '        if getattr(request, "skip_writing_prefix_cache", False):  # [glm53-apc-no-store]\n'


# ---------------------------------------------------------------- part B ----


class CapturingLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def _rec(self, level, msg, *args):
        self.lines.append(f"{level}: {msg % args if args else msg}")

    def info_once(self, msg, *args, **kw):
        self._rec("INFO", msg, *args)

    def warning_once(self, msg, *args, **kw):
        self._rec("WARN", msg, *args)

    def warning(self, msg, *args, **kw):
        self._rec("WARN", msg, *args)

    def info(self, msg, *args, **kw):
        self._rec("INFO", msg, *args)

    def debug(self, msg, *args, **kw):
        self._rec("DEBUG", msg, *args)


def helper_namespace(env_value: str | None) -> tuple[dict, CapturingLogger]:
    """exec the injected helper block from a freshly patched replica."""
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        staged = stage(tmp, None, pristine_only=True)
        r = run_patcher(staged)
        assert r.returncode == 0, r.stderr
        text = staged["GLM53_SAMPLING_PARAMS_PY"].read_text()
    begin = text.index("# [glm53-apc-no-store] helper-begin")
    end = text.index("# [glm53-apc-no-store] helper-end")
    block = text[begin:end]
    log = CapturingLogger()
    ns = {"logger": log}
    saved = os.environ.pop("GLM53_APC_NO_STORE", None)
    try:
        if env_value is not None:
            os.environ["GLM53_APC_NO_STORE"] = env_value
        exec(compile(block, "<glm53-no-store-helpers>", "exec"), ns)  # noqa: S102
    finally:
        os.environ.pop("GLM53_APC_NO_STORE", None)
        if saved is not None:
            os.environ["GLM53_APC_NO_STORE"] = saved
    return ns, log


class Params:
    def __init__(self, typed=None, extra=None):
        self.skip_writing_prefix_cache = typed
        self.extra_args = extra


def raises_value_error(fn, *a):
    try:
        fn(*a)
    except ValueError:
        return True
    return False


def part_b() -> None:
    print("Part B: resolver semantics (helper block exec'd in a bare namespace)")
    ns, log = helper_namespace(None)
    parse = ns["_glm53_parse_no_store"]
    accepted = {True: True, False: False, 1: True, 0: False, "1": True, "0": False}
    for value, want in accepted.items():
        check(parse(value, "t") is want, f"B1 accept {value!r} -> {want}")
    rejected = [2, -1, 1.0, 0.0, "true", "false", "yes", "no", " 1", "1 ", "01", "", None, [1], {"a": 1}, b"1", "True"]
    for value in rejected:
        check(raises_value_error(parse, value, "t"), f"B1 reject {value!r}")

    try:
        from pydantic import TypeAdapter
        xargs_type = dict[str, str | int | float | list[str | int | float]] | None
        ta = TypeAdapter(xargs_type)
        coerced = {j: ta.validate_json(j)["skip_writing_prefix_cache"] for j in ('{"skip_writing_prefix_cache": true}', '{"skip_writing_prefix_cache": false}', '{"skip_writing_prefix_cache": "1"}', '{"skip_writing_prefix_cache": 1.0}')}
        check(coerced['{"skip_writing_prefix_cache": true}'] == 1 and coerced['{"skip_writing_prefix_cache": false}'] == 0 and all(not isinstance(v, bool) for v in coerced.values()), f"B1 pydantic coerces a JSON boolean in the vllm_xargs type to int 1/0 (never a bool) -> {coerced}")
        check(parse(coerced['{"skip_writing_prefix_cache": true}'], "t") is True and parse(coerced['{"skip_writing_prefix_cache": false}'], "t") is False, "B1 ... which the strict parser accepts with the intended meaning")
        check(raises_value_error(parse, coerced['{"skip_writing_prefix_cache": 1.0}'], "t"), "B1 ... while JSON 1.0 stays a float and is rejected")
    except ImportError:
        print("  --   B1 pydantic not importable here: vllm_xargs coercion leg skipped (runs in the image / venv)")

    validate = ns["_glm53_validate_no_store_params"]
    resolve = ns["_glm53_resolve_no_store"]
    p = Params(typed="1")
    validate(p)
    check(p.skip_writing_prefix_cache is True, "B2 typed field normalised to bool by validation")
    check(raises_value_error(validate, Params(typed="yes")), "B2 typed 'yes' rejected at the API boundary")
    check(raises_value_error(validate, Params(extra={"skip_writing_prefix_cache": 1.0})), "B2 extra_args 1.0 rejected at the API boundary")

    check(resolve(None, "r") is False, "B3 no sampling params -> False")
    check(resolve(Params(), "r") is False, "B3 unset -> False")
    check(resolve(Params(extra={"skip_writing_prefix_cache": 1}), "r") is True, "B3 extra_args 1 -> True")
    check(resolve(Params(extra={"skip_writing_prefix_cache": "0"}), "r") is False, "B3 extra_args '0' -> False")
    check(resolve(Params(typed=True), "r") is True, "B3 typed True -> True")
    check(resolve(Params(typed=False, extra={"skip_writing_prefix_cache": 1}), "r") is False, "B3 typed False wins over extra_args 1 (precedence typed > extra_args)")
    check(resolve(Params(typed=True, extra={"skip_writing_prefix_cache": 0}), "r") is True, "B3 typed True wins over extra_args 0")
    n_before = len(log.lines)
    check(resolve(Params(extra={"skip_writing_prefix_cache": "yes"}), "r") is False, "B3 unparseable value in the engine -> False, never raises")
    check(any(ln.startswith("WARN:") for ln in log.lines[n_before:]), "B3 ... and it is logged as a warning")
    check(any("INFO" in ln and "first request resolved skip_writing_prefix_cache=1" in ln for ln in log.lines), "B3 resolution receipt logged")

    # Kill switch matrix. Rule: parse/reject BEFORE the switch; the switch only
    # decides whether a valid 1 is honoured.
    for env_value, enabled in ((None, True), ("1", True), ("0", False)):
        ns2, log2 = helper_namespace(env_value)
        got = ns2["_glm53_resolve_no_store"](Params(extra={"skip_writing_prefix_cache": 1}), "r")
        check(got is enabled, f"B4 valid 1 under GLM53_APC_NO_STORE={env_value!r} -> {enabled}")
        check(raises_value_error(ns2["_glm53_validate_no_store_params"], Params(extra={"skip_writing_prefix_cache": "yes"})), f"B4 malformed value still rejected under GLM53_APC_NO_STORE={env_value!r}")
        if not enabled:
            check(any("ignoring skip_writing_prefix_cache=1" in ln for ln in log2.lines), "B4 kill switch logs the ignore once")
    for bad in ("", " 1", "yes", "2", "true", "01"):
        try:
            helper_namespace(bad)
            ok = False
        except ValueError:
            ok = True
        check(ok, f"B4 GLM53_APC_NO_STORE={bad!r} fails closed at import")


# ---------------------------------------------------------------- part C ----

PART_C = r'''
import logging, os, sys, importlib.util, shutil, subprocess
from pathlib import Path

ROOT = Path(sys.argv[1]); PATCH = Path(sys.argv[2]); TMP = Path(sys.argv[3])
REL = {"GLM53_SAMPLING_PARAMS_PY": "sampling_params.py", "GLM53_REQUEST_PY": "v1/request.py", "GLM53_BLOCK_POOL_PY": "v1/core/block_pool.py"}
staged = {}
for env, rel in REL.items():
    dst = TMP / Path(rel).name
    shutil.copyfile(ROOT / rel, dst)
    staged[env] = dst
env = {k: v for k, v in os.environ.items() if not k.startswith("GLM53_")}
env.update({k: str(v) for k, v in staged.items()})
r = subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True, text=True)
assert r.returncode == 0, r.stderr

FAIL = []
N = 0
def check(cond, label):
    global N
    N += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAIL.append(label)

import vllm  # noqa: F401  (package init only)

def inject(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod

sp_mod = inject("vllm.sampling_params", staged["GLM53_SAMPLING_PARAMS_PY"])
import vllm.v1  # noqa: F401
req_mod = inject("vllm.v1.request", staged["GLM53_REQUEST_PY"])
import vllm.v1.core  # noqa: F401
bp_mod = inject("vllm.v1.core.block_pool", staged["GLM53_BLOCK_POOL_PY"])

import torch
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash, get_request_block_hasher, get_block_hash
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core import kv_cache_coordinator as kvc_mod
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, FullAttentionSpec, MambaSpec, SlidingWindowSpec, MLAAttentionSpec
try:
    from vllm.v1.kv_cache_interface import KpoolTailSpec  # fork-only
except Exception:
    KpoolTailSpec = None
SamplingParams = sp_mod.SamplingParams
Request = req_mod.Request
BlockPool = bp_mod.BlockPool
check(kvc_mod.BlockPool is BlockPool, "C0 the coordinator (and so KVCacheManager) uses the patched BlockPool")
init_none_hash(sha256)

# Provenance + deterministic fixture config. This file runs in-image AFTER
# patch_hybrid_prefix_hit.py and patch_apc_per_group_retention.py; record what
# was imported, and pin the global retention to dense -- the pinned runtime's
# default and the .env.example guidance ("leave GLM53_APC_RETENTION_INTERVAL
# unset") -- so the gate does not silently inherit a launcher's deployment
# value. The env-driven SWA override is still exercised per mode in C3L.
import vllm.envs as vllm_envs
previous_global_retention = getattr(vllm_envs, "VLLM_PREFIX_CACHE_RETENTION_INTERVAL", None)
vllm_envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL = None
coordinator_path = str(getattr(kvc_mod, "__file__", ""))
retention_mark = "unreadable"
if coordinator_path.endswith(".py"):
    try:
        retention_mark = "# [glm53-apc-per-group]" in Path(coordinator_path).read_text()
    except OSError as exc:
        retention_mark = f"unreadable ({exc})"
print(f"       provenance: vllm={getattr(vllm, '__file__', '?')} coordinator={coordinator_path} retention_mark={retention_mark} src_root={ROOT} global_retention={previous_global_retention!r} -> None (dense)")

records = []
class H(logging.Handler):
    def emit(self, rec):
        records.append(rec.getMessage())
for name in ("vllm.sampling_params", "vllm.v1.core.block_pool"):
    lg = logging.getLogger(name); lg.addHandler(H()); lg.setLevel(logging.DEBUG)

def mk(rid, toks, no_store=None, extra=None, skip_reading=None, hbs=2):
    sp = SamplingParams(max_tokens=17, skip_writing_prefix_cache=no_store, extra_args=extra, skip_reading_prefix_cache=skip_reading)
    return Request(request_id=rid, prompt_token_ids=list(toks), sampling_params=sp, pooling_params=None, block_hasher=get_request_block_hasher(hbs, sha256))

def full_cfg(block, num_blocks):
    """Full attention (block > hash) + one mamba(align) group: the smallest
    layout with a partial tail in the attention group (UnitaryKVCacheCoordinator
    requires hash_block_size == block_size, so a single group cannot do it)."""
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["full"], FullAttentionSpec(block_size=block, num_kv_heads=1, head_size=1, dtype=torch.float32)),
        KVCacheGroupSpec(["mamba"], MambaSpec(block_size=block, shapes=(1, 1), dtypes=(torch.float32,), mamba_cache_mode="align")),
    ])

def hybrid_cfg(hbs, block, num_blocks):
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["full"], FullAttentionSpec(block_size=hbs, num_kv_heads=1, head_size=1, dtype=torch.float32)),
        KVCacheGroupSpec(["mamba"], MambaSpec(block_size=block, shapes=(1, 1), dtypes=(torch.float32,), mamba_cache_mode="align")),
    ])

class LayoutError(RuntimeError):
    pass

# The live GLM-5.3 group layout this fixture reproduces, by manager class. The
# fork adds the KpoolTailSpec group; the upstream shape has six groups. A pass
# on one shape is never reported as the other.
LIVE_MANAGERS = {
    True: ["FullAttentionManager", "KpoolTailManager", "MambaManager", "MambaManager", "MambaManager", "MambaManager", "SlidingWindowManager"],
    False: ["FullAttentionManager", "MambaManager", "MambaManager", "MambaManager", "MambaManager", "SlidingWindowManager"],
}

def live_cfg(hbs, block, num_blocks):
    """Live-shape fixture: MLA + [KpoolTail] + 4 Mamba(align) + EAGLE SWA drafter.

    Returns ``(config, spec_names, fork_shaped)``. ``KpoolTailSpec`` is fork-only
    and part of the layout under test: when this runtime exposes it, it must also
    be constructible. Dropping it would run a six-group upstream shape while the
    checks claim the fork's seven-group composition.
    """
    names = ["MLAAttentionSpec"]
    groups = [KVCacheGroupSpec(["mla"], MLAAttentionSpec(block_size=block, num_kv_heads=1, head_size=1, dtype=torch.float32))]
    if KpoolTailSpec is not None:
        try:
            tail = KpoolTailSpec(block_size=hbs, num_kv_heads=1, head_size=1, dtype=torch.float32, sliding_window=block)
        except Exception as exc:
            raise LayoutError(
                "KpoolTailSpec is exposed by this runtime but not constructible "
                f"({exc!r}); a six-group fixture must not stand in for the fork's "
                "seven-group layout"
            ) from None
        groups.append(KVCacheGroupSpec(["kpool"], tail))
        names.append("KpoolTailSpec")
    for i in range(4):
        groups.append(KVCacheGroupSpec([f"m{i}"], MambaSpec(block_size=block, shapes=((1, 1),), dtypes=(torch.float32,), mamba_cache_mode="align")))
        names.append("MambaSpec")
    groups.append(KVCacheGroupSpec(["swa"], SlidingWindowSpec(block_size=hbs, num_kv_heads=1, head_size=1, dtype=torch.float32, sliding_window=block), is_eagle_group=True))
    names.append("SlidingWindowSpec")
    return (
        KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups),
        names,
        KpoolTailSpec is not None,
    )

def manager(cfg, hbs, sched, **kw):
    return KVCacheManager(cfg, max_model_len=8192, scheduler_block_size=sched, hash_block_size=hbs, enable_caching=True, **kw)

def hashes(m, rid):
    return [[b.block_hash for b in grp] for grp in m.get_blocks(rid).blocks]

def any_hash(m, rid):
    return any(h is not None for grp in hashes(m, rid) for h in grp)

def managers(m):
    return list(m.coordinator.single_type_managers)

def ncb(m, rid):
    return [mgr.num_cached_block.get(rid) for mgr in managers(m)]

def block_ids(m, rid):
    return {b.block_id for grp in m.get_blocks(rid).blocks for b in grp if not b.is_null}

def prefill(m, req, chunks):
    """Chunked prefill: returns the per-step (num_cached_block vector, has-hash) trace."""
    cb, n, _ = m.get_computed_blocks(req)
    trace = []
    done = 0
    for i, c in enumerate(chunks):
        out = m.allocate_slots(req, c, n if i == 0 else 0, cb if i == 0 else None)
        assert out is not None, "allocate_slots returned None"
        done += c
        req.num_computed_tokens = n + done if i == 0 else req.num_computed_tokens + c
        trace.append((ncb(m, req.request_id), any_hash(m, req.request_id)))
    return n, trace

# C1 -- BlockPool alone: LIFO front vs LRU back, isolated legs (distinct tokens).
pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
def store(pool, rid, toks, no_store):
    req = mk(rid, toks, no_store=no_store, hbs=4)
    blocks = pool.get_new_blocks(len(toks) // 4)
    pool.cache_full_blocks(request=req, blocks=blocks, num_cached_blocks=0, num_full_blocks=len(blocks), block_size=4, kv_cache_group_id=0)
    return blocks
normal = store(pool, "n", range(100, 112), False)
quiet = store(pool, "q", range(200, 212), True)
check(all(b.block_hash is not None for b in normal), "C1 normal request: every full block hashed")
check(all(b.block_hash is None for b in quiet), "C1 no-store request (distinct tokens): no block hashed")
check(pool.get_cached_block(mk("x", range(200, 212), hbs=4).block_hashes[0], [0]) is None, "C1 no-store hashes are not reachable in the cache map")
check(pool.get_cached_block(mk("y", range(100, 112), hbs=4).block_hashes[0], [0]) is not None, "C1 normal hashes are reachable in the cache map")
pool.free_blocks(reversed(normal)); pool.free_blocks(reversed(quiet))
reused = pool.get_new_blocks(3)
check({b.block_id for b in reused} == {b.block_id for b in quiet}, "C1 after freeing both, the next 3 blocks recycled are exactly the no-store ones (LIFO front)")
check(all(b.block_hash is not None for b in normal), "C1 the normal request's blocks are still cached after the recycle")
blk = pool.get_new_blocks(1)[0]
out = pool.cache_partial_block(request=mk("p", range(300, 316), no_store=True, hbs=4), block=blk, num_tokens=4, kv_cache_group_id=0, block_size=8)
check(out is None and blk.block_hash is None, "C1 cache_partial_block returns None and inserts nothing for a no-store request")
out = pool.cache_partial_block(request=mk("p2", range(400, 416), hbs=4), block=blk, num_tokens=4, kv_cache_group_id=0, block_size=8)
check(out is not None and blk.block_hash is not None, "C1 control: cache_partial_block inserts for a normal request")

# C2 -- KVCacheManager (full attention, hash 2 / block 4), multi-step, isolated legs.
toks = list(range(1000, 1014))  # 14 tokens: 3 full blocks + 2-token partial tail
legs = {}
for label, ns in (("normal", False), ("nostore", True)):
    m = manager(full_cfg(4, 20), 2, 4)
    req = mk(label, toks, no_store=ns)
    n, trace = prefill(m, req, (4, 4, 4, 2))
    check(n == 0, f"C2 [{label}] cold start: 0 computed")
    mgr = managers(m)[0]
    check([t[0][0] for t in trace] == [1, 2, 3, 3], f"C2 [{label}] attention-group num_cached_block advances 1,2,3,3 over the four steps (sentinel) -> {[t[0] for t in trace]}")
    check(req.request_id in mgr.num_cached_block, f"C2 [{label}] request is in num_cached_block after step 1 (fast path from step 2)")
    legs[label] = (m, req, trace)
mN, rN, tN = legs["normal"]; mQ, rQ, tQ = legs["nostore"]
check(all(t[1] for t in tN), "C2 [normal] hashes present after every step")
check(not any(t[1] for t in tQ), "C2 [nostore] no hash after any step")
check([t[0] for t in tN] == [t[0] for t in tQ], "C2 the two legs' num_cached_block traces are identical (placement invariant)")
blkN = mN.get_blocks("normal").blocks[0]; blkQ = mQ.get_blocks("nostore").blocks[0]
check(blkN[3].block_hash is not None and blkN[3].block_hash_num_tokens == 14, "C2 [normal] partial tail entry at 14 tokens")
check(blkQ[3].block_hash is None and blkQ[3].block_hash_num_tokens is None, "C2 [nostore] no partial tail entry")
fN = mk("fN", toks + [7, 7]); fQ = mk("fQ", toks + [7, 7])
_, nN, _ = mN.get_computed_blocks(fN); _, nQ, _ = mQ.get_computed_blocks(fQ)
check(nN == 14, f"C2 [normal] same-prefix follow-up hits 14 (12 full + 2-token partial tail) -> {nN}")
check(nQ == 0, f"C2 [nostore] same-prefix follow-up hits 0 -> {nQ}")
idsN = block_ids(mN, "normal"); idsQ = block_ids(mQ, "nostore")
mN.free(rN); mQ.free(rQ)
gN = mk("gN", range(5000, 5004)); gQ = mk("gQ", range(5000, 5004))
cb, n, _ = mN.get_computed_blocks(gN); mN.allocate_slots(gN, 4, n, cb)
cb, n, _ = mQ.get_computed_blocks(gQ); mQ.allocate_slots(gQ, 4, n, cb)
check(block_ids(mQ, "gQ") <= idsQ, "C2 [nostore] the next allocation recycles the freed no-store block first (front of the free queue)")
check(not (block_ids(mN, "gN") & idsN), "C2 [normal] the next allocation does NOT touch the freed cached blocks (they sit at the back)")
# C2r: sparse retention interval on the same shape -> still no hashes for no-store, sentinel still advances.
# Upstream KVCacheConfig (22df3a3) carries prefix_cache_retention_interval as a dataclass field; the fork's
# in-image build drives retention through VLLM_PREFIX_CACHE_RETENTION_INTERVAL instead and has no such field,
# so this sub-check feature-detects it (found by the in-fork receipt run, 2026-09-01).
from dataclasses import replace, fields
if any(f.name == "prefix_cache_retention_interval" for f in fields(type(full_cfg(4, 24)))):
    m = manager(replace(full_cfg(4, 24), prefix_cache_retention_interval=4), 2, 4)
    req = mk("ret", toks, no_store=True)
    _, trace = prefill(m, req, (4, 4, 4, 2))
    check(not any(t[1] for t in trace) and [t[0][0] for t in trace] == [1, 2, 3, 3], "C2 [nostore + retention_interval] no hashes, sentinel unchanged")
else:
    print("  skip C2 [nostore + retention_interval]: this fork uses the env-driven retention cases below, not the upstream dataclass field")

# C3 -- hybrid Full(hash 2) + Mamba(align, block 4): the upstream partial-tail fixture shape.
def mamba(m):
    return managers(m)[1]
# (a) no-store PRODUCER of an unaligned tail, then continue-decode (running request).
m = manager(hybrid_cfg(2, 4, 24), 2, 4)
r0 = mk("p0", [0, 0, 1, 1, 2, 2], no_store=True)
cb, n, _ = m.get_computed_blocks(r0)
check(m.allocate_slots(r0, 6, n, cb) is not None, "C3a no-store producer: allocate 6 tokens (partial tail at 6)")
check(not any_hash(m, "p0"), "C3a no hashes in either group")
check("p0" not in mamba(m)._partial_hit_reqs and "p0" not in mamba(m)._producer_partial_tail_reqs, "C3a not registered as a partial-tail producer (no _partial_hit_reqs / _producer_partial_tail_reqs entry)")
check(ncb(m, "p0") == [3, 1], f"C3a num_cached_block advanced as for a normal producer ([3,1]) -> {ncb(m, 'p0')}")
ph = r0.block_hashes[6 // 2 - 1]
check(m.block_pool.get_cached_block(ph, [1]) is None and m.block_pool.get_cached_block(ph, [0]) is None, "C3a partial hash not in the cache map for either group")
r0.num_computed_tokens = 6; r0.append_output_token_ids([3])
before_ids = block_ids(m, "p0")
nb = m.allocate_slots(r0, 1)
check(nb is not None, "C3a continue-decode allocates (get_num_blocks_to_allocate / allocate_new_blocks agree, no assertion)")
copies, _ = m.take_kv_cache_block_copies()
check(not any(c.src_block_id in before_ids for c in copies), "C3a no CoW copy sourced from the no-store producer's blocks (move_block_hashes path unreachable)")
check(not any_hash(m, "p0"), "C3a still no hash on any of its blocks after the continue step")
r0.num_computed_tokens = 7; r0.append_output_token_ids([4]); m.allocate_slots(r0, 1)
check(not any_hash(m, "p0") and "p0" not in mamba(m)._partial_hit_reqs, "C3a second decode step: still unhashed, still no partial-hit registration")
# control on a fresh manager: a NORMAL producer does get the partial entry and the CoW.
m = manager(hybrid_cfg(2, 4, 24), 2, 4)
c0 = mk("c0", [0, 0, 1, 1, 2, 2])
cb, n, _ = m.get_computed_blocks(c0); m.allocate_slots(c0, 6, n, cb)
check(m.block_pool.get_cached_block(c0.block_hashes[2], [1]) is not None, "C3a control: normal producer registers the mamba partial tail")
c0.num_computed_tokens = 6; c0.append_output_token_ids([3]); m.allocate_slots(c0, 1)
copies, _ = m.take_kv_cache_block_copies()
check(len(copies) >= 1, "C3a control: normal producer's continue step queues a CoW copy")
# (b)/(c) no-store READER of a normal producer's partial tail (with and without delay_cache_blocks)
for delay in (False, True):
    m = manager(hybrid_cfg(2, 4, 24), 2, 4)
    c0 = mk("c0", [0, 0, 1, 1, 2, 2])
    cb, n, _ = m.get_computed_blocks(c0); m.allocate_slots(c0, 6, n, cb); m.free(c0); m.new_step_starts()
    ph = c0.block_hashes[2]
    src = m.block_pool.get_cached_block(ph, [1])[0]
    r1 = mk("r1", [0, 0, 1, 1, 2, 2, 3, 3], no_store=True)
    cb, n, _ = m.get_computed_blocks(r1)
    check(n == 6, f"C3b[delay={delay}] no-store reader read-hits 6 (lookups enabled)")
    nb = m.allocate_slots(r1, 2, n, cb, delay_cache_blocks=delay)
    check(nb is not None, f"C3b[delay={delay}] allocation ok")
    cow = m.get_blocks("r1").blocks[1][1]
    check(cow.block_id != src.block_id, f"C3b[delay={delay}] reader CoWs into a private mamba block")
    copies, _ = m.take_kv_cache_block_copies()
    check(any(c.src_block_id == src.block_id and c.dst_block_id == cow.block_id for c in copies), f"C3b[delay={delay}] CoW copy src->private queued")
    check(src.block_hash is not None and get_block_hash(src.block_hash) == ph and src.block_hash_num_tokens == 6, f"C3b[delay={delay}] source partial entry kept its hash (no move onto the no-store request)")
    check(cow.block_hash is None and cow.block_hash_num_tokens is None, f"C3b[delay={delay}] the private CoW block is unhashed (normal reader would carry the 8-token hash)")
    own = {b.block_id for grp in m.get_blocks("r1").blocks for b in grp if not b.is_null} - {b.block_id for grp in cb.blocks for b in grp if not b.is_null}
    check(own and not any(b.block_hash is not None for grp in m.get_blocks("r1").blocks for b in grp if b.block_id in own), f"C3b[delay={delay}] every block the no-store reader allocated itself is unhashed")
    m.free(r1)
    nxt = m.block_pool.get_new_blocks(1)[0]
    check(nxt.block_id in own, f"C3b[delay={delay}] after free, the next block recycled is one of the reader's own (front of queue)")
    m.block_pool.free_blocks([nxt])
# (d) control: normal reader gets the 8-token hash on its CoW block
m = manager(hybrid_cfg(2, 4, 24), 2, 4)
c0 = mk("c0", [0, 0, 1, 1, 2, 2]); cb, n, _ = m.get_computed_blocks(c0); m.allocate_slots(c0, 6, n, cb); m.free(c0); m.new_step_starts()
r1 = mk("r1", [0, 0, 1, 1, 2, 2, 3, 3]); cb, n, _ = m.get_computed_blocks(r1); m.allocate_slots(r1, 2, n, cb)
check(m.get_blocks("r1").blocks[1][1].block_hash_num_tokens == 8, "C3d control: normal reader's CoW block is hashed at 8 tokens")

# C3L -- live-shape layout (see live_cfg): chunked prefill + two decode steps +
# free-queue recycling, once per retention mode this runtime supports. The
# composed gate runs this file AFTER patch_hybrid_prefix_hit.py and
# patch_apc_per_group_retention.py, so these legs run on the real hybrid +
# per-group retention + injected-no-store stack.
SWA_ENV = "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA"
RETENTION_HELPERS = ("_glm53_swa_retention_env", "_glm53_min_exempt_group_ids", "_glm53_resolve_retention_by_group")
retention_overlay = all(hasattr(kvc_mod, _name) for _name in RETENTION_HELPERS)
require_composition = os.environ.get("GLM53_REQUIRE_COMPOSITION", "") == "1"
if retention_overlay:
    live_modes = [
        ("inherit", ""),
        ("SWA=0", "0"),
    ]
else:
    live_modes = [("no-retention-runtime", None)]
    check(
        not require_composition,
        "C3L the per-group retention overlay is missing from the loaded coordinator "
        "(hybrid + per-group retention must be installed before this run)"
        + (" -- GLM53_REQUIRE_COMPOSITION=1" if require_composition else "; the live leg still runs without it"),
    )


def partial_state(m, rid):
    """Managers holding this request as a partial-tail reader/producer."""
    out = []
    for mgr in managers(m):
        hit = rid in getattr(mgr, "_partial_hit_reqs", {})
        prod = rid in getattr(mgr, "_producer_partial_tail_reqs", {})
        if hit or prod:
            out.append(type(mgr).__name__)
    return out

def live_manager(mode, print_layout=True):
    """Build the live-shape manager; ``(manager, group_count)`` or ``None``."""
    try:
        cfg, spec_names, fork_shaped = live_cfg(2, 4, 96)
        m = manager(cfg, 2, 4, use_eagle=True)
    except LayoutError as exc:
        check(False, f"C3L [{mode}] {exc}")
        return None
    except Exception as exc:  # e.g. the retention overlay's fail-closed boot
        check(False, f"C3L [{mode}] manager construction raised {type(exc).__name__}: {exc}")
        return None
    got = [type(x).__name__ for x in managers(m)]
    want = LIVE_MANAGERS[fork_shaped]
    if print_layout:
        partial_hits = getattr(m.coordinator, "enable_partial_hash_hits", "<missing>")
        print(f"       layout [{mode}]: specs={spec_names} managers={got} fork_seven_group={fork_shaped} eagle={sorted(m.coordinator.eagle_group_ids)} partial_hits={partial_hits}")
    check(got == want, f"C3L [{mode}] runtime group layout is explicit: {want} -> {got}")
    return m, len(want)

def live_mode(mode, env_value):
    saved_swa = os.environ.get(SWA_ENV)
    if env_value is not None:
        os.environ[SWA_ENV] = env_value
    try:
        legs = {}
        for label, ns in (("normal", False), ("nostore", True)):
            made = live_manager(f"{mode}/{label}", print_layout=(label == "normal"))
            if made is None:
                return
            m, _ = made
            req = mk(label, list(range(2000, 2014)), no_store=ns)
            n, trace = prefill(m, req, (4, 4, 4, 2))
            req.append_output_token_ids([9])
            check(m.allocate_slots(req, 1) is not None, f"C3L [{mode}/{label}] decode step after chunked prefill")
            req.num_computed_tokens += 1
            req.append_output_token_ids([9])
            check(m.allocate_slots(req, 1) is not None, f"C3L [{mode}/{label}] second decode step")
            legs[label] = (m, req, trace)
        mN, rN, tN = legs["normal"]
        mQ, rQ, tQ = legs["nostore"]
        check([t[0] for t in tN] == [t[0] for t in tQ], f"C3L [{mode}] per-group num_cached_block traces identical across the normal and no-store legs -> {[t[0] for t in tQ]}")
        check(all(t[1] for t in tN), f"C3L [{mode}] [normal] hashes present (control)")
        check(not any_hash(mQ, "nostore"), f"C3L [{mode}] [nostore] no hash in ANY group after prefill + decode (full site)")
        check(all(mgr.cached_blocks_this_step == set() for mgr in managers(mQ) if hasattr(mgr, "cached_blocks_this_step")), f"C3L [{mode}] [nostore] mamba cached_blocks_this_step stays empty")
        # Partial-site provenance: the normal leg's registrations are printed,
        # not asserted (this runtime decides whether the partial path is reached
        # in this shape); the discriminating partial-site legs are C2 and C3a.
        print(f"       partial-tail readers/producers on the normal leg [{mode}]: {partial_state(mN, 'normal') or 'none'}")
        check(partial_state(mQ, "nostore") == [], f"C3L [{mode}] [nostore] no manager registered it as a partial-tail reader/producer")
        f = mk("f", list(range(2000, 2016)))
        _, nQ, _ = mQ.get_computed_blocks(f)
        _, nN, _ = mN.get_computed_blocks(mk("f2", list(range(2000, 2016))))
        check(nQ == 0 and nN > 0, f"C3L [{mode}] same-prefix follow-up: normal hits {nN}, no-store hits {nQ}")
        idsQ = block_ids(mQ, "nostore")
        mQ.free(rQ)
        nxt = mQ.block_pool.get_new_blocks(4)
        check({b.block_id for b in nxt} <= idsQ, f"C3L [{mode}] [nostore] freed blocks are the very next ids recycled")
        if env_value is not None:
            # Cross the drafter replay window: query retained cache entries,
            # rather than asserting internal retention vectors or free-order fields.
            for label, ns in (("normal", False), ("nostore", True)):
                cfg, _, _ = live_cfg(2, 4, 512)
                retained = manager(cfg, 2, 4, use_eagle=True)
                req = mk(label, list(range(3000, 3080)), no_store=ns)
                prefill(retained, req, (4,) * 20)
                group = len(managers(retained)) - 1
                retained.free(req)
                old_hit = retained.block_pool.get_cached_block(req.block_hashes[0], [group]) is not None
                boundary_hit = retained.block_pool.get_cached_block(req.block_hashes[-2], [group]) is not None
                check(old_hit == (not ns and env_value == "") and boundary_hit == (not ns),
                      f"C3L [{mode}/{label}] released drafter cache lookups: old={old_hit}, boundary={boundary_hit}")
    finally:
        if env_value is not None:
            if saved_swa is None:
                os.environ.pop(SWA_ENV, None)
            else:
                os.environ[SWA_ENV] = saved_swa

for _mode, _swa in live_modes:
    live_mode(_mode, _swa)

# C4 -- preemption bookkeeping: cold no-store recomputes from 0; a read-hit no-store resumes from what it read.
m = manager(full_cfg(4, 20), 2, 4)
req = mk("pre", toks, no_store=True)
n, trace = prefill(m, req, (4, 4))
m.free(req); req.num_computed_tokens = 0
cb, n, _ = m.get_computed_blocks(req)
check(n == 0, "C4 cold no-store request preempted mid-prefill resumes with 0 computed (nothing was stored)")
check(m.allocate_slots(req, 4, n, cb) is not None and ncb(m, "pre")[0] == 1, "C4 ... and re-allocates cleanly from zero")
m = manager(full_cfg(4, 20), 2, 4)
a = mk("a", toks[:8]); cb, n, _ = m.get_computed_blocks(a); m.allocate_slots(a, 8, n, cb); m.free(a)
b = mk("b", toks, no_store=True); cb, n, _ = m.get_computed_blocks(b)
check(n == 8, "C4 read-hit no-store request hits the 8 cached tokens")
m.allocate_slots(b, 6, n, cb); m.free(b); b.num_computed_tokens = 0
cb, n2, _ = m.get_computed_blocks(b)
check(n2 == 8, "C4 ... and after preemption resumes from those same 8 (its own 6 were never stored)")

# C5 -- Request-level resolution on real SamplingParams; boundary rejection; zero-interaction combination.
check(mk("x1", toks, extra={"skip_writing_prefix_cache": 1}).skip_writing_prefix_cache is True, "C5 extra_args 1 -> True on a real Request")
check(mk("x2", toks, extra={"skip_writing_prefix_cache": "0"}).skip_writing_prefix_cache is False, "C5 extra_args '0' -> False")
check(mk("x3", toks, no_store=True).skip_writing_prefix_cache is True, "C5 typed True -> True")
check(mk("x4", toks).skip_writing_prefix_cache is False, "C5 default -> False")
try:
    SamplingParams(max_tokens=1, extra_args={"skip_writing_prefix_cache": "yes"}); rejected = False
except ValueError as exc:
    rejected = "skip_writing_prefix_cache" in str(exc)
check(rejected, "C5 SamplingParams(extra_args={...: 'yes'}) raises ValueError naming the field (API boundary -> 400)")
try:
    SamplingParams(max_tokens=1, skip_writing_prefix_cache=1.0); rejected = False
except ValueError:
    rejected = True
check(rejected, "C5 SamplingParams(skip_writing_prefix_cache=1.0) rejected")
sp = SamplingParams(max_tokens=1, extra_args={"skip_writing_prefix_cache": 1}); sp.extra_args["skip_writing_prefix_cache"] = "yes"
rr = Request(request_id="mut", prompt_token_ids=toks, sampling_params=sp, pooling_params=None, block_hasher=get_request_block_hasher(2, sha256))
check(rr.skip_writing_prefix_cache is False, "C5 a value mutated after validation resolves to False without raising in Request")
m = manager(full_cfg(4, 20), 2, 4)
a = mk("a", toks[:8]); cb, n, _ = m.get_computed_blocks(a); m.allocate_slots(a, 8, n, cb); m.free(a)
z = mk("z", toks, no_store=True, skip_reading=True); cb, n, _ = m.get_computed_blocks(z)
check(n == 0, "C5 skip_reading + skip_writing: no read hit despite a cached prefix")
m.allocate_slots(z, 14, n, cb)
check(not any_hash(m, "z"), "C5 skip_reading + skip_writing: nothing stored either (zero cache interaction)")
sp_mod._GLM53_NO_STORE_ENABLED = False
check(mk("ks", toks, no_store=True).skip_writing_prefix_cache is False, "C5 kill switch off: typed True resolves to False (ignored)")
sp_mod._GLM53_NO_STORE_ENABLED = True

# C6 -- receipts
res = [x for x in records if "first request resolved skip_writing_prefix_cache=1" in x]
full = [x for x in records if "suppressing prefix-cache store (full site)" in x]
part = [x for x in records if "suppressing prefix-cache store (partial site)" in x]
check(len(res) == 1, f"C6 resolution receipt logged exactly once for the whole run ({len(res)})")
check(len(full) == 1 and len(part) == 1, f"C6 suppression receipts logged exactly once per site (full={len(full)} partial={len(part)})")
check(any("ignoring skip_writing_prefix_cache=1" in x for x in records), "C6 kill-switch ignore receipt logged")

print(f"PARTC_RESULT checks={N} failures={len(FAIL)}")
for f in FAIL:
    print("PARTC_FAIL " + f)
sys.exit(1 if FAIL else 0)
'''


def part_c(root: Path | None) -> None:
    print("Part C: behaviour on a real vLLM (patched copies injected via sys.modules)")
    require = os.environ.get("GLM53_REQUIRE_VLLM", "") == "1"
    if root is None:
        check(not require, "C0 no vLLM source root -> Part C skipped" + (" (GLM53_REQUIRE_VLLM=1: failing)" if require else ""))
        return
    probe = subprocess.run([sys.executable, "-c", "import vllm.v1.core.kv_cache_manager, torch"], capture_output=True, text=True)
    if probe.returncode != 0:
        msg = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else "?"
        check(not require, f"C0 `import vllm` failed under {sys.executable}: {msg} -> Part C skipped" + (" (GLM53_REQUIRE_VLLM=1: failing)" if require else ""))
        return
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = tmp / "part_c.py"
        script.write_text(PART_C)
        env = {k: v for k, v in os.environ.items() if not k.startswith("GLM53_")}
        env.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")
        # GLM53_* is stripped above (patch-path overrides); carry the strictness
        # flag through explicitly so C3L can require the retention composition.
        if os.environ.get("GLM53_REQUIRE_COMPOSITION", "") == "1":
            env["GLM53_REQUIRE_COMPOSITION"] = "1"
        r = subprocess.run([sys.executable, str(script), str(root), str(PATCH), str(tmp)], capture_output=True, text=True, env=env)
        for line in r.stdout.splitlines():
            if line.startswith(("  ok", "  FAIL", "  skip", "       ")):
                print(line)
        summary = next((ln for ln in r.stdout.splitlines() if ln.startswith("PARTC_RESULT")), "")
        fails = [ln[len("PARTC_FAIL "):] for ln in r.stdout.splitlines() if ln.startswith("PARTC_FAIL ")]
        FAILURES.extend(fails)
        global CHECKS
        try:
            CHECKS += int(summary.split("checks=")[1].split()[0])
        except (IndexError, ValueError):
            pass
        if r.returncode != 0 and not fails:
            tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-25:])
            check(False, f"C0 Part C crashed (rc={r.returncode}):\n{tail}")
        else:
            check(r.returncode == 0, f"C summary: {summary or 'no summary line'}")


# ---------------------------------------------------------------- part D ----


def guard_source() -> str:
    text = START.read_text()
    begin = text.index("# GLM53 numeric config guard (begin)")
    end_marker = "# GLM53 numeric config guard (end)"
    return text[begin : text.index(end_marker, begin) + len(end_marker)]


def caller_gate(export: str | None, dotenv: str) -> tuple[int, str, str]:
    """Run start.sh's real caller-override block against a controlled ``.env``
    and then the lifted numeric-config guard, and report the value the ranks
    would receive. The launcher snapshots caller exports and re-exports them --
    explicit empties included -- AFTER sourcing ``.env`` (generic precedence,
    #161/#92); the guard is what then refuses an empty kill switch."""
    src = START.read_text()
    begin = src.index(CALLER_BEGIN)
    end = src.index(CALLER_END, begin) + len(CALLER_END)
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        (tmp / ".env").write_text(dotenv)
        script = (
            "SCRIPT_DIR=$GLM53_ENV_DIR\n"
            + src[begin:end]
            + "\nGPU_MEM_UTIL=0.87; MAX_MODEL_LEN=1000000; MAX_NUM_SEQS=4; MAX_NUM_BATCHED_TOKENS=1024\n"
            + "GLM53_INDEXER_WORKSPACE=stock; GLM53_SPINWAIT_MS=stock; SPEC_METHOD=none\n"
            + guard_source()
            + '\nGLM53_APC_NO_STORE="${GLM53_APC_NO_STORE-1}"\n'
            + "validate_numeric_config || exit $?\n"
            + 'printf "%s\\n" "$GLM53_APC_NO_STORE"\n'
        )
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LC_ALL": "C",
            "GLM53_ENV_DIR": str(tmp),
        }
        if export is not None:
            env["GLM53_APC_NO_STORE"] = export
        r = subprocess.run(["bash", "-c", script], text=True, capture_output=True, env=env)
        return r.returncode, r.stdout.strip(), r.stderr.strip()


def part_d() -> None:
    print("Part D: launcher knob GLM53_APC_NO_STORE (numeric config guard)")
    if not START.is_file():
        # The Dockerfile copies the test into the image WITHOUT start.sh, so
        # the launcher legs are host-only; a checkout missing its launcher is
        # still an error (PATCH sits next to the test only in the flat image).
        check(PATCH.parent == HERE, "D0 no start.sh next to the test: launcher legs are host-only (in-image run)")
        return
    guard = guard_source()
    for label, export, dotenv, want in (
        ("caller unset: the .env value is what the ranks get", None, "GLM53_APC_NO_STORE=0\n", (0, "0")),
        ("a caller export wins over .env", "1", "GLM53_APC_NO_STORE=0\n", (0, "1")),
        ("an explicitly empty caller export wins, then the guard rejects it", "", "GLM53_APC_NO_STORE=0\n", (2, "")),
    ):
        rc, out, err = caller_gate(export, dotenv)
        check((rc, out) == want, f"D1 {label} (rc={rc} out={out!r} {err[:80]!r})")

    def run(value: str | None) -> tuple[int, str, str]:
        script = guard + "\nGPU_MEM_UTIL=0.87; MAX_MODEL_LEN=1000000; MAX_NUM_SEQS=4; MAX_NUM_BATCHED_TOKENS=1024; GLM53_INDEXER_WORKSPACE=stock; GLM53_SPINWAIT_MS=stock\n" + "validate_numeric_config || exit $?\n" + 'printf "%s\\n" "${GLM53_APC_NO_STORE-unset}"\n'
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"}
        if value is not None:
            env["GLM53_APC_NO_STORE"] = value
        r = subprocess.run(["bash", "-c", script], text=True, capture_output=True, env=env)
        return r.returncode, r.stdout.strip(), r.stderr.strip()

    for value, out in ((None, "unset"), ("0", "0"), ("1", "1")):
        rc, o, e = run(value)
        check(rc == 0 and o == out, f"D2 GLM53_APC_NO_STORE={value!r} accepted (rc={rc} out={o!r} {e[:60]})")
    for value in ("", " ", "01", "2", "yes", "true", "1 ", " 1", "1\r", "0x1", "-1"):
        rc, o, e = run(value)
        check(rc == 2 and "GLM53_APC_NO_STORE" in e, f"D3 GLM53_APC_NO_STORE={value!r} rejected rc=2 with a named error (rc={rc} {e[:60]!r})")


# ------------------------------------------------------------------ main -----


def main() -> int:
    if PATCH is None:
        raise SystemExit("missing overlay/patch_apc_no_store.py")
    root = source_root()
    print(f"overlay: {PATCH}\nsources: {root or '(none: replicas only, Part C skipped)'}  python: {sys.executable}")
    part_a(root)
    part_b()
    part_c(root)
    part_d()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}/{CHECKS}): " + "; ".join(FAILURES))
        return 1
    print(f"no-store overlay OK ({CHECKS} checks)")
    return 0


def test_apc_no_store() -> None:
    """pytest entry point."""
    FAILURES.clear()
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
