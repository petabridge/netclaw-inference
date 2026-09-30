#!/usr/bin/env python3
"""CPU-only numerical-screen regressions using normalized synthetic evidence."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("bench_numerical_calibrated.py")
SPEC = importlib.util.spec_from_file_location("panel", SCRIPT)
panel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panel)


def row(gap, swap=False):
    z = math.log1p(math.exp(-gap))
    return {"b" if swap else "a": {"logprob": -z},
            "a" if swap else "b": {"logprob": -gap-z}}


def receipt(rows, top_k=2):
    rec = {"mode": "prompt_logprobs", "prompt_sha256": "a"*64,
           "positions": [None] + rows}
    if top_k is not None:
        rec["prompt_logprobs_k"] = top_k
    return {"model": "synthetic-model", "texts": {"probe": rec}}


class NumericalPanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = receipt([row(.5) for _ in range(30)] + [row(3) for _ in range(10)])
        self.stock = [self.save(f"s{i}.json", self.base) for i in range(4)]
        self.clean = self.save("clean.json", self.base)
        self.changed = self.save("changed.json", receipt(
            [row(.5, True) for _ in range(30)] + [row(3) for _ in range(10)]))
        self.cal = str(self.root/"cal.json")
        result = self.cli("calibrate", "--out", self.cal, *self.stock)
        self.assertEqual(result.returncode, 0, result.stderr)

    def save(self, name, value):
        path = self.root/name
        path.write_text(json.dumps(value))
        return str(path)

    def cli(self, *args, env=None):
        return subprocess.run([sys.executable, str(SCRIPT), *args],
                              capture_output=True, text=True, timeout=20, env=env)

    def compare(self, candidate=None, calibration=None, stock=None):
        return self.cli("compare", "--calibration", calibration or self.cal,
                        "--stock", *(stock or self.stock), "--cand", candidate or self.clean)

    def test_explicit_groups_and_old_syntax_rejected(self):
        result = self.cli("compare", "--calibration", self.cal,
                          "--stock", *self.stock[:2], "--cand", self.clean, self.clean, self.clean)
        self.assertEqual(result.returncode, 0, result.stderr)
        # An outlying candidate must not leak into the stock reference range.
        result = self.cli("compare", "--calibration", self.cal,
                          "--stock", *self.stock[:2], "--cand", self.changed, self.changed, self.clean)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        result = self.cli("compare", "--calibration", self.cal,
                          *self.stock[:2], self.clean, self.clean, self.clean)
        self.assertEqual(result.returncode, 2)

    def test_unstable_position_cannot_hide_unrelated_flips(self):
        self.assertEqual(self.compare(self.changed).returncode, 1)
        poison = copy.deepcopy(self.base)
        poison["texts"]["probe"]["positions"][1] = row(.6, True)
        poison_path = self.save("poison.json", poison)
        cal = str(self.root/"poison-cal.json")
        result = self.cli("calibrate", "--out", cal, *self.stock[:3], poison_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.compare(self.changed, cal)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(self.compare(calibration=cal).returncode, 0)

    def test_noisy_position_cannot_hide_unrelated_probability_shifts(self):
        quiet = receipt([row(.5) for _ in range(40)])
        quiet_paths = [self.save(f"quiet{i}.json", quiet) for i in range(4)]
        noisy = copy.deepcopy(quiet)
        noisy["texts"]["probe"]["positions"][1] = row(10)
        noisy_path = self.save("noisy.json", noisy)
        candidate = copy.deepcopy(quiet)
        candidate["texts"]["probe"]["positions"][2:] = [row(.35) for _ in range(39)]
        candidate_path = self.save("flattened.json", candidate)
        for label, paths in (("quiet", quiet_paths), ("noisy", quiet_paths[:3] + [noisy_path])):
            with self.subTest(calibration=label):
                cal = str(self.root/(label + "-cal.json"))
                self.assertEqual(self.cli("calibrate", "--out", cal, *paths).returncode, 0)
                result = self.compare(candidate_path, cal, stock=[quiet_paths[0]])
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(self.compare(quiet_paths[0], cal, stock=[quiet_paths[0]]).returncode, 0)

    def test_input_shapes_are_unqualified(self):
        for value in ([], None, 2, {"model": "m", "texts": {"probe": []}}):
            for kind in ("capture", "calibration"):
                with self.subTest(value=value, kind=kind):
                    path = self.save("bad.json", value)
                    result = self.compare(candidate=path) if kind == "capture" else self.compare(calibration=path)
                    self.assertEqual(result.returncode, 2, result.stderr)
            result = self.cli("calibrate", "--out", str(self.root/"bad-cal.json"),
                              *self.stock[:3], path)
            self.assertEqual(result.returncode, 2, result.stderr)
        path = self.root/"bad.json"
        path.write_text("{bad")
        self.assertEqual(self.compare(str(path)).returncode, 2)

    def test_calibration_classes_and_thresholds_are_validated(self):
        original = json.loads(Path(self.cal).read_text())
        for key, value in (("classes", [False]*40), ("classes", [99]*40),
                           ("n_confident", 0), ("stock_spread_by_position", [float("nan")]*40),
                           ("null_kl_by_position", None), ("r_up", float("inf"))):
            with self.subTest(key=key, value=value):
                cal = copy.deepcopy(original)
                cal["texts"]["probe"][key] = value
                self.assertEqual(self.compare(calibration=self.save("bad-cal.json", cal)).returncode, 2)
        original["schema"] = "calibrated-numerical-panel/v2"
        self.assertEqual(self.compare(calibration=self.save("old-cal.json", original)).returncode, 2)

    def test_prompt_identity_required_and_equal(self):
        for fingerprint in (None, "", "b"*64):
            with self.subTest(fingerprint=fingerprint):
                changed = copy.deepcopy(self.base)
                changed["texts"]["probe"]["prompt_sha256"] = fingerprint
                path = self.save("mismatch.json", changed)
                self.assertEqual(self.compare(path).returncode, 2)
                self.assertEqual(self.cli("calibrate", "--out", str(self.root/"bad-cal.json"),
                                          *self.stock[:3], path).returncode, 2)
        cal = json.loads(Path(self.cal).read_text())
        del cal["texts"]["probe"]["prompt_sha256"]
        changed = copy.deepcopy(self.base)
        del changed["texts"]["probe"]["prompt_sha256"]
        self.assertEqual(self.compare(self.save("missing.json", changed),
                                     self.save("missing-cal.json", cal)).returncode, 2)
        changed = copy.deepcopy(self.base)
        changed["texts"]["other-prompt"] = changed["texts"].pop("probe")
        self.assertEqual(self.compare(self.save("other.json", changed)).returncode, 2)

    def test_network_failure_is_not_candidate_flag(self):
        # Reserve but do not listen: a deterministic refused localhost connection.
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            env = {**os.environ, "GLM53_BENCH_BASE": f"http://127.0.0.1:{reserved.getsockname()[1]}",
                   "NO_PROXY": "127.0.0.1"}
            result = self.cli("capture", "--out", str(self.root/"capture.json"), env=env)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse((self.root/"capture.json").exists())

    def test_poisson_exact_smallest_tail(self):
        # Independent log-PMF upper-tail sum, not the implementation's CDF recurrence.
        for lam in (.5, 5, 30, 500, 501, 1000):
            for alpha in (.01, .05, .2):
                with self.subTest(lam=lam, alpha=alpha):
                    critical = panel.poisson_critical(lam, alpha)
                    stop = int(lam + 16*math.sqrt(lam)) + 100
                    def tail(start):
                        return math.fsum(math.exp(-lam + k*math.log(lam) - math.lgamma(k+1))
                                         for k in range(start, stop))
                    self.assertLessEqual(tail(critical), alpha)
                    self.assertGreater(tail(critical-1), alpha)
        self.assertEqual(panel.poisson_critical(0), 1)

    def test_incomplete_stock_reference_is_unqualified(self):
        incomplete = copy.deepcopy(self.base)
        for i in (1, 2):
            incomplete["texts"]["probe"]["positions"][i] = {
                "b": {"logprob": math.log(.8)}, "c": {"logprob": math.log(.2)}}
        path = self.save("incomplete-stock.json", incomplete)
        result = self.compare(stock=[self.stock[0], self.stock[1], path])
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    def test_missing_shared_support_is_unqualified(self):
        changed = copy.deepcopy(self.base)
        changed["texts"]["probe"]["positions"][1] = {
            "c": {"logprob": math.log(.8)}, "d": {"logprob": math.log(.2)}}
        self.assertEqual(self.compare(self.save("disjoint.json", changed)).returncode, 2)

    def test_candidate_absence_uses_bounds_not_infinity(self):
        # KL on shared support is exactly zero (one shared token); two flips
        # are below the count bar. Only a provable range separation can FLAG.
        near = copy.deepcopy(self.base)
        far = copy.deepcopy(self.base)
        for i in (1, 2):
            near["texts"]["probe"]["positions"][i] = {
                "b": {"logprob": math.log(.50)}, "c": {"logprob": math.log(.49)}}
            far["texts"]["probe"]["positions"][i] = {
                "b": {"logprob": math.log(.8)}, "c": {"logprob": math.log(.1)}}
        self.assertEqual(self.compare(self.save("near.json", near)).returncode, 0)
        self.assertEqual(self.compare(self.save("far.json", far)).returncode, 1)

    def test_top20_plus_observed_does_not_invent_separation(self):
        # Each mapping is top-20 plus (optionally) the observed prompt token.
        # Omitted mass can be spread over 100 tail tokens below the boundary.
        stock = {"observed": .001*.23/.35, "a": .032, "b": .023,
                 **{f"x{i}": .022 for i in range(18)}}
        candidate = {"observed": .001, "b": .035, "c": .034,
                     **{f"x{i}": .022*.35/.23 for i in range(18)}}
        for present in (False, True):
            s, c = dict(stock), dict(candidate)
            if present:
                # The observed token is now a itself: inside stock top-k, but
                # outside candidate top-k. Its actual shift is only .06454 nats.
                del s["observed"]
                del c["observed"]
                c["a"] = .030
            for top_k in (20, None):  # recorded k and recoverable legacy capture
                with self.subTest(reference_present=present, top_k=top_k):
                    sr = {k: {"logprob": math.log(v)} for k, v in s.items()}
                    cr = {k: {"logprob": math.log(v)} for k, v in c.items()}
                    sp = self.save("top20-stock.json", receipt([sr, sr], top_k))
                    cp = self.save("top20-candidate.json", receipt([cr, cr], top_k))
                    cal = str(self.root/"top20-cal.json")
                    self.assertEqual(self.cli("calibrate", "--out", cal, sp, sp, sp, sp).returncode, 0)
                    result = self.compare(cp, cal, stock=[sp])
                    # Present a adds real shared-support KL, so that case
                    # legitimately flags KL. Neither layout justifies separation.
                    self.assertEqual(result.returncode, int(present), result.stdout + result.stderr)
                    report = next(line.split() for line in result.stdout.splitlines()
                                  if line.startswith("probe "))
                    self.assertEqual(int(report[6]), 0)  # observable separation count

    def test_censored_consensus_tie_does_not_impute_observed_floor(self):
        def distribution(values):
            return {k: {"logprob": math.log(v)} for k, v in values.items()}
        # One vote each. Neither tied token is present in the other's capture.
        # Moving the extra observed token cannot choose the winner of that tie.
        for observed in (.001, .02):
            a = receipt([distribution({"a": .4, "x": .3, "observed": .01})])
            b = receipt([distribution({"b": .5, "x": .3, "observed": observed})])
            parsed = [panel.parsed_texts(r)["probe"] for r in (a, b)]
            self.assertEqual(panel._candidate_argmax(parsed, 0), "a")
            self.assertEqual(panel._candidate_argmax(list(reversed(parsed)), 0), "a")
        # When both tied tokens are observed everywhere, their actual mean
        # logprobs (rather than lexicographic order) choose b.
        a = receipt([distribution({"a": .4, "b": .3})])
        b = receipt([distribution({"b": .5, "a": .1})])
        self.assertEqual(panel._candidate_argmax(
            [panel.parsed_texts(r)["probe"] for r in (a, b)], 0), "b")

    def test_invalid_boundary_and_provenance_are_unqualified(self):
        bad = copy.deepcopy(self.base)
        bad["texts"]["probe"]["prompt_logprobs_k"] = 20  # only two entries
        self.assertEqual(self.compare(self.save("bad-width.json", bad)).returncode, 2)
        cal = json.loads(Path(self.cal).read_text())
        for count, hashes in ((0, []), (4, []), (4, ["not-a-hash"]*4)):
            with self.subTest(count=count, hashes=hashes):
                cal["stock_captures"], cal["inputs_sha256"] = count, hashes
                self.assertEqual(self.compare(calibration=self.save("bad-provenance.json", cal)).returncode, 2)


if __name__ == "__main__":
    unittest.main()
