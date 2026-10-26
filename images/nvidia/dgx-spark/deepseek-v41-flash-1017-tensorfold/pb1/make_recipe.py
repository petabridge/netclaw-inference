#!/usr/bin/env python3
"""Write Petabridge's build recipe k2a-pb1 into an extracted upstream recipe tree (Petabridge).

k2a-pb1 is upstream's k2a recipe with the files in this directory as extra overlays. It goes through upstream's
own machinery unchanged: variants/patch.py applies and verifies the overlays, variants/build.py builds and runs
the CPU checks, and launch.py compares BUILD.json with recipes/k2a-pb1.json at start. This script writes:

  tensorfold/variants/objects/<sha256>.zig     each overlay file, named by its hash as upstream's objects are
  tensorfold/variants/recipes/k2a-pb1.json     k2a.json plus those overlays, renamed, with its tree hash redone
  tensorfold/profiles/1.2.5-pb1.json           profile 1.2.5 with build_variant k2a-pb1, for build.py install

It refuses to run unless k2a.json is the pinned upstream file and each replaced file is still the upstream base
file k2a leaves alone, so a new upstream commit cannot silently drop or misapply a patch.

The overlays:
  zig/src/server/tool_parse.zig       a V4.1 DSML invoke naming a tool the request did not offer is a call
                                      under its written name (unlimited calls) instead of the reply's text
  zig/src/server/tool_parse_test.zig  the upstream test updated for that, plus a mixed offered/unoffered block
Upstream report: https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/67
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "k2a-pb1"
# recipes/k2a.json at upstream commit b4e476d6 (the hash the Dockerfile and selfcheck.py already pin).
K2A_JSON_SHA256 = "600c83c88d72f22506facc7cc1883f8d9e4439268b7e1d198efdbcb4536de9bc"
# target path -> (file in this directory, the upstream base file's sha256 as k2a.json's source_files records it)
OVERLAYS = {
    "zig/src/server/tool_parse.zig":
        ("tool_parse.zig", "211918aed38ccbf5081f9427374d2a33c7e5b903a1481ebd8859950e8a3b5bb0"),
    "zig/src/server/tool_parse_test.zig":
        ("tool_parse_test.zig", "53e8a544a1604af7f01715148800af0cdcad09874c94f15123cd723fee0c0fa4"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tree_digest(files):
    # upstream's variants/patch.py tree_digest, byte for byte
    return digest(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe-root", required=True, type=Path, help="the upstream checkout (holds tensorfold/)")
    args = parser.parse_args()
    variants = args.recipe_root / "tensorfold" / "variants"
    raw = (variants / "recipes" / "k2a.json").read_bytes()
    if digest(raw) != K2A_JSON_SHA256:
        sys.exit("recipes/k2a.json is not the pinned upstream file; review make_recipe.py")
    spec = json.loads(raw)
    for target, (local, base_sha) in OVERLAYS.items():
        if target in spec["overlays"]:
            sys.exit(f"k2a already overlays {target}; rebase the k2a-pb1 patch")
        if spec["source_files"].get(target) != base_sha:
            sys.exit(f"{target} is no longer the upstream base file this patch was written against")
        data = (HERE / local).read_bytes()
        sha = digest(data)
        (variants / "objects" / f"{sha}.zig").write_bytes(data)
        spec["overlays"][target] = {"object": f"objects/{sha}.zig", "sha256": sha}
        spec["source_files"][target] = sha
    spec["recipe"] = NAME
    spec["source_tree_sha256"] = tree_digest(spec["source_files"])
    # k2a's recorded binaries and CPU-check logs describe k2a's build, not this one; BUILD.json records ours.
    spec["recorded_binaries"] = {}
    spec["recorded_cpu_checks"] = {}
    spec["petabridge"] = {
        "base_recipe": "k2a",
        "base_recipe_sha256": K2A_JSON_SHA256,
        "overlays": sorted(OVERLAYS),
        "upstream_report": "https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/67",
    }
    out = json.dumps(spec, indent=2) + "\n"
    (variants / "recipes" / f"{NAME}.json").write_text(out, encoding="utf-8", newline="\n")
    profiles = args.recipe_root / "tensorfold" / "profiles"
    profile = json.loads((profiles / "1.2.5.json").read_bytes())
    if profile.get("build_variant") != "k2a":
        sys.exit("profile 1.2.5 no longer builds k2a; review make_recipe.py")
    profile["build_variant"] = NAME
    (profiles / "1.2.5-pb1.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"recipe": NAME, "recipe_sha256": digest(out.encode()),
                      "source_tree_sha256": spec["source_tree_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
