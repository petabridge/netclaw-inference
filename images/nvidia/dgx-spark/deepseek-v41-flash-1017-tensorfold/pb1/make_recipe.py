#!/usr/bin/env python3
"""Write Petabridge's build recipe k2a-pb1 into an extracted upstream recipe tree (Petabridge).

k2a-pb1 is upstream's k2a recipe with the files under overlay/ as extra overlays, each at its target path. It goes
through upstream's own machinery unchanged: variants/patch.py applies and verifies the overlays, variants/build.py builds
and runs the CPU checks, and launch.py compares BUILD.json with recipes/k2a-pb1.json at start. This script writes:

  tensorfold/variants/objects/<sha256>.zig     each overlay file, named by its hash as upstream's objects are
  tensorfold/variants/recipes/k2a-pb1.json     k2a.json plus those overlays, renamed, with its tree hash redone
  tensorfold/profiles/1.2.5-pb1.json           profile 1.2.5 with build_variant k2a-pb1, for build.py install

It refuses to run unless k2a.json is the pinned upstream file and each replaced file is still the exact file k2a
would build (upstream's base file, or k2a's own overlay of it), so a new upstream commit cannot silently drop or
misapply a patch. Each overlay file is the replaced file plus every change below that touches it.

The changes:
  unoffered tool calls (Petabridge; upstream report MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks#67)
    server/tool_parse.zig, tool_parse_test.zig: a V4.1 DSML invoke naming a tool the request did not offer is a call
      under its written name (unlimited calls) instead of the reply's text; the upstream test updated, plus a mixed block
  prefill interleave (TF_DS_PREFILL_INTERLEAVE_MS; MiaAI-Lab zig/patches/0005-prefill-interleave.patch, by Coinupbtc)
    core/lane_host.zig: ported by hand onto k2a's lane host: between two chunks of a prompt pass, decode rounds for the
      other live streams (the learner's clock advanced as run() does), their tokens delivered and cancels taken
    core/lanes/engine.zig, core/lanes/stream.zig, core/lanes/fake.zig, families/dsv41/lanes.zig, families/dsv41/model.zig:
      the patch's hunks (runnableCount, Stream.between, the pass offering rounds between chunks with the Engram
      read-ahead joined first, workers re-arming it, fillChunk keeping a joined read-ahead, verify joining it)
  image prompt cache (TF_DS_IMAGE_CACHE, unset: on; MiaAI-Lab zig/patches/0006-image-prompt-cache.patch)
    server/prompt.zig: a prompt with pictures gets a cache key with each picture's positions made from a hash of its
      payload (the same picture always the same ids, above every vocabulary id), and its history length counted in
      the expanded prompt
    server/chat.zig (k2a's file), core/engine_api.zig (k2a's file): Request.cache_key, passed through
    core/lane_host.zig: ported by hand: such a request looks up, keeps and reports by its key within k2a's cache
      namespace (k2a's text keeps by the stream's context, unchanged); a request with pictures and no key is refused
      by the cache as before
    families/dsv41/lanes.zig: a prompt with pictures resumes a kept state, and the tower and workers get rows only for
      pictures after the resume point
    core/lane_host_reuse_test.zig: the patch's test (a picture resumes its conversation, another picture does not, no
      key neither resumes nor keeps)
  retention (TF_DS_RETENTION, unset: on; Petabridge, for MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks#27)
    families/dsv41/lanes.zig: placeKept gives up kept states by retention order instead of oldest first: never
      resumed, then superseded (an older prefix of a conversation that moved on), then resumed conversations' newest
      states, least recently used first in each; a new state inherits its conversation's resume count. Rank 0 alone
      decides placement, so no frame changes; TF_DS_RETENTION=0 restores upstream's oldest-first order
  stale Engram read-ahead (Petabridge)
    families/dsv41/model.zig: fillBegin clears a read-ahead the previous pass left joined (a pass that ended early
      between chunks), so 0005's joined-read-ahead reuse can never hand another prompt's Engram rows to this pass's
      first chunk
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
# target path (its file is overlay/<target path>) -> the sha256 k2a.json's source_files records for it: the file it replaces
OVERLAYS = {
    "zig/src/server/tool_parse.zig": "211918aed38ccbf5081f9427374d2a33c7e5b903a1481ebd8859950e8a3b5bb0",
    "zig/src/server/tool_parse_test.zig": "53e8a544a1604af7f01715148800af0cdcad09874c94f15123cd723fee0c0fa4",
    "zig/src/core/lane_host.zig": "63a04d2285d41d23de7179ba66fd1f9943618f78636c679307e4fb6d9d8c185d",
    "zig/src/core/lanes/engine.zig": "7b6a7b6db774aded6887e33617894511b085f5ce3a2ab16284cc3b05bb7cac29",
    "zig/src/core/lanes/stream.zig": "1eee608d15f66d0844e2d4673ebbab8d979821041d209828e4810ae1278b269e",
    "zig/src/core/lanes/fake.zig": "b4e88c57fcc8f31bbd462320818ce44734be0b93da616ae6105597344a42a302",
    "zig/src/families/dsv41/lanes.zig": "00506fc47eeb39b06f414c92be81c2d661d4cc818d5324d4cff7b681e7f9de95",
    "zig/src/families/dsv41/model.zig": "9fbd6a092c2e62661b08a16990a441a4b610c4c789a60b01e78e078b65ca20c2",
    "zig/src/core/engine_api.zig": "198ec7a4b25bf13a62c55e70116c4112e79cf326e21a55525250280926a4c925",
    "zig/src/core/lane_host_reuse_test.zig": "69adf37d7fe392f34e24e7be0e6ca4bced32fcb41bdbb3f91cc72da928b17d9a",
    "zig/src/server/prompt.zig": "9a3f360224a1b0f0d19d07b070652b09776fbbcb294587ae305818a89eb93798",
    "zig/src/server/chat.zig": "f96ab77e5d3013249cc312940489f74d9f99e1325a13e85dd691ab25e0321914",
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
    for target, replaced_sha in OVERLAYS.items():
        if spec["source_files"].get(target) != replaced_sha:
            sys.exit(f"{target} is no longer the file this patch was written against; rebase the k2a-pb1 patch")
        if target in spec["overlays"] and spec["overlays"][target]["sha256"] != replaced_sha:
            sys.exit(f"k2a's overlay of {target} is not the one this patch was written against")
        data = (HERE / "overlay" / target).read_bytes()
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
        "ported_patches": [
            "https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/blob/main/zig/patches/0005-prefill-interleave.patch",
            "https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/blob/main/zig/patches/0006-image-prompt-cache.patch",
        ],
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
