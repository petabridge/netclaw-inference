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
  spill v2, step P2: the disk block store, inert (MiaAI-Lab zig/patches/0013-core-spill-store.patch, from wojo's GLM
  0088/0110 store; nothing opens a store until a later step wires it)
    core/spill/{format,fsio,mutants,root,slots,store,store_test}.zig (new files) and core/root.zig (the export):
      verbatim, except two tests (fsio's partial direct write, store_test's write faults): their simulated short
      write is a real unaligned pwrite on an O_DIRECT descriptor, which Linux refuses on ext4, so the direct
      variant skips where the kernel refuses the simulation; the buffered variants still run
  spill v2, step P3: the dsv41 host modules, compiled and tested, not wired (MiaAI-Lab 0016/0019 tip)
    families/dsv41/{fingerprint,parts,spill_state}.zig (new files): verbatim; parts.zig only as a type dependency of
      rank_wire and rank_identity (KV sharding is not ported)
    families/dsv41/rank_identity.zig (new file): world 3 added to splitText/rankSplitText (plain TP3, as prompt2d's
      split has it), plus a TP3 test
    families/dsv41/rank_wire.zig (new file, v2): verbatim outside its tests; test-only follower fields and world-3
      tests (agreement, a point missing or damaged on one follower aborting with forget on both, rank vs common fields)
    families/dsv41/dsv41.zig: exports the modules and runs their tests with the family's; link.zig: MiaAI 0016's
      bounded reply reads (readable, recvWithin) that rank_wire needs; rank_cache.zig: writeFake made pub for a test
  spill v2, step P4: the spill state layout against our model (MiaAI-Lab 0016's model side)
    families/dsv41/model.zig: SpillDev, spillGenerations, spillPacker, spillCheckResumed and spillSelfCheck from 0016;
      the small state is our Snap less its tail (the drafter rings and prompt logits, which a pre-replay disk point's
      replay and final chunk rebuild; spillSnapTail, spillClearSnapTail). At the end of Model.open every rank prints
      one "spill layout: ok|MISMATCH ..." line: host arithmetic only while spill is off, never failing the start
      (TF_DS_SPILL_SELF_CHECK=1 adds a device round trip on the fresh pool)
    families/dsv41/dsv41.zig: 0016's compile references for the model side
  spill v2, step P5: the cache tier in k2a's prompt cache and lane host (MiaAI-Lab 0014, by hand), no tier bound
    core/prompt_cache.zig (k2a's file): prompt_cache.Tier, durable lookup with the grid and window rules (disk points
      only, so k2a's off-grid memory resumes stay), restore bytes in the budget, chats continued (0111) for each disk
      point's conversation; salted (namespaced) requests never reach a tier; every memory drop (evictBudget,
      prune_dead, dead_at_keep, a failed resume) calls Tier.release, only forgetPoint calls Tier.forget; a cut the
      --learn imprint learns is never written to a tier, a point it holds never restored
    core/lane_host.zig: restore parking (TF_SPILL_LOAD_MS, TF_SPILL_LOAD_WAIT_MS), polled on the engine thread after
      the 0005 interleave's rounds; flush at stop; lone-driver pumps counted only while a tier is servicing
    core/engine_api.zig (k2a's file): 0014's status and request additions
    core/prompt_cache_fake_tier.zig, prompt_cache_tier_test.zig, lane_host_tier_test.zig (new files): the tip tests,
      bound to k2a's live/compatible where the tip used usable, 0020's full hook dropped
    core/spill_p5_k2a_test.zig (new file): a salted request never reaches the tier; drops release, never forget
  spill v2, step P6: spill wired into dsv41 on rank 0 and the followers (MiaAI-Lab 0016, 0019's spill half, 0020's
  exit on a broken agreement), off unless spill flags are given
    families/dsv41/spill_tier.zig (new file, tip) and lanes.zig: Kind.rank_wire = 10, spillReserve, loading Snaps,
      the follower SpillFollower in follow(); tier.bump before every KV-pool write (fill chunks, verify and draft rows
      including the 0005 interleave's rounds, copyExtent; widened to ratio-2 groups and the extent's scratch row) and
      bumpSlot on slot reuse; staling and drops guard pending captures first; with the tier on, a pass that resumes
      off the 2,048 grid cuts its first chunk at the next grid boundary and only grid points go to disk; complete,
      decoded, off-grid and picture points stay memory-only; a restored state enters retention's probation
    families/dsv41/native.zig, lanes_main.zig, native/cuda.zig, build/cuda.zig (k2a's or upstream's files):
      openSpill, spillPreflight (worlds 2 and 3; TF_DS_KV_SHARD refused unless unset or 0), the rank exchange, the
      flags and env, the tier bound to the prompt cache; rank 0 exits 75 when the ranks' agreement breaks
    server/cli.zig, engines.zig, metrics.zig, status_routes.zig: 0014's deferred server hunks (the spill flags, the
      tier's counters in /health and /metrics when a tier is bound)
    core/spill/{fsio,store,store_test}.zig: 0016's read-failure slot fix and 0019's scan-progress hooks
    core/lane_host.zig: picture requests never ask the tier
    families/dsv41/spill_p6_test.zig, spill_tier_test.zig (new files): the grid realignment and refusal rules, world 3
  spill v2, per-rank Engram shards (Petabridge): with TP3 each rank holds its own Engram shard folder
    families/dsv41/fingerprint.zig, rank_identity.zig: the common engram field is the recorded revision and the kit's
      engram.json digest; the shard folder's listing moves to a new rank-scoped field, engram_shard (31), so the ranks
      agree while each rank's store still names its own shards
    families/dsv41/rank_wire.zig: the world-3 loopback test with per-rank shards (agree) and a differing revision (refused)
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
    "zig/src/core/root.zig": "ffcf279a324ad3a236fdb6aac3aed63ce295aa02dcd0037fc86d922dd19e5917",
    "zig/src/families/dsv41/dsv41.zig": "e0be7b5fb7afcd6945eb1512073cc2f4e5d38052e624b1fa1682e20c56086e7a",
    "zig/src/families/dsv41/link.zig": "bd0509c979e4adac66a3705657ed2e6155a35969153da91ce1b26cf91cb3f6ab",
    "zig/src/families/dsv41/rank_cache.zig": "a7e6794e4ecf7c20bcce1bb9effef374382be53794d7578f815b26b458ca529e",
    "zig/src/core/prompt_cache.zig": "9446351e980bd206837ab625bfa521f51a4bb2f4feac9e79bf6c5a7b4bf936a1",
    "zig/build/cuda.zig": "c7ca819827054bc4d4e0fc68be7f3b485d2579cfe61c46df7404389fc742c215",
    "zig/src/families/dsv41/lanes_main.zig": "9f27f79e4520008471556ef9deba6db7d1bc9edc2f1ad5fd1ac48f8a1582be80",
    "zig/src/families/dsv41/native.zig": "1cd93d5d40d3c6d9dab7f263c97ba640c358e87c2ebcd8d5f447b61a816d7b20",
    "zig/src/native/cuda.zig": "fa9efe5e0edbd464f6dd97434fa7029b906fc6cf63de6adcce117294f56853fa",
    "zig/src/server/cli.zig": "255728ac09811a0b3f865cd4afb5367cc2ae09535ae93a4beb1f7fa3c92747f8",
    "zig/src/server/engines.zig": "34d357c8054029cc3ffacc0e7ce207a00b041cf637b35e58b9a7e02f3f680265",
    "zig/src/server/metrics.zig": "ead6c8493fe0a705aba3bbf0d18cad7b5f75d81f77969913d686ec584fa8c212",
    "zig/src/server/status_routes.zig": "bc11bb724d33221c03d5e4df6ecbf9467fab76058395d4450637fa8c5b12d246",
}
# Files k2a-pb1 adds that upstream's tree does not have (their file is overlay/<target path>): each must be absent
# from k2a.json's source_files, so an upstream that starts shipping one makes this script stop instead of clobbering it.
NEW_FILES = [
    "zig/src/core/spill/format.zig",
    "zig/src/core/spill/fsio.zig",
    "zig/src/core/spill/mutants.zig",
    "zig/src/core/spill/root.zig",
    "zig/src/core/spill/slots.zig",
    "zig/src/core/spill/store.zig",
    "zig/src/core/spill/store_test.zig",
    "zig/src/families/dsv41/fingerprint.zig",
    "zig/src/families/dsv41/parts.zig",
    "zig/src/families/dsv41/rank_identity.zig",
    "zig/src/families/dsv41/rank_wire.zig",
    "zig/src/families/dsv41/spill_state.zig",
    "zig/src/core/prompt_cache_fake_tier.zig",
    "zig/src/core/prompt_cache_tier_test.zig",
    "zig/src/core/lane_host_tier_test.zig",
    "zig/src/core/spill_p5_k2a_test.zig",
    "zig/src/families/dsv41/spill_p6_test.zig",
    "zig/src/families/dsv41/spill_tier.zig",
    "zig/src/families/dsv41/spill_tier_test.zig",
]


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
    listed = set(OVERLAYS) | set(NEW_FILES)
    if len(listed) != len(OVERLAYS) + len(NEW_FILES):
        sys.exit("a target is in both OVERLAYS and NEW_FILES")
    present = {p.relative_to(HERE / "overlay").as_posix() for p in (HERE / "overlay").rglob("*") if p.is_file()}
    if present != listed:
        sys.exit(f"overlay/ and the lists disagree: unlisted {sorted(present - listed)}, missing {sorted(listed - present)}")
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
    for target in NEW_FILES:
        if target in spec["source_files"]:
            sys.exit(f"upstream now has {target}; move it from NEW_FILES to OVERLAYS with its sha256 and rebase")
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
        "new_files": sorted(NEW_FILES),
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
