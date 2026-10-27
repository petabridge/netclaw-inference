# Release notes

Repository releases may contain one or more independently promoted inference
images. Every release section lists the image ids affected by that release.

## v0.3.0 (Unreleased)

Images:

- `tensorfold-deepseek-v4-1-flash-1017-gb10` (candidate; not promotable until GPU qualification passes)
- `tensorfold-qwen3-8-flash-gb10` (candidate; not promotable until GPU qualification passes)

Changes:

- **Spill v2, step P2: add the disk block store to the DeepSeek V4.1 Flash 1017 image, inert (recipe `k2a-pb1`).**
  The first step of porting MiaAI-Lab's spill-to-disk prompt cache (`0013-core-spill-store`) onto the TP3
  K2a engine: seven new files under `zig/src/core/spill/` and their export, verbatim. Nothing opens a store
  yet, so serving is unchanged. Two of its tests simulate a short write with a real unaligned `pwrite` on an
  O_DIRECT descriptor, which Linux refuses on ext4; their direct variant now skips where the kernel refuses
  the simulation. `make_recipe.py` gains `NEW_FILES` for overlay files upstream does not have.
- **Fix a stale Engram read-ahead in the DeepSeek V4.1 Flash 1017 image (recipe `k2a-pb1`).** The prefill
  interleave port (MiaAI-Lab 0005) lets `fillChunk` reuse an already-joined read-ahead with a matching
  `start` and `n`. A pass that ended early between chunks left that state set, so a later pass whose
  first chunk began at the same position (a resume there) used the other prompt's Engram rows for that
  chunk, silently degrading its output. `fillBegin` now joins and clears the read-ahead on every rank.
- **Retain resumed conversations in the DeepSeek V4.1 Flash 1017 image's prompt cache (recipe `k2a-pb1`).**
  Kept prompt states live in the shared KV pool, and upstream gave up the oldest first when a new stream
  needed room, so an agent's long main conversation, idle while its subagents ran, always lost its state
  and re-read up to 268K tokens on its next turn. `placeKept` now gives up the least recently used state
  of the lowest tier: never resumed, then superseded prefixes, then resumed conversations' newest states.
  `TF_DS_RETENTION` is on when unset; `0` restores the old order. Petabridge's own change, for
  MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks#27; one overlay (`dsv41/lanes.zig`) changes.
- **Add an image prompt cache to recipe `k2a-pb1` of the DeepSeek V4.1 Flash 1017 image.** The prompt
  cache finds kept states by token ids, and every picture's positions hold the same placeholder token,
  so a prompt with pictures was never looked up or kept. Every turn of a conversation with an image in
  its history re-read the whole prompt from 0, and because image prompts run whole, each turn froze
  every other stream: an agent conversation of about 30K tokens carrying email-attachment images froze
  all streams for 15 to 19 s per turn. The server now gives such a request a cache key with each
  picture's positions replaced by ids made from a hash of its payload, and the lane host looks up and
  keeps by that key. `TF_DS_IMAGE_CACHE` is on when unset and off at `0`. Image prompts still run
  without the prefill interleave, but only prefill the new part of each turn. Six new overlay targets
  (`server/prompt.zig` and the patch's test among them) and six replaced k2a overlays bring `k2a-pb1`
  to 33 overlays, and every overlay file now lives at its target path under `pb1/overlay/`. Ported from
  MiaAI-Lab's `zig/patches/0006-image-prompt-cache.patch`; `lane_host.zig` is ported by hand.
- **Add an opt-in prefill interleave to recipe `k2a-pb1` of the DeepSeek V4.1 Flash 1017 image.**
  Without it, every prompt pass froze all decoding streams for its full length, at about 2.2K prompt
  tok/s, so a cold 135K-token prompt froze every reply for about a minute. With
  `TF_DS_PREFILL_INTERLEAVE_MS` set (the deployment sets it), the head runs decode rounds for the other
  live streams between the 2,048-row chunks of a prompt pass, for up to that many ms. Unset or 0, the
  engine is unchanged; prompts with images run whole, and the interleave is off while the core logs a
  trail. Six overlay files (`pb1/interleave/`) bring `k2a-pb1` to 31 overlays. Ported from MiaAI-Lab's
  `zig/patches/0005-prefill-interleave.patch` (author Coinupbtc); `lane_host.zig` is ported by hand
  because the patch's hunks misapplied onto k2a.
- **Build the DeepSeek V4.1 Flash 1017 image from recipe `k2a-pb1`, which returns tool calls to unoffered
  tools.** Upstream's `k2a` rewrote a DSML invoke naming a tool the request did not offer into the reply
  text, so a streamed reply with no reasoning came back as raw DSML text (`finish_reason=stop`) and one
  with reasoning lost the call. `k2a-pb1` is `k2a` plus two overlay files (`tool_parse.zig`,
  `tool_parse_test.zig`) that return the call under its written name when calls are unlimited;
  `parallel_tool_calls: false` keeps upstream's behaviour. Launcher profiles must now use
  `build_variant: "k2a-pb1"`. Reported upstream as MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks#67.
- **Add Pillow 12.3.0 to the DeepSeek V4.1 Flash 1017 image.** The engine decodes non-PNG images
  through a `python3` child that imports Pillow (`dsv41/pil.zig`); without it the engine logged
  `vision: no Pillow (NoPillow): images other than PNG are refused` and rejected JPEG/WebP inputs.
  Pinned to the version the qualified serving image used, installed hash-locked from the manylinux
  wheel, with a build-time JPEG/WebP encode-decode check.
- **Fix the DeepSeek V4.1 Flash 1017 candidate build:** install curl
  (8.5.0-2ubuntu10.16, from the pinned snapshot) in the NCCL and tools stages.
  NVIDIA's CUDA images purge curl after setup, so the first candidate build
  failed at the NCCL download.
- **Fix a broken line continuation in the DS41F tools stage** that passed a
  stray `n` to apt (second candidate build).
- **Add a DeepSeek V4.1 Flash 1017 TensorFold image definition for three DGX Sparks (candidate)**
  - Build LakyFx's DS41F 1.2.5 recipe, build recipe `k2a` (tag `v1.2.5`,
    commit `b4e476d6`), from source in container stages: NCCL `v2.30.7-1`
    for `sm_121` from a sha256-pinned tarball, the TensorFold Zig engine
    (`tensorfold-native`, `tf-dsv41-lanes`) with Zig 0.17.0, and the 10M-pool
    RoPE tables from hash-locked CPU PyTorch 2.13.0, checked against the
    upstream hashes. The upstream publishes no image for this path.
  - Digest-pinned `nvcr.io/nvidia/cuda:13.0.3` devel (NCCL stage) and runtime
    (final image) arm64 bases; Ubuntu packages come from a dated apt snapshot
    at exact versions; no base-image build arguments.
  - The final image carries the engine binaries and build receipt, the kernel
    kit with the RoPE tables, `libcublas.so.13`/`libcublasLt.so.13` from the
    CUDA base, NCCL 2.30.7 in place of the base image's 2.28.3, libibverbs
    with the mlx5 provider, and the launcher files under `/opt/ds41f/recipe`
    (run with `python3 -m tensorfold.launch`). Weights are bind-mounted.
  - The build checks the library loads, `ncclGetVersion`, the binaries against
    `BUILD.json`, the RoPE hashes and the launcher import path. The image
    has not been built or run on GPUs; the build is enabled so a
    `sha-<commit>` candidate can be qualified on three DGX Sparks, and
    promotion waits until those tests pass.
- **Add a Qwen3.8-Flash-Next TensorFold image definition with image prompt reuse (prototype)**
  - Vendor the MiaAI-Lab Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold recipe
    unmodified at commit `be3ed857` (37 files, per-file hash lock), and
    materialize its inline Dockerfile heredoc: digest-pinned
    `nvcr.io/nvidia/pytorch:26.07-py3` arm64 base in both stages, build
    arguments as pinned defaults, and the `tf.patches` label recorded and
    verified at build time.
  - Add Petabridge patch `0010-media-prompt-reuse`, applied after the
    recipe's 0001-0009: image and video prompts keep and resume prompt states
    by per-picture content keys instead of always prefilling from token 0; a
    fixed per-image visual-token cap (`TENSORFOLD_IMAGE_TOKENS_EACH`, default
    1024) so an added image never refits earlier ones; an optional host-side
    tower-feature cache; the media rotary table attached before a resume's
    MTP absorb; and a `"prompt_cache": false` request field for exactness
    checks.
  - Host-side tests pass (Zig unit and server tests, the CUDA server compiles
    and links, Python helper tests on CPU). GPU exactness and reuse tools are
    included under `petabridge-tools/` but have not been run. The build is
    enabled so a `sha-<commit>` candidate can be qualified on two DGX Sparks;
    promotion waits until those tests pass.

## v0.2.0 (2026-10-10)

Images:

- `tensorfold-glm-5-3-flash-gb10`
- `llama-cpp-rocm-gfx1201-avx2`

Changes:

- **Qualify the AMD v0.6.0 image with embedded MTP**
  - Verify readiness, non-thinking chat, tool calls, vision, and four
    simultaneous requests on two gfx1201 GPUs with an AVX2 CPU.
  - Retain Qwen3.8-27B UD-Q4_K_XL, q8_0 KV, two slots per server and MTP/3;
    observe draft acceptance, stable memory, and zero restarts.
  - Record the tested candidate digest in the image README. DFlash and
    extended or maximum-context qualification remain separate.
- **Check the AMD serving CLI during the image build**
  - Parse the multimodal MTP serving arguments with `--help`, catching removed
    or incompatible flags without requiring model weights or GPU hardware.
  - Document the v0.6.0 migration from `--no-mmap` to `--load-mode none` and
    the separate draft-model requirement for DFlash. Model loading and draft
    acceptance remain hardware qualification checks before promotion.
- **Bump llama.cpp to v0.6.0 for the AMD ROCm gfx1201 AVX2 image**
  - Move the upstream llama.cpp source pin from commit `9b05354` (build
    `b10433`) to `d812350` (build `b11429`), the tagged `v0.6.0` release.
  - Update the GitHub archive URL and SHA-256 checksum in the Dockerfile and
    the dependency lock; the build stays on the digest-pinned ROCm 7.2.3 base
    with the AVX2 (`x86-64-v3`) CPU baseline and AVX-512 disabled.
  - Adds the `llama_batch_ext` API and DFlash speculative decoding support
    (including `--spec-type draft-dflash`), on the same proven gfx1201
    backend as before.
- **Replace the vLLM GLM-5.3-Flash image with the MiaAI-Lab TensorFold recipe**
  - Remove `vllm-glm-5-3-flash-exl3-gb10`; the TensorFold runtime supersedes it
    for the same two-DGX-Spark deployment.
  - Vendor the MiaAI-Lab GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold recipe
    unmodified at commit `05397e24`, with a per-file hash lock of the upstream
    tree.
  - Materialize the image the upstream builds as a shell heredoc in
    `scripts/prepare.sh` as a checked-in Dockerfile: digest-pinned
    `nvcr.io/nvidia/pytorch:26.07-py3` arm64 base, no base-image build
    arguments, TensorFold pinned by commit instead of the `v0.6.0` tag, and the
    recipe's `tf.patches` kernel-cache label recorded and verified against the
    vendored patches at build time.
  - Record the recipe's Apache-2.0 license (not the AGPL-3.0 of the sibling
    MiaAI-Lab vLLM recipe), TensorFold's Apache-2.0 and pre-`v0.6.0` MIT terms,
    and the runtime weights' Apache-2.0, CC BY-NC-ND 4.0 and ShapleyMCG terms in
    the image folder.

## v0.1.0 (2026-08-14)

**Initial Release**

Images:

- `vllm-deepseek-v4-flash-0731-dspark-gb10`
- `llama-cpp-rocm-gfx1201-avx2`

Changes:

- **Add the first AMD ROCm image: AVX2-baseline llama.cpp for gfx1201**
  - Build `llama-server`/`llama-cli` from a pinned llama.cpp commit against a
    digest-pinned ROCm 7.2.3 base, targeting the `gfx1201` (RDNA4) GPU backend.
  - Compile the CPU backend at an explicit `x86-64-v3` (AVX2) baseline with all
    AVX-512 paths disabled, so the binary runs on AVX2-only hosts that stock
    upstream toolbox images crash on at startup.
  - Fail the build if any AVX-512 opcode remains in the CPU backend.
  - Run the image build under bash so the `pipefail` and `[[ ]]` build guards
    execute instead of failing under the base image's `/bin/sh`.
  - Install the `rocblas-dev`/`hipblas-dev` ROCm math libraries so ggml-hip's
    `find_package()` resolves and the runtime libraries are present; build in a
    single stage so those runtime libraries ship with `llama-server`.
  - Build the full default llama.cpp target set so `cmake --install` is
    consistent (and `libmtmd` is present for the `--mmproj` multimodal path).
  - Register `/opt/llama.cpp/lib` with `ldconfig` so `llama-server` resolves
    its own shared objects (e.g. `libllama-server-impl.so`) at run time.
- **Select the candidate build runner from the manifest's target platform**
  - Build `linux/amd64` images natively on x64 runners and `linux/arm64` images
    on ARM64 runners, instead of hardcoding an ARM64 builder.
- **Warm all observed DeepSeek V4 portable sparse-MLA layouts**
  - Add the production 512-wide Triton specialization to the existing 576-wide
    startup warmup for both 32- and 64-head TP-local layouts.
  - Prevent the first long request from compiling the missed multihead
    accumulate kernel after the API starts.
- **Publish public images through GitHub Container Registry**
  - Remove the private registry and self-hosted runner dependency.
  - Build on native GitHub-hosted ARM64 and promote through GitHub-hosted x64.
- **Bootstrap gated, multi-platform inference image CI/CD** (#1)
  - Add separate NVIDIA DGX Spark and future AMD ROCm image families.
  - Add immutable candidate builds, SBOM/provenance, and digest-only promotion.
- **Plan the first DeepSeek V4 Flash 0731 DGX Spark image**
  - Pin the proven r0b0tlab SM121 runtime digest and its complete provenance.
  - Enable a fast-path downstream candidate for immediate dual-Spark testing.
- **Rebuild the patched vLLM package from pinned source**
  - Materialize and checksum the cumulative DeepSeek V4 / SM121 source overlay.
  - Build a new vLLM package while retaining the proven native CUDA, FlashInfer,
    B12X, NCCL, and extension artifacts from the exact pinned parent digest.
  - Pin the build-only `setuptools-rust`, `setuptools-scm`, and
    `semantic-version` wheels by version, URL, and SHA-256 instead of resolving
    them dynamically.
  - Preserve the qualified `0.26.0+dspark.sm121.3` package identity when
    reusing native artifacts, matching vLLM's upstream container build policy.
  - Install the rebuilt package as an overlay and verify every inherited native
    vLLM binary remains present and byte-identical after installation.
  - Verify installed runtime source hashes during the image build.
- **Document the public project and its supply-chain boundary**
  - Add image status, public pull guidance, security properties, and attribution.
  - Explicitly identify which artifacts Petabridge builds and which remain
    inherited pending complete public kernel provenance.

---
