# Release notes

Repository releases may contain one or more independently promoted inference
images. Every release section lists the image ids affected by that release.

## v0.2.0 (2026-10-10)

Images:

- `vllm-glm-5-3-flash-exl3-gb10`
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
- **Add GLM-5.3-Flash EXL3 for two DGX Sparks (TP=2)**
  - Vendor the MiaAI-Lab GLM-5.3-Flash-EXL3-2x-DGX-Sparks recipe unmodified at
    commit `674155de`, with a per-file hash lock of the upstream tree.
  - Pin the base by literal digest, the ExLlamaV3 source archive and the
    InstantTensor wheel by checksum, and bake the files upstream `start.sh`
    bind-mounts at runtime plus its generated per-rank launch scripts, so the
    image runs under a plain compose deployment.
  - Record the AGPL-3.0 recipe license and the runtime weights' ShapleyMCG and
    CC BY-NC-ND terms in the image folder.

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
