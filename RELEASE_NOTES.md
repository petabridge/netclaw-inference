# Release notes

Repository releases may contain one or more independently promoted inference
images. Every release section lists the image ids affected by that release.

## v0.1.0 (Unreleased)

**Initial Release**

Images:

- `vllm-deepseek-v4-flash-0731-dspark-gb10`

Changes:

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
