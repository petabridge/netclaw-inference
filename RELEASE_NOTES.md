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

---
