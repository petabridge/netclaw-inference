# Release notes

Repository releases may contain one or more independently promoted inference
images. Every release section lists the image ids affected by that release.

## v0.1.0 (Unreleased)

**Initial Release**

Images:

- `vllm-deepseek-v4-flash-0731-dspark-gb10`

Changes:

- **Bootstrap gated, multi-platform inference image CI/CD** (#1)
  - Add separate NVIDIA DGX Spark and future AMD ROCm image families.
  - Add immutable candidate builds, SBOM/provenance, and digest-only promotion.
- **Plan the first DeepSeek V4 Flash 0731 DGX Spark image**
  - Keep the image disabled until its vLLM and SM121 dependency lock is audited.

---
