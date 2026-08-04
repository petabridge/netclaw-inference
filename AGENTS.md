# Agent instructions

This repository builds inference container images for Petabridge hardware.

## Safety and supply chain

- Never add secrets, model weights, Hugging Face tokens, registry credentials,
  Tailscale credentials, or machine-specific addresses.
- Pin every external base image by `sha256` digest. `latest`, floating tags,
  and Dockerfile base-image build arguments are prohibited.
- Pin source dependencies by immutable Git commit and retain their licenses.
- Do not publish images from pull-request workflows.
- Candidate builds must originate from `master` and use `sha-<commit>` tags.
- Promote an already-built digest; never rebuild during promotion.
- Treat the AMD and NVIDIA image families independently. Do not assume that a
  CUDA setting applies to ROCm or vice versa.

## Changes

- Use feature branches and pull requests. Do not push changes directly to
  `master` after the initial repository bootstrap.
- Update the image README and lock data whenever runtime behavior changes.
- Run `./scripts/validate-repository.sh` before committing.
- Shell scripts use `#!/bin/bash`, `set -euo pipefail`, quoted paths, and usage
  checks.

## Image definitions

- NVIDIA DGX Spark images live under `images/nvidia/dgx-spark/` and target
  `linux/arm64`.
- AMD ROCm images live under `images/amd/rocm/` and target `linux/amd64`.
- Each buildable directory contains `image.json`, `Dockerfile`, `README.md`,
  dependency lock data, and all downstream patches or overlays.
- Set `build_enabled` to `false` while an image definition is incomplete or
  unqualified. Enabling it requires all referenced build files to exist.
