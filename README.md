# Netclaw Inference Images

Public, reproducible container images for running open inference workloads on
Petabridge hardware. Images are built on GitHub-hosted runners and published to
the public GitHub Container Registry under [`ghcr.io/petabridge`](https://github.com/orgs/petabridge/packages).

This repository contains build recipes and release automation, not model
weights, credentials, hostnames, or private network configuration.

## Built for NetClaw

These images are built first to power [Netclaw](https://netclaw.dev/),
Petabridge's open-source AI agent. Netclaw runs on your hardware, connects to
the communications tools your team already uses, and keeps agent execution and
inference under your control. The images remain usable as standalone inference
runtimes wherever their documented hardware and model requirements fit.

## Images

| Image | Platform | Intended hardware | Status |
| --- | --- | --- | --- |
| [`vllm-deepseek-v4-flash-0731-dspark-gb10`](images/nvidia/dgx-spark/deepseek-v4-flash-0731/README.md) | `linux/arm64` | Two NVIDIA DGX Sparks, TP=2 | Source-build candidate; hardware qualification pending |
| AMD ROCm family | `linux/amd64` | Future AMD inference hosts | Reserved; no image built yet |

The DGX Spark image targets `deepseek-ai/DeepSeek-V4-Flash-0731`. Model weights
are downloaded separately and should be pinned to the revision recorded in the
image's `dependency.lock.json`.

## What Petabridge builds

The first image started as a thin, labeled copy of an already exercised
r0b0tlab runtime. The current recipe goes materially further:

1. Downloads vLLM source from the immutable upstream commit recorded in the
   lock file and verifies the source archive checksum.
2. Applies the repository-owned, checksum-locked DeepSeek V4 / SM121 source
   overlay derived from the cumulative public patch.
3. Builds and installs a new vLLM Python package from that source.
4. Verifies every installed runtime overlay file before the image is emitted.

The native CUDA, FlashInfer, B12X, NCCL, and vLLM extension artifacts are still
inherited from one pinned r0b0tlab image digest. That boundary is intentional:
the public upstream recipe does not contain enough material to reproduce its
patched FlashInfer kernel parent byte-for-byte. Retaining the proven native
stack gives us a testable source-owned vLLM image without pretending those
native inputs are Petabridge-built. The exact boundary and every known version
are documented in the image lock file.

## Pulling images

Candidates use an immutable source-commit tag:

```bash
docker pull ghcr.io/petabridge/vllm-deepseek-v4-flash-0731-dspark-gb10:sha-<12-character-commit>
```

Deployments should use the published manifest digest rather than a tag:

```text
ghcr.io/petabridge/vllm-deepseek-v4-flash-0731-dspark-gb10@sha256:<digest>
```

Release tags are added only by promoting an existing candidate digest. The
promotion workflow never rebuilds an image.

## Repository layout

```text
images/
  nvidia/dgx-spark/   ARM64 / GB10 / SM121 images
  amd/rocm/           x64 ROCm images (reserved for future use)
scripts/              Manifest, source-overlay, and release validation
.github/workflows/    Public-runner validation, builds, and promotion
```

Each buildable image has a declarative `image.json`, a digest-pinned
Dockerfile, dependency locks, retained licenses, and any downstream source
patches or overlays needed to reproduce it.

## CI/CD and supply chain

- Pull requests perform static validation only; they never receive registry
  credentials and never publish images.
- Candidate builds run only from `master` on GitHub-hosted ARM64 runners.
- GitHub Actions and external base images are pinned by full commit or digest.
- Candidate tags are immutable `sha-<commit>` tags.
- BuildKit publishes an SBOM and maximum-mode provenance attestation.
- Promotion accepts an exact digest and adds a version tag without rebuilding.

See [the CI/CD contract](docs/ci-cd.md) and [release notes](RELEASE_NOTES.md) for
the operational and release process.

## License and attribution

Repository-authored material is MIT licensed. vLLM is Apache-2.0, and the
image retains upstream notices and third-party attribution alongside the
runtime. Individual dependencies remain subject to their respective licenses.
