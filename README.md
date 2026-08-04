# NetClaw Inference Images

Reproducible, validated, open-source container images for Petabridge inference
hardware. Images are published publicly through GitHub Container Registry.

## Repository layout

```text
images/
  nvidia/dgx-spark/   ARM64 / GB10 / SM121 images
  amd/rocm/           x64 ROCm images (reserved for future use)
scripts/              Manifest validation, builds, and digest promotion
.github/workflows/    Cheap validation and gated registry operations
```

Every buildable image has an `image.json` manifest. The manifest identifies the
target platform, Dockerfile, build context, and registry repository without
executing repository-controlled shell as configuration.

The first image is
`images/nvidia/dgx-spark/deepseek-v4-flash-0731/image.json`. Its initial
candidate is a provenance-pinned downstream build of the exact r0b0tlab
runtime already exercised on two DGX Sparks. Model weights are not included.

## CI/CD contract

1. Pull requests run static validation only on a GitHub-hosted runner.
2. Candidate builds are manual, master-only, run on a native GitHub-hosted
   ARM64 runner, and are scoped to the `inference-image-build` environment.
3. Candidate images receive only an immutable `sha-<commit>` tag.
4. Promotion is a separate manual workflow. It accepts an exact
   `sha256:<digest>` and adds a semantic-version tag without rebuilding.
5. BuildKit emits an SBOM and maximum-mode provenance with the candidate image.
6. Promotion requires a dated entry in [`RELEASE_NOTES.md`](RELEASE_NOTES.md)
   that explicitly lists the image id being released.

See [docs/ci-cd.md](docs/ci-cd.md) for operational details.

## Registry naming

Image manifests use repository paths below the shared registry, for example:

```text
ghcr.io/petabridge/vllm-deepseek-v4-flash-0731-dspark-gb10
```

Deployments should pin the resulting manifest digest, not a mutable tag.
