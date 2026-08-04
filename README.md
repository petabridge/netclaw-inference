# NetClaw Inference Images

Reproducible, validated container images for Petabridge inference hardware.
Images are published to the tailnet-only registry at
`docker.testlab.petabridge.net`.

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

The first planned image is
`images/nvidia/dgx-spark/deepseek-v4-flash-0731/image.json`. It is intentionally
disabled until the DeepSeek V4 Flash 0731-native vLLM, FlashInfer, b12x, and
DSpark revisions have been selected and audited.

## CI/CD contract

1. Pull requests run static validation on a GitHub-hosted runner. PR code does
   not execute inside the testlab tailnet.
2. Candidate builds are manual, master-only, and scoped to the
   `inference-image-build` GitHub environment.
3. Candidate images receive only an immutable `sha-<commit>` tag.
4. Promotion is a separate manual workflow. It accepts an exact
   `sha256:<digest>` and adds a semantic-version tag without rebuilding.
5. BuildKit emits an SBOM and maximum-mode provenance with the candidate image.

See [docs/ci-cd.md](docs/ci-cd.md) for operational details.

## Registry naming

Image manifests use repository paths below the shared registry, for example:

```text
docker.testlab.petabridge.net/petabridge/inference/deepseek-v4-flash-0731-dspark
```

Deployments should pin the resulting manifest digest, not a mutable tag.
