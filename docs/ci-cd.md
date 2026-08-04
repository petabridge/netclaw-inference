# CI/CD and release gates

## Why validation and publishing use different runners

Pull-request code runs on a GitHub-hosted runner. The ARC runners can reach the
tailnet and private registry, so they are reserved for trusted, master-only
workflows. This prevents an unmerged workflow or script from gaining testlab
network access.

Registry jobs use the same `arc-petabridge-x64` scale set as `netclaw-images`.
The scale set supplies an ephemeral runner and a privileged Docker-in-Docker
sidecar. GitHub's `private-testlab` runner group must explicitly allow this
private repository.

## Candidate build

Run **Build inference image candidate** from the Actions tab on `master` and
provide the path to an enabled `image.json`. The workflow:

1. validates the repository on a GitHub-hosted runner;
2. enters the master-only `inference-image-build` environment;
3. starts an ephemeral ARC runner inside the testlab network;
4. configures QEMU and Buildx for the manifest's target platform;
5. builds and pushes `sha-<12-character commit>`;
6. publishes BuildKit SBOM/provenance attestations;
7. verifies the resulting digest and target platform; and
8. stores BuildKit metadata as a workflow artifact.

The workflow never publishes `latest`, a release tag, or `validated`.

## Promotion

Run **Promote inference image digest** on `master`. Supply the same manifest
path, the exact candidate digest, and a new semantic version such as `v0.1.0`.
The master-only `inference-image-promote` environment then creates the new tag
from the existing manifest digest using `docker buildx imagetools create`.

Promotion fails if the destination tag already points to a different digest.
This makes releases idempotent and prevents silent tag replacement.

The current GitHub plan does not support required reviewers on private-repo
environments. The enforced gates are therefore the protected `master` branch,
disabled-by-default image definitions, explicit manual dispatch, immutable
candidate tags, and the independent exact-digest promotion workflow.

## Builder capacity

The current ARC Docker sidecar is x64 with a 2 CPU / 8 GiB limit. It is suitable
for validation, registry operations, ordinary x64 images, and thin ARM64
overlays. A full ARM64 CUDA/vLLM source build under QEMU is expected to be slow
and may exceed that memory limit. Before enabling the DGX Spark definition,
qualify one of these build strategies:

- increase the gated ARC builder's resources;
- attach a native ARM64 remote BuildKit worker; or
- add a separate native ARM64 ARC scale set while retaining the same workflow
  and registry gates.

Do not silently move the full build onto a production Spark node. Builder
infrastructure belongs in Git and must have explicit resource isolation.
