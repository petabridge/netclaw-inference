# CI/CD and release gates

## Public runner boundary

All workflows use GitHub-hosted runners. The public repository has no access to
Petabridge self-hosted runners, private registries, or private networks.

## Candidate build

Run **Build inference image candidate** from the Actions tab on `master` and
provide the path to an enabled `image.json`. The workflow:

1. validates the repository on a GitHub-hosted runner;
2. enters the master-only `inference-image-build` environment;
3. starts a native GitHub-hosted ARM64 runner;
4. configures pinned BuildKit tooling for the manifest's target platform;
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

Promotion also requires a matching dated section in `RELEASE_NOTES.md`. The
section must list the selected image id. An `Unreleased` entry cannot be
promoted; preparing the dated notes is a reviewed release-preparation change.

Build-affecting pull requests must update `RELEASE_NOTES.md`. This includes
Dockerfiles, image manifests, dependency locks, patches, overlays, and the
build/promotion scripts. Documentation-only image changes do not require a
release-note entry.

The enforced gates are the protected `master` branch, disabled-by-default image
definitions, explicit manual dispatch, immutable candidate tags, and the
independent exact-digest promotion workflow.

## Builder capacity

Candidate builds use native GitHub-hosted ARM64 capacity. The first full build
must establish whether the selected runner has sufficient memory and scratch
space for the CUDA/vLLM source build; if it does not, move to a larger
GitHub-hosted ARM64 runner without granting the repository private-network
access.
