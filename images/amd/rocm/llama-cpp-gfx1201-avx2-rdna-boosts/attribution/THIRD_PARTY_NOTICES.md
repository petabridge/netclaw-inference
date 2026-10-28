# Third-party notices

This image bundles software from the projects below. Each remains subject to
its own license.

## llama.cpp

- Project: https://github.com/ggml-org/llama.cpp
- Pinned commit: `6050b467c07a95c7b87debd4ddac2f8ba8d9b807` (2026-10-25)
- License: MIT
- The upstream MIT license text is retained in the image at
  `/opt/llama.cpp/share/doc/llama.cpp/LICENSE`, copied from the pinned source
  tree at build time.

llama.cpp vendors the `ggml` tensor library (also MIT), whose notices are
included in the upstream source tree referenced above.

## rdna-boosts patch series

- Project: https://github.com/stew675/llama-cpp-rdna-boosts
- Release: `v15-6050b467c-r161`
- Release commit: `ecf48ff08197d92bb88f1642aab72eeefd87cd99`
- License: MIT (full text retained at `patches/LICENSE` and in the image at
  `/usr/share/doc/netclaw-inference/rdna-boosts/LICENSE`)
- Author: stew675 and contributors.

The patch series is **vendored byte-for-byte** under `patches/`. Every file's
SHA-256 is recorded in `patches/SHA256SUMS` and in `dependency.lock.json`, and
is re-verified during the image build before any patch is applied. The patches
are applied to a pinned upstream llama.cpp tree with `git am`; the patches are
not modified.

Blocks 00-08 are applied. Blocks 09-14 are not vendored; the reason for each
exclusion is recorded in `dependency.lock.json` under
`source_build.patch_series.excluded_blocks`.

## ROCm base image

- Base: `rocm/dev-ubuntu-24.04` (digest-pinned in the image `Dockerfile` and
  `dependency.lock.json`)
- The ROCm runtime, hipBLAS, and rocBLAS components are distributed by AMD under
  their respective licenses; consult the base image for the complete texts.

## Repository-authored material

The `Dockerfile`, manifests, and build recipe in this directory are authored by
Petabridge and released under the MIT license (see the repository `LICENSE`).
