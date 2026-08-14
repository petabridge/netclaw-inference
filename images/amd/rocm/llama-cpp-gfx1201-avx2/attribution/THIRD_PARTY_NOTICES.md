# Third-party notices

This image bundles software from the projects below. Each remains subject to
its own license.

## llama.cpp

- Project: https://github.com/ggml-org/llama.cpp
- Pinned commit: `9b05354ec6fb58b4e665e9a39ebc40285c015638` (build `b10433`)
- License: MIT
- The upstream MIT license text is retained in the image at
  `/opt/llama.cpp/share/doc/llama.cpp/LICENSE`, copied from the pinned source
  tree at build time.

llama.cpp vendors the `ggml` tensor library (also MIT), whose notices are
included in the upstream source tree referenced above.

## ROCm base image

- Base: `rocm/dev-ubuntu-24.04` (digest-pinned in the image `Dockerfile` and
  `dependency.lock.json`)
- The ROCm runtime, hipBLAS, and rocBLAS components are distributed by AMD under
  their respective licenses; consult the base image for the complete texts.

## Repository-authored material

The `Dockerfile`, manifests, and build recipe in this directory are authored by
Petabridge and released under the MIT license (see the repository `LICENSE`).
