# llama.cpp ROCm gfx1201 AVX2 server

`linux/amd64` llama.cpp `llama-server` built from pinned upstream source for AMD
RDNA4 GPUs (`gfx1201`, e.g. the Radeon AI PRO R9700), with an **AVX2 CPU
baseline**.

## Why this image exists

Stock upstream llama.cpp ROCm toolbox images have been shipping a `libggml-cpu`
compiled with an AVX-512 CPU baseline. On an inference host whose CPU has AVX2
but **not** AVX-512, `ggml_cpu_init()` executes an AVX-512 instruction at
process startup, and `llama-server` dies with an illegal instruction (SIGILL)
before it loads a model or touches the GPU — the whole serving stack
crash-loops.

This image removes that hazard at the source: it compiles llama.cpp with
`GGML_NATIVE=OFF` and an explicit `x86-64-v3` (AVX / AVX2 / FMA / F16C) baseline
with **all AVX-512 code paths disabled**, so the same binary runs on AVX2-only
hosts. The GPU backend still targets `gfx1201` via ROCm. A build-time guard
fails the image if any AVX-512 (`zmm`) opcode remains in the CPU backend.

## What is built

| Input | Pin |
| --- | --- |
| Base image | `rocm/dev-ubuntu-24.04:7.2.3` (digest-pinned; matches the proven-good production ROCm 7.2.3) |
| llama.cpp | commit `9b05354ec6fb58b4e665e9a39ebc40285c015638` (build `b10433`), GitHub archive verified by SHA-256 |
| GPU target | `gfx1201` (RDNA4) |
| CPU baseline | `x86-64-v3` — AVX, AVX2, FMA, F16C; AVX-512 off |
| Binaries | `llama-server`, `llama-cli` under `/opt/llama.cpp` |

Every inference binary is compiled here from the pinned llama.cpp commit;
nothing is inherited prebuilt. The exact inputs and build flags are recorded in
[`dependency.lock.json`](dependency.lock.json).

## Status

Source-build candidate; hardware qualification pending. The image is
`build_enabled` so CI can produce an immutable `sha-<commit>` candidate; promote
to a release tag only after it is validated on `gfx1201` hardware.

## Running

This is an engine image. GGUF model weights are **not** included — supply them
at runtime through a read-only bind mount and pass the full `llama-server`
command (model path, ports, sampling) yourself. `llama-server` is on `PATH`.

```bash
docker pull ghcr.io/petabridge/llama-cpp-rocm-gfx1201-avx2:sha-<12-character-commit>
```

Deployments should reference the published manifest digest rather than a tag:

```text
ghcr.io/petabridge/llama-cpp-rocm-gfx1201-avx2@sha256:<digest>
```

## License

Repository-authored material is MIT. llama.cpp is MIT and its license is
retained in the image at `/opt/llama.cpp/share/doc/llama.cpp/LICENSE`; see
[`attribution/`](attribution/THIRD_PARTY_NOTICES.md). The ROCm base image and
its components remain subject to their respective licenses.
