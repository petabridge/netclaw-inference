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
| llama.cpp | commit `d81235049384534c167caea52b85a694f6103d14` (build `b11429`, v0.6.0), GitHub archive verified by SHA-256 |
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

### Migrating to llama.cpp v0.6.0

The CLI is part of the deployment contract. v0.6.0 removed `--no-mmap`;
replace it with `--load-mode none` to preserve loading without memory mapping.
Leaving the old flag in a deployment makes `llama-server` exit before loading
the model. `--load-mode auto` is the default and may enable memory mapping.

The image build parses a representative multimodal serving command with
`--load-mode none`, q8_0 KV, two slots, reasoning disabled, and
`--spec-type draft-mtp --spec-draft-n-max 3`. It appends `--help`, so this
check needs neither model weights nor GPU hardware and fails on unsupported
arguments. It does not establish model compatibility, GPU health, or draft
acceptance; those still require hardware qualification before promotion.

`draft-dflash` is a separate speculative decoding implementation, not a
replacement name for `draft-mtp`. It needs a compatible DFlash draft GGUF,
supplied with `--model-draft`, containing the target-layer metadata expected
by that implementation. An embedded MTP head alone does not qualify a model
for DFlash. Keep `draft-mtp` when upgrading an existing embedded-MTP deployment;
qualify DFlash with its draft model in a separate deployment change.

For either mode, qualification must verify stable container state, readiness,
non-empty chat completions, memory usage under representative load, and actual
draft acceptance in the server logs. A successful image build or container
start alone is insufficient.

## License

Repository-authored material is MIT. llama.cpp is MIT and its license is
retained in the image at `/opt/llama.cpp/share/doc/llama.cpp/LICENSE`; see
[`attribution/`](attribution/THIRD_PARTY_NOTICES.md). The ROCm base image and
its components remain subject to their respective licenses.
