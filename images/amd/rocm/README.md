# AMD ROCm images

`linux/amd64` ROCm inference images for AMD RDNA4 GPUs (`gfx1201`, e.g. the
Radeon AI PRO R9700).

## Images

- [`llama-cpp-gfx1201-avx2`](llama-cpp-gfx1201-avx2/README.md) — AVX2-baseline
  llama.cpp `llama-server` built from pinned upstream source. Exists because
  stock upstream ROCm toolbox images ship an AVX-512 CPU backend that SIGILLs on
  AVX2-only hosts at startup.

These images use the same manifest, candidate, SBOM/provenance, and digest-only
promotion gates as the DGX Spark family. Candidate builds run natively on x64
GitHub-hosted runners; the build workflow selects the runner from each
manifest's `platform`.
