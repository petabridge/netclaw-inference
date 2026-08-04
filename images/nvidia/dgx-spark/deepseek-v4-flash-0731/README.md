# DeepSeek V4 Flash 0731 on DGX Spark

This image packages vLLM `0.26.0+dspark.sm121.3` for
`deepseek-ai/DeepSeek-V4-Flash-0731` on two NVIDIA DGX Sparks using tensor
parallelism 2. Model weights are not included.

## Build boundary

Petabridge builds the vLLM Python package from the exact upstream commit and a
vendored, checksum-locked SM121 source overlay. The build verifies the source
archive, the cumulative patch, every materialized source file, the installed
package version, and every installed runtime override.

The image intentionally inherits native artifacts from this pinned parent:

```text
ghcr.io/r0b0tlab/deepseek-v4-flash-dspark-v026-sm121
sha256:ef852781efd6278cb3d908a4298874fb8b34fe84f0e1a03181ed415ec233a9d4
```

Those inherited artifacts include CUDA 13.0.2, PyTorch 2.11.0+cu130,
FlashInfer 0.6.14 plus its SM120 sparse-MLA kernel patch, B12X 0.15.3, NCCL
2.30.4, and the native vLLM/Rust extensions. The public r0b0tlab known-fix
recipe overlays an unpublished local parent, so its native kernel stack cannot
currently be reproduced from public source alone. This recipe makes that
provenance gap explicit rather than silently depending on it.

## Runtime capabilities

- `linux/arm64`, GB10 / SM121
- two-node TP=2
- NVFP4 DS-MLA KV cache
- FlashInfer B12X MoE backend
- DSpark speculative decoding, K=6
- NCCL 2.30.4
- model revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`

Request concurrency, maximum model length, maximum batched tokens, and
`--max-num-seqs` are deployment settings and are not forced by this image.

## Reproducibility inputs

- `dependency.lock.json` records upstream commits, package versions, archive
  checksums, and the inherited parent digest.
- `patches/` contains the cumulative downstream vLLM patch.
- `source-overlay/` is the deterministic materialization of that patch over the
  pinned upstream source.
- `source-overlay.sha256` and `runtime-overlay.sha256` protect the source and
  installed runtime files.
- `manifests/` retains the original known-fix and runtime audit manifests.

The image remains a candidate until it passes dual-Spark startup, health,
correctness, and sustained-concurrency qualification. Promotion always uses
the exact tested digest.
