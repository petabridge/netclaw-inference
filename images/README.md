# Image families

Image definitions are grouped by accelerator vendor and hardware family.

- [`nvidia/dgx-spark`](nvidia/dgx-spark/README.md) — ARM64 CUDA images for GB10
  and SM121.
- [`amd/rocm`](amd/rocm/README.md) — reserved for future x64 ROCm derivatives.

An image directory becomes buildable only when its `image.json` sets
`build_enabled` to `true` and all referenced files pass repository validation.
