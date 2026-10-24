# Third-party notices

This folder builds LakyFx's DS41F 1.2.5 / K2a TensorFold recipe into a container image. It vendors
nothing: the Dockerfile fetches the upstream repository at a pinned commit and builds from it. The
upstream license and notice files are copied into the image (see "In the image" below).

Upstream: https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark, tag `v1.2.5`, commit
`b4e476d60f01fda564d760c5906ac1358390c728`.

## Components

| Component | License | Where it appears |
| --- | --- | --- |
| Recipe repository (integration code, launcher, docs, benchmark tooling) | MIT, Copyright (c) 2026 LakyFx. It does not relicense third-party code or the model. | Launcher files in `/opt/ds41f/recipe`, with `LICENSE` and `THIRD_PARTY_NOTICES.md` |
| TensorFold engine (`tensorfold-source.tar.gz`, 1,130 files, plus the 27-file K2a overlay) | Apache-2.0 from v0.6.0, Copyright 2026 TensorFold contributors (https://github.com/ashhart/TensorFold); earlier code keeps its MIT notice. The dsv41 family, RDMA ring and `tf-dsv41-lanes` are a fork by BertholomusAI (Albert Lee), Apache-2.0, with a changes list in its `NOTICE`. | `tensorfold-native`, `tf-dsv41-lanes`; `/usr/share/doc/ds41f-tensorfold/tensorfold/{LICENSE,NOTICE,THIRD_PARTY_NOTICES.md,LICENSES/}` |
| Kernel kit (`tensorfold-kernels.tar.gz`) | Triton AOT and EXL3 cubins Apache-2.0 (Triton MIT; the EXL3 format follows ExLlamaV3, MIT, (c) 2025 Turboderp); `engram.json` and `vision/gate_bias_vl.safetensors` DeepSeek MIT ((c) 2023 DeepSeek); `vision/torch_fmha_sm120.cubin` PyTorch BSD-3-Clause on NVIDIA CUTLASS BSD-3-Clause; kit tools and docs Apache-2.0 | `/srv/ds41f-tensorfold/kit` with its own `KIT-LICENSES.md`, `NOTICE`, `LICENSE`, `LICENSES/` |
| RoPE tables (`rope-*.f32`) | Apache-2.0. Generated here from the RoPE reference archive (`tensorfold-rope-reference.tar.gz`, Apache-2.0) with CPU PyTorch. | `/srv/ds41f-tensorfold/kit`; reference license in `/usr/share/doc/ds41f-tensorfold/rope-reference/LICENSE` |
| NVIDIA NCCL 2.30.7 | Apache-2.0 for most of the project; some parts keep their original BSD-3-Clause license. Copyright NVIDIA CORPORATION & AFFILIATES. | `/usr/local/lib/libnccl.so.2.30.7`; `/usr/share/doc/nccl/{LICENSE.txt,ThirdPartyNotices.txt}`. Built from the source tarball of tag `v2.30.7-1` for `sm_121` only. |
| NVIDIA CUDA 13.0.3 runtime libraries, including cuBLAS 13.1.1.3 | NVIDIA CUDA Toolkit EULA / NGC container license | Base image `nvcr.io/nvidia/cuda:13.0.3-runtime-ubuntu24.04` (`NGC-DL-CONTAINER-LICENSE` at `/` in the base); `libcublas.so.13`, `libcublasLt.so.13` |
| NVIDIA CUDA 13.0.3 devel image (nvcc) | Same | NCCL build stage only; not in the final image |
| rdma-core 50.0 (`libibverbs1`, `ibverbs-providers`) | GPL-2 or BSD-2-Clause (dual) | Ubuntu 24.04 packages; copyright files in `/usr/share/doc/libibverbs1` and `/usr/share/doc/ibverbs-providers` |
| Ubuntu 24.04 userland, Python 3.12 | Various; see `/usr/share/doc/*/copyright` | Base image and `python3` |
| Zig 0.17.0 | MIT (Expat) | Build stage only; not in the final image |
| PyTorch 2.13.0 (CPU), NumPy and their Python dependencies (see `../requirements-rope.txt`) | BSD-3-Clause and others (permissive) | Build stage only: they generate the RoPE tables and are not in the final image |

## What Petabridge adds

| Item | License | Notes |
| --- | --- | --- |
| `Dockerfile`, `apt-snapshot-install.sh`, `selfcheck.py`, `requirements-rope.txt`, `image.json`, `dependency.lock.json`, `README.md` | Apache-2.0 (this repository) | Build and verification glue only. No upstream code is modified. |

## In the image

- `/opt/ds41f/recipe/LICENSE`, `/opt/ds41f/recipe/THIRD_PARTY_NOTICES.md` (the recipe's MIT license and its notices)
- `/usr/share/doc/ds41f-tensorfold/tensorfold/` (the engine's `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `LICENSES/`)
- `/usr/share/doc/ds41f-tensorfold/rope-reference/LICENSE`
- `/usr/share/doc/ds41f-tensorfold/THIRD_PARTY_NOTICES.md` (this file)
- `/usr/share/doc/nccl/` (NCCL's license and third-party notices)
- `/srv/ds41f-tensorfold/kit/{KIT-LICENSES.md,NOTICE,LICENSE,LICENSES/}`

## Not in the image

The model weights are separate downloads under their own license, which they carry (see upstream's
`THIRD_PARTY_NOTICES.md`). The recipe's optional gateway, its vLLM overlay under `runtime/`, and the
`licenses/` for those components are not shipped.
