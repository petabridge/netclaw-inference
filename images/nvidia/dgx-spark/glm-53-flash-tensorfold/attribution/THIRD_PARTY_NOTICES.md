# Third-party notices

This image folder vendors a third-party recipe. Its license is **not** the repository-wide
MIT statement, and it is **not** the AGPL-3.0 that applies to the sibling
`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` vLLM recipe: this recipe, the TensorFold one, is
**Apache-2.0**.

## Included in the image

- **MiaAI-Lab GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold recipe** (`recipe/`: `patches/`,
  `scripts/`, `tools/`, `tests/`, `docs/`, `start.sh`, `stop.sh`, `start-tp3.sh`), vendored at
  commit `05397e24ee1fa0a60df11ff5dcb7a68f45cff6cd`. Licensed under the **Apache License 2.0**
  ([`LICENSE.APACHE-2.0`](LICENSE.APACHE-2.0)), with its third-party notices in the vendored
  `recipe/NOTICE`. The upstream `recipe/README.md` and `recipe/CREDITS.md` state the same. The
  recipe's own code is the only part Petabridge changes, and only in the materialized
  `../Dockerfile`, marked `PETABRIDGE:` and recorded in `../petabridge-dockerfile.diff`.
- **TensorFold** (ashhart/TensorFold) at commit
  `c4646171139ee8a3c38103eaa1699dad226ec12b` (the annotated tag `v0.6.0`), installed from
  source and patched in place by the vendored `patches/`: Apache-2.0, with the MIT notice for
  code written before v0.6.0 reproduced in [`LICENSE.MIT`](LICENSE.MIT).
- **b12x** (local-inference-lab, Apache-2.0) — the `COMM=roce` one-shot RoCE all-gather in
  `patches/0006-cuda-roce-allgather.patch` is derived from its RDMA proxy and all-gather
  kernel, as the vendored `recipe/NOTICE` and `recipe/CREDITS.md` describe.
- **glm53-tensorfold-spark** (Jay Leaton, Apache-2.0) — parts of patches 0036, 0046, 0047 and
  0058, credited in the vendored `recipe/NOTICE` and `recipe/CREDITS.md`.
- **PyAV / av 18.1.0** (BSD-3-Clause) with its FFmpeg libraries (LGPL) — video input.
- **xgrammar** `>=0.2.8,<0.3` (Apache-2.0) — structured outputs.
- **PyTorch** (BSD-3-Clause), **Triton** (MIT), **NCCL** (BSD-3-Clause), **CUDA**, **cuBLAS**,
  **cuDNN** and the other **NVIDIA PyTorch container** components, inherited from the base
  image: governed by the **NVIDIA Software License Agreement** and the Product-Specific Terms
  for NVIDIA AI Products, which the container prints at every start.
- **ExLlamaV3** (turboderp-org/exllamav3, MIT) — the EXL3 quantization format the routed
  experts are stored in. Credited by TensorFold and by the recipe; **not bundled** in this
  image (TensorFold implements the EXL3 load and inference path itself).
- **vLLM** — not applicable. This image runs TensorFold, not vLLM; no vLLM code is included.

## Not included in the image (runtime models)

The image contains no model weights. A deployment downloads these separately, and each stays
governed by its own license:

- **Target checkpoint** `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` @
  `078455ffe6472f9a52fbc1139f58b9db2881b25c`: **Apache-2.0**. EXL3 routed experts at 4 bits a
  weight, BF16 elsewhere, ~176 GB.
- **Base model** `zai-org/GLM-5.3-Flash`: **MIT**, Copyright (c) 2026 Z.AI Co., Ltd. The
  checkpoint derives from it.
- **Draft model** `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4eacc1810f76656d1811693ff6c6737d2a`:
  **CC BY-NC-ND 4.0**, released "for research and evaluation"; commercial licensing is
  available from Inco AI. Deployments without a commercial license should set `DRAFTER=mtp`,
  which uses the target checkpoint's own MTP head and downloads no draft model.
- **Alternate checkpoint** `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (a byte-identical mirror of
  `brandonmusic/GLM-5.3-Flash-tr3-4bpw`), selectable with `MODEL_ID`: **ShapleyMCG License
  1.0**, attribution required. Its required notice is reproduced in
  [`SHAPLEYMCG-NOTICE.md`](SHAPLEYMCG-NOTICE.md). It is not the default.
- **Ablit weights** `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit`, selectable with
  `ABLIT=1` and gated on Hugging Face: under the license on its own model page.

NVIDIA and DGX are trademarks of NVIDIA Corporation. No endorsement is implied.
