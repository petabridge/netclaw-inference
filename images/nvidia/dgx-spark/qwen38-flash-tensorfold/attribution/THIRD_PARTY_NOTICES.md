# Third-party notices

This image folder builds the MiaAI-Lab Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold recipe with one
Petabridge patch. The vendored recipe carries its own notices unmodified:

- [`../recipe/LICENSE`](../recipe/LICENSE): Apache License 2.0, the recipe's own code, scripts,
  documentation and patches 0001-0009 (Copyright 2026 MiaAI-Lab).
- [`../recipe/NOTICE`](../recipe/NOTICE): the full third-party list, including TensorFold's
  Apache-2.0 terms and its MIT notice for code written before v0.6.0, Zig (MIT, build stage only),
  and the model licenses.
- [`../recipe/CREDITS.md`](../recipe/CREDITS.md): the recipe's credits.

What Petabridge adds:

| Item | License | Notes |
|---|---|---|
| `petabridge-patches/0010-media-prompt-reuse.patch` | Apache-2.0 | Modifies TensorFold's source, which stays under TensorFold's licenses (Apache-2.0; pre-v0.6.0 code also MIT, per `recipe/NOTICE`). The approach follows the GLM-5.3-Flash TensorFold recipe's image prompt reuse (issue #11 of that recipe, Apache-2.0); no code is copied from it. |
| `Dockerfile` | Apache-2.0 | A materialization of the recipe's inline Dockerfile (`recipe/scripts/prepare.sh`); changes are recorded in `petabridge-dockerfile.diff`. |
| `petabridge-tools/` | Apache-2.0 | Test tools; not in the image. They import the recipe's `tools/client.py`. |

In the image (from the upstream build, unchanged by Petabridge): TensorFold's Zig engine and its
LICENSE, LICENSES/, NOTICE and THIRD_PARTY_NOTICES.md under `/opt/tensorfold/share/doc/tensorfold`;
TensorFold's Python tree for the vision helper; `huggingface_hub` (Apache-2.0), `transformers`
5.17.0 (Apache-2.0) and PyAV 19.0.1 (BSD-3-Clause; its FFmpeg libraries are LGPL) on NVIDIA's
PyTorch container, `nvcr.io/nvidia/pytorch:26.07-py3`, under NVIDIA's container license.

Not in the image: the weights. `nvidia/Qwen3.8-Flash-Next-NVFP4` is governed by the NVIDIA Open
Model License, with the Qwen Community License 1.0 as additional information (per its model card;
see `recipe/NOTICE`).
