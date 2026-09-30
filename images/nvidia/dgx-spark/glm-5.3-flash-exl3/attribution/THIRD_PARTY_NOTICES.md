# Third-party notices

This image folder, unlike most of this repository, contains AGPL-3.0 material.
The repository-wide statement that repository-authored material is MIT licensed
does not apply to the vendored recipe under `recipe/`.

## Included in the image

- **MiaAI-Lab GLM-5.3-Flash-EXL3-2x-DGX-Sparks recipe** (`recipe/`: Dockerfile,
  `overlay/`, `files/`, `tests/`, `ablit/`), vendored at commit
  `674155dec2f2f62bb879801b5ce2cfc759a0bebf`. Licensed under AGPL-3.0
  ([`LICENSE.AGPL-3.0`](LICENSE.AGPL-3.0)); contributions made before
  2026-09-07 are MIT ([`LICENSE.MIT`](LICENSE.MIT)). Petabridge's modifications
  are marked `PETABRIDGE:` in `recipe/Dockerfile` and recorded in
  `../petabridge-dockerfile.diff`. Because this image runs modified AGPL code as
  a network service, its complete corresponding source is this public folder.
- **ExLlamaV3** (turboderp-org/exllamav3) at commit
  `c5d9c657966ffeeaa9353f0cc899f18629da4a13`, MIT licensed, compiled with the
  recipe's SM121 kernel overlays.
- **vLLM** (Apache-2.0), inherited from the digest-pinned
  `vllm/vllm-openai:glm53-flash-arm64-cu130` base and patched in place by the
  recipe overlays.
- **InstantTensor** 0.2.0, installed from its checksum-pinned PyPI wheel under
  its own license.
- FlashInfer, PyTorch, CUDA, NCCL, and other base-image components retain
  their respective upstream copyrights and licenses.
- The recipe's `ablit/` direction vectors derive from drowzeys' published
  abliteration recipe (see upstream README credits). They are inert unless a
  deployment sets `ABLIT=1`.

## Not included in the image (runtime models)

The image contains no model weights. A deployment downloads these separately,
and each remains governed by its own license:

- **GLM-5.3-Flash** base model, `zai-org/GLM-5.3-Flash`: MIT,
  Copyright (c) 2026 Z.AI Co., Ltd.
- **EXL3 TR3 4bpw checkpoint**, `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` @
  `25a44fdbf16862a46b7cc9921142c6c81350af2f`: ShapleyMCG License 1.0, an
  attribution-required, source-available license. Its required attribution
  notice is reproduced in [`SHAPLEYMCG-NOTICE.md`](SHAPLEYMCG-NOTICE.md).
- **DFlash2 draft model**, `incoai/GLM-5.3-Flash-DFlash2` @
  `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`: CC BY-NC-ND 4.0, released "for
  research and evaluation"; commercial licensing is available from Inco AI
  (contact@inco.ai). Deployments that do not hold a commercial license should
  set `SPEC_METHOD=mtp`, which uses the MIT-licensed target's built-in MTP head.

NVIDIA and DGX are trademarks of NVIDIA Corporation. No endorsement is implied.
