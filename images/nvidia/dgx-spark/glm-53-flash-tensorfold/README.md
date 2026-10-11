# GLM-5.3-Flash EXL3 TensorFold on DGX Spark

Image id: `tensorfold-glm-5-3-flash-gb10` (`ghcr.io/petabridge/tensorfold-glm-5-3-flash-gb10`)

A reproducible build of the
[MiaAI-Lab GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
serving recipe for two NVIDIA DGX Sparks (GB10, `sm_121a`) running tensor parallel 2 over the
CX7 RoCE link. It runs [TensorFold](https://github.com/ashhart/TensorFold) v0.6.0 on both
Sparks, in [NVIDIA's PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
`nvcr.io/nvidia/pytorch:26.07-py3`, with the recipe's 97 patches applied to the installed
engine. It serves
[`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold)
— EXL3 routed experts at 4 bits a weight, BF16 elsewhere, ~176 GB — through TensorFold's
OpenAI-compatible API, with the model's full 1,048,576-token context, up to 8 concurrent
requests, DFlash2 or MTP drafting, image and video input, tool calling and structured outputs.

> **License notice:** the recipe under `recipe/` is **Apache-2.0**, not the repository-wide
> MIT and not the AGPL-3.0 of the sibling vLLM GLM recipe. The runtime weights carry their own
> licenses. Read [Licensing](#licensing) before deploying.

## Build boundary

The upstream recipe is vendored unmodified under `recipe/` at commit
`05397e24ee1fa0a60df11ff5dcb7a68f45cff6cd`; `upstream-recipe.sha256` records every vendored
file's hash at that commit. The upstream does not ship a Dockerfile: it materializes its image
as a shell heredoc inside `recipe/scripts/prepare.sh`. Petabridge materializes that heredoc as
`Dockerfile` in this folder, and every change against it is marked `PETABRIDGE:` and recorded
in `petabridge-dockerfile.diff`:

| Change | Why |
|---|---|
| Literal digest-pinned `FROM` with a marked placeholder digest | Repository rule: no base-image build arguments, no floating tags. The digest is still outstanding (see below) |
| The engine spec, the extra packages and the patch hash materialized as `ARG` defaults | Upstream passed them as build arguments; `scripts/build-image.sh` passes only the arguments declared in `image.json` |
| TensorFold pinned by commit `c4646171139ee8a3c38103eaa1699dad226ec12b` instead of the mutable tag `v0.6.0` | Repository rule: pin source dependencies by immutable commit |
| `COPY patches/ /opt/tf-patches/` instead of `COPY . /opt/tf-patches` | The build context here is the whole vendored `recipe/` tree, not the `patches/` directory |
| `tf.patches` recorded literally and verified against the copied patches at build time | Upstream passed it in; the label must not drift from the patch set it names |
| `org.opencontainers.image.*` and `io.petabridge.*` provenance labels | Upstream sets none |

**Not done yet — one operator step.** `nvcr.io/nvidia/pytorch:26.07-py3` is auth-gated on NGC
and its `sha256` digest could not be resolved when this folder was written, so the `FROM` line
carries the sentinel digest `sha256:0000…0000`. That digest can never resolve, so a build
fails closed rather than silently using a floating tag; nothing in
`scripts/validate-repository.sh` catches it (the sentinel satisfies the shape check). Resolve
it with NGC credentials and replace it in `Dockerfile` and in `dependency.lock.json`. The
manifest is therefore `build_enabled: false`, and so is the image: its declared context is
`recipe/` while the Dockerfile sits at this folder's root, which
`scripts/validate-image-manifest.sh` rejects for an enabled manifest. Move the Dockerfile to
`recipe/Dockerfile` (as the removed vLLM GLM image did) or widen the context before enabling.

What the build does (upstream): installs TensorFold v0.6.0 plus `av==18.1.0` and
`xgrammar>=0.2.8,<0.3` into the PyTorch container, then applies the recipe's 97 patches to the
installed package with `patch -p0`: the two-rank CUDA engine for GLM-5.3-Flash, the EXL3
routed-expert kernels, DFlash2 and copy drafting, 4-bit dense weights, an FP8 KV cache, the
one-shot RoCE all-gather, vision, tool calling, `/tokenize`, `/metrics` and more. No GPU is
needed to build.

## Runtime contract

The image does not launch anything by itself (no entrypoint and no `CMD` beyond the base
image's). The recipe's `recipe/start.sh` on the head Spark does everything: it runs
`recipe/scripts/prepare.sh` on both Sparks when the image or a checkpoint is not ready, starts
rank 1 on the worker, starts rank 0 on the head (which serves the API), waits for `/health`,
then smoke-tests. `recipe/stop.sh` stops them.

Each rank is a `docker run` of this image with the command
`tensorfold serve <model> --tp 2 --rank <0|1> --master <head address on the link> --master-port 29551 …`
— rank 1 goes up first and joins the head; rank 0 binds the API. Both ranks get
`--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK
--ulimit memlock=-1 --ulimit stack=67108864`, the rank's NCCL/RoCE environment, and two mounts:

| Container path | Host content |
|---|---|
| `/root/.cache/huggingface` | HF cache holding the pre-staged weights (the ranks run with `HF_HUB_OFFLINE=1`); on the worker, either its own copy or the head's cache over NFS |
| `/cache` | Compiled CUDA kernel cache, one directory per patch set (`$HOME/.cache/tensorfold-glm53/<tf.patches>`); the image's `TORCH_EXTENSIONS_DIR` and `TRITON_CACHE_DIR` point inside it |

Everything else is environment. `recipe/scripts/config.sh` holds the defaults and each
setting's measured effect; `prepare.sh` rebuilds or re-pulls the image when `patches/*.patch`
or `IMAGE_EXTRAS` change, keyed on the very `tf.patches` label this image sets. The upstream
defaults at the pinned commit are:

```text
SERVED_NAME=GLM-5.3-Flash-EXL3  HOST=0.0.0.0  PORT=8888  TP=2  MASTER_PORT=29551
MODEL_ID=Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold
MODEL_REVISION=078455ffe6472f9a52fbc1139f58b9db2881b25c
DRAFTER=dflash2  DFLASH2_ID=incoai/GLM-5.3-Flash-DFlash2
DFLASH2_REVISION=bf582e4eacc1810f76656d1811693ff6c6737d2a
PARALLEL=4  CONTEXT=1048576  KV=fp8  DENSE=q4  THINKING=1  MAX_TOKENS=32768  VISION=1  VISION_URLS=0
COMM=roce  KV_POOL_GIB=12.5  MEMORY_RESERVE_GIB=14.5  DISPLAY_KV_MIB=0  DISPLAY_KV_BACKEND=drm
TF_GLM_MTP=auto  TF_GLM_KV=fp8  TF_GLM_DENSE=q4  TF_GLM_CACHE_GIB=12.5
WORKER_WEIGHTS=copy  HF_HUB_OFFLINE=1  MIN_FREE_GB=180  IMAGE_FREE_GB=35
```

Switches worth knowing: `PARALLEL=1…8`, `CONTEXT` up to 1,048,576, `KV=bf16|fp8`, `DENSE=bf16|fp8|q4`,
`DRAFTER=dflash2|mtp`, `ABLIT=1` for the gated abliterated checkpoint, `WORKER_WEIGHTS=nfs` to
skip the ~166 GiB copy to the worker, `TF_GLM_L2PF`, `TF_GLM_EXL3_LOADS`, `TF_ROCE_MAX_KB`,
`TF_GLM_MULTI_WINDOW`. At `PARALLEL=8` the memory reserve grows, so the shared pool is ~1.5M
tokens instead of the 2,922,496 measured at 4.

## Runtime models

Weights are downloaded separately and pinned by revision:

| Role | Repository | Revision | Size |
|---|---|---|---|
| Target | `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` | `078455ffe6472f9a52fbc1139f58b9db2881b25c` | ~176 GB |
| Draft (optional) | `incoai/GLM-5.3-Flash-DFlash2` | `bf582e4eacc1810f76656d1811693ff6c6737d2a` | |

`DRAFTER=mtp` uses the target's built-in MTP head instead of the draft and serves one request
at a time. `MODEL_ID` also accepts the earlier `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`
checkpoint and the gated Ablit checkpoint (`ABLIT=1`); see [Licensing](#licensing).

## Reproducibility inputs

See `dependency.lock.json`: the base image reference and its outstanding digest, the upstream
recipe commit and per-file hash lock, the Dockerfile diff hash, the TensorFold commit, and the
extra package versions.

## Licensing

- **Image contents (the recipe).** `recipe/` is the MiaAI-Lab
  GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold recipe, **Apache-2.0**
  ([`attribution/LICENSE.APACHE-2.0`](attribution/LICENSE.APACHE-2.0)), with its third-party
  notices vendored unmodified in `recipe/NOTICE` and `recipe/CREDITS.md`. This is not the
  AGPL-3.0 license of the sibling MiaAI-Lab vLLM recipe and not MIT.
- **TensorFold (in the image).** Apache-2.0 from v0.6.0; code written before v0.6.0 keeps its
  MIT notice, kept in [`attribution/LICENSE.MIT`](attribution/LICENSE.MIT).
- **Target weights (not in the image).** `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`:
  **Apache-2.0**.
- **Base model.** GLM-5.3-Flash is **MIT**, Copyright (c) 2026 Z.AI Co., Ltd.
- **Draft weights (not in the image, optional).** `incoai/GLM-5.3-Flash-DFlash2`: CC BY-NC-ND
  4.0, released for research and evaluation; commercial use requires a license from Inco AI.
  Set `DRAFTER=mtp` without one.
- **Alternate checkpoint (not in the image).** `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, a
  byte-identical mirror of `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, still selectable with
  `MODEL_ID`: **ShapleyMCG License 1.0**, which requires attribution. The required notice
  is in [`attribution/SHAPLEYMCG-NOTICE.md`](attribution/SHAPLEYMCG-NOTICE.md):

  > This work includes or was produced using ShapleyMcg, created by Brandon M. Music
  > (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the
  > ShapleyMcg License v1.0, an attribution-required license that grants no rights to the
  > person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

- **Everything else.** The full party list — b12x, glm53-tensorfold-spark, ExLlamaV3 (credited
  for the EXL3 format, not bundled), PyAV/FFmpeg, xgrammar, the NVIDIA container, and vLLM
  (not applicable) — is in
  [`attribution/THIRD_PARTY_NOTICES.md`](attribution/THIRD_PARTY_NOTICES.md).

## Credits

The serving recipe, patches, kernels, and measurements are the work of Mia's AI Lab and the
recipe's contributors; TensorFold by Ash Hart and the TensorFold contributors; the EXL3
checkpoint by Mia's AI Lab on exllamav3 by turboderp; the ShapleyMcG TR3 checkpoint by
Brandon M. Music; DFlash2 by Inco AI; GLM-5.3-Flash by Z.ai; b12x by local-inference-lab. The
vendored `recipe/CREDITS.md` has the full list.
