# GLM-5.3-Flash EXL3 on DGX Spark

Image id: `vllm-glm-5-3-flash-exl3-gb10` (`ghcr.io/petabridge/vllm-glm-5-3-flash-exl3-gb10`)

A reproducible build of the
[MiaAI-Lab GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
serving recipe for two NVIDIA DGX Sparks (GB10, `sm_121a`) running tensor
parallel 2 over the CX7 RoCE link. It serves
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) from
4-bpw EXL3/TR3 routed-expert weights through vLLM's OpenAI-compatible API.

> **License notice:** this folder contains AGPL-3.0 material and the runtime
> weights carry their own non-MIT licenses. Read [Licensing](#licensing) before
> deploying. The repository-wide MIT statement does not cover `recipe/`.

## Build boundary

The upstream recipe is vendored unmodified under `recipe/` at commit
`674155dec2f2f62bb879801b5ce2cfc759a0bebf`; `upstream-recipe.sha256` records
every vendored file's hash at that commit. Petabridge changes only
`recipe/Dockerfile` (every change is marked `PETABRIDGE:` and recorded in
`petabridge-dockerfile.diff`) and adds `recipe/petabridge/`:

| Change | Why |
|---|---|
| Literal digest-pinned `FROM` instead of `ARG BASE` | Repository rule: no base-image build arguments |
| ExLlamaV3 source via `ADD --checksum` instead of `curl \| tar` | Checksum-pinned source |
| InstantTensor via a checksum-pinned wheel instead of `pip install` | Checksum-pinned dependency |
| Bake `patch_adaptive_k.py`, `patch_dense_fp8.py`, `patch_default_max_new_tokens.py`, `/opt/glm53/exl3.py`, and `ablit/` | Upstream `start.sh` bind-mounts these at container start; baking them lets a plain `docker run`/compose apply the identical overlay set |
| Bake `/opt/petabridge/glm53-head.sh` and `glm53-worker.sh` | The per-rank scripts upstream `start.sh` generates (`write_inner_scripts()`) and mounts at `/start.sh`, captured unmodified |

What the build does (upstream): patches the base vLLM for GLM-5.3's NoPE
sparse MLA on SM12x, registers the `exl3` quantization method, compiles
`exllamav3_ext` for `sm_121a` with the recipe's fused/fat MoE and thin-decode
kernels, applies and self-tests roughly thirty source overlays (scheduler,
hybrid prefix cache, DFlash2 drafter slot-sharing, xgrammar backports, KV and
indexer fixes), and installs InstantTensor. The GPU self-check is skipped at
build time (`EXL3_SELFCHECK_GPU=0`); no GPU is needed to build.

## Runtime contract

The image does not launch anything by itself (base entrypoint unchanged). Run
the head with `--entrypoint bash <image> /opt/petabridge/glm53-head.sh` and the
worker with `/opt/petabridge/glm53-worker.sh`. Start the worker first; it
joins headless (`--node-rank 1 --headless`) and the head serves the API.

Both ranks need host networking, `--ipc=host`, `--gpus all`,
`--device /dev/infiniband`, `--cap-add IPC_LOCK`, `--ulimit memlock=-1`,
`--ulimit stack=67108864`, `--shm-size 32g`, and these mounts:

| Container path | Host content |
|---|---|
| `/root/.cache/huggingface` | HF cache holding the pre-staged weights (runs offline) |
| `/root/.cache/vllm` | Persistent vLLM compile cache for this image |
| `/root/.triton/cache`, `/root/.tilelang/cache`, `/root/.nv/ComputeCache` | Persistent kernel caches |

Everything else is environment. The launch scripts read the serve settings
(`SERVED_MODEL_NAME`, `PORT`, `TP`, `NNODES`, `HEAD_IP`, `MASTER_PORT`,
`MODEL_DIR`, `QUANTIZATION=exl3`, `MAX_MODEL_LEN`, `GPU_MEM_UTIL`,
`MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `KV_CACHE_DTYPE`, `LOAD_FORMAT`,
`SPEC_METHOD`, `MTP_TOKENS`, `DFLASH_*`, `CHAT_TEMPLATE`, multimodal limits,
`ENFORCE_EAGER`, `EXTRA_ARGS`), and the overlays read `GLM53_*` / `EXL3_*`
knobs. The upstream defaults at the pinned commit are:

```text
SERVED_MODEL_NAME=GLM-5.3-Flash-EXL3  TP=2  NNODES=2  QUANTIZATION=exl3
MAX_MODEL_LEN=850000  GPU_MEM_UTIL=0.85  MAX_NUM_SEQS=4  MAX_NUM_BATCHED_TOKENS=7168
KV_CACHE_DTYPE=fp8  LOAD_FORMAT=instanttensor  ENFORCE_EAGER=0
SPEC_METHOD=dflash  DFLASH_TOKENS=7  DFLASH_DRAFT_TP=2  MTP_TOKENS=2
EXTRA_ARGS=--kv-cache-memory-bytes 11811160064
LANGUAGE_MODEL_ONLY=0  LIMIT_MM={"image":48,"video":1}  MM_IMAGE_TOKENS=2048
MM_PROCESSOR_CACHE_GB=1  SKIP_MM_PROFILING=1  CHAT_TEMPLATE=/opt/glm53/chat_template.jinja
DEFAULT_MAX_NEW_TOKENS=65536  GLM53_DENSE_FP8=all  GLM53_KDA_BF16_LARGE_M=1
GLM53_MIXED_PREFILL_CHUNK=fair  GLM53_FAIR_PREFILL_CHUNK=256  GLM53_FAIR_PREFILL_SHARE=0.30
GLM53_FAIR_PREFILL_MAX_INTERVAL_MS=2000  GLM53_FAIR_PREFILL_MAX_STEP_MS=2000
GLM53_FAIR_PREFILL_MAX_CHUNKS=1  GLM53_SUPPRESS_STOPS_IN_REASONING=1  GLM53_APC_NO_STORE=1
GLM53_KV_CAPACITY_LOG=1  GLM53_INDEXER_WORKSPACE=rightsize  GLM53_DRAFT_KV_COMPACT=1
GLM53_SPINWAIT_MS=stock  GLM53_LOAD_CLONE=1  GLM53_LOAD_PREFETCH=0  GLM53_ADAPTIVE_K=off
EXL3_FUSED_MOE=1  EXL3_FAT_KERNEL=1  EXL3_FAT_GROUPED=1  EXL3_TEMP_ROWS_FUSED=32  ABLIT=0
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800  VLLM_ENGINE_READY_TIMEOUT_S=3600
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1
```

plus per-rank NCCL/GLOO interface pins (`NCCL_SOCKET_IFNAME`,
`GLOO_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `VLLM_HOST_IP`) and
the upstream NCCL set (`NCCL_NET=IB`, `NCCL_NET_PLUGIN=none`,
`NCCL_IB_ROCE_VERSION_NUM=2`, `NCCL_NVLS_ENABLE=0`, `NCCL_CUMEM_ENABLE=0`,
`NCCL_IB_MERGE_NICS=0`, `NCCL_CROSS_NIC=0`, `NCCL_IGNORE_CPU_AFFINITY=1`).
Upstream `start.sh` also runs host memory hygiene before launch and a
post-`/health` warmup (`scripts/boot-shape-warmup.sh`); neither is part of this
image.

## Runtime models

Weights are downloaded separately and pinned by revision:

| Role | Repository | Revision |
|---|---|---|
| Target | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (~164 GiB) | `25a44fdbf16862a46b7cc9921142c6c81350af2f` |
| Draft (optional) | `incoai/GLM-5.3-Flash-DFlash2` (~2.3 GB) | `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |

`SPEC_METHOD=mtp` uses the target's built-in MTP head instead of the draft.

## Reproducibility inputs

See `dependency.lock.json`: base image digest, upstream recipe commit and file
lock, the Dockerfile diff hash, the ExLlamaV3 commit and archive checksum, and
the InstantTensor wheel checksum.

## Licensing

- **Image contents.** `recipe/` is the MiaAI-Lab recipe, **AGPL-3.0**
  ([`attribution/LICENSE.AGPL-3.0`](attribution/LICENSE.AGPL-3.0); pre-2026-09-07
  contributions MIT, [`attribution/LICENSE.MIT`](attribution/LICENSE.MIT)).
  This image runs that code, modified as described above, as a network
  service, so this public folder is its corresponding source. ExLlamaV3 (MIT),
  vLLM (Apache-2.0), and the other components are listed in
  [`attribution/THIRD_PARTY_NOTICES.md`](attribution/THIRD_PARTY_NOTICES.md).
- **Target weights (not in the image).** ShapleyMCG License 1.0, which requires
  attribution; the required notice is in
  [`attribution/SHAPLEYMCG-NOTICE.md`](attribution/SHAPLEYMCG-NOTICE.md):

  > This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

- **Draft weights (not in the image, optional).** CC BY-NC-ND 4.0, released for
  research and evaluation; commercial use requires a license from Inco AI.
  Use `SPEC_METHOD=mtp` without one.
- **Base model.** GLM-5.3-Flash is MIT, Copyright (c) 2026 Z.AI Co., Ltd.

## Credits

The serving recipe, overlays, kernels, and measurements are the work of Mia's
AI Lab and the recipe's contributors; EXL3/TR3 weights by Brandon M. Music;
ExLlamaV3 by turboderp; DFlash2 by Inco AI; GLM-5.3-Flash by Z.ai. See the
upstream README for full credits.
