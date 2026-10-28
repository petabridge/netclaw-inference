# llama.cpp ROCm gfx1201 AVX2 server (rdna-boosts patches)

`linux/amd64` llama.cpp `llama-server` for AMD RDNA4 GPUs (`gfx1201`, e.g. the
Radeon AI PRO R9700), built from pinned upstream source with the **vendored
rdna-boosts patch blocks 00-08** applied, on the **same ROCm 7.2.3 runtime** as
the production image.

## Status

**Experimental and unqualified.** No hardware qualification has been recorded
for this image. It exists to answer one question — do the rdna-boosts patches
make our workload faster on this hardware — and it should not be deployed to a
shared endpoint until that question is answered and the results are recorded
here.

## Why this image exists

Upstream llama.cpp leaves substantial RDNA4 performance unclaimed, in patch
families that were never upstreamed. Community tuning on the R9700 reports
MTP-3 decode in the high-50s to low-90s tok/s on Qwen3.8-27B UD-Q4_K_XL,
against a measured **36.6 tok/s** for the production image on this fleet. That
gap is the entire motivation.

The alternative was pulling a prebuilt third-party container. This image exists
because that container ships **ROCm 7.14.1 or 10.1 only** — and ROCm 7.14
SIGILLed on gfx1201 during GPU init in our own incident
([testlab-setup#492](https://github.com/petabridge/testlab-setup/pull/492),
reverted in [#493](https://github.com/petabridge/testlab-setup/pull/493)).
Testing the patches on 7.14 would confound patch performance with a known engine
fault. Here, **ROCm is held constant at 7.2.3** and the patches are the only
variable.

Second reason: supply chain. The third-party image is built on an off-host
worker with no CI provenance or SBOM. This one goes through the same candidate /
digest / promotion gates as every other image in this repository.

## What is built

| Input | Pin |
| --- | --- |
| Base image | `rocm/dev-ubuntu-24.04:7.2.3` (digest-pinned; identical to production) |
| llama.cpp | commit `6050b467c07a95c7b87debd4ddac2f8ba8d9b807` (2026-10-25), archive verified by SHA-256 |
| Patch series | `stew675/llama-cpp-rdna-boosts` `v15-6050b467c-r161` (commit `ecf48ff0`), vendored, per-file SHA-256 verified |
| Blocks applied | `00-08` (see below) |
| GPU target | `gfx1201` (RDNA4) |
| CPU baseline | `x86-64-v3` — AVX, AVX2, FMA, F16C; AVX-512 off |
| Binaries | `llama-server`, `llama-cli` under `/opt/llama.cpp` |

## Which blocks, and why not all 15

The blocks are a **sequential `git am` series**, not independent patches. Blocks
05, 08 and 13 were verified to fail application when blocks 01 and 04 are
skipped. Blocks 00-08 form a contiguous prefix that applies cleanly.

Applied: `00` foundation, `01` hybrid all-reduce, `02` adaptive MTP draft depth,
`03` chunked gated-delta-net prefill, `04` BF16 KV, `05` flash-attention band /
native quantized K-V, `06` MMB quantized prefill GEMM, `07` k-quant boosts,
`08` fused kernels.

Excluded, with reasons:

| Block | Why excluded |
| --- | --- |
| 09 | host-resident MoE — MoE-only |
| 10 | slab / fusion dispatcher infrastructure — consumed by 09 and 11 |
| 11 | `qwen4exp` (Qwen3.8-Flash-Next) architecture — different model family |
| 12 | campaign remainder — no dependency from 00-08 |
| 13 | opt-in MTP draft-vocabulary subset — requires a re-cut GGUF with `draft_head` tensors |
| 14 | temporary relaxed tensor-split mode — tensor split is not used here |

This is deliberately **not** the full 15-block delivery. Upstream
[issue #30](https://github.com/stew675/llama-cpp-rdna-boosts/issues/30) reports
the full delivery costs ~14% MTP decode versus stock at the same fork point on
dense qwen35 27B / gfx1201, so taking the whole set was never the plan.

## Two variables, not one

This image changes **two** things relative to production: the llama.cpp commit
(`d812350`/v0.6.0 → `6050b467c`) and the patch series. The engine bump is
unavoidable — the patches are `git am` commits against `6050b467c` and will not
apply to the older tree.

Mitigation: the community reference reports no measurable change from a
four-day-older llama.cpp on this model, and the build's `--help` smoke check
proves our serving flags survived the bump. A rigorous A/B would build a second
image at `6050b467c` with **no** patches; that is worth doing if the patched
result is close to the baseline.

## New knobs

| Knob | Source | Purpose |
| --- | --- | --- |
| `--spec-type draft-mtp-adaptive` | block 02 | adaptive draft depth (replaces fixed `--spec-draft-n-max`) |
| `GGML_CUDA_GDN_CHUNKED=0` | block 03 | A/B the chunked gated-delta-net prefill off |
| `GGML_HIP_FA_BAND_WMMA` | block 05 | RDNA4 FA band — the knob for head-256 / GQA-6 / q8_0 KV |
| `GGML_HIP_FA_BAND_WIDE`, `_SPLIT`, `_FETCH_AHEAD`, `_WMMA_GQA_MAX` | block 05 | band tuning |

## Running

Engine image only — GGUF weights are supplied at runtime through a read-only
bind mount.

```bash
docker pull ghcr.io/petabridge/llama-cpp-rocm-gfx1201-avx2-rdna-boosts:sha-<12-character-commit>
```

Deployments should reference the published manifest digest rather than a tag:

```text
ghcr.io/petabridge/llama-cpp-rocm-gfx1201-avx2-rdna-boosts@sha256:<digest>
```

## Qualification plan

Compare against the production baseline recorded in
`testlab-setup/services/big-gpu/README.md`: **36.64 tok/s** single-stream decode
(MTP-3, 256-token completions, technical prompt) and **89.7 tok/s** at four
concurrent requests.

1. Readiness: `/health` 200.
2. Non-empty `content` at `max_tokens: 4` with a system message present.
3. Forced tool call returns valid `tool_calls` JSON.
4. Vision via `--mmproj` still works.
5. Decode tok/s at 256 tokens, three runs — the headline number.
6. Decode and prefill at ~30K and ~100K context, where the FA band should matter.
7. Four concurrent requests — aggregate throughput.
8. VRAM and GTT: VRAM stays primary, GTT does not grow (this box runs at 97-98%
   VRAM, so a workspace-buffer increase can OOM).
9. Greedy-output sanity: the series makes rounding-level changes. Diff a long
   greedy completion against the production image and read it, do not just hash
   it.

## Files

```
llama-cpp-gfx1201-avx2-rdna-boosts/
├── README.md                       this file
├── Dockerfile                      build recipe
├── image.json                      image manifest
├── dependency.lock.json            pins, patch provenance, exclusion reasons
├── patches/                        vendored patch series (blocks 00-08)
│   ├── 0000-...patch               one file per block
│   ├── SHA256SUMS                  verified during the build
│   ├── upstream-release.json       the release manifest the series came from
│   └── LICENSE                     MIT, stew675
└── attribution/
    ├── THIRD_PARTY_NOTICES.md
    └── rdna-boosts/LICENSE         retained MIT text
```
