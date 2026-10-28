## What

Replaces the vLLM-based GLM-5.3-Flash image with the **MiaAI-Lab TensorFold recipe** (the faster engine) and hosts it on GHCR for the test-lab DGX Sparks.

## Why

TensorFold decodes ~1.8–2× faster than the vLLM variant on prose/code (60.4 tok/s C1 vs ~36; 108.8 aggregate @ C4), with a 1M-token context and up to 8 concurrent requests on two Sparks. This swaps the old `glm-5.3-flash-exl3` vLLM image for the TensorFold build.

## Changes

- **Add** `images/nvidia/dgx-spark/glm-53-flash-tensorfold/` — upstream MiaAI-Lab `GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold` recipe vendored unmodified at commit `05397e24` (145 files, 97 patches).
- **Materialize** the upstream inline-Dockerfile heredoc as a checked-in `recipe/Dockerfile`:
  - digest-pinned `nvcr.io/nvidia/pytorch:26.07-py3` arm64 base (`sha256:7531d90b…`, anonymously pullable — no NGC auth)
  - TensorFold pinned by commit `c4646171…` (not the `v0.6.0` tag)
  - recipe's `tf.patches` kernel-cache label recorded and verified at build time
- **Attribution** (`attribution/`): recipe Apache-2.0 (not AGPL), TensorFold Apache-2.0 + pre-0.6.0 MIT, and runtime-weight terms (Apache-2.0 / CC BY-NC-ND 4.0 / ShapleyMCG).
- **Remove** the superseded `images/nvidia/dgx-spark/glm-5.3-flash-exl3/` vLLM image.
- **Update** `README.md` + `RELEASE_NOTES.md`.

## Image

- id: `tensorfold-glm-5-3-flash-gb10`
- registry: `ghcr.io/petabridge/tensorfold-glm-5-3-flash-gb10`
- platform: `linux/arm64`
- `build_enabled: true` (hardware qualification still pending)

## Validation

`./scripts/validate-repository.sh` passes (3 enabled manifests).

## Note

Weights are **not** baked into the image — the ~176 GB `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` checkpoint + DFlash2 drafter are pre-staged on the Spark hosts' HF cache at runtime.
