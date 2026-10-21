# Qwen3.8-Flash-Next TensorFold on DGX Spark, with image prompt reuse (candidate)

Image id: `tensorfold-qwen3-8-flash-gb10` (`ghcr.io/petabridge/tensorfold-qwen3-8-flash-gb10`)

> **Candidate, not qualified.** The build is enabled so a `sha-<commit>` candidate can be run
> through the GPU steps of the [test plan](#test-plan) on two DGX Sparks. The Petabridge patch has
> passed host-side tests only so far. Do not promote this image until the GPU steps pass.

A reproducible build of the
[MiaAI-Lab Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold)
recipe: TensorFold's Zig engine (`tensorfold-native`, branch `zig-flashnext` at
`db281878ddb836fd0df510d8771ecb7e0fe47d26`) with the recipe's patches 0001-0009 and its Triton
kernel set, serving `nvidia/Qwen3.8-Flash-Next-NVFP4` over two DGX Sparks (tensor parallel 2).
Petabridge adds one patch, `petabridge-patches/0010-media-prompt-reuse.patch`, so that
conversations with images reuse their kept prompt states the way text conversations already do.

## Why

Upstream, a prompt with images or video is never resumed from the prompt cache and never kept
(`zig/src/core/lane_host.zig`'s lookup is gated on `media == null`; `cuda_engine.zig` refuses
`keep_at`/`resume_from` with `error.MediaPromptKept`). Multi-turn agent chats with screenshots
(225k-255k-token prompts, 1-22 images) therefore prefill from token 0 on every turn, about three
minutes each. A second cost: once a request has more than four images, they share
`TENSORFOLD_IMAGE_TOKENS` (16,384) equally (each image also stays under its 4,096-token ceiling).
So adding an image shrinks every earlier one; production showed a 22-image chat at 15,466 visual
tokens, about 703 an image (16,384 / 22). The earlier images' tokens change, and no prefix could
match even with the gate removed.

## What 0010 changes

| Layer | Change |
|---|---|
| Vision helper (Python) | Each picture and video frame group gets a 64-bit content key: the decoded pixels' hash, the pixel cap it was fitted under and its processed grid (`tensorfold/vision/media_keys.py`), sent as `item_keys` in the reply header. |
| Per-image cap | `TENSORFOLD_IMAGE_TOKENS_EACH` (default 1024): every image is fitted under this fixed cap while `images x cap <= TENSORFOLD_IMAGE_TOKENS`, so an added image never refits earlier ones. See [settings](#settings). |
| Feature cache | `TENSORFOLD_VISION_FEATURE_CACHE_MB` (default 512, `0` = off): each picture is encoded alone and its tower rows are kept by key in the helper's host memory; a later turn runs the tower only for new pictures. |
| Server | The ids the prompt cache matches are the prompt's, with every picture row replaced by `0x80000000 \| (hash(key, row) & 0x7fffffff)` and row 0 tagged (`prompt_cache.mediaKeyed`, `Request.cache_key`). Two pictures with the same grid never share a state, and image and text states are never mixed. An image prompt's history length is found as the text path finds it (the template's generation suffix), so keep points get planned. A request field `"prompt_cache": false` prefills from 0 and keeps nothing (for exactness checks). |
| Engine | The two `MediaPromptKept` refusals are gone. The media rotary table is attached before a resume's MTP absorb, which rotates through it as a fresh pass did; upstream attached it after. Keep segments carry the whole prompt's media (`Media.check` accepts a prefix; the table is cut to the segment). `TF_FLASHNEXT_PREFILL_DIGEST=1` now also prints the sliced fill's chunk digests. |

The keying follows the GLM-5.3-Flash TensorFold recipe's image prompt reuse (issue #11 of that
recipe): kept states are keyed by token ids with each picture's content key substituted into its
rows. The engine side, the cap and the MTP ordering are specific to the Qwen Zig engine.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `TENSORFOLD_IMAGE_TOKENS_EACH` | `1024` | Fixed visual-token cap per image; `0` restores upstream's shared split. |
| `TENSORFOLD_IMAGE_TOKENS` | `16384` (recipe `config.sh`) | A request's visual-token budget (upstream). |
| `TENSORFOLD_MAX_IMAGES` | `50` (recipe `config.sh`) | Images a request may carry (upstream). |
| `TENSORFOLD_VISION_FEATURE_CACHE_MB` | `512` | The helper's feature cache; `0` = off (the tower then runs a request's images together, as upstream). |

How they interact:

- **Fixed cap applies** while `images x TENSORFOLD_IMAGE_TOKENS_EACH <= TENSORFOLD_IMAGE_TOKENS`.
  Each image gets `min(cap, 4096-token ceiling, processor max)`, and 256 tokens for
  `detail: low`. With the defaults that covers up to 16 images.
- **Past that budget** the request falls back to upstream's split (`TENSORFOLD_IMAGE_TOKENS /
  images`). It is served, but its images refit whenever the count changes, so it resumes nothing
  past its first image. Nothing is refused that upstream would serve.
- **To cover the full image limit**, set `TENSORFOLD_IMAGE_TOKENS >= TENSORFOLD_MAX_IMAGES x
  TENSORFOLD_IMAGE_TOKENS_EACH` (50 x 1,024 = 51,200; the helper accepts up to 65,536). The helper
  logs the covered image count at start-up when the budget is smaller.
- **Quality trade-off:** with 1-15 images, each image is now capped at 1,024 tokens instead of
  up to 4,096 (a 1920x1080 screenshot is about 2,000 tokens at full resolution). With 22 images
  it is 1,024 instead of about 703. Raise the cap for fidelity, at the cost of more prompt tokens
  and tower time; lower it for longer conversations.

## Risks and open questions

- **Not run on GPUs.** Bit-exactness of resumed and fresh passes is argued from the code
  (positions are prefix-causal, injection is offset-aware, prompt chunks are chunk-invariant) and
  from host tests. The [test plan](#test-plan) is what proves it.
- **Text-to-image transition.** Row 0 tagging keeps text-only and image states apart. Text rows
  of an image prompt rotate through the table kernel mode rather than the plain one, and bit
  equality between the two is unverified. So a conversation that turns into an image conversation
  re-prefills once, at its first image turn. Relax this only if the GPU test shows identical bits.
- **Tower batching.** Without the feature cache, the tower encodes a request's images in shared
  runs, so an earlier image's features could differ in the last bits once more images join its
  run. A resumed state would then differ from a fresh pass. The feature cache (on by default)
  encodes each image alone, which removes this.
- **Keep segments re-attach media.** Each keep segment uploads (and on two ranks broadcasts) the
  whole prompt's features again. If a later segment's attachment is refused for memory mid-pass,
  the lane backend treats it as a lost fill, not a refused request. It is rare, since the first
  segment's attach has the same size.
- **Snapshot tokens are real ids.** The engine's own resume check compares real token ids. The
  picture-content check lives in the prompt cache (keyed ids), so only the cache may hand out
  resumes, as it does today.
- **Exactness reference changes.** Encoding each picture alone (feature cache on) gives
  different, still deterministic, feature bits than upstream's batched encode. Upstream's
  captured digests for multi-image prompts no longer apply. The single-image self-test is
  unchanged.
- **Deployment label.** `tf.patches` is `50cb44760834`, upstream's `image_hash()` with 0010 in
  `patches/`. A recipe checkout that does not carry 0010 computes `f3e1300f4f62` and would rebuild
  or re-pull. testlab-setup's `coordinator.sh` also pins the upstream image digest.
- **Python packages** are version-pinned, not hash-pinned, as upstream ships them (see
  `dependency.lock.json`).

## Test plan

Host-side (done for this prototype; no GPU):

- `zig build test` (180 tests), `zig/tests/server/zig_test.sh` (server 50 and engine API 22,
  including new tests: keyed ids, same-grid pictures never sharing a state, a new picture in the
  suffix, keep inside an image span, lane-host resume by cache key and `prompt_cache: false`,
  `Media.check` prefixes, `item_keys` parsing and media history length), and
  `zig build native` (the CUDA server compiles and links). Run with Zig 0.17.0 on x86_64 Linux,
  over TensorFold `db281878` plus 0001-0010 applied with `git apply`.
- `tests/test_media_prompt_reuse.py` (10 tests, CPU torch): keys, fixed cap versus shared split,
  fallback past the budget, the feature cache (LRU, per-image encode, hits equal fresh rows).

On two DGX Sparks (to do before promotion):

1. Build this image; the build stage verifies `tf.patches`. Serve with
   `TF_FLASHNEXT_PREFILL_DIGEST=1`.
2. `petabridge-tools/media_exact.py --log-cmd '<print rank 0 log>'`: four scenarios (new image in
   the suffix, a turn that ends on an image, a same-grid picture swap, 12 then 13 images). Each
   compares a resumed turn with the same turn sent with `"prompt_cache": false`. First-token
   logits digests, greedy reply text, finish reason and token sha must all be identical, and the
   swap must resume nothing past the swapped picture.
3. `petabridge-tools/media_prompt_reuse.py 200k 6 --first-images 8`: each follow-up turn must
   resume at least 90% of its prompt. Prefill time per turn should fall from minutes to seconds.
   Repeat with `TENSORFOLD_IMAGE_TOKENS=51200` and `--first-images 20` for the 22-image case.
4. The recipe's own `tools/exact.py`, `tools/prompt_reuse.py` and `tools/needle.py`: text
   behavior unchanged.
5. Watch the helper's log line (`N picture(s) keyed, M from the feature cache, fixed cap yes`) and
   the prompt cache's `resumed at` lines on a real agent conversation.

## Build boundary

`recipe/` is the upstream recipe, unmodified at `be3ed857ab1055ca47b36134d22691f00f2bb94a`;
`upstream-recipe.sha256` records every vendored file's hash. Upstream has no Dockerfile: it
builds from a heredoc in `recipe/scripts/prepare.sh`. `Dockerfile` materializes it with these
changes, each marked `PETABRIDGE:` and recorded in `petabridge-dockerfile.diff`:

| Change | Why |
|---|---|
| Literal digest-pinned `FROM` (both stages), `nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90b...` | No base-image build arguments, no floating tags. The digest resolves anonymously, unlike the placeholder the GLM folder's README once described. |
| `ZIG_VERSION`, `ZIG_SHA256`, `TF_REPO`, `TF_REF`, `KERNEL_SOURCE_MTIME` as `ARG` defaults | Upstream passed them from `config.sh`; `scripts/build-image.sh` passes only `image.json`'s. |
| `COPY recipe/patches/` plus `COPY petabridge-patches/` | The context is this folder; 0010 sorts after 0009 and goes through upstream's own apply loop. |
| `tf.patches` and `tf.kernels` recorded literally; `tf.patches` verified in the build stage | Upstream passed them in; the label must match the patch set. |
| Provenance labels | Upstream sets none. |

## Runtime contract

As upstream: no entrypoint. The recipe's `start.sh` / `prepare.sh` run `tensorfold-native serve`
on each Spark with `--vision`. The image puts TensorFold's Python tree at
`/opt/tensorfold/python` for the vision helper (`TENSORFOLD_VISION_PYTHONPATH`) and the kernel set
at `TENSORFOLD_CUDA_KERNELS`. The 0010 settings above are read from the server's environment by
the helper it starts.

## Licensing

The recipe and the patch are Apache-2.0; TensorFold is Apache-2.0, with an MIT notice for code
written before v0.6.0; the weights are under the NVIDIA Open Model License (not in the image).
See [`attribution/THIRD_PARTY_NOTICES.md`](attribution/THIRD_PARTY_NOTICES.md) and the vendored
`recipe/NOTICE`.

## Credits

The serving recipe, patches 0001-0009, kernels and measurements are the work of Mia's AI Lab and
the recipe's contributors. TensorFold is by its contributors, and the Zig CUDA serving path is
credited in `recipe/NOTICE`. The keyed-ids approach follows the GLM-5.3-Flash TensorFold recipe's
image prompt reuse.
