# DeepSeek V4.1 Flash 1017 on TensorFold, three DGX Sparks (candidate)

Image id: `tensorfold-deepseek-v4-1-flash-1017-gb10`
(`ghcr.io/petabridge/tensorfold-deepseek-v4-1-flash-1017-gb10`)

> **Candidate, not yet GPU-qualified.** The build is enabled so a `sha-<commit>` candidate can be
> pulled onto three DGX Sparks and tested. Nothing in this folder has been built or run yet. Upstream
> says a local rebuild "still needs functional GPU qualification" and records its own qualification
> against the binaries it built, not these. Do not promote until the [acceptance steps](#acceptance)
> pass.

A source build of [LakyFx/deepseek-v41-flash-exl3-tp3-spark](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark)
at tag `v1.2.5` (commit `b4e476d60f01fda564d760c5906ac1358390c728`), variant `1.2.5`, build recipe
`k2a-pb1` (upstream's `k2a` plus Petabridge overlay files, see
[Petabridge patches](#petabridge-patches-recipe-k2a-pb1)): the TensorFold Zig engine serving DeepSeek V4.1 Flash (`TF351-1017`) with tensor parallel 3, one
rank per Spark. Upstream ships no image for this path; its docs build and prepare on the host. This image
does the host-side work once, in container stages, so a rank only needs the weights.

## What is in the image

| Path | What |
| --- | --- |
| `/srv/ds41f-tensorfold/bin-1.2.5/` | `tensorfold-native` (rank 0, HTTP server), `tf-dsv41-lanes` (ranks 1 and 2) and `BUILD.json`, from `build.py build --recipe k2a-pb1` and `build.py install --variant 1.2.5-pb1`. Every rank carries both binaries and the receipt: the launcher validates the receipt on each rank. |
| `/srv/ds41f-tensorfold/kit/` | The extracted kernel kit (Triton AOT cubins, vision kit, token map) plus the four RoPE tables for the 10M-position pool (`rope-*.f32`, 5,121,245,184 bytes) and the rewritten `rope.json`. Identical on all ranks. |
| `/opt/ds41f/recipe/` | The upstream files `launch.py` needs, with their relative paths: `tensorfold/launch.py`, `profiles/`, `variants/patch.py`, `variants/recipes/k2a.json`, Petabridge's `variants/recipes/k2a-pb1.json` (the recipe this image's `BUILD.json` names), `templates/`, `prepare_template.py`, `cluster.example.json`, `LICENSE`, `THIRD_PARTY_NOTICES.md`. This is the `WORKDIR`. |
| `/usr/local/lib/libnccl.so.2` | NCCL 2.30.7, built from source for `sm_121`. The base image's NCCL 2.28.3 is removed. |
| `/usr/local/cuda/lib64/libcublas.so.13`, `libcublasLt.so.13` | From the CUDA 13.0.3 runtime base. The engine dlopens both and the container toolkit does not inject them. |
| `libibverbs.so.1`, the mlx5 provider | `libibverbs1` and `ibverbs-providers` from rdma-core 50.0 (Ubuntu 24.04). |
| `/opt/ds41f/selfcheck.py` | The checks the build runs; `--runtime` adds a `libcuda.so.1` load. |
| `/usr/share/doc/ds41f-tensorfold/`, `/usr/share/doc/nccl/` | Licenses and notices, and the build's CPU check logs. |

Not in the image: the weights, the gateway (`ds41f-gateway`), PyTorch, and the Zig toolchain.
- Pillow 12.3.0 in the system `python3` (`/usr/local/lib/python3.12/dist-packages`), for the engine's image
  decoder child (`dsv41/pil.zig`). Without it the engine refuses every non-PNG image (`vision: no Pillow`).
  The version matches the qualified serving image, so decoded pixels match.

## How the controller runs it

One container per Spark, started by the controller. The launcher builds the argv and environment from a
cluster config and then `exec`s the engine:

```bash
docker run -d --name ds41f-rank0 \
  --runtime nvidia --gpus all \
  --network host \
  --device /dev/infiniband \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -v /srv/ds41f-tensorfold/weights:/srv/ds41f-tensorfold/weights \
  -v /etc/ds41f-tensorfold.json:/etc/ds41f-tensorfold.json:ro \
  ghcr.io/petabridge/tensorfold-deepseek-v4-1-flash-1017-gb10@sha256:<digest> \
  python3 -m tensorfold.launch --config /etc/ds41f-tensorfold.json --variant 1.2.5 --rank 0 --run
```

Ranks 1 and 2 use `--rank 1` and `--rank 2`. Without `--run` the launcher only prints the argv and
environment as JSON, which is a safe dry run.

- Run `python3 -m tensorfold.launch` from the image's `WORKDIR` (`/opt/ds41f/recipe`). Upstream has no
  `__init__.py`; `launch.py` does `from tensorfold.variants import patch`, so running it as a script
  (as upstream's `tp3@.service` does) fails to import. Do not override the working directory.
- The cluster config is upstream's `cluster.example.json` schema (`/opt/ds41f/recipe/tensorfold/cluster.example.json`
  in the image). Every rank uses `"root": "/srv/ds41f-tensorfold"`. `master_address` must be a dotted IPv4
  address that exists on rank 0's host. Leave `library_path` empty: setting it replaces `LD_LIBRARY_PATH`,
  and NCCL is already on the linker path.
- The launcher validates files and hashes on `--run` and refuses `REPLACE_*` placeholders.
- Start all three ranks within the engine's `TF_DS_START_TIMEOUT` (1,800 s). There is no rejoin: if a
  follower dies the head fails; restart all three.
- `GET /health` on rank 0's `api_port` reports `warming`; wait for `false` before testing chat, tools and
  vision.

### Mounts, devices and limits

| Need | Why |
| --- | --- |
| NVIDIA runtime, `--gpus all` | GB10 GPU and `libcuda.so.1` from the host driver (580.178.04, CUDA 13.0). |
| `--network host` | The engine binds `master_address`, NCCL and the RDMA ring use the host's NICs and RoCE addresses. Bridge networking does not work. |
| `--device /dev/infiniband` | `uverbs*` for libibverbs. `/sys/class/infiniband` is read from the host's sysfs. |
| `--ulimit memlock=-1:-1`, `--cap-add IPC_LOCK` | Pinned memory for RDMA and CUDA host registration (upstream's unit uses `LimitMEMLOCK=infinity`). `IPC_LOCK` is an inference from the code; confirm it is needed and drop it if not. |
| The weights mount at `/srv/ds41f-tensorfold/weights` | See the layout below. Use a local ext4 or xfs bind mount: the Engram tables are sparse files read with `O_DIRECT` and Linux AIO. Docker's default seccomp profile should permit `io_setup`/`io_submit`; confirm on first start. |

No `--memory` limit below what the engine needs: see [memory](#memory-and-pages).

### Weights layout (bind-mounted, not in the image)

Per rank, under `/srv/ds41f-tensorfold/weights` (from upstream `docs/DS41F-1.2.0.md`, `docs/DS41F-1.2.5.md`):

```
weights/
  models/TF351-1017/              HF models/TF351-1017/* (config.json, tokenizer, chat_template.jinja, vision.safetensors, ...); all ranks
  models/TF351-1017-1.2.5/        head-only overlay from prepare_template.py (rank 0 only)
  rank-cache/TF351-1017/
    rank{N}of3.bin               from the HF repo; must exist and not be a symlink
    rank{N}of3-1017-eb0abe4d9d54.bin   hardlink of the line above, same filesystem
  engram-tp3/                     SOURCE.json plus two sparse table files, this rank only
```

Preparation that stays on the host or runs in a container with the weights mounted writable:

1. Verify downloads against the release manifest (size and sha256), per upstream.
2. Engram extraction (GNU tar): `tar --sparse -xf engram-tp3/rank${RANK}.sparse.tar -C engram-tp3 --strip-components=1`,
   then `cmp engram-tp3/SOURCE.json engram-tp3/rank${RANK}/SOURCE.json`. Never extract two ranks into one directory.
3. The rank-binary hardlink: `ln rank${RANK}of3.bin rank${RANK}of3-1017-eb0abe4d9d54.bin` inside
   `rank-cache/TF351-1017/`. A symlink does not work (the loader accepts only regular files).
4. **Rank 0 only: the head template overlay.** Run it in the same container layout the engine will see:

   ```bash
   docker run --rm -v /srv/ds41f-tensorfold/weights:/srv/ds41f-tensorfold/weights \
     ghcr.io/petabridge/tensorfold-deepseek-v4-1-flash-1017-gb10@sha256:<digest> \
     python3 tensorfold/prepare_template.py --runtime /srv/ds41f-tensorfold
   ```

   It creates `models/TF351-1017-1.2.5/` with one **absolute symlink** per entry of `models/TF351-1017/`,
   and points `chat_template.jinja` at `/opt/ds41f/recipe/tensorfold/templates/1017-direct-output.jinja`
   inside the image. The links therefore dangle on the host and resolve only inside a container that has
   the weights at `/srv/ds41f-tensorfold/weights` and this image's recipe at `/opt/ds41f/recipe`. It checks
   the original template's sha256 (`7d135667...`) and refuses to overwrite an existing overlay directory.

The RoPE tables and the engine binaries are already in the image; no host step is needed for them.

## Memory and pages

- GB10 is an integrated GPU. The engine budgets from the **host's** `/proc/meminfo` (`MemAvailable`, page
  cache counted) minus a reserve, and refuses to start if it cannot parse it. The reserve defaults to
  `max(4 GiB, MemTotal / 10)`, about 12 GiB on 128 GB. `TENSORFOLD_MEMORY_RESERVE_GIB` (at least 2) overrides
  it and `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` caps it.
- A Docker `--memory` limit does not change `/proc/meminfo`. A limit lower than the engine needs ends in an
  OOM kill, not a refusal. Prefer no limit and keep other workloads off these GPUs.
- Margins are thin: about 86 GB of rank weights on about 128 GB of unified memory.
- Upstream qualified `K2a` on a **64K-page kernel** with a signed memory-saver module. The target hosts here
  run a **4K-page** kernel, and upstream has no native-TensorFold data for it. The pool (7,000,000 tokens),
  12 streams, 48 round rows and 3 GiB prompt snapshots are upstream's configured capacity, not a stress
  result, even on its own host. Expect to lower the pool or snapshot size on first bring-up and watch
  `MemAvailable`: copy `tensorfold/profiles/1.2.5.json` to a new profile name that sets `build_variant: "k2a-pb1"`
  (this image's `BUILD.json` names that recipe, so `k2a` fails the receipt check) and keeps
  `adaptive_learning: true` (the receipt check keys on those, not the name), or emit the same argv and
  environment from the controller. The pool must stay at least `alignUp(context + 50, 2048)` (403,456 at
  the default 401,408 context) and identical on all ranks; the 10M RoPE tables cover any pool up to 9,999,951.

## Acceptance

Do these before any promotion:

1. `python3 /opt/ds41f/selfcheck.py --runtime` in a container with the NVIDIA runtime: all libraries load,
   including `libcuda.so.1`.
2. On each Spark: `uname -r`, `getconf PAGESIZE`, `nvidia-smi`, `ibv_devices`, `ibv_devinfo`, and the RoCE
   GID at the configured `gid_index` on every selected device.
3. Start all three ranks with `--run`. The launcher must accept the receipt: it compares `BUILD.json` with
   `tensorfold/variants/recipes/k2a-pb1.json` and the binaries' sha256.
4. All ranks connect within 1,800 s, `/health` reaches `warming=false`, `/v1/models` lists
   `deepseek-v4.1-flash`.
5. Chat, MAX reasoning, a tool-call round trip, and an image request; then 12 concurrent short requests.
6. Watch `MemAvailable` per rank during a long-context run.

## Petabridge patches (recipe k2a-pb1)

The image builds `k2a-pb1`, upstream's `k2a` plus overlay files from `pb1/`. They go through upstream's own
`patch.py` and `build.py` unchanged, so the launcher checks `k2a-pb1` the way it checks `k2a`.

| Overlay | Change |
| --- | --- |
| `zig/src/server/tool_parse.zig` | A V4.1 DSML invoke naming a tool the request did not offer is returned as a tool call under its written name when calls are unlimited. Upstream rewrote it into the reply text: a streamed reply with no reasoning came back as raw DSML text (`finish_reason=stop`), and one with reasoning lost the call. The one-call policy (`parallel_tool_calls: false`) keeps upstream's behaviour. |
| `zig/src/server/tool_parse_test.zig` | Upstream's test updated for the new behaviour, plus a block mixing offered and unoffered tools. |
| `zig/src/core/lane_host.zig` | Between two 2,048-row chunks of a prompt pass, the head runs decode rounds for the other live streams for up to `TF_DS_PREFILL_INTERLEAVE_MS` ms, advancing the adaptive learner's clock as `run()` does. It delivers their tokens and takes their cancels, but leaves the filling job's own cancel for its pass. Ported by hand onto k2a's lane host because the patch's hunks misapplied there. Includes the patch's regression test. |
| `zig/src/core/lanes/engine.zig` | k2a's file plus `runnableCount`. |
| `zig/src/core/lanes/stream.zig` | k2a's file plus `Stream.between`. |
| `zig/src/core/lanes/fake.zig` | Upstream's file plus the fake backend calling between rounds. |
| `zig/src/families/dsv41/lanes.zig` | k2a's file plus the patch's hunks. The prompt pass offers rounds between chunks, joining the Engram read-ahead first, and the workers re-arm the read-ahead on `fill_chunk`. |
| `zig/src/families/dsv41/model.zig` | Upstream's file plus the patch's hunks. `fillChunk` keeps a joined read-ahead, and `verify` joins it. |

The prefill interleave is a port of MiaAI-Lab's [`zig/patches/0005-prefill-interleave.patch`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/blob/main/zig/patches/0005-prefill-interleave.patch) (author Coinupbtc). Without it, every prompt pass froze all decoding streams for its full length, at about 2.2K prompt tok/s, so a cold 135K-token prompt froze every reply for about a minute. With `TF_DS_PREFILL_INTERLEAVE_MS` set, the decoding streams keep getting rounds between the chunks of a prompt pass. It is off by default: unset or 0, the engine is unchanged. Prompts with images run whole, and the interleave is disabled while the core logs a trail.

`pb1/make_recipe.py` runs in the `engine` stage and writes the overlay objects, `variants/recipes/k2a-pb1.json`
(31 overlays: k2a has 27; the two `tool_parse` files, `fake.zig` and `model.zig` are new targets, and `lane_host.zig`, `engine.zig`, `stream.zig` and `dsv41/lanes.zig` replace k2a's own overlays of those files) and `profiles/1.2.5-pb1.json` (profile 1.2.5 with `build_variant: "k2a-pb1"`,
used only by `build.py install`). It refuses to run unless `k2a.json` is the pinned upstream file (`600c83c8...`)
and each replaced file is still the one it was written against (upstream's base, or k2a's own overlay), so a new upstream commit cannot silently
drop or misapply a patch. The Dockerfile asserts the written recipe file hashes to `b78a1a94...`. The image
carries both `recipes/k2a.json` and `recipes/k2a-pb1.json`.

Upstream report: [MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks#67](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/67).

To drop the patch once upstream fixes it: bump the pinned upstream commit, remove the `pb1/` steps from the
`engine` and `final` stages, build `--recipe k2a` and `install --variant 1.2.5` again, and point profiles back at
`build_variant: "k2a"`.

## Build

`Dockerfile` runs the upstream commands in stages. Departures from upstream's host procedure are marked
`PETABRIDGE:` in the file.

| Stage | Base | Does |
| --- | --- | --- |
| `nccl` | `nvcr.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:2396393d...` (arm64) | Builds NCCL `v2.30.7-1` (tarball pinned by sha256) with `NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"`, as upstream's `build/Dockerfile` L18-L19; asserts version code 23007. |
| `runtime-base`, `tools` | `nvcr.io/nvidia/cuda:13.0.3-runtime-ubuntu24.04@sha256:760753ab...` (arm64) | Pinned `python3`; `git`, `xz-utils`, `python3-venv`. |
| `recipe` | `tools` | Fetches the upstream commit (asserts `git rev-parse HEAD`), runs `tensorfold/extract.py --out /build/ds41f`. |
| `engine` | `recipe` | Zig 0.17.0 (sha256-pinned), `pb1/make_recipe.py` writes recipe `k2a-pb1`, then `build.py build --recipe k2a-pb1` and `build.py install --variant 1.2.5-pb1`. |
| `ropetools` | `tools` | Hash-locked CPU PyTorch 2.13.0 and numpy (`requirements-rope.txt`). |
| `final` | `runtime-base` | Installs libibverbs, swaps NCCL, copies the engine, kit and launcher files, generates and verifies the RoPE tables, runs `selfcheck.py --full` and `python3 -m tensorfold.launch --help`. |

Deviations from the upstream procedure:

| Change | Why |
| --- | --- |
| Builds recipe `k2a-pb1`, not upstream's `k2a`; `pb1/make_recipe.py` writes it in the `engine` stage | `k2a-pb1` is `k2a` plus overlays that fix an unoffered-tool-call bug and add an opt-in prefill interleave; everything else in `k2a` is unchanged. See [Petabridge patches](#petabridge-patches-recipe-k2a-pb1). |
| `make_rope.py` runs in the `final` stage against the copied kit, with the venv and the upstream scripts mounted from other stages | The 5.12 GB of tables is written once, into one layer, instead of being copied between stages (a second copy would put about 5 GB more in the builder's cache). The script reads only `kit/models/TF351/config.json` and writes the four tables and `rope.json`, so the result is the same; the tables are checked against `rope-hashes.json` by the script and again afterwards. |
| `extract.py` runs once in `recipe`, not inside the engine stage | `build.py` patches only `source/`; the engine and RoPE steps read `kit/` and `reference/` unchanged. |
| The recipe file is asserted to hash to `600c83c8...`, not `ed9c269a...` | `ed9c269a...` is the earlier file measured as `1.2.4k2a`. The blob at `v1.2.5` hashes to `600c83c8...` and is what `BUILD.json` binds. See `dependency.lock.json`. |
| Partial, sparse fetch of the upstream repo | Only `tensorfold/` and the root files are read; `results/` is large. HEAD is asserted. |
| Apt installs come from a dated Ubuntu snapshot with exact versions | Reproducible dependencies; see `dependency.lock.json`. |

## Licensing

Mixed. The recipe is MIT (LakyFx). The TensorFold engine is Apache-2.0 (earlier code MIT) with a NOTICE
that is shipped. NCCL is Apache-2.0 with BSD-3-Clause parts. CUDA runtime libraries (including cuBLAS) come
from NVIDIA's CUDA base image under NVIDIA's license terms. rdma-core is GPL-2 or BSD-2-Clause. Zig (MIT)
and PyTorch (BSD-3-Clause) are used at build time only. The weights carry their own license and are not in
the image. See [`attribution/THIRD_PARTY_NOTICES.md`](attribution/THIRD_PARTY_NOTICES.md).

## Credits

The recipe, the K2a overlay, the kernel kit and the qualification work are LakyFx's. The engine is TensorFold
and its contributors; the dsv41 family and the RDMA ring are credited in the engine's `NOTICE`.
