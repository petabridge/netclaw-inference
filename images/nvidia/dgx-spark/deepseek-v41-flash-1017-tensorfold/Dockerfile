# DeepSeek V4.1 Flash 1017 on TensorFold's Zig engine (variant 1.2.5, build recipe k2a-pb1: upstream's k2a plus
# the Petabridge overlays in ./pb1/) for three NVIDIA
# DGX Sparks (GB10, sm_121), tensor parallel 3. CANDIDATE: built from source, not yet GPU-qualified.
#
# Upstream: LakyFx/deepseek-v41-flash-exl3-tp3-spark, tag v1.2.5, commit
# b4e476d60f01fda564d760c5906ac1358390c728 (MIT; the engine it builds is TensorFold, Apache-2.0).
# Upstream publishes no image for this path. Its docs build and prepare on the host:
#   docs/DS41F-1.2.5.md (build, install), docs/TENSORFOLD-INSTALL.md (NCCL, RoPE), and build/Dockerfile
#   lines 18-19 (the NCCL 2.30.7 build, from the vLLM path). This file runs the same commands in
#   container stages. Every line that departs from them is marked "PETABRIDGE:". See ./README.md and
#   ./dependency.lock.json.
#
# The build context is this folder (image.json -> context).
#
# Stages:
#   nccl       NVIDIA CUDA 13.0.3 devel: NCCL v2.30.7-1 from source for sm_121
#   tools      CUDA 13.0.3 runtime base + git/xz/venv: shared by the stages below
#   recipe     the upstream repo at the pinned commit; extract.py unpacks source/, kit/, reference/
#   engine     Zig 0.17.0: pb1/make_recipe.py, then build.py build + install -> tensorfold-native, tf-dsv41-lanes,
#              BUILD.json
#   ropetools  hash-locked CPU PyTorch 2.13.0 + numpy venv (never copied into the image)
#   final      CUDA 13.0.3 runtime base + everything the three ranks need; weights are bind-mounted

# PETABRIDGE: the same snapshot of the Ubuntu 24.04 archive is used by every apt install below, so the
# versions pinned here (and the dependencies they pull in) resolve to the same packages on every build.
# This repository prohibits base-image build arguments; ARGs below are version pins, not base images.

# --- NCCL ---------------------------------------------------------------------------------------------
# PETABRIDGE: literal digest-pinned base: the linux/arm64 manifest of nvcr.io/nvidia/cuda:13.0.3-devel-ubuntu24.04.
FROM nvcr.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:2396393dc8a031a6faf810894ec8e40c8d80e2a1177cfda0c2bf81acac9aa240 AS nccl
ARG UBUNTU_SNAPSHOT=20261024T000000Z
ARG NCCL_TAG=v2.30.7-1
# PETABRIDGE: upstream pipes the tag tarball straight into tar; it is checked against a recorded sha256 here
# (codeload.github.com/NVIDIA/nccl/tar.gz/refs/tags/v2.30.7-1, tag object -> commit 73cf112295c3...).
ARG NCCL_TARBALL_SHA256=292a7f7a27b6754acaf46b5506a60758ca7b18cc1dfbd3d1d4e1e229d0863b4e
ENV PYTHONDONTWRITEBYTECODE=1
# NCCL's device-code generator (src/device/generate.py) needs python3, which the devel image lacks. gcc, g++
# and make are installed only if the devel image turns out not to carry them.
# PETABRIDGE: NVIDIA's CUDA images purge curl after setup, so it is installed here for the tarball download.
RUN --mount=type=bind,source=apt-snapshot-install.sh,target=/tmp/apt-snapshot-install.sh \
    set -eux; \
    extra=""; \
    for tool in gcc g++ make; do command -v "$tool" >/dev/null || extra="$extra $tool"; done; \
    bash /tmp/apt-snapshot-install.sh "${UBUNTU_SNAPSHOT}" python3=3.12.3-0ubuntu2.1 python3.12=3.12.3-1ubuntu0.17 curl=8.5.0-2ubuntu10.16 $extra
RUN set -eux; \
    mkdir -p /opt/nccl-src /opt/nccl/lib /opt/nccl/share/doc/nccl; \
    curl -fsSL --retry 5 -o /tmp/nccl.tar.gz "https://codeload.github.com/NVIDIA/nccl/tar.gz/refs/tags/${NCCL_TAG}"; \
    echo "${NCCL_TARBALL_SHA256}  /tmp/nccl.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/nccl.tar.gz -C /opt/nccl-src --strip-components=1; \
    rm /tmp/nccl.tar.gz
# Upstream build/Dockerfile L18-L19, with the sm_121-only gencode.
RUN set -eux; \
    make -C /opt/nccl-src -j2 src.build CUDA_HOME=/usr/local/cuda NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"; \
    grep -Eq '^#define NCCL_VERSION_CODE[[:space:]]+23007$' /opt/nccl-src/build/include/nccl.h; \
    cp -a /opt/nccl-src/build/lib/libnccl.so* /opt/nccl/lib/; \
    cp /opt/nccl-src/LICENSE.txt /opt/nccl-src/ThirdPartyNotices.txt /opt/nccl/share/doc/nccl/; \
    ls -l /opt/nccl/lib

# --- shared runtime-side base ---------------------------------------------------------------------------
# PETABRIDGE: literal digest-pinned base: the linux/arm64 manifest of nvcr.io/nvidia/cuda:13.0.3-runtime-ubuntu24.04.
# It ships libcublas.so.13 and libcublasLt.so.13 (libcublas-13-0 13.1.1.3-1, held by apt), which the engine
# dlopens and the NVIDIA container toolkit does not inject. The stages below share its layers (disk on CI).
FROM nvcr.io/nvidia/cuda:13.0.3-runtime-ubuntu24.04@sha256:760753ab2971b5c4b5ba8e055334e3c1126c3d88597fc702623eae402257e737 AS runtime-base
ARG UBUNTU_SNAPSHOT=20261024T000000Z
ENV PYTHONDONTWRITEBYTECODE=1
RUN --mount=type=bind,source=apt-snapshot-install.sh,target=/tmp/apt-snapshot-install.sh \
    bash /tmp/apt-snapshot-install.sh "${UBUNTU_SNAPSHOT}" python3=3.12.3-0ubuntu2.1 python3.12=3.12.3-1ubuntu0.17

# --- build tools --------------------------------------------------------------------------------------
FROM runtime-base AS tools
ARG UBUNTU_SNAPSHOT=20261024T000000Z
# curl: the engine stage downloads Zig (NVIDIA's CUDA images purge curl after setup).
RUN --mount=type=bind,source=apt-snapshot-install.sh,target=/tmp/apt-snapshot-install.sh \
    bash /tmp/apt-snapshot-install.sh "${UBUNTU_SNAPSHOT}" \
      git=1:2.43.0-1ubuntu7.4 git-man=1:2.43.0-1ubuntu7.4 \
      xz-utils=5.6.1+really5.4.5-1ubuntu0.4 \
      python3-venv=3.12.3-0ubuntu2.1 python3.12-venv=3.12.3-1ubuntu0.17 \
      curl=8.5.0-2ubuntu10.16

# --- upstream recipe, extracted ------------------------------------------------------------------------
FROM tools AS recipe
ARG UPSTREAM_REPO=https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark.git
ARG UPSTREAM_COMMIT=b4e476d60f01fda564d760c5906ac1358390c728
# Immutable commit, asserted after the fetch. core.autocrlf=false: upstream binds exact bytes
# (.gitattributes: tensorfold/** -text), so no line-ending conversion may happen.
# PETABRIDGE: a blobless sparse checkout of tensorfold/ plus the root files. The full tree carries results/
# evidence the build does not read; every file the build reads is below tensorfold/.
RUN set -eux; \
    git init -q /src/recipe; \
    cd /src/recipe; \
    git config core.autocrlf false; \
    git remote add origin "${UPSTREAM_REPO}"; \
    git config remote.origin.promisor true; \
    git config remote.origin.partialclonefilter blob:none; \
    git fetch -q --depth 1 --filter=blob:none origin "${UPSTREAM_COMMIT}"; \
    git sparse-checkout set --cone tensorfold; \
    git checkout -q --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "${UPSTREAM_COMMIT}"; \
    test -z "$(git status --porcelain)"
# Verifies size and sha256 of the three shipped archives (tensorfold/inputs.json), then unpacks source/, kit/
# and reference/. Local file work only. A build needs a fresh extraction, so this stage extracts once and the
# engine stage patches its own copy of source/.
RUN cd /src/recipe && python3 tensorfold/extract.py --out /build/ds41f

# --- engine -------------------------------------------------------------------------------------------
FROM recipe AS engine
ARG ZIG_VERSION=0.17.0
ARG ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
# Zig 0.17.0 (the repo's .zig-version), the same aarch64 tarball the Qwen3.8-Flash image pins.
RUN set -eux; \
    curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz"; \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c -; \
    mkdir -p /opt/zig; \
    tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1; \
    rm /tmp/zig.tar.xz; \
    /opt/zig/zig version
# PETABRIDGE: recipe k2a-pb1 is upstream's k2a plus the overlay files under ./pb1/overlay/, each at its target path.
# make_recipe.py writes their objects, recipes/k2a-pb1.json and profiles/1.2.5-pb1.json into the recipe tree, and
# refuses to run unless k2a.json and each replaced base file are the pinned upstream ones. build.py, patch.py and
# launch.py then check k2a-pb1 exactly as they check k2a. The overlay: a V4.1 DSML call to a tool the request did not
# offer is returned as a tool call instead of the reply's text (server/tool_parse.zig; upstream report MiaAI-Lab/
# DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks#67). The prefill interleave (opt-in through TF_DS_PREFILL_INTERLEAVE_MS) runs
# decode rounds for other streams between prompt chunks; it ports MiaAI-Lab's 0005-prefill-interleave.patch. The image
# prompt cache (on unless TF_DS_IMAGE_CACHE=0) keys a prompt with pictures by a hash of each picture's payload so its
# conversation resumes from a kept state; it ports 0006-image-prompt-cache.patch. The hash below is the recipe file
# make_recipe.py writes from those inputs.
COPY pb1/ /src/pb1/
RUN set -eux; \
    python3 /src/pb1/make_recipe.py --recipe-root /src/recipe; \
    echo "123514692e43e2a6ecb172033fc228a1cc8c348aeaa054f111c0dd6fd138999b  /src/recipe/tensorfold/variants/recipes/k2a-pb1.json" | sha256sum -c -
# build: applies the k2a-pb1 overlays to source/, runs the CPU checks (zig build test, four zig test runs), builds
# native + dsv41-tools for aarch64-linux-gnu with the prebuilt kit fatbins, and writes BUILD.json. No network.
# PETABRIDGE: build.py writes each CPU check's output to /build/built/build-<n>.log and fails the step without
# printing it, so a failed check is invisible in CI; on failure, print the tail of each log (the RDMA mock's
# per-rank lines dropped) and fail as before.
RUN cd /src/recipe && { python3 tensorfold/variants/build.py build --recipe k2a-pb1 \
      --build-root /build/ds41f --out /build/built --zig /opt/zig/zig \
    || { for f in /build/built/build-*.log; do echo "== $f"; grep -v '^RDMA rank' "$f" | tail -150; done; exit 1; }; }
# install: re-verifies BUILD.json against the recipe and copies the two binaries and BUILD.json to
# <runtime>/bin-1.2.5 (profile 1.2.5-pb1: profile 1.2.5 with build_variant k2a-pb1, same binary_subdir). The
# binaries are not stripped or touched afterwards.
RUN cd /src/recipe && python3 tensorfold/variants/build.py install \
      --built /build/built --runtime /srv/ds41f-tensorfold --variant 1.2.5-pb1 && \
    ls -l /srv/ds41f-tensorfold/bin-1.2.5

# --- RoPE generator (build-time only) ---------------------------------------------------------------------
FROM tools AS ropetools
COPY requirements-rope.txt /tmp/requirements-rope.txt
# PETABRIDGE: torch 2.13.0 (CPU) and numpy, plus their dependencies, all by sha256 (--require-hashes).
RUN set -eux; \
    python3 -m venv /opt/rope-venv; \
    /opt/rope-venv/bin/pip install --no-cache-dir --require-hashes --no-deps --only-binary=:all: \
      --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
      -r /tmp/requirements-rope.txt; \
    /opt/rope-venv/bin/python -c "import torch; assert torch.__version__ == '2.13.0+cpu', torch.__version__"

# --- Pillow for the engine's image decoder (installed into the system python3's dist-packages) -------------
FROM tools AS pillow
COPY requirements-pillow.txt /tmp/requirements-pillow.txt
# PETABRIDGE: the engine decodes non-PNG images through a `python3` child that imports Pillow
# (dsv41/pil.zig); without it, JPEG/WebP/... are refused ("vision: no Pillow (NoPillow)" at startup).
# Hash-locked, binary-only, no dependencies; --target keeps it out of any venv.
RUN set -eux; \
    python3 -m venv /opt/pip-venv; \
    /opt/pip-venv/bin/pip install --no-cache-dir --require-hashes --no-deps --only-binary=:all: \
      --target /opt/pillow -r /tmp/requirements-pillow.txt

# --- final image ----------------------------------------------------------------------------------------
FROM runtime-base AS final
ARG UBUNTU_SNAPSHOT=20261024T000000Z
# libibverbs1 + the mlx5 provider (ConnectX-7) from rdma-core 50.0 on Ubuntu 24.04. python3 is in runtime-base.
# PETABRIDGE: the CUDA runtime base carries libnccl2 2.28.3 (held by apt). The engine dlopens libnccl.so.2 and
# needs 2.30.7, so the distro package is removed to leave exactly one libnccl on the linker path.
RUN --mount=type=bind,source=apt-snapshot-install.sh,target=/tmp/apt-snapshot-install.sh \
    set -eux; \
    bash /tmp/apt-snapshot-install.sh "${UBUNTU_SNAPSHOT}" libibverbs1=50.0-2ubuntu0.2 ibverbs-providers=50.0-2ubuntu0.2; \
    apt-mark unhold libnccl2; \
    DEBIAN_FRONTEND=noninteractive apt-get purge -y libnccl2; \
    if dpkg -s libnccl2 >/dev/null 2>&1; then exit 1; fi; \
    rm -rf /var/lib/apt/lists/*

# NCCL 2.30.7 built for sm_121. Only the versioned file is copied; the soname links come from ldconfig.
COPY --from=nccl /opt/nccl/lib/libnccl.so.2.30.7 /usr/local/lib/libnccl.so.2.30.7
COPY --from=nccl /opt/nccl/share/doc/nccl/ /usr/share/doc/nccl/
RUN set -eux; \
    ldconfig; \
    ln -sf libnccl.so.2.30.7 /usr/local/lib/libnccl.so.2; \
    ldconfig; \
    test "$(ldconfig -p | grep -c 'libnccl\.so')" = 1; \
    ldconfig -p | grep 'libnccl\.so'

# Engine: bin-1.2.5/{tensorfold-native,tf-dsv41-lanes,BUILD.json}. All three ranks carry both binaries and
# the receipt (the launcher validates the receipt on every rank); rank 0 runs native, ranks 1-2 run lanes.
COPY --from=engine /srv/ds41f-tensorfold/bin-1.2.5/ /srv/ds41f-tensorfold/bin-1.2.5/
# The CPU check logs BUILD.json hashes (cpu_checks), kept as build evidence.
COPY --from=engine /build/built/build-*.log /usr/share/doc/ds41f-tensorfold/build-logs/

# Kernel kit (Triton AOT cubins, vision kit, token map, ...). Its rope.json is the stale 3.3M one until the
# RoPE step below rewrites it.
COPY --from=recipe /build/ds41f/kit/ /srv/ds41f-tensorfold/kit/

# Upstream files the launcher needs at runtime, with their relative paths (spec: tensorfold/launch.py,
# profiles/, variants/patch.py, variants/recipes/k2a.json, templates/, prepare_template.py), plus Petabridge's
# variants/recipes/k2a-pb1.json from the engine stage, plus
# cluster.example.json for reference and the recipe's license files. variants/objects/ and artifacts/ are
# build-time only and are not copied. No __init__.py exists upstream: run `python3 -m tensorfold.launch`
# from /opt/ds41f/recipe (the WORKDIR), never `python3 .../launch.py` as upstream's tp3@.service does.
COPY --from=recipe /src/recipe/LICENSE /src/recipe/THIRD_PARTY_NOTICES.md /opt/ds41f/recipe/
COPY --from=recipe /src/recipe/tensorfold/launch.py /src/recipe/tensorfold/prepare_template.py /src/recipe/tensorfold/cluster.example.json /opt/ds41f/recipe/tensorfold/
COPY --from=recipe /src/recipe/tensorfold/profiles/ /opt/ds41f/recipe/tensorfold/profiles/
COPY --from=recipe /src/recipe/tensorfold/templates/ /opt/ds41f/recipe/tensorfold/templates/
COPY --from=recipe /src/recipe/tensorfold/variants/patch.py /opt/ds41f/recipe/tensorfold/variants/patch.py
COPY --from=recipe /src/recipe/tensorfold/variants/recipes/k2a.json /opt/ds41f/recipe/tensorfold/variants/recipes/k2a.json
# The recipe file the launcher compares with BUILD.json: the blob at the pinned commit (git c688435a...).
RUN echo "600c83c88d72f22506facc7cc1883f8d9e4439268b7e1d198efdbcb4536de9bc  /opt/ds41f/recipe/tensorfold/variants/recipes/k2a.json" | sha256sum -c -
# PETABRIDGE: the recipe this image's BUILD.json names (profiles with build_variant k2a-pb1 select it).
COPY --from=engine /src/recipe/tensorfold/variants/recipes/k2a-pb1.json /opt/ds41f/recipe/tensorfold/variants/recipes/k2a-pb1.json
RUN echo "123514692e43e2a6ecb172033fc228a1cc8c348aeaa054f111c0dd6fd138999b  /opt/ds41f/recipe/tensorfold/variants/recipes/k2a-pb1.json" | sha256sum -c -

# RoPE tables for the 10M-position pool, generated on the CPU straight into the kit (5,121,245,184 bytes in
# four files plus a rewritten rope.json). PETABRIDGE: upstream runs make_rope.py against the extracted kit/;
# here it runs against the copied kit, with the venv, the upstream scripts and the RoPE reference mounted from
# earlier stages, so the 5.12 GB is written once into one layer of this image and the venv is never copied.
# make_rope.py checks each table's sha256 against tensorfold/rope-hashes.json while it writes; the assertion
# after it repeats that check from the finished files and asserts the 10M rope.json.
RUN --mount=type=bind,from=ropetools,source=/opt/rope-venv,target=/opt/rope-venv \
    --mount=type=bind,from=recipe,source=/src/recipe/tensorfold,target=/mnt/tensorfold \
    --mount=type=bind,from=recipe,source=/build/ds41f/reference,target=/mnt/reference \
    set -eux; \
    /opt/rope-venv/bin/python /mnt/tensorfold/make_rope.py \
      --reference /mnt/reference --kit /srv/ds41f-tensorfold/kit --pool 10000000; \
    python3 -c "import hashlib, json; e = json.load(open('/mnt/tensorfold/rope-hashes.json'))['10000000']; k = '/srv/ds41f-tensorfold/kit/'; assert e['rows'] == 10002432, e['rows']; assert json.load(open(k + 'rope.json')) == e, 'rope.json'; bad = [n for n in e if n.endswith('.f32') and hashlib.file_digest(open(k + n, 'rb'), 'sha256').hexdigest() != e[n]]; assert not bad, bad; print('rope tables verified', sorted(n for n in e if n.endswith('.f32')))"

# Licenses of what was built into the binaries and the kit, and Petabridge's notices.
COPY --from=recipe /build/ds41f/source/LICENSE /build/ds41f/source/NOTICE /build/ds41f/source/THIRD_PARTY_NOTICES.md /usr/share/doc/ds41f-tensorfold/tensorfold/
COPY --from=recipe /build/ds41f/source/LICENSES/ /usr/share/doc/ds41f-tensorfold/tensorfold/LICENSES/
COPY --from=recipe /build/ds41f/reference/LICENSE /usr/share/doc/ds41f-tensorfold/rope-reference/LICENSE
COPY attribution/THIRD_PARTY_NOTICES.md /usr/share/doc/ds41f-tensorfold/THIRD_PARTY_NOTICES.md
COPY selfcheck.py /opt/ds41f/selfcheck.py
# Pillow 12.3.0 for the image decoder child; Ubuntu's python3 searches /usr/local/lib/python3.12/dist-packages.
COPY --from=pillow /opt/pillow/ /usr/local/lib/python3.12/dist-packages/
RUN python3 -c "import io, PIL; from PIL import Image; \
    assert PIL.__version__ == '12.3.0', PIL.__version__; \
    bufs = {f: io.BytesIO() for f in ('JPEG', 'WEBP')}; \
    [Image.new('RGB', (8, 8), (10, 20, 30)).save(b, f) for f, b in bufs.items()]; \
    [Image.open(io.BytesIO(b.getvalue())).convert('RGB').load() for b in bufs.values()]; \
    print('Pillow', PIL.__version__, 'JPEG and WebP encode/decode OK')"

# Build-time checks: dlopen of libcublas.so.13, libcublasLt.so.13, libnccl.so.2 (ncclGetVersion == 23007) and
# libibverbs.so.1 by soname; BUILD.json against the binaries; the 10M RoPE tables by hash; and the launcher's
# import path (`python3 -m tensorfold.launch`, which needs the recipe root as cwd). libcuda.so.1 comes from
# the host driver at run time and is not checked here (`python3 /opt/ds41f/selfcheck.py --runtime` does).
WORKDIR /opt/ds41f/recipe
RUN set -eux; \
    python3 /opt/ds41f/selfcheck.py --full; \
    python3 -m tensorfold.launch --help >/dev/null

# PETABRIDGE: provenance labels. Upstream sets none for this path.
LABEL org.opencontainers.image.source="https://github.com/petabridge/netclaw-inference" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0 AND BSD-3-Clause AND LicenseRef-NVIDIA-CUDA-EULA" \
      io.petabridge.upstream.recipe="https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark" \
      io.petabridge.upstream.recipe.tag="v1.2.5" \
      io.petabridge.upstream.recipe.commit="b4e476d60f01fda564d760c5906ac1358390c728" \
      io.petabridge.upstream.recipe.variant="1.2.5" \
      io.petabridge.upstream.recipe.build="k2a" \
      io.petabridge.upstream.recipe.manifest.sha256="600c83c88d72f22506facc7cc1883f8d9e4439268b7e1d198efdbcb4536de9bc" \
      io.petabridge.recipe.build="k2a-pb1" \
      io.petabridge.recipe.manifest.sha256="123514692e43e2a6ecb172033fc228a1cc8c348aeaa054f111c0dd6fd138999b" \
      io.petabridge.engine="TensorFold (Zig engine: tensorfold-native, tf-dsv41-lanes), frozen fork shipped in the recipe" \
      io.petabridge.engine.source.sha256="8d4b2f808827fd247048082d122b8fab31b75e798b5ed86fd370103ef178469a" \
      io.petabridge.nccl.version="2.30.7-1" \
      io.petabridge.nccl.gencode="compute_121,sm_121" \
      io.petabridge.rope.pool="10000000" \
      io.petabridge.status="candidate, not GPU-qualified"
