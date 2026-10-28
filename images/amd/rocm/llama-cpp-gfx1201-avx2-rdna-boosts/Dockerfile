# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

# AVX2-baseline llama.cpp server for AMD RDNA4 (gfx1201), with the vendored
# rdna-boosts patch series (blocks 00-08) applied.
#
# WHY THIS IMAGE EXISTS: upstream llama.cpp leaves substantial RDNA4 decode and
# prefill performance unclaimed, in patch families that were never upstreamed.
# The sibling image (`llama-cpp-gfx1201-avx2`) builds stock upstream at v0.6.0
# and is the production default. THIS image exists to test whether a pinned
# third-party patch series, compiled against the SAME proven-good ROCm 7.2.3
# runtime and the SAME AVX2 CPU baseline, is faster on our workload.
#
# WHY ROCm 7.2.3 IS HELD CONSTANT: the publicly published third-party container
# for these patches ships ROCm 7.14.1 / 10.1 only. Our production runtime is
# 7.2.3, and 7.14 SIGILLed on gfx1201 during GPU init (testlab-setup#492).
# Testing the patches on 7.14 would confound patch performance with a known
# engine fault, so this image deliberately pins the 7.2.3 base instead. That
# is the entire reason to build here instead of pulling theirs.
#
# WHY BLOCKS 00-08 AND NOT 15: the blocks are a sequential `git am` series, not
# independent patches. Blocks 05, 08 and 13 fail to apply when blocks 01 and 04
# are skipped -- verified by application testing on the pinned base. 00-08 is a
# contiguous prefix and applies cleanly. Blocks 09-14 are excluded deliberately:
# 09/10 are host-resident-MoE and slab infrastructure, 11 is the qwen4exp
# (Qwen3.8-Flash-Next) architecture, 12 is a mixed campaign remainder, 13 is an
# opt-in draft-vocabulary subset that needs a re-cut GGUF, and 14 is a temporary
# tensor-split relaxation. None of them serve a dense single-GPU 27B replica.
#
# SUPPLY CHAIN: the ROCm base image is pinned by digest; llama.cpp is pinned to
# an immutable upstream commit verified by SHA-256; the patch series is VENDORED
# into this directory (patches/) with every file's SHA-256 recorded in
# dependency.lock.json and re-verified during the build. The upstream patches
# repository publishes a new release roughly hourly and its `main` branch is not
# a stable reference, which is why the series is vendored rather than fetched.

FROM rocm/dev-ubuntu-24.04@sha256:8ab378c7264c7e15496f548392c518a6b2feeb322d6233364c6c2ebc8e7ffb42

USER root

# Run every build step under bash so `set -o pipefail` and [[ ]] work; the
# base image's default /bin/sh is dash, which rejects both.
SHELL ["/bin/bash", "-c"]

ENV PATH=/opt/rocm/bin:${PATH}
WORKDIR /opt/petabridge/llama.cpp

RUN set -euxo pipefail; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install --yes --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        rocblas-dev \
        hipblas-dev; \
    rm -rf /var/lib/apt/lists/*

# Pinned, checksum-verified llama.cpp source. This is the base the patch series
# was exported against (see patches/upstream-release.json "base"). It is NOT the
# v0.6.0 commit the sibling image builds -- moving forward is unavoidable here,
# because the patches are `git am` commits against this tree.
ADD --checksum=sha256:64a750f710b8cf922d7dd931a1297a9cc236039f7e0f10759e9bfa555966e7ca \
    https://github.com/ggml-org/llama.cpp/archive/6050b467c07a95c7b87debd4ddac2f8ba8d9b807.tar.gz \
    /tmp/llama.cpp.tar.gz

RUN set -euxo pipefail; \
    tar --extract --gzip --file /tmp/llama.cpp.tar.gz --strip-components=1; \
    rm /tmp/llama.cpp.tar.gz

# Vendored patch series (blocks 00-08). Copied before the build so the
# checksum verification and the `git am` happen against identical bytes.
COPY patches/ /opt/petabridge/patch-series/

# Verify every vendored patch against the SHA-256 recorded in the lock data,
# then apply the series. Fail loudly on a checksum mismatch or a rejected hunk:
# a partially applied series produces a binary that is not the tested config,
# which is worse than a failed build.
#
# The base tree is committed before patching because `git am --3way` resolves
# against a real ancestor; on a bare `git init` with no HEAD it fails with
# "sha1 information is lacking or useless" and "could not build fake ancestor".
# Verified: without this commit the build fails on block 00.
RUN set -euxo pipefail; \
    cd /opt/petabridge/patch-series; \
    sha256sum --check --strict SHA256SUMS; \
    cd /opt/petabridge/llama.cpp; \
    git init --quiet .; \
    git config user.email "build@petabridge.invalid"; \
    git config user.name "netclaw-inference build"; \
    git add --all; \
    git commit --quiet --message "pinned llama.cpp source (base for patch series)"; \
    applied=0; \
    for p in /opt/petabridge/patch-series/000?-*.patch; do \
        echo "applying $(basename "$p")"; \
        git am --3way --keep-cr --quiet "$p"; \
        applied=$((applied + 1)); \
    done; \
    echo "applied ${applied} patch blocks"

# ROCm / HIP build targeting gfx1201, with the SAME explicit AVX2 CPU baseline
# as the sibling image. The AVX-512 guard below is unchanged: the patch series
# touches ggml-cuda and cross-cutting CPU paths, so the baseline must be proven
# after patching, not assumed.
RUN set -euxo pipefail; \
    export HIPCXX="$(hipconfig -l)/clang"; \
    export HIP_PATH="$(hipconfig -R)"; \
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/llama.cpp \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS=gfx1201 \
        -DGGML_HIP_ROCWMMA_FATTN=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX=ON \
        -DGGML_AVX2=ON \
        -DGGML_FMA=ON \
        -DGGML_F16C=ON \
        -DGGML_AVX512=OFF \
        -DGGML_AVX512_VBMI=OFF \
        -DGGML_AVX512_VNNI=OFF \
        -DGGML_AVX512_BF16=OFF \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TESTS=OFF; \
    cmake --build build --config Release --parallel "$(nproc)"; \
    cmake --install build; \
    install -D -m 0644 LICENSE /opt/llama.cpp/share/doc/llama.cpp/LICENSE; \
    echo "/opt/llama.cpp/lib" > /etc/ld.so.conf.d/petabridge-llama.conf; \
    ldconfig

# Regression guard: assert the CPU backend contains no AVX-512 (zmm) opcodes.
# The patch series builds the CPU paths too, so this is re-proven post-patch.
RUN set -euxo pipefail; \
    target="$(find /opt/llama.cpp -name 'libggml-cpu*.so*' -type f | head -n1)"; \
    if [[ -z "$target" ]]; then target=/opt/llama.cpp/bin/llama-server; fi; \
    if objdump -d --no-show-raw-insn "$target" | grep -qE '%zmm[0-9]'; then \
        echo "AVX-512 (zmm) instruction found in $target -- CPU baseline is wrong" >&2; \
        exit 1; \
    fi; \
    echo "Verified AVX2 baseline: no AVX-512 (zmm) opcodes in $target"

# Parse the serving arguments before emitting an image. This is the contract
# check that catches an upstream flag being removed or renamed by the engine
# bump -- the exact failure that shipped once before with --no-mmap.
# Keep this command in sync with services/big-gpu/docker-compose.yml.
RUN set -euxo pipefail; \
    /opt/llama.cpp/bin/llama-server \
        -m /models/model.gguf \
        --mmproj /models/mmproj.gguf \
        --host 0.0.0.0 --port 8080 \
        -ngl 999 -fa 1 --load-mode none --jinja \
        --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
        --presence-penalty 1.5 --reasoning off \
        --ctx-size 262144 --cache-type-k q8_0 --cache-type-v q8_0 \
        --spec-type draft-mtp --spec-draft-n-max 3 -np 2 \
        --help >/dev/null; \
    cd /; \
    rm -rf /opt/petabridge/llama.cpp /opt/petabridge/patch-series

LABEL org.opencontainers.image.title="Petabridge llama.cpp ROCm gfx1201 AVX2 server (rdna-boosts)" \
      org.opencontainers.image.description="AVX2-baseline llama.cpp llama-server for AMD RDNA4 (gfx1201) with the vendored rdna-boosts patch blocks 00-08" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/petabridge/netclaw-inference" \
      org.opencontainers.image.base.name="rocm/dev-ubuntu-24.04" \
      org.opencontainers.image.base.digest="sha256:8ab378c7264c7e15496f548392c518a6b2feeb322d6233364c6c2ebc8e7ffb42" \
      io.petabridge.llamacpp.build.kind="pinned-source-full-build-plus-vendored-patches" \
      io.petabridge.llamacpp.upstream.commit="6050b467c" \
      io.petabridge.llamacpp.patch.series="rdna-boosts-v15-6050b467c-r161" \
      io.petabridge.llamacpp.patch.blocks="00-08" \
      io.petabridge.rocm.version="7.2.3" \
      io.petabridge.gpu.targets="gfx1201" \
      io.petabridge.cpu.baseline="x86-64-v3-avx2-no-avx512" \
      io.petabridge.status="experimental-unqualified"

COPY dependency.lock.json /usr/local/share/petabridge/dependency.lock.json
COPY attribution/ /usr/share/doc/netclaw-inference/

ENV PATH=/opt/llama.cpp/bin:/opt/rocm/bin:${PATH}

WORKDIR /models
