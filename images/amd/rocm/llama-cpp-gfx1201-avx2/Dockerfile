# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

# AVX2-baseline llama.cpp server for AMD RDNA4 (gfx1201 / Radeon AI PRO R9700).
#
# WHY THIS IMAGE EXISTS: stock upstream llama.cpp ROCm toolbox images have been
# shipping a libggml-cpu compiled with an AVX-512 CPU baseline. On an inference
# host whose CPU has AVX2 but not AVX-512, ggml_cpu_init() executes an AVX-512
# instruction at process startup, and llama-server dies with an illegal
# instruction (SIGILL) before it loads a model or touches the GPU. This image
# rebuilds llama.cpp from a pinned upstream commit with an explicit AVX2
# (x86-64-v3) CPU baseline -- AVX-512 disabled -- so the same binary runs on
# those hosts, while still targeting the gfx1201 GPU backend via ROCm.
#
# SUPPLY CHAIN: the ROCm base image is pinned by digest; the llama.cpp source
# is pinned to an immutable upstream commit and its GitHub archive is verified
# by SHA-256 before the build. Every inference binary in the runtime is
# compiled here from that pinned source -- nothing is inherited prebuilt. The
# ROCm 7.2.3 base intentionally matches the proven-good production runtime.
#
# Single-stage: the ROCm dev base already carries the compiler, so a second
# stage on the same base saves little; keeping one stage guarantees the rocBLAS
# and hipBLAS runtime libraries llama-server links are present at run time.

FROM rocm/dev-ubuntu-24.04@sha256:8ab378c7264c7e15496f548392c518a6b2feeb322d6233364c6c2ebc8e7ffb42

USER root

# Run every build step under bash so `set -o pipefail` and [[ ]] work; the
# base image's default /bin/sh is dash, which rejects both.
SHELL ["/bin/bash", "-c"]

ENV PATH=/opt/rocm/bin:${PATH}
WORKDIR /opt/petabridge/llama.cpp

# Build toolchain plus the ROCm math libraries. The base image provides hipcc
# and the HIP runtime, but not rocBLAS/hipBLAS -- the `-dev` packages supply
# the CMake config files ggml-hip's find_package() needs and pull in the
# runtime libraries llama-server dynamically links.
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

# Pinned, checksum-verified llama.cpp source (immutable upstream commit).
# GitHub commit archives are byte-stable, so BuildKit refuses the download
# unless the archive bytes match this digest.
ADD --checksum=sha256:3757add5ecea48727edf81b56d3e233eb82332d30ac475f23aa3efec4c647d1c \
    https://github.com/ggml-org/llama.cpp/archive/9b05354ec6fb58b4e665e9a39ebc40285c015638.tar.gz \
    /tmp/llama.cpp.tar.gz

RUN set -euxo pipefail; \
    tar --extract --gzip --file /tmp/llama.cpp.tar.gz --strip-components=1; \
    rm /tmp/llama.cpp.tar.gz

# ROCm / HIP build targeting gfx1201, with an explicit AVX2 CPU baseline.
#   GGML_NATIVE=OFF            -- do NOT use -march=native (the build runner's
#                                 CPU may have AVX-512; the deployment target
#                                 does not -- that mismatch is the whole bug)
#   GGML_AVX/AVX2/FMA/F16C=ON   -- x86-64-v3 baseline
#   GGML_AVX512*=OFF            -- the entire point: no AVX-512 in the CPU backend
#   GGML_HIP=ON + AMDGPU_TARGETS=gfx1201 -- RDNA4 GPU backend
#   BUILD_SHARED_LIBS=ON        -- emit libggml-cpu.so so the AVX-512 guard can
#                                  inspect the CPU backend directly
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

# Regression guard: assert the CPU backend contains no AVX-512 (zmm) opcodes --
# the exact failure mode this image exists to prevent. Fail loudly rather than
# ship a binary that would SIGILL on an AVX2-only host.
RUN set -euxo pipefail; \
    target="$(find /opt/llama.cpp -name 'libggml-cpu*.so*' -type f | head -n1)"; \
    if [[ -z "$target" ]]; then target=/opt/llama.cpp/bin/llama-server; fi; \
    if objdump -d --no-show-raw-insn "$target" | grep -qE '%zmm[0-9]'; then \
        echo "AVX-512 (zmm) instruction found in $target -- CPU baseline is wrong" >&2; \
        exit 1; \
    fi; \
    echo "Verified AVX2 baseline: no AVX-512 (zmm) opcodes in $target"

# Standalone smoke check, then drop the source/build tree.
RUN set -euxo pipefail; \
    /opt/llama.cpp/bin/llama-server --help >/dev/null; \
    cd /; \
    rm -rf /opt/petabridge/llama.cpp

LABEL org.opencontainers.image.title="Petabridge llama.cpp ROCm gfx1201 AVX2 server" \
      org.opencontainers.image.description="AVX2-baseline llama.cpp llama-server built from pinned source for AMD RDNA4 (gfx1201) ROCm hosts" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/petabridge/netclaw-inference" \
      org.opencontainers.image.base.name="rocm/dev-ubuntu-24.04" \
      org.opencontainers.image.base.digest="sha256:8ab378c7264c7e15496f548392c518a6b2feeb322d6233364c6c2ebc8e7ffb42" \
      io.petabridge.llamacpp.build.kind="pinned-source-full-build" \
      io.petabridge.llamacpp.upstream.commit="9b05354ec6fb58b4e665e9a39ebc40285c015638" \
      io.petabridge.llamacpp.upstream.build="b10433" \
      io.petabridge.llamacpp.source.archive.sha256="3757add5ecea48727edf81b56d3e233eb82332d30ac475f23aa3efec4c647d1c" \
      io.petabridge.rocm.version="7.2.3" \
      io.petabridge.gpu.targets="gfx1201" \
      io.petabridge.cpu.baseline="x86-64-v3-avx2-no-avx512"

COPY dependency.lock.json /usr/local/share/petabridge/dependency.lock.json
COPY attribution/ /usr/share/doc/netclaw-inference/

ENV PATH=/opt/llama.cpp/bin:/opt/rocm/bin:${PATH}

WORKDIR /models
