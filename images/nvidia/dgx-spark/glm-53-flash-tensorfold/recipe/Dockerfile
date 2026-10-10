# GLM-5.3-Flash EXL3 TensorFold runtime for two NVIDIA DGX Sparks (GB10, sm_121a).
#
# Materialized by Petabridge from the upstream recipe's inline Dockerfile:
# MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold, recipe/scripts/prepare.sh, the
# `docker build ... -f - patches` (an inline Dockerfile heredoc), at commit
# 05397e24ee1fa0a60df11ff5dcb7a68f45cff6cd. The heredoc is not a file upstream, so this is a
# materialization, not a copy: every change against it is marked "PETABRIDGE:" below and is
# recorded in ../petabridge-dockerfile.diff. See ../README.md and ../dependency.lock.json.
#
# PETABRIDGE: literal digest-pinned base image. Upstream: `ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3`
# followed by `FROM ${BASE_IMAGE}`; this repository prohibits base-image build arguments.
# The digest is the linux/arm64 manifest of nvcr.io/nvidia/pytorch:26.07-py3 (anonymously pullable).
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3

# PETABRIDGE: upstream took the engine spec, the extra packages and the patch hash as build
# arguments (`ARG TF_SPEC`, `ARG EXTRAS`, `ARG PATCHES_HASH`) and recipe/scripts/prepare.sh
# passed them in. This image is built by scripts/build-image.sh, which passes only the build
# arguments declared in image.json, so the three values are materialized here instead.
# PETABRIDGE: the TensorFold spec is pinned by immutable commit. Upstream installs the mutable
# tag (recipe/scripts/config.sh: TF_VERSION=v0.6.0, TF_REPO=https://github.com/ashhart/TensorFold.git);
# c4646171139ee8a3c38103eaa1699dad226ec12b is the commit that annotated tag points at.
ARG TF_REPO=https://github.com/ashhart/TensorFold.git
ARG TF_COMMIT=c4646171139ee8a3c38103eaa1699dad226ec12b
ARG TF_EXTRAS="av==18.1.0 xgrammar>=0.2.8,<0.3"

RUN pip install --no-cache-dir --upgrade "git+${TF_REPO}@${TF_COMMIT}" \
 && pip install --no-cache-dir ${TF_EXTRAS} \
 && tensorfold --version

# PETABRIDGE: upstream `COPY . /opt/tf-patches` with the patches/ directory itself as the build
# context (`docker build ... -f - patches`). This repository builds with the vendored recipe/
# directory as the context (image.json -> context), where `COPY .` would drag the whole recipe
# tree into the image, so the copy is narrowed to patches/. The patch application below is
# upstream's, unchanged. /opt/tf-patches is a build-time path only; no runtime code reads it.
COPY patches/ /opt/tf-patches/

RUN cd "$(python -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; patch -p0 --forward < "$p" || exit 1; done && \
    python -c "import tensorfold.cuda.server, tensorfold.families.glm5_next.cuda.engine, tensorfold.vision.glm, av, xgrammar"

# PETABRIDGE: upstream `ARG PATCHES_HASH` + `LABEL tf.patches=${PATCHES_HASH}`, with the value
# passed in by recipe/scripts/prepare.sh. Nothing passes it here, so the value is recorded
# literally and verified against the vendored patches at build time, so the label cannot drift
# from the patch set it names. The label is what the recipe's start.sh uses as its compiled
# kernel cache key (recipe/scripts/config.sh image_hash(): sha256 over patches/*.patch plus
# IMAGE_EXTRAS, first 12 hex characters), so keeping the recipe's own value lets start.sh
# reuse kernels compiled by an image built from the same patch set.
RUN set -eux; \
    computed="$( { cat /opt/tf-patches/*.patch; printf '%s\n' 'av==18.1.0 xgrammar>=0.2.8,<0.3'; } | sha256sum | cut -c1-12 )"; \
    test "$computed" = "0718b016e3c6"; \
    printf 'patch set hash %s\n' "$computed"
LABEL tf.patches="0718b016e3c6"

ENV HF_HOME=/root/.cache/huggingface TORCH_EXTENSIONS_DIR=/cache/torch_extensions TRITON_CACHE_DIR=/cache/triton
WORKDIR /workspace

# PETABRIDGE: provenance labels. Upstream sets none. The recipe's own license is Apache-2.0;
# see ../attribution/THIRD_PARTY_NOTICES.md.
LABEL org.opencontainers.image.source="https://github.com/petabridge/netclaw-inference" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.petabridge.upstream.recipe="https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold" \
      io.petabridge.upstream.recipe.commit="05397e24ee1fa0a60df11ff5dcb7a68f45cff6cd" \
      io.petabridge.engine="TensorFold" \
      io.petabridge.engine.commit="c4646171139ee8a3c38103eaa1699dad226ec12b"
