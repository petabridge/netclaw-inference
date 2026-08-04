# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM ghcr.io/r0b0tlab/deepseek-v4-flash-dspark-v026-sm121@sha256:ef852781efd6278cb3d908a4298874fb8b34fe84f0e1a03181ed415ec233a9d4 AS vllm-build

USER root
WORKDIR /opt/petabridge/vllm

# GitHub source archives are generated from an immutable commit, and BuildKit
# rejects the download unless the archive bytes match this checksum.
ADD --checksum=sha256:00cd0a71ecdfcdc96bcc6c2cddaa3e548d024f96c1a8354ed69c8331ae9154cf \
    https://github.com/vllm-project/vllm/archive/568afb3a13806beb53bb2e6bd518269357b237c0.tar.gz \
    /tmp/vllm.tar.gz

COPY source-overlay/ /tmp/source-overlay/
COPY source-overlay.sha256 runtime-overlay.sha256 /tmp/
COPY patches/vllm-v0.26.0-dspark-sm121.patch /tmp/

SHELL ["/bin/bash", "-c"]

RUN set -euxo pipefail; \
    echo "17432713400bcda550063bd11939a625d290d5d28e53ef05fd1360166749acd5  /tmp/vllm-v0.26.0-dspark-sm121.patch" | sha256sum --check -; \
    tar --extract --gzip --file /tmp/vllm.tar.gz --strip-components=1; \
    cp --archive /tmp/source-overlay/. .; \
    sha256sum --check --quiet /tmp/source-overlay.sha256

# The downstream SM121 changes are Python source. The CUDA, FlashInfer, B12X,
# and Rust/native vLLM artifacts remain byte-for-byte inherited from the pinned
# and already exercised parent. Package those native files as the explicit
# precompiled input to vLLM's own source build instead of downloading a wheel.
RUN python3 <<'PY'
import importlib.util
import pathlib
import zipfile

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("The pinned parent does not contain an importable vllm package")

package = pathlib.Path(next(iter(spec.submodule_search_locations)))
root = package.parent
with zipfile.ZipFile(
    "/tmp/parent-vllm-native.whl", "w", zipfile.ZIP_DEFLATED
) as archive:
    for path in sorted(package.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(root))
PY

RUN set -euxo pipefail; \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0+dspark.sm121.3 \
    VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/parent-vllm-native.whl \
    VLLM_USE_PRECOMPILED=1 \
    python3 -m pip wheel \
        --no-build-isolation \
        --no-deps \
        --wheel-dir /opt/petabridge/wheels \
        .

FROM ghcr.io/r0b0tlab/deepseek-v4-flash-dspark-v026-sm121@sha256:ef852781efd6278cb3d908a4298874fb8b34fe84f0e1a03181ed415ec233a9d4

USER root

LABEL org.opencontainers.image.title="Petabridge DeepSeek V4 Flash 0731 DGX Spark runtime" \
      org.opencontainers.image.description="Pinned-source vLLM 0.26.0 SM121 runtime for dual DGX Spark TP=2 qualification" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      org.opencontainers.image.base.name="ghcr.io/r0b0tlab/deepseek-v4-flash-dspark-v026-sm121" \
      org.opencontainers.image.base.digest="sha256:ef852781efd6278cb3d908a4298874fb8b34fe84f0e1a03181ed415ec233a9d4" \
      io.petabridge.vllm.build.kind="pinned-source-with-inherited-native-artifacts" \
      io.petabridge.vllm.upstream.commit="568afb3a13806beb53bb2e6bd518269357b237c0" \
      io.petabridge.vllm.source.archive.sha256="00cd0a71ecdfcdc96bcc6c2cddaa3e548d024f96c1a8354ed69c8331ae9154cf" \
      io.petabridge.vllm.patch.sha256="17432713400bcda550063bd11939a625d290d5d28e53ef05fd1360166749acd5" \
      io.petabridge.vllm.native-artifacts="inherited-from-pinned-base"

COPY --from=vllm-build /opt/petabridge/wheels/ /tmp/petabridge-wheels/
COPY runtime-overlay.sha256 /usr/local/share/petabridge/runtime-overlay.sha256
COPY dependency.lock.json /usr/local/share/petabridge/dependency.lock.json
COPY manifests/ /usr/local/share/petabridge/manifests/
COPY patches/ /usr/local/share/petabridge/patches/
COPY attribution/ /usr/share/doc/netclaw-inference/

SHELL ["/bin/bash", "-c"]

RUN set -euxo pipefail; \
    python3 -m pip uninstall --yes vllm; \
    python3 -m pip install --no-cache-dir --no-deps /tmp/petabridge-wheels/vllm-*.whl; \
    rm -rf /tmp/petabridge-wheels; \
    python3 -c 'from importlib.metadata import version; assert version("vllm") == "0.26.0+dspark.sm121.3"'; \
    cd /usr/local/lib/python3.12/dist-packages; \
    sha256sum --check --quiet /usr/local/share/petabridge/runtime-overlay.sha256

WORKDIR /vllm-workspace
