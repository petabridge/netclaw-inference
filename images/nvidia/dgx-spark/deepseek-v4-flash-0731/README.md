# DeepSeek V4 Flash 0731 on DGX Spark

Two-node TP=2 vLLM runtime for `deepseek-ai/DeepSeek-V4-Flash-0731` on two
NVIDIA DGX Sparks.

The first candidate is deliberately a thin, reproducible derivative of the
exact r0b0tlab runtime digest already exercised in the testlab. It preserves
the vLLM `0.26.0+dspark.sm121.3`, FlashInfer B12X, NVFP4 DS-MLA, NCCL 2.30.4,
and DSpark K=6 implementation while moving distribution to Petabridge's public
GHCR namespace.

The base image, model revision, upstream repository commit, integrated vLLM
commit, NCCL version, and B12X commit are pinned in `dependency.lock.json`.
Model weights are not included.

This fast-path candidate is intended to unblock dual-Spark qualification. A
future release may replace the pinned downstream base with a Petabridge-owned
source build without changing the package name or release gates.
