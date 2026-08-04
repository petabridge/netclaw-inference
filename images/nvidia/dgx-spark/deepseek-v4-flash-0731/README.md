# DeepSeek V4 Flash 0731 on DGX Spark

Planned two-node TP=2 vLLM runtime for `deepseek-ai/DeepSeek-V4-Flash-0731` on
two NVIDIA DGX Sparks.

The definition remains disabled while the vLLM baseline and the required SM121
FlashInfer, b12x, NVFP4 DS-MLA, 0731 compatibility, and DSpark changes are
audited. No image can be built from this directory yet.

No vLLM version is pinned here today. The current live testlab runtime is based
on vLLM `0.26.0`, and `v0.26.0` is the likely upstream baseline for this image.
That upstream release predates the 0731 checkpoint, so all post-release 0731
changes must be explicit in the future dependency lock and patch provenance.
