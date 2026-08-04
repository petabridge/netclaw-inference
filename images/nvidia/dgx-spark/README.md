# NVIDIA DGX Spark images

This family targets `linux/arm64`, NVIDIA GB10, and SM121 CUDA kernels.

The first planned runtime is a source-built DeepSeek V4 Flash 0731 image. The
older Anemll `0.1.1` image is useful implementation evidence, but it predates
the 0731 model architecture and is not an acceptable base for the Petabridge
release. Its relevant Apache-2.0/MIT work may be incorporated only with pinned
provenance and retained attribution.
