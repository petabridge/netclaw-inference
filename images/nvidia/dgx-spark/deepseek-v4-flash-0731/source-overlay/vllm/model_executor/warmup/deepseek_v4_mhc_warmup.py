# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek V4 mHC TileLang kernels before serving requests.

Ported from lucifer1004/vllm-jasl with the two env-var knobs removed
(`VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP`, `VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES`).
Gating is intrinsic: non-DSv4 models and layers without hc_* attributes
return early, so the warmup is a no-op except where it's needed.
"""

import time
from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.tracing import instrument

logger = init_logger(__name__)

_AUTO_WARMUP_MAX_TOKENS = 16_384
# DeepGEMM path sets n_splits = n_sms // cdiv(num_tokens, block_m=64).
# TileLang specializes on n_splits, so powers-of-two alone miss remainder
# chunks (e.g. 265 after an 8K first chunk on a 16K prompt).
_DEEPGEMM_BLOCK_M = 64
_DEFAULT_TOKEN_SIZE_CANDIDATES = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    265,  # observed fail-closed remainder on 16K ladder rung
    512,
    1024,
    2048,
    4096,
    8192,
    8187,  # K6-adjusted max scheduled tokens
    16_384,
)


def _normalize_token_sizes(
    token_sizes: Iterable[int],
    *,
    max_tokens: int,
) -> list[int]:
    return sorted({size for size in token_sizes if 1 <= size <= max_tokens})


def _deepgemm_grid_bucket_token_sizes(max_tokens: int) -> list[int]:
    """One representative token count per DeepGEMM split-k grid bucket."""
    if max_tokens <= 0:
        return []
    sizes: set[int] = set()
    # grid_size = cdiv(num_tokens, 64); cover start/end of each bucket.
    max_grid = (max_tokens + _DEEPGEMM_BLOCK_M - 1) // _DEEPGEMM_BLOCK_M
    for grid in range(1, max_grid + 1):
        start = (grid - 1) * _DEEPGEMM_BLOCK_M + 1
        end = min(max_tokens, grid * _DEEPGEMM_BLOCK_M)
        sizes.add(start)
        sizes.add(end)
    return sorted(sizes)


def _select_mhc_warmup_token_sizes(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int],
) -> list[int]:
    if max_tokens <= 0:
        return []

    max_auto_tokens = min(max_tokens, _AUTO_WARMUP_MAX_TOKENS)
    candidates: list[int] = list(_DEFAULT_TOKEN_SIZE_CANDIDATES)
    candidates.extend(cudagraph_capture_sizes)
    candidates.append(max_auto_tokens)
    # Cover every DeepGEMM n_splits bucket up to the scheduler token ceiling.
    candidates.extend(_deepgemm_grid_bucket_token_sizes(max_auto_tokens))
    return _normalize_token_sizes(candidates, max_tokens=max_auto_tokens)


def _find_first_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    # NVIDIA path inlines mhc_pre_tilelang / mhc_fused_post_pre_tilelang in
    # forward(); there is no layer.hc_pre/hc_post method. Gate on the real
    # parameter attributes instead.
    required = (
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
        "attn_norm",
        "ffn_norm",
        "hidden_size",
        "hc_mult",
        "rms_norm_eps",
        "hc_eps",
        "hc_post_alpha",
        "hc_sinkhorn_iters",
    )
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4DecoderLayer":
            continue
        if all(hasattr(module, attr) for attr in required):
            return module
    return None


def _find_deepseek_v4_model(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4Model":
            continue
        if all(
            hasattr(module, attr)
            for attr in ("hc_head_fn", "hc_head_scale", "hc_head_base")
        ):
            return module
    return None


def _warmup_layer_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    """Warm the NVIDIA-inlined TileLang mHC kernels used on the real path."""
    from vllm.model_executor.kernels.mhc.tilelang import (
        mhc_fused_post_pre_tilelang,
        mhc_pre_tilelang,
    )

    max_tokens = max(token_sizes)
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device
    residual = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    attn_norm_weight = layer.attn_norm.weight.data
    attn_norm_eps = float(layer.attn_norm.variance_epsilon)
    ffn_norm_weight = layer.ffn_norm.weight.data
    ffn_norm_eps = float(layer.ffn_norm.variance_epsilon)

    for size in token_sizes:
        residual_slice = residual[:size]
        # First-layer style pure pre with fused norm (DeepGEMM n_splits path).
        post_mix, res_mix, x = mhc_pre_tilelang(
            residual_slice,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            float(layer.rms_norm_eps),
            float(layer.hc_eps),
            float(layer.hc_eps),
            float(layer.hc_post_alpha),
            int(layer.hc_sinkhorn_iters),
            norm_weight=attn_norm_weight,
            norm_eps=attn_norm_eps,
        )
        # Subsequent fused post+pre path used for FFN (and later layers).
        mhc_fused_post_pre_tilelang(
            x,
            residual_slice,
            post_mix,
            res_mix,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            float(layer.rms_norm_eps),
            float(layer.hc_eps),
            float(layer.hc_eps),
            float(layer.hc_post_alpha),
            int(layer.hc_sinkhorn_iters),
            n_splits=1,
            tile_n=1,
            norm_weight=ffn_norm_weight,
            norm_eps=ffn_norm_eps,
        )


def _warmup_hc_head(
    model: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    # Upstream a8887c208 ("[DSV4] aiter mhc support (ROCm)") refactored
    # ``hc_head`` from a free function into the ``HCHeadOp`` CustomOp
    # instance attached to the model as ``hc_head_op``. We call through
    # that instance so the warmup exercises the same dispatched
    # implementation as the inference path.
    hc_head_op = getattr(model, "hc_head_op", None)
    if hc_head_op is None:
        return

    max_tokens = max(token_sizes)
    hidden_size = int(model.config.hidden_size)
    hc_mult = int(model.hc_mult)
    device = model.hc_head_fn.device
    hidden_states = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        hc_head_op(
            hidden_states[:size],
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base,
            model.rms_norm_eps,
            model.hc_eps,
        )


@instrument(span_name="DeepSeek V4 mHC warmup")
def deepseek_v4_mhc_warmup(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    # Cheap model-type gate before walking ``model.modules()``. The class
    # walk below is O(num_layers) and shows up in startup time on very
    # large checkpoints; bail out for any model that is not DeepSeek V4.
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None) if config is not None else None
    if model_type is not None and model_type != "deepseek_v4":
        return

    layer = _find_first_mhc_layer(model)
    if layer is None:
        return

    device = layer.hc_attn_fn.device
    if device.type != "cuda":
        return

    deepseek_model = _find_deepseek_v4_model(model)
    token_sizes = _select_mhc_warmup_token_sizes(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
    )
    if not token_sizes:
        return

    started = time.perf_counter()
    logger.info(
        "Warming up DeepSeek V4 mHC TileLang kernels for token sizes: %s",
        token_sizes,
    )
    with torch.inference_mode():
        _warmup_layer_mhc(layer, token_sizes)
        if deepseek_model is not None:
            _warmup_hc_head(deepseek_model, token_sizes)
        torch.accelerator.synchronize()
    logger.info(
        "DeepSeek V4 mHC TileLang warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )
