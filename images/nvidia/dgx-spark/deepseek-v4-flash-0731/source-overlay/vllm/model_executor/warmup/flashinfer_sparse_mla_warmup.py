# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup and autotune helpers for FlashInfer sparse MLA backends."""

from contextlib import suppress
from typing import TYPE_CHECKING, cast

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_file,
    write_flashinfer_autotune_cache,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import autotune as flashinfer_autotune
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2GPUModelRunner
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_DEEPSEEK_V4_SPARSE_MLA_BACKENDS = frozenset(
    {
        "FLASHMLA_SPARSE_DSV4",
        "FLASHINFER_MLA_SPARSE_DSV4",
        "ROCM_FLASHMLA_SPARSE_DSV4",
        "DEEPSEEK_SPARSE_SWA",
    }
)
_FLASHINFER_MLA_SPARSE_BACKENDS = frozenset({"FLASHINFER_MLA_SPARSE_SM120"})
_DEEPSEEK_V4_FLASHINFER_MLA_SPARSE_BACKENDS = frozenset({"FLASHINFER_MLA_SPARSE_DSV4"})

_FLASHINFER_SM120_SPARSE_MLA_DECODE_LABELS = {
    "FLASHINFER_MLA_SPARSE_SM120": "DSv3.2",
    "FLASHINFER_MLA_SPARSE_DSV4": "DSv4",
}

_SPARSE_MLA_MIXED_WARMUP_TOKENS = 16


def _attention_backend_name(backend: object) -> str | None:
    get_name = getattr(backend, "get_name", None)
    if get_name is None:
        return None
    try:
        return get_name()
    except NotImplementedError:
        return None


def _has_deepseek_v4_sparse_mla_backend(runner: "GPUModelRunner") -> bool:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name in _DEEPSEEK_V4_SPARSE_MLA_BACKENDS:
                return True
    return False


def _flashinfer_sparse_mla_decode_label(
    runner: "GPUModelRunner",
    allowed_backends: frozenset[str],
) -> str | None:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name in allowed_backends:
                return _FLASHINFER_SM120_SPARSE_MLA_DECODE_LABELS.get(name)
    return None


def _clamp_warmup_tokens(num_tokens: int, max_tokens: int) -> int:
    return max(0, min(num_tokens, max_tokens))


def _uses_v2_model_runner(runner: "GPUModelRunner") -> bool:
    vllm_config = getattr(runner, "vllm_config", None)
    return bool(getattr(vllm_config, "use_v2_model_runner", False))


def _resolve_dsv4_portable_prefill_num_heads(
    runner: "GPUModelRunner",
) -> tuple[int, ...]:
    """Return every TP-local padded Q-head count used by loaded attention.

    Production DeepSeek-V4 stacks can expose more than one padding policy at
    once (for example FlashMLA pads 32 local heads to 64, while the SM120
    FlashInfer path keeps 32). Triton's multihead accumulate kernel treats
    ``num_heads`` and contiguous ``stride_q_t = num_heads * d_qk`` as
    constexpr compile keys, so warming only one layout leaves the other as a
    fail-closed inference JIT. Warm every reachable padded layout.
    """
    model = getattr(runner, "model", None)
    if model is None:
        raise RuntimeError("Cannot resolve DeepSeek V4 padded Q-head count: no model")

    config = runner.vllm_config
    hf_config = config.model_config.hf_config
    tp_size = int(config.parallel_config.tensor_parallel_size)
    global_heads = int(hf_config.num_attention_heads)
    if global_heads % tp_size != 0:
        raise RuntimeError(
            f"DeepSeek V4 attention heads {global_heads} are not divisible "
            f"by TP={tp_size}"
        )
    local_heads = global_heads // tp_size
    head_dim = int(hf_config.head_dim)
    padded_heads: set[int] = set()
    for module in model.modules():
        if getattr(module, "n_local_heads", None) != local_heads:
            continue
        if getattr(module, "head_dim", None) != head_dim:
            continue
        instance_pad = getattr(module, "padded_heads", None)
        if instance_pad is not None:
            padded_heads.add(int(instance_pad))  # type: ignore[arg-type]
        getter = getattr(type(module), "get_padded_num_q_heads", None)
        if callable(getter):
            # Abstract bases may raise; ignore unsupported head counts.
            with suppress(Exception):
                padded_heads.add(int(type(module).get_padded_num_q_heads(local_heads)))
    if not padded_heads:
        raise RuntimeError(
            "Cannot resolve DeepSeek V4 padded Q-head count from loaded "
            "attention modules"
        )
    # Portable SM12x FlashMLA prefill pads local<=64 heads to 64 even when
    # some loaded modules report unpadded local counts (e.g. SM120 FlashInfer
    # siblings). Always warm both layouts when local heads fit the FlashMLA
    # 64-head pad rule so fail-closed inference cannot miss stride_q_t.
    if local_heads <= 64:
        padded_heads.add(64)
    padded_heads.add(local_heads)
    ordered = tuple(sorted(padded_heads))
    if any(num_heads < local_heads for num_heads in ordered):
        raise RuntimeError(
            "DeepSeek V4 padded Q-head count "
            f"{[n for n in ordered if n < local_heads]} is below local heads "
            f"{local_heads}"
        )
    return ordered


def _run_flashinfer_sparse_mla_decode_autotune(
    worker: "Worker",
    num_tokens: int,
    allowed_backends: frozenset[str],
) -> bool:
    """Autotune FlashInfer's SM120 sparse-MLA decode path."""
    runner = worker.model_runner
    log_label = _flashinfer_sparse_mla_decode_label(runner, allowed_backends)
    if log_label is None:
        return False
    if worker.vllm_config.kernel_config.enable_flashinfer_autotune is not True:
        return False
    if not has_flashinfer() or not current_platform.is_device_capability_family(120):
        return False

    try:
        from flashinfer.autotuner import AutoTuner
    except ImportError:
        logger.warning(
            "Skipping FlashInfer SM120 sparse MLA decode autotune because "
            "FlashInfer autotuner is unavailable."
        )
        return False

    from vllm.distributed.parallel_state import get_world_group

    world = get_world_group()
    is_leader = world.rank_in_group == 0
    cache_path = resolve_flashinfer_autotune_file(runner)

    dummy_run_kwargs = dict(
        num_tokens=num_tokens,
        skip_eplb=True,
        is_profile=True,
        force_attention=True,
        create_mixed_batch=True,
    )

    if is_leader:
        logger.info(
            "Autotuning FlashInfer SM120 sparse MLA %s decode with cache: %s",
            log_label,
            cache_path,
        )

    with torch.inference_mode():
        warmup_executed = True
        if is_leader:
            if _uses_v2_model_runner(runner) and runner.max_num_reqs >= 2:
                v2_runner = cast("V2GPUModelRunner", runner)
                warmup_executed = run_mixed_prefill_decode_warmup(
                    v2_runner,
                    worker.execute_model,
                    worker.sample_tokens,
                    num_tokens,
                    mixed_step_context=flashinfer_autotune(True, cache=str(cache_path)),
                    req_id_prefix="_sparse_mla_v2_warmup",
                )
            else:
                with flashinfer_autotune(True, cache=str(cache_path)):
                    runner._dummy_run(**dummy_run_kwargs)
        else:
            if _uses_v2_model_runner(runner) and runner.max_num_reqs >= 2:
                v2_runner = cast("V2GPUModelRunner", runner)
                warmup_executed = run_mixed_prefill_decode_warmup(
                    v2_runner,
                    worker.execute_model,
                    worker.sample_tokens,
                    num_tokens,
                    req_id_prefix="_sparse_mla_v2_warmup",
                )
            else:
                runner._dummy_run(**dummy_run_kwargs)

    if not warmup_executed:
        return False

    tune_results: bytes | None = None
    if is_leader and cache_path.exists():
        with open(cache_path, "rb") as f:
            tune_results = f.read()

    tune_results = world.broadcast_object(tune_results, src=0)
    if tune_results is None:
        logger.warning(
            "No FlashInfer SM120 sparse MLA %s decode autotune cache entries found. "
            "Falling back to FlashInfer's default tactic heuristic.",
            log_label,
        )
        world.barrier()
        return True

    write_flashinfer_autotune_cache(cache_path, tune_results)
    world.barrier()

    AutoTuner.get().load_configs(str(cache_path))
    logger.info(
        "FlashInfer SM120 sparse MLA %s decode autotune cache loaded on rank %d "
        "from %s.",
        log_label,
        world.rank_in_group,
        cache_path,
    )
    return True


def _flashinfer_sparse_mla_decode_autotune(
    worker: "Worker",
    num_tokens: int,
) -> bool:
    return _run_flashinfer_sparse_mla_decode_autotune(
        worker, num_tokens, _FLASHINFER_MLA_SPARSE_BACKENDS
    )


def _deepseek_v4_sparse_mla_decode_autotune(
    worker: "Worker",
    num_tokens: int,
) -> bool:
    return _run_flashinfer_sparse_mla_decode_autotune(
        worker, num_tokens, _DEEPSEEK_V4_FLASHINFER_MLA_SPARSE_BACKENDS
    )


_DSV4_PORTABLE_PREFILL_WARM_SHAPES = (
    # tokens, num_candidates, num_kv_rows
    # Triton specializes non-constexpr integer arguments on equal-to-one and
    # divisible-by-16 attributes. The production chunk planner can emit every
    # reachable (candidate-width, flattened-KV-row) class below, so warm each
    # class explicitly before the fail-closed JIT monitor is armed.
    (1, 512, 1024),  # divisible_16, divisible_16
    (1, 22, 22),  # unaligned, unaligned
    (2, 8, 16),  # unaligned, divisible_16
    (1, 1, 1),  # equal_one, equal_one
    (2, 1, 2),  # equal_one, unaligned
    # Retain representative production widths as execution coverage even
    # though they share the divisible-by-16 Triton specialization class.
    (16, 512, 2048),
    (16, 640, 2048),  # topk + window
    (256, 2048, 4096),
    (1024, 16384 + 512, 32768),
    (16, 16384 + 4096 + 128, 65536),  # long c4a-style chunk_M
)


_DSV4_PORTABLE_PREFILL_WARM_HEAD_DIMS = (
    # The portable Triton helper is generic. Production uses the 512-wide
    # value/latent layout as well as the full 512 NoPE + 64 RoPE Q/K layout.
    # Both head_dim and the derived contiguous strides are compile-time keys.
    512,
    576,
)


def _warm_portable_sparse_prefill_accumulate_kernels(runner: "GPUModelRunner") -> None:
    """Compile portable sparse-prefill accumulate kernels before serving.

    ``runner._dummy_run`` with ``attn_metadata is None`` reserves workspace but
    skips ``flash_mla_sparse_fwd``. Long DSv4 prefills then JIT
    ``_accumulate_indexed_attention_chunk_multihead_kernel`` under the
    fail-closed inference monitor. Call the Triton entrypoint directly with
    production head/dim/scale and representative candidate widths.
    """
    device = getattr(runner, "device", None)
    if device is None or not torch.cuda.is_available():
        return

    from vllm.v1.attention.backends.mla.sm12x_sparse_mla_attn import (
        flash_mla_sparse_fwd_triton,
    )

    model_config = getattr(runner.vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    head_dim = int(getattr(hf_config, "head_dim", 512) or 512)
    d_v = 512
    sm_scale = float(head_dim**-0.5)
    head_layouts = _resolve_dsv4_portable_prefill_num_heads(runner)
    warmed = 0
    logger.info(
        "Warming portable sparse-MLA accumulate kernels on %s "
        "(heads=%s, head_dims=%s, scale=%s).",
        device,
        list(head_layouts),
        list(_DSV4_PORTABLE_PREFILL_WARM_HEAD_DIMS),
        sm_scale,
    )
    with torch.inference_mode():
        for num_heads in head_layouts:
            for warm_head_dim in _DSV4_PORTABLE_PREFILL_WARM_HEAD_DIMS:
                for num_tokens, num_candidates, num_kv_rows in (
                    _DSV4_PORTABLE_PREFILL_WARM_SHAPES
                ):
                    q = torch.empty(
                        (num_tokens, num_heads, warm_head_dim),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    kv = torch.empty(
                        (num_kv_rows, 1, warm_head_dim),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    indices = torch.zeros(
                        (num_tokens, 1, num_candidates),
                        dtype=torch.int32,
                        device=device,
                    )
                    base = torch.arange(
                        num_candidates, device=device, dtype=torch.int32
                    ) % max(num_kv_rows - 1, 1)
                    indices.copy_(
                        base.view(1, 1, -1).expand(num_tokens, 1, num_candidates)
                    )
                    if num_candidates > 1:
                        indices[:, :, -1] = -1
                    flash_mla_sparse_fwd_triton(
                        q=q,
                        kv=kv,
                        indices=indices,
                        sm_scale=sm_scale,
                        d_v=d_v,
                    )
                    warmed += 1
        torch.cuda.synchronize()
    logger.info(
        "Warmed %s portable sparse-MLA accumulate specializations before serving "
        "(head_layouts=%s, head_dims=%s).",
        warmed,
        list(head_layouts),
        list(_DSV4_PORTABLE_PREFILL_WARM_HEAD_DIMS),
    )


def flashinfer_sparse_mla_decode_autotune_warmup(worker: "Worker") -> None:
    """Autotune generic FlashInfer sparse MLA decode when selected."""
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    mixed_tokens = _clamp_warmup_tokens(_SPARSE_MLA_MIXED_WARMUP_TOKENS, max_tokens)
    if mixed_tokens <= 0:
        return
    _flashinfer_sparse_mla_decode_autotune(worker, mixed_tokens)


def deepseek_v4_sparse_mla_attention_warmup(worker: "Worker") -> None:
    """Warm DSv4 sparse-MLA mixed prefill+decode attention."""
    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    mixed_tokens = _clamp_warmup_tokens(_SPARSE_MLA_MIXED_WARMUP_TOKENS, max_tokens)
    if mixed_tokens <= 0:
        return

    logger.info(
        "Warming up DeepSeek V4 sparse MLA attention for mixed tokens=%s.",
        mixed_tokens,
    )
    mixed_warmup_done = _deepseek_v4_sparse_mla_decode_autotune(worker, mixed_tokens)
    if not mixed_warmup_done:
        if _uses_v2_model_runner(runner) and runner.max_num_reqs >= 2:
            v2_runner = cast("V2GPUModelRunner", runner)
            run_mixed_prefill_decode_warmup(
                v2_runner,
                worker.execute_model,
                worker.sample_tokens,
                mixed_tokens,
                req_id_prefix="_sparse_mla_v2_warmup",
            )
        else:
            runner._dummy_run(
                num_tokens=mixed_tokens,
                skip_eplb=True,
                is_profile=True,
                force_attention=True,
                create_mixed_batch=True,
            )

    # Decode autotune success must not suppress portable sparse-prefill warmup.
    # In C1, create_mixed_batch=True profiles a synthetic mixed layout whose
    # q/indices strides differ from a real single-request prefill. The first
    # production request would otherwise JIT the multihead indexed-accumulate
    # kernel after the fail-closed JIT monitor is armed. Run one additive pure
    # prefill with the real model runner so both TP ranks compile the exact
    # production tensor layout before serving starts.
    logger.info(
        "Warming up DeepSeek V4 portable sparse MLA pure prefill for tokens=%s.",
        mixed_tokens,
    )
    with torch.inference_mode():
        runner._dummy_run(
            num_tokens=mixed_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=False,
        )

    # Dummy-run with attn_metadata=None skips flash_mla_sparse_fwd. Directly
    # compile the portable multihead accumulate path with DSv4 production
    # head/dim/scale and several candidate widths before the JIT monitor arms.
    _warm_portable_sparse_prefill_accumulate_kernels(runner)
