# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up sparse-MLA Triton metadata kernels."""

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
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
_GENERIC_SPARSE_MLA_BACKENDS = frozenset(
    {
        "FLASHMLA_SPARSE",
        "FLASHINFER_MLA_SPARSE",
        "FLASHINFER_MLA_SPARSE_SM120",
    }
)
_INDEXER_PREFILL_CHUNK_METADATA_BACKENDS = frozenset({"DEEPSEEK_V32_INDEXER"})

_SPARSE_PREFILL_METADATA_NUM_PREFILLS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16)
_SPARSE_PREFILL_METADATA_NUM_DECODES = (0, 1, 2, 3, 4, 6, 8, 12, 16)
_DSV4_PREFILL_CHUNK_METADATA_COMPRESS_RATIOS = (4, 128)
_DSV4_FIRST_REQUEST_PREFILL_TOKENS = 300_024
_PREFILL_CHUNK_METADATA_BLOCK_SIZE = 1024
_PREFILL_CHUNK_METADATA_SEQ_LEN_MULTIPLIERS = (2, 3)
_PREFILL_CHUNK_METADATA_QUERY_SLICE_OFFSETS = (
    # query_slice_start offset, query_slice_stop offset
    (0, 0),
    (0, -1),
    (1, 0),
    (1, -1),
)
_COMBINE_TOPK_SWA_INPUT_VARIANTS = (
    # offset_topk, offset_query_and_seq, offset_gather
    (False, False, False),
    (False, True, False),
    (True, True, True),
)
_DSV4_COMBINE_TOPK_SWA_WARMUP_CASES = (
    # compress_ratio, topk, topk_width, N
    # Production clamps config 0 -> max(1, ratio). Never pass 0 (div-by-zero).
    (1, 0, 512, 512),
    (1, 0, 512, 16384),
    (4, 512, 512, 512 * 4),
    # First long C1 chunk: ~16K query tokens over empty/growing prefix.
    (4, 512, 512, 4096),
    (4, 512, 512, 16384),
    # Late long C1 chunk: compressed prefix ≈ 300024/4.
    (4, 512, 512, 75006),
    # DSv4-Pro C4A traffic uses top-k 1024 with N=1024.
    (4, 1024, 1024, 1024),
    (128, 512, 512, 1),
    (128, 512, 512, 2344),
    (128, 8192, 8192, 8192 * 128),
    # Real C128A traffic also specializes N=1 in one call path.
    (128, 8192, 8192, 1),
)


def _clamp_warmup_tokens(num_tokens: int, max_tokens: int) -> int:
    return max(0, min(num_tokens, max_tokens))


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _hf_config_int(runner: "GPUModelRunner", name: str, default: int) -> int:
    model_config = getattr(runner.vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    return int(getattr(hf_config, name, default) or default)


def _dsv4_compress_ratios(runner: "GPUModelRunner") -> tuple[int, ...]:
    """Return the configured nontrivial DSV4 indexer compression ratios."""
    model_config = getattr(runner.vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    configured = getattr(hf_config, "compress_ratios", None)
    if configured is None:
        return _DSV4_PREFILL_CHUNK_METADATA_COMPRESS_RATIOS

    ratios = tuple(sorted({int(ratio) for ratio in configured if int(ratio) > 1}))
    return ratios or _DSV4_PREFILL_CHUNK_METADATA_COMPRESS_RATIOS


def _attention_backend_name(backend: object) -> str | None:
    get_name = getattr(backend, "get_name", None)
    if get_name is None:
        return None
    try:
        return get_name()
    except NotImplementedError:
        return None


def _has_attention_backend(
    runner: "GPUModelRunner",
    backend_names: frozenset[str],
) -> bool:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name in backend_names:
                return True
    return False


def _warm_sparse_swa_prefill_metadata_kernel(
    device: torch.device,
    window_size: int,
    prefill_tokens: int,
) -> None:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _compute_prefill_metadata_kernel,
    )

    for num_prefills in _SPARSE_PREFILL_METADATA_NUM_PREFILLS:
        for num_decodes in _SPARSE_PREFILL_METADATA_NUM_DECODES:
            query_lens = [1] * num_decodes
            query_lens += [prefill_tokens] * num_prefills
            query_start_locs = [0]
            for query_len in query_lens:
                query_start_locs.append(query_start_locs[-1] + query_len)
            query_start_loc = torch.tensor(
                query_start_locs,
                dtype=torch.int32,
                device=device,
            )
            seq_lens = torch.tensor(
                [1] * num_decodes + [window_size + q for q in query_lens[num_decodes:]],
                dtype=torch.int32,
                device=device,
            )
            prefill_gather_lens = torch.empty(
                num_prefills, dtype=torch.int32, device=device
            )
            _compute_prefill_metadata_kernel[(1,)](
                prefill_gather_lens,
                seq_lens,
                query_start_loc,
                num_prefills,
                num_decodes,
                window_size,
                BLOCK_SIZE=16,
            )


def _warm_prefill_chunk_metadata_kernel(
    device: torch.device,
    compress_ratio: int,
    query_len: int,
) -> None:
    from vllm.v1.attention.backends.mla.indexer import build_prefill_chunk_metadata

    num_reqs = 2
    query_start_loc_cpu = torch.arange(
        0, (num_reqs + 1) * query_len, query_len, dtype=torch.int32
    )
    query_start_loc = query_start_loc_cpu.to(device=device)

    uncompressed_seq_lens_cpu = torch.tensor(
        [
            compress_ratio * multiplier + query_len
            for multiplier in _PREFILL_CHUNK_METADATA_SEQ_LEN_MULTIPLIERS
        ],
        dtype=torch.int32,
    )
    compressed_seq_lens_cpu = uncompressed_seq_lens_cpu // compress_ratio
    uncompressed_seq_lens = uncompressed_seq_lens_cpu.to(device=device)
    compressed_seq_lens = compressed_seq_lens_cpu.to(device=device)
    block_table = torch.zeros(
        (num_reqs, int(compressed_seq_lens_cpu.max().item())),
        dtype=torch.int32,
        device=device,
    )

    offset_uncompressed_seq_lens = torch.empty(
        num_reqs + 1, dtype=torch.int32, device=device
    )[1:]
    offset_uncompressed_seq_lens.copy_(uncompressed_seq_lens)
    query_slices = tuple(
        slice(start, num_reqs * query_len + stop)
        for start, stop in _PREFILL_CHUNK_METADATA_QUERY_SLICE_OFFSETS
    )
    for warmup_uncompressed_seq_lens in (
        uncompressed_seq_lens,
        offset_uncompressed_seq_lens,
    ):
        for query_slice in query_slices:
            build_prefill_chunk_metadata(
                0,
                num_reqs,
                query_start_loc,
                query_start_loc_cpu,
                warmup_uncompressed_seq_lens,
                compressed_seq_lens,
                compressed_seq_lens_cpu,
                block_table,
                compress_ratio,
                query_slice=query_slice,
            )


def _first_request_query_slices(
    runner: "GPUModelRunner",
    prompt_tokens: int,
    compressed_seq_len: int,
) -> tuple[slice, ...]:
    """Select representative slices from the real first-request chunk plan.

    A 300K request can produce hundreds of query slices for C4A under the
    logits budget. The Triton compile key only needs the first, an interior,
    the final, and the full prompt boundary; compiling every slice would turn
    warmup into an O(24K) operation without adding useful coverage.
    """
    from vllm.v1.attention.backends.mla.indexer import (
        get_max_prefill_buffer_size,
        split_indexer_prefill_chunks,
    )

    max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    chunks = split_indexer_prefill_chunks(
        torch.tensor([compressed_seq_len], dtype=torch.int32),
        torch.tensor([prompt_tokens], dtype=torch.int32),
        get_max_prefill_buffer_size(runner.vllm_config),
        max_logits_bytes,
    )
    planned = tuple(query_slice for _, query_slice in chunks)
    if not planned:
        return (slice(0, prompt_tokens),)

    selected = {
        (planned[0].start, planned[0].stop),
        (planned[len(planned) // 2].start, planned[len(planned) // 2].stop),
        (planned[-1].start, planned[-1].stop),
        (0, prompt_tokens),
    }
    # Include the 1024-aligned first boundary used by the BLOCK_SIZE loop when
    # it differs from the planner's exact logits boundary.
    first_stop = min(
        prompt_tokens,
        cdiv(planned[0].stop, _PREFILL_CHUNK_METADATA_BLOCK_SIZE)
        * _PREFILL_CHUNK_METADATA_BLOCK_SIZE,
    )
    selected.add((0, first_stop))
    return tuple(slice(start, stop) for start, stop in sorted(selected))


def _warm_first_request_prefill_chunk_metadata(
    runner: "GPUModelRunner",
    prompt_tokens: int,
) -> None:
    """Warm the exact DSV4 first-request metadata shapes without enumeration."""
    if prompt_tokens <= 0:
        return

    device = getattr(runner, "device", torch.device("cuda"))
    from vllm.v1.attention.backends.mla.indexer import build_prefill_chunk_metadata

    query_start_loc_cpu = torch.tensor([0, prompt_tokens], dtype=torch.int32)
    query_start_loc = query_start_loc_cpu.to(device=device)
    for compress_ratio in _dsv4_compress_ratios(runner):
        compressed_seq_len = prompt_tokens // compress_ratio
        if compressed_seq_len <= 0:
            continue
        uncompressed_seq_lens_cpu = torch.tensor(
            [prompt_tokens], dtype=torch.int32
        )
        compressed_seq_lens_cpu = torch.tensor(
            [compressed_seq_len], dtype=torch.int32
        )
        uncompressed_seq_lens = uncompressed_seq_lens_cpu.to(device=device)
        compressed_seq_lens = compressed_seq_lens_cpu.to(device=device)
        block_table = torch.zeros(
            (1, compressed_seq_len), dtype=torch.int32, device=device
        )
        for query_slice in _first_request_query_slices(
            runner, prompt_tokens, compressed_seq_len
        ):
            build_prefill_chunk_metadata(
                0,
                1,
                query_start_loc,
                query_start_loc_cpu,
                uncompressed_seq_lens,
                compressed_seq_lens,
                compressed_seq_lens_cpu,
                block_table,
                compress_ratio,
                query_slice=query_slice,
            )


def _warm_combine_topk_swa_indices_kernel(
    device: torch.device,
    num_tokens: int,
    window_size: int,
    compress_ratio: int,
    topk: int,
    topk_width: int,
    n: int,
) -> None:
    from vllm.models.deepseek_v4.common.ops.cache_utils import combine_topk_swa_indices

    if num_tokens <= 0:
        return

    def _make_topk_indices(*, offset: bool) -> torch.Tensor:
        if offset:
            topk_storage = torch.full(
                (num_tokens * topk_width + 1,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            topk_indices = topk_storage[1:].reshape(num_tokens, topk_width)
        else:
            topk_indices = torch.full(
                (num_tokens, topk_width), -1, dtype=torch.int32, device=device
            )
        if topk > 0:
            topk_indices.copy_(
                torch.arange(num_tokens * topk_width, dtype=torch.int32, device=device)
                .reshape(num_tokens, topk_width)
                .remainder(topk_width)
            )
        return topk_indices

    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    seq_lens = torch.tensor(
        [window_size + num_tokens], dtype=torch.int32, device=device
    )
    gather_lens = torch.tensor(
        [min(window_size + num_tokens, window_size + num_tokens - 1)],
        dtype=torch.int32,
        device=device,
    )
    offset_query_start_loc = torch.empty(3, dtype=torch.int32, device=device)[1:]
    offset_query_start_loc.copy_(query_start_loc)
    offset_seq_lens = torch.empty(2, dtype=torch.int32, device=device)[1:]
    offset_seq_lens.copy_(seq_lens)
    offset_gather_lens = torch.empty(2, dtype=torch.int32, device=device)[1:]
    offset_gather_lens.copy_(gather_lens)

    for (
        offset_topk,
        offset_query_and_seq,
        offset_gather,
    ) in _COMBINE_TOPK_SWA_INPUT_VARIANTS:
        warmup_topk_indices = _make_topk_indices(offset=offset_topk)
        warmup_query_start_loc = (
            offset_query_start_loc if offset_query_and_seq else query_start_loc
        )
        warmup_seq_lens = offset_seq_lens if offset_query_and_seq else seq_lens
        warmup_gather_lens = offset_gather_lens if offset_gather else gather_lens
        n_values = (n,) if n == 1 else (n, n + 1)
        # Include production long-chunk M widths. Prior warmup only used
        # window+num_tokens / topk_width, which misses C1 300K chunk_M.
        m_values = {
            window_size + num_tokens,
            topk_width,
            max(window_size + num_tokens, topk_width),
            max(n, 1) + max(num_tokens, 1),
            max(n, 1) + max(num_tokens, 1) + max(window_size - 1, 0),
        }
        for m in sorted(m_values):
            for n_value in n_values:
                combine_topk_swa_indices(
                    warmup_topk_indices,
                    warmup_query_start_loc,
                    warmup_seq_lens,
                    warmup_gather_lens,
                    window_size,
                    compress_ratio,
                    topk,
                    M=m,
                    N=n_value,
                )


@torch.inference_mode()
def sparse_mla_triton_warmup(
    runner: "GPUModelRunner",
    num_tokens: int,
    *,
    compress_ratios: tuple[int, ...],
    combine_topk_swa_cases: tuple[tuple[int, int, int, int], ...] = (),
    first_request_prompt_tokens: int | None = None,
) -> None:
    device = getattr(runner, "device", torch.device("cuda"))
    window_size = _hf_config_int(runner, "sliding_window", 128)

    _warm_sparse_swa_prefill_metadata_kernel(device, window_size, num_tokens)
    for compress_ratio in compress_ratios:
        _warm_prefill_chunk_metadata_kernel(device, compress_ratio, num_tokens)
    if first_request_prompt_tokens is not None:
        _warm_first_request_prefill_chunk_metadata(runner, first_request_prompt_tokens)
    for compress_ratio, topk, topk_width, n in combine_topk_swa_cases:
        _warm_combine_topk_swa_indices_kernel(
            device,
            num_tokens,
            window_size,
            compress_ratio,
            topk,
            topk_width,
            n,
        )


def deepseek_v4_sparse_triton_warmup(
    runner: "GPUModelRunner",
    num_tokens: int,
    first_request_prompt_tokens: int | None = None,
) -> None:
    sparse_mla_triton_warmup(
        runner,
        num_tokens,
        compress_ratios=_dsv4_compress_ratios(runner),
        combine_topk_swa_cases=_DSV4_COMBINE_TOPK_SWA_WARMUP_CASES,
        first_request_prompt_tokens=first_request_prompt_tokens,
    )


def sparse_mla_triton_warmup_if_needed(worker: "Worker") -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    num_tokens = _clamp_warmup_tokens(8, max_tokens)
    if num_tokens <= 0:
        return

    try:
        if _has_attention_backend(runner, _DEEPSEEK_V4_SPARSE_MLA_BACKENDS):
            deepseek_v4_sparse_triton_warmup(
                runner,
                num_tokens,
                first_request_prompt_tokens=_DSV4_FIRST_REQUEST_PREFILL_TOKENS,
            )
            # Re-run combine-topk path at the production batched-token width so
            # PADDED_TOP_K/M specializations used by C1 long chunked prefill are
            # compiled before the fail-closed inference JIT monitor arms.
            long_tokens = _clamp_warmup_tokens(max_tokens, max_tokens)
            if long_tokens > num_tokens:
                for compress_ratio, topk, topk_width, n in _DSV4_COMBINE_TOPK_SWA_WARMUP_CASES:
                    _warm_combine_topk_swa_indices_kernel(
                        getattr(runner, "device", torch.device("cuda")),
                        long_tokens,
                        _hf_config_int(runner, "sliding_window", 128),
                        compress_ratio,
                        topk,
                        topk_width,
                        n,
                    )
            logger.info(
                "Completed DeepSeek-V4 sparse MLA Triton warmup "
                "(metadata tokens=%s, combine long tokens=%s).",
                num_tokens,
                long_tokens,
            )
        elif _has_attention_backend(runner, _GENERIC_SPARSE_MLA_BACKENDS):
            sparse_mla_triton_warmup(
                runner,
                num_tokens,
                compress_ratios=(1,),
            )
        elif _has_attention_backend(runner, _INDEXER_PREFILL_CHUNK_METADATA_BACKENDS):
            _warm_prefill_chunk_metadata_kernel(
                getattr(runner, "device", torch.device("cuda")),
                compress_ratio=1,
                query_len=num_tokens,
            )
    except Exception:
        logger.warning("Skipping sparse MLA Triton warmup.", exc_info=True)
