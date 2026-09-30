#!/bin/bash
set -euo pipefail
say() { echo "[glm53-exl3-head] $*"; }

ARGS=(
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --nnodes "${NNODES}"
    --node-rank 0
    --enable-prompt-tokens-details
    --master-addr "${HEAD_IP}"
    --master-port "${MASTER_PORT}"
    --distributed-executor-backend mp
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --enable-prefix-caching
    --no-enable-flashinfer-autotune
)
[ "${ENFORCE_EAGER:-1}" = "1" ] && ARGS+=(--enforce-eager)
[ -n "${QUANTIZATION:-}" ] && [ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${MAX_NUM_SEQS:-}" ] && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ] && ARGS+=(--long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}")
[ -n "${KV_CACHE_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
[ -n "${LOAD_FORMAT:-}" ] && ARGS+=(--load-format "${LOAD_FORMAT}")
[ -n "${PREFIX_MATCH_UNIT:-}" ] && ARGS+=(--prefix-match-unit "${PREFIX_MATCH_UNIT}")
if [ -n "${GLM53_DEFAULT_REASONING_EFFORT:-}" ]; then
    ARGS+=(--default-chat-template-kwargs "{\"reasoning_effort\":\"${GLM53_DEFAULT_REASONING_EFFORT}\"}")
fi
if [ "${SPEC_METHOD:-mtp}" = "dflash" ]; then
    ARGS+=(--speculative-config "$(python3 -S -c 'import json,os
spec={"method":"dflash","model":os.environ["DFLASH_MODEL_DIR"],"num_speculative_tokens":int(os.environ.get("DFLASH_TOKENS","7")),"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard"}
tp=os.environ.get("DFLASH_DRAFT_TP","").strip()
if tp:
    spec["draft_tensor_parallel_size"]=int(tp)
print(json.dumps(spec,separators=(",",":")))')")
elif [ "${SPEC_METHOD:-mtp}" = "none" ]; then
    :
elif [ "${MTP_TOKENS:-0}" != "0" ]; then
    ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi
if [ -n "${CHAT_TEMPLATE:-}" ] && [ -f "${CHAT_TEMPLATE}" ]; then
    ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
    ARGS+=(--language-model-only)
    say "language-model-only: no vision tower"
else
    [ -n "${LIMIT_MM:-}" ] && ARGS+=(--limit-mm-per-prompt "${LIMIT_MM}")
    [ -n "${MM_IMAGE_TOKENS:-}" ] && ARGS+=(--mm-processor-kwargs "{\"max_image_tokens\":${MM_IMAGE_TOKENS}}")
    [ -n "${VIDEO_NUM_FRAMES:-}" ] && ARGS+=(--media-io-kwargs "{\"video\":{\"num_frames\":${VIDEO_NUM_FRAMES}}}")
    [ -n "${MM_PROCESSOR_CACHE_GB:-}" ] && ARGS+=(--mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB}")
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
    say "vision on: limit-mm=${LIMIT_MM:-} image-tokens=${MM_IMAGE_TOKENS:-8000} video-frames=${VIDEO_NUM_FRAMES:-32} mm-cache-gb=${MM_PROCESSOR_CACHE_GB:-4} skip-mm-profiling=${SKIP_MM_PROFILING:-1} chat-template=${CHAT_TEMPLATE:-}"
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
if [ -f /opt/glm53/patch_cold_load_uma.py ]; then
    python3 /opt/glm53/patch_cold_load_uma.py
fi
if [ -f /opt/glm53/patch_skip_cudagraph_profile.py ]; then
    python3 /opt/glm53/patch_skip_cudagraph_profile.py
fi
if [ -f /opt/glm53/patch_glm_video_placeholders.py ]; then
    python3 /opt/glm53/patch_glm_video_placeholders.py
fi
if [ -f /opt/glm53/patch_suppress_stops_in_reasoning.py ]; then
    python3 /opt/glm53/patch_suppress_stops_in_reasoning.py
fi
if [ -f /opt/glm53/patch_scheduler_decode_floor.py ]; then
    python3 /opt/glm53/patch_scheduler_decode_floor.py
fi
if [ -f /opt/glm53/patch_mamba_align_chunking.py ]; then
    python3 /opt/glm53/patch_mamba_align_chunking.py
fi
if [ -f /opt/glm53/patch_glm5_drafter_group.py ]; then
    python3 /opt/glm53/patch_glm5_drafter_group.py
fi
if [ -f /opt/glm53/patch_hybrid_prefix_hit.py ]; then
    python3 /opt/glm53/patch_hybrid_prefix_hit.py
fi
if [ -f /opt/glm53/patch_apc_per_group_retention.py ]; then
    python3 /opt/glm53/patch_apc_per_group_retention.py
fi
if [ -f /opt/glm53/patch_apc_no_store.py ]; then
    python3 /opt/glm53/patch_apc_no_store.py
fi
if [ -f /opt/glm53/patch_mamba_align_state_free.py ]; then
    python3 /opt/glm53/patch_mamba_align_state_free.py
fi
if [ -f /opt/glm53/patch_kv_capacity_log.py ]; then
    python3 /opt/glm53/patch_kv_capacity_log.py
fi
if [ -f /opt/glm53/patch_tool_choice_none.py ]; then
    python3 /opt/glm53/patch_tool_choice_none.py
fi
if [ -f /opt/glm53/patch_xgrammar_termination.py ]; then
    python3 /opt/glm53/patch_xgrammar_termination.py
fi
if [ -f /opt/glm53/patch_kpool_tail_slotmap.py ]; then
    python3 /opt/glm53/patch_kpool_tail_slotmap.py
fi
if [ -f /opt/glm53/patch_kpool_tail_seed_stride.py ]; then
    python3 /opt/glm53/patch_kpool_tail_seed_stride.py
fi
if [ -f /opt/glm53/patch_spinwait.py ]; then
    python3 /opt/glm53/patch_spinwait.py
fi
if [ -f /opt/glm53/patch_adaptive_k.py ]; then
    python3 /opt/glm53/patch_adaptive_k.py
fi
if [ -f /opt/glm53/patch_dense_fp8.py ]; then
    python3 /opt/glm53/patch_dense_fp8.py
fi
if [ -f /opt/glm53/patch_dflash2_exl3.py ]; then
    python3 /opt/glm53/patch_dflash2_exl3.py
fi
if [ -f /opt/glm53/patch_loadclone.py ]; then
    python3 /opt/glm53/patch_loadclone.py
fi
if [ -f /opt/glm53/patch_default_max_new_tokens.py ]; then
    python3 /opt/glm53/patch_default_max_new_tokens.py
fi
if [ -f /opt/glm53/patch_indexer_workspace.py ]; then
    python3 /opt/glm53/patch_indexer_workspace.py
fi
if [ -f /opt/glm53/patch_indexer_warmup_range.py ]; then
    python3 /opt/glm53/patch_indexer_warmup_range.py
fi
if [ -f /opt/glm53/patch_cache_reset.py ]; then
    python3 /opt/glm53/patch_cache_reset.py
fi
if [ -f /opt/glm53/patch_ablit.py ]; then
    python3 /opt/glm53/patch_ablit.py
fi
if [ "${ABLIT:-0}" = "1" ]; then
    say "ablit: o_proj orthogonalization ON (method=${ABLIT_METHOD:-auto} direction=${ABLIT_DIRECTION:-dealign} layers=${ABLIT_LAYERS:-15-45} alpha=${ABLIT_ALPHA:-3.0})"
else
    say "runtime ablit: off; checkpoint o_proj unchanged"
fi
say "launching: vllm serve ${MODEL_DIR} ${ARGS[*]}"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
