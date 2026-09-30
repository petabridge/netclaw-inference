#!/usr/bin/env python3
"""Optional FlashKDA prefill port for the pinned vLLM 487ecf187 KDA layout.

Adapted from NNNtrance/GLM-5.3-Flash-EXL3-DGX-Spark at
7f1bd29cf473548212b8b731c52dcf2c7aa5a033,
tracks/tp3/patches/flashkda/patch-flashkda-tp3.py; original vLLM integration:
https://github.com/vllm-project/vllm/pull/55737 (JaredforReal).

Local adaptations preserve the 14-argument extension ABI, FP32 recurrent state,
preallocated graph-safe workspace/capacity guards, explicit bounded-gate and
layout checks, and composition with the existing dense-FP8 patch. The original
HAREM_KDA_FLASHKDA environment key is retained for source compatibility.
Unset/0 keeps Triton; 1 explicitly selects FlashKDA or fails on unsupported
inputs. This changes chunked prefill, not recurrent decode.

Use --root DIST_PACKAGES --in-place inside a compatible fresh container.
Without --in-place this only checks patch anchors. Reapplication is idempotent;
missing/duplicate anchors or partial patches fail. Native GPU validation of
this upstream-based branch remains pending; see docs/tp3-throughput-results.md.
"""

import argparse
import os
import sys

REL = "vllm/models/glm5next/nvidia/kda.py"
MARK = "HAREM-FLASHKDA"

# --- anchor 1: the import block ----------------------------------------------
A1_OLD = "from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata\n"
A1_NEW = (
    "from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata\n"
    "from vllm.v1.worker.workspace import current_workspace_manager\n"
)

# --- anchor 2: backend resolver, right after _cast_sigmoid -------------------
A2_OLD = "    return x.float().sigmoid()\n"
A2_NEW = '''    return x.float().sigmoid()


_HAREM_KDA_BACKEND_LOGGED = set()


def _harem_log_kda_backend(backend: str) -> str:
    """HAREM-FLASHKDA.  Say once per process which chunked-prefill kernel won.

    The resolver runs per KDA layer (34 of them), so the decision is printed the
    first time only; a boot log without this line means the patch is not
    installed at all.
    """
    if backend not in _HAREM_KDA_BACKEND_LOGGED:
        _HAREM_KDA_BACKEND_LOGGED.add(backend)
        print(
            f"[HAREM-FLASHKDA] kda_prefill_backend={backend} "
            f"(HAREM_KDA_FLASHKDA={os.environ.get('HAREM_KDA_FLASHKDA', '')!r})",
            flush=True,
        )
    return backend


def _harem_kda_prefill_backend(
    additional_config,
    head_dim: int,
    dtype: "torch.dtype",
    lower_bound: float | None,
) -> str:
    """HAREM-FLASHKDA.  Pick the chunked-prefill kernel.

    ``HAREM_KDA_FLASHKDA=1`` asks for FlashKDA (``vllm._flashkda_C``, the fused
    CUDA kernel Kimi-K3 already uses); unset/0 keeps the Triton
    ``chunk_kda_with_fused_gate`` chain, byte for byte upstream.
    ``additional_config["kda_prefill_backend"]`` (auto/triton/flashkda) is
    honoured too and wins over the env gate, so the PR's knob keeps working.
    """
    backend = os.environ.get("HAREM_KDA_FLASHKDA", "").strip().lower()
    backend = {"1": "auto", "flashkda": "flashkda", "": "triton", "0": "triton"}.get(
        backend, backend
    )
    if isinstance(additional_config, dict):
        backend = additional_config.get("kda_prefill_backend", backend)
    if backend not in ("auto", "triton", "flashkda"):
        raise ValueError(f"Unsupported KDA prefill backend: {backend}")
    if backend == "triton":
        return _harem_log_kda_backend("triton")
    capability = current_platform.get_device_capability()
    supported = (
        current_platform.is_cuda()
        and capability is not None
        # SM90 is architecture-specific; SM10x and SM12x are built as family
        # binaries ("10.0f"/"12.0f" in cmake/external_projects/flashkda.cmake),
        # so the sm_120 cubin in this image runs on GB10 (sm_121) -- measured.
        and capability.major in (9, 10, 12)
        and head_dim == 128
        and dtype == torch.bfloat16
        and lower_bound is not None
    )
    if not supported:
        raise RuntimeError(
            "HAREM-FLASHKDA: FlashKDA requires CUDA SM90/SM10x/SM12x, bfloat16, "
            f"head_dim=128 and a bounded KDA gate; got capability={capability}, "
            f"dtype={dtype}, head_dim={head_dim}, lower_bound={lower_bound}"
        )
    return _harem_log_kda_backend("flashkda")
'''

# --- anchor 3: end of __init__ (vllm_config is in scope here) ---------------
A3_OLD = "        self._conv_state_dim_first = is_conv_state_dim_first()\n"
A3_NEW = '''        self._conv_state_dim_first = is_conv_state_dim_first()

        # HAREM-FLASHKDA.  Resolve the chunked-prefill kernel once, and size the
        # three workspace buffers FlashKDA needs (recurrent final state, scratch,
        # and an output buffer for steps that also carry spec-decode tokens).
        self.kda_prefill_backend = _harem_kda_prefill_backend(
            vllm_config.additional_config,
            self.head_dim,
            vllm_config.model_config.dtype,
            self.kda_lower_bound if self.kda_safe_gate else None,
        )
        self._flashkda_buffer_specs = None
        if self.kda_prefill_backend == "flashkda":
            import vllm._flashkda_C  # noqa: F401

            schema = torch.ops._flashkda_C.fwd.default._schema
            expected_args = ["q", "k", "v", "g", "beta", "scale", "out", "workspace", "A_log", "dt_bias", "lower_bound", "initial_state", "final_state", "cu_seqlens"]
            if [a.name for a in schema.arguments] != expected_args:
                raise RuntimeError(f"GLM53 FlashKDA ABI mismatch: {schema}")
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            max_seqs = vllm_config.scheduler_config.max_num_seqs
            self._flashkda_max_tokens = max_tokens
            self._flashkda_max_seqs = max_seqs
            if self.get_state_dtype()[1] != torch.float32:
                raise RuntimeError("GLM53 FlashKDA requires FP32 recurrent state")
            workspace_size = torch.ops._flashkda_C.get_workspace_size(
                max_tokens, self.local_num_heads, max_seqs
            )
            self._flashkda_buffer_specs = (
                (
                    (max_seqs, self.local_num_heads, self.head_dim, self.head_dim),
                    self.get_state_dtype()[1],
                ),
                ((workspace_size,), torch.uint8),
                (
                    (1, max_tokens, self.local_num_heads, self.head_dim),
                    vllm_config.model_config.dtype,
                ),
            )
'''

# --- anchor 4: insert the kernel wrapper just before forward() --------------
A4_OLD = "    def forward(\n"
A4_NEW = '''    def _harem_flashkda_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """HAREM-FLASHKDA.  Fused KDA chunked prefill.

        Takes the RAW gate logits ``g`` and RAW ``beta`` logits, l2-normalizes
        q/k in-kernel and applies the bounded gate
        ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` -- the same thing
        ``chunk_kda_with_fused_gate(..., safe_gate=True)`` computes.  Writes the
        attention output into ``out`` (a workspace buffer when ``None``) and
        returns ``(out, final_state)``.

        NOTE the 14-argument call: this image's ``_flashkda_C::fwd`` ends at
        ``cu_seqlens``; upstream's newer ``checkpoint_state`` /
        ``checkpoint_offsets`` tail does not exist here.
        """
        if self._flashkda_buffer_specs is None:
            raise RuntimeError("FlashKDA workspace not initialized")
        if (q.ndim != 4 or q.shape[0] != 1
                or q.shape[1] > self._flashkda_max_tokens
                or initial_state.shape[0] > self._flashkda_max_seqs
                or cu_seqlens.numel() != initial_state.shape[0] + 1):
            raise RuntimeError("FlashKDA input exceeds preallocated token/sequence capacity")
        final_state, workspace, workspace_out = current_workspace_manager(
        ).get_simultaneous(*self._flashkda_buffer_specs)
        final_state = final_state[: initial_state.shape[0]]
        if out is None:
            out = workspace_out[:, : q.shape[1]]
        # FlashKDA hardcodes dense q/k/v/g strides; beta may be row-strided.
        torch.ops._flashkda_C.fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta,
            self.head_dim**-0.5,
            out,
            workspace,
            self.A_log.reshape(-1).contiguous(),
            self.dt_bias.reshape(-1, self.head_dim).contiguous(),
            self.kda_lower_bound,
            initial_state.contiguous(),
            final_state,
            cu_seqlens.contiguous(),
        )
        return out, final_state

    def forward(
'''

# --- anchor 5: the chunked-prefill call --------------------------------------
A5_OLD = '''            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_kda_with_fused_gate(
                q=_rearr(q_ns),
                k=_rearr(k_ns),
                v=_rearr(v_ns),
                raw_g=g1_ns,
                # Chunk path wants the pre-sigmoided fp32 beta (its kernels
                # don't sigmoid); beta_ns is raw bf16 from forward.
                beta=_cast_sigmoid(beta_ns.squeeze(0)).unsqueeze(0),
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
            )
'''
A5_NEW = '''            if self.kda_prefill_backend == "flashkda":
                # HAREM-FLASHKDA.  Non-spec step: write straight into the layer
                # output buffer (dense token order, no merge copy) -- ns_out
                # stays non-None so the tail copy below is skipped.  A step that
                # also carries spec-decode tokens writes to the workspace buffer
                # and is scattered by non_spec_token_indx in the merge below.
                ns_out = None if use_spec else core_attn_out[:, :num_actual_tokens]
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = self._harem_flashkda_prefill(
                    q=_rearr(q_ns),
                    k=_rearr(k_ns),
                    v=_rearr(v_ns),
                    g=g1_ns,
                    beta=beta_ns,
                    initial_state=initial_state,
                    cu_seqlens=non_spec_query_start_loc,
                    out=ns_out,
                )
            else:
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = chunk_kda_with_fused_gate(
                    q=_rearr(q_ns),
                    k=_rearr(k_ns),
                    v=_rearr(v_ns),
                    raw_g=g1_ns,
                    # Chunk path wants the pre-sigmoided fp32 beta (its kernels
                    # don't sigmoid); beta_ns is raw bf16 from forward.
                    beta=_cast_sigmoid(beta_ns.squeeze(0)).unsqueeze(0),
                    A_log=self.A_log,
                    g_bias=self.dt_bias,
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=non_spec_query_start_loc,
                    safe_gate=safe_gate,
                    lower_bound=lower_bound,
                )
'''

ANCHORS = [
    ("A1-import", A1_OLD, A1_NEW),
    ("A2-resolver", A2_OLD, A2_NEW),
    ("A3-init", A3_OLD, A3_NEW),
    ("A4-wrapper", A4_OLD, A4_NEW),
    ("A5-prefill-call", A5_OLD, A5_NEW),
]


def apply(src: str, where: str) -> str:
    if MARK in src:
        if not all(src.count(new) == 1 for _name, _old, new in ANCHORS):
            raise RuntimeError(f"partial or different FlashKDA patch in {where}")
        print(f"patch-flashkda: already applied ({where})")
        return src
    if "import os\n" not in src:
        # kda.py imports os lazily inside __init__ in the HAREM-FULLSCOPE build;
        # the resolver needs it at module scope.
        src = src.replace("import torch\n", "import os\n\nimport torch\n", 1)
    for name, old, new in ANCHORS:
        n = src.count(old)
        if n != 1:
            print(
                f"patch-flashkda: {name} count={n} (expected 1) in {where} "
                "-- refusing",
                file=sys.stderr,
            )
            sys.exit(3)
    for _name, old, new in ANCHORS:
        src = src.replace(old, new, 1)
    return src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dist-packages root")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--out", default="", help="write here instead of in place")
    a = ap.parse_args()
    p = os.path.join(a.root, REL)
    src = open(p).read()
    out = apply(src, REL)
    if out is src:
        return
    dst = a.out or (p if a.in_place else "")
    if not dst:
        print(
            "patch-flashkda: dry run OK (pass --in-place or --out to write)",
        )
        return
    open(dst, "w").write(out)
    print(f"patch-flashkda: applied to {dst} (HAREM_KDA_FLASHKDA honoured)")


if __name__ == "__main__":
    main()
