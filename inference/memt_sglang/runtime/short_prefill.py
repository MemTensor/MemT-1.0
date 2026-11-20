"""Experiment: single-kernel recurrent GDN prefill for short prompts.

Native GDN prefill runs the chunked delta rule: about eight Triton launches per
layer (l2norm x2, cumsum, intra-chunk solve, state, output). The model has 30 GDN
layers and they execute eagerly between prefill-graph segments, so for a short
prompt the step is bound by launch overhead on the host, not by GPU work: TTFT
is the same for 16 and 512 prompt tokens.

Decode and speculative verify already use the recurrent form of the same update
(`fused_sigmoid_gating_delta_rule_update`): one launch per layer, reading and
writing the same state pool by slot index. It is the token-by-token definition
the chunked algorithm reformulates, so the two agree up to rounding. For
sequences short enough that the per-token loop is cheap, use it for prefill too.

Scope is deliberately narrow; anything else falls through to the native path:
target EXTEND only, Triton extend kernel, no radix state tracking in the batch
(the recurrent kernel does not produce the per-chunk states tracking needs),
contiguous state pool, and every sequence at most the configured length.
"""

import logging
import os
from functools import wraps

_CALL = {"eligible": False, "gating": None}
STATS = {"recurrent": 0, "native": 0}


def install_recurrent_short_prefill():
    import sglang.srt.layers.attention.linear.gdn_backend as backend
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    cls = backend.GDNAttnBackend
    if getattr(cls.forward_extend, "_memt_short_prefill", False):
        return
    limit = int(os.environ.get("MEMT_RECURRENT_SHORT_PREFILL_TOKENS", "128"))
    if limit < 1:
        raise ValueError("MEMT_RECURRENT_SHORT_PREFILL_TOKENS must be positive")
    native_forward = cls.forward_extend
    native_gating = backend.fused_gdn_gating
    native_extend = backend.GDNKernelDispatcher.extend

    @wraps(native_forward)
    def forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        lengths = forward_batch.extend_seq_lens_cpu
        _CALL["eligible"] = bool(
            forward_batch.forward_mode.name == "EXTEND"
            and lengths
            and max(lengths) <= limit
            and not self.forward_metadata.has_mamba_track_mask
            and type(self.kernel_dispatcher.extend_kernel) is TritonGDNKernel
        )
        _CALL["gating"] = None
        try:
            return native_forward(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)
        finally:
            _CALL["eligible"] = False
            _CALL["gating"] = None

    def gating(A_log, a, b, dt_bias):
        if not _CALL["eligible"]:
            return native_gating(A_log, a, b, dt_bias)
        # The recurrent kernel computes the gates itself from the raw inputs.
        _CALL["gating"] = (A_log, a, b, dt_bias)
        return None, None

    @wraps(native_extend)
    def extend(self, q, k, v, g, beta, *, ssm_states, cache_indices,
               query_start_loc, **kwargs):
        stashed = _CALL["gating"]
        if stashed is None:
            return native_extend(
                self, q, k, v, g, beta, ssm_states=ssm_states,
                cache_indices=cache_indices, query_start_loc=query_start_loc,
                **kwargs,
            )
        A_log, a, b, dt_bias = stashed
        _CALL["gating"] = None
        if not ssm_states.is_contiguous():
            STATS["native"] += 1
            g, beta = native_gating(A_log, a, b, dt_bias)
            return native_extend(
                self, q, k, v, g, beta, ssm_states=ssm_states,
                cache_indices=cache_indices, query_start_loc=query_start_loc,
                **kwargs,
            )
        STATS["recurrent"] += 1
        # Same call the decode path makes, over whole sequences: reads each
        # sequence's slot, applies its tokens in order, writes the slot back.
        output = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )
        return output, None, None

    forward_extend._memt_short_prefill = True
    cls.forward_extend = forward_extend
    backend.fused_gdn_gating = gating
    backend.GDNKernelDispatcher.extend = extend
    logging.getLogger(__name__).info(
        "MemT recurrent short prefill installed: sequences <= %d tokens", limit
    )
