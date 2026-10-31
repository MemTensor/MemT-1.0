"""Force the native exact/stable C4 top-k entry point into the DSV4 indexer.

Why this file exists
--------------------
The DSV4 indexer picks the top-512 keys per token with `topk_transform_512`.
The stock implementation is free to break ties and to emit indices in
whatever order the kernel happens to produce, because for a *single* forward
that choice does not change the result: any k of the tied candidates is
equally valid.

That freedom becomes a correctness problem once the forward is captured into a
CUDA graph. Capture records the kernel's addresses and control flow once; every
replay re-executes those same device nodes. If the selection of tied candidates
depends on anything that differs between capture and replay -- a nondeterministic
reduction order, a lookup order over a hash-like structure -- then replay can
read a *different* set of KV rows than the ones it captured against, and the
memory path decodes from rows it never validated.

The overlay ships `topk_transform_512_stable`, which pins both the tie-break
(logical id) and the output order. This module swaps it in for the indexer's
two call sites so capture and replay make identical choices.

Scope of the change
-------------------
Only the DSV4 indexer is patched, and only when C4_STABLE_TOPK is on. Nothing
else in SGLang is touched: the patch is applied through the indexer module
object, not through a global import hook, so a process that never enables the
flag behaves exactly as stock.

The checks below are deliberately fail-closed. Each one corresponds to a
property the graph relies on; silently proceeding without it would produce a
graph that runs, produces plausible-looking numbers, and is wrong.
"""

import logging
from functools import wraps
from importlib import import_module

from memt_sglang.runtime.features import enabled


# Overlay-provided API version marks. The overlay is a separate deliverable
# from this package, so every capability is probed by an explicit version
# constant rather than by hasattr(): a missing constant means "this overlay
# predates the API", which must stop startup instead of degrading silently.
_GRAPH_PROFILE_MARKS = (
    ("sglang.kernels.ops.layernorm.mhc", "FIXED_MHC_REDUCTION_API_VERSION"),
    ("sglang.srt.managers.scheduler", "DSV4_PHASE_ISOLATION_API_VERSION"),
    ("sglang.srt.server_args", "DSV4_DETERMINISTIC_PROFILE_API_VERSION"),
    ("sglang.srt.models.deepseek_v2", "DSV4_MOE_COMBINE_FP32_API_VERSION"),
    ("sglang.srt.models.deepseek_v4", "DSV4_DECODE_OWNERS_API_VERSION"),
)


def install(config):
    """Replace the indexer's top-k with the stable variant.

    Called once during Flash model construction, before capture, so the graph
    records the stable kernel and replays it.
    """
    import torch

    # The stable entry point is a 512-wide kernel; a different index_topk would
    # need a different kernel that the overlay does not currently provide.
    if config.get("index_topk", 512) != 512:
        raise ValueError("MemT stable C4 currently validates index_topk=512 only")

    # SM103 is the only architecture this has been validated on. The kernel
    # itself carries no architecture branch (it only rejects HIP), so this is a
    # conservative qualification gate, not a hardware requirement -- widening
    # it requires measuring numerics on the target part first.
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        raise ValueError("MemT stable C4 profile currently validates CUDA SM103 only")

    native = import_module("sglang.jit_kernel.dsv4.topk")
    function = getattr(native, "topk_transform_512_stable", None)
    if getattr(native, "STABLE_EXACT_TOPK_API_VERSION", None) != 1 or not callable(
        function
    ):
        raise ValueError(
            "C4_STABLE_TOPK requires the separately supplied SGLang stable exact top-k API v1"
        )

    # The graph additionally depends on the rest of the isolated precision
    # profile. Checking a version mark is not a behavioural test: it only proves
    # the overlay was built from the revision that has these fixes.
    if enabled("MEMORY_DECODE_GRAPH"):
        for module, attribute in _GRAPH_PROFILE_MARKS:
            if getattr(import_module(module), attribute, None) != 1:
                raise ValueError(
                    "Flash graph requires the complete isolated DSV4 precision profile v1"
                )

    indexer = import_module("sglang.srt.layers.attention.dsv4.indexer")
    if getattr(indexer, "_memt_c4_stable_topk", False):
        return

    # Two call sites with different arities: the v2 variant takes a `metadata`
    # argument that the stable kernel does not need, so it forwards without it.
    @wraps(function)
    def v1(
        scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices=None
    ):
        # A zero-row call happens for idle ranks; the kernel would divide by an
        # empty shape, so short-circuit rather than call it.
        if scores.shape[0] == 0:
            return
        return function(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )

    def v2(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        metadata,
        out_raw_indices=None,
    ):
        return v1(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )

    indexer.topk_transform_512, indexer.topk_transform_512_v2 = v1, v2
    indexer._memt_c4_stable_topk = True
    logging.getLogger(__name__).info(
        "MemT C4 selection: native stable exact top-k API v1; logical-id ties and output order"
    )
