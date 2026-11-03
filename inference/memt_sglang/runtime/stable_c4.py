"""Install the overlay's stable exact C4 selection implementation.

Stable tie ordering is the reference selection policy used by this adapter.
CUDA graphs can consume changing index values at replay. Operators may select
the native policy explicitly; selection numerics and graph pointer ownership
are independent. Supported devices are determined by the CUDA implementation.
"""

import logging
from functools import wraps
from importlib import import_module

# The overlay ships exactly one kernel width, and the width appears in the
# symbol name as well as in the check. Deriving the name from the constant keeps
# them from drifting apart when the overlay gains another width.
STABLE_TOPK_WIDTH = 512
STABLE_TOPK_SYMBOL = f"topk_transform_{STABLE_TOPK_WIDTH}_stable"
# Version of the overlay's stable top-k API this module is written against.
STABLE_TOPK_API_VERSION = 1


def install(config):
    """Replace the indexer's top-k with the stable variant.

    Called once during Flash model construction, before capture, so the graph
    records the stable kernel and replays it.
    """
    import torch

    log = logging.getLogger(__name__)

    # The stable entry point is one fixed width; a different index_topk would
    # need a different kernel that the overlay does not currently provide.
    width = config.get("index_topk", STABLE_TOPK_WIDTH)
    if width != STABLE_TOPK_WIDTH:
        raise ValueError(
            f"MemT stable C4 needs index_topk={STABLE_TOPK_WIDTH}: the overlay "
            f"provides {STABLE_TOPK_SYMBOL} and no kernel for width {width}"
        )

    if not torch.cuda.is_available():
        raise ValueError("MemT stable C4 requires CUDA; no device is available")
    native = import_module("sglang.jit_kernel.dsv4.topk")
    function = getattr(native, STABLE_TOPK_SYMBOL, None)
    if getattr(
        native, "STABLE_EXACT_TOPK_API_VERSION", None
    ) != STABLE_TOPK_API_VERSION or not callable(function):
        raise ValueError(
            "C4_STABLE_TOPK requires the separately supplied SGLang stable exact "
            f"top-k API v{STABLE_TOPK_API_VERSION}"
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

    # Upstream's own symbol names, which carry the width too. Derived from the
    # same constant so that widening the overlay is one edit here, not four.
    entry = f"topk_transform_{STABLE_TOPK_WIDTH}"
    setattr(indexer, entry, v1)
    setattr(indexer, f"{entry}_v2", v2)
    indexer._memt_c4_stable_topk = True
    log.info(
        "MemT C4 selection: native stable exact top-k API v%d; "
        "logical-id ties and output order pinned",
        STABLE_TOPK_API_VERSION,
    )
