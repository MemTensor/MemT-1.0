"""Install the overlay's stable exact C4 selection implementation.

Stable tie ordering is the reference selection policy used by this adapter.
CUDA graphs can consume changing index values at replay: stable selection is
not a general CUDA graph requirement. Keep this implementation selected until
an alternative has passed the same model-level numerical comparison.
"""

import logging
import os
from functools import wraps
from importlib import import_module

# The overlay ships exactly one kernel width, and the width appears in the
# symbol name as well as in the check. Deriving the name from the constant keeps
# them from drifting apart when the overlay gains another width.
STABLE_TOPK_WIDTH = 512
STABLE_TOPK_SYMBOL = f"topk_transform_{STABLE_TOPK_WIDTH}_stable"
# Version of the overlay's stable top-k API this module is written against.
STABLE_TOPK_API_VERSION = 1

# Compute capabilities on which this kernel's numerics have actually been
# measured. The kernel carries no architecture branch of its own (it only
# rejects HIP), so this is a record of what was qualified, NOT a hardware
# requirement -- see `_allowed_capabilities`.
QUALIFIED_CAPABILITIES = ((10, 3),)

# Optional operator allowlist. Unset accepts devices supported by the kernel.
ARCH_OVERRIDE_VARIABLE = "MEMT_STABLE_C4_ARCH"


def _format_capability(capability):
    """(10, 3) -> 'sm_103', the spelling nvidia-smi and nvcc use."""
    major, minor = capability
    return f"sm_{major}{minor}"


def _allowed_capabilities(environ=None):
    """Capabilities `install` will accept, or None for "do not gate".

    Unset  -> no architecture allowlist; the kernel validates its capabilities.
    "any"  -> None, i.e. run anywhere and say so in the log.
    list   -> exactly those, e.g. "10.3,9.0".

    Returning None rather than a wildcard keeps the caller's branch explicit:
    there is a real difference between "this part is qualified" and "the
    operator took responsibility for an unqualified part", and the log has to
    be able to tell them apart afterwards.
    """
    raw = (
        (environ if environ is not None else os.environ)
        .get(ARCH_OVERRIDE_VARIABLE, "")
        .strip()
    )
    if not raw:
        return None
    if raw.lower() == "any":
        return None
    allowed = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        major, _, minor = item.partition(".")
        if not major.isdecimal() or not minor.isdecimal():
            raise ValueError(
                f"{ARCH_OVERRIDE_VARIABLE} takes 'any' or a comma list of "
                f"major.minor capabilities (e.g. '10.3,9.0'); got {item!r}"
            )
        allowed.append((int(major), int(minor)))
    if not allowed:
        raise ValueError(f"{ARCH_OVERRIDE_VARIABLE} was set but lists no capability")
    return tuple(allowed)


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
    capability = torch.cuda.get_device_capability()
    allowed = _allowed_capabilities()
    qualified = ", ".join(_format_capability(c) for c in QUALIFIED_CAPABILITIES)
    if allowed is not None and capability not in allowed:
        raise ValueError(
            f"MemT stable C4 is qualified on {qualified}; this device reports "
            f"{_format_capability(capability)}. The kernel has no architecture "
            f"branch, so it will probably run -- set "
            f"{ARCH_OVERRIDE_VARIABLE}={capability[0]}.{capability[1]} (or 'any') "
            "to allow it, accepting that its numerics are unmeasured there."
        )
    if capability not in QUALIFIED_CAPABILITIES:
        # Loud, and on every start: a run that produced numbers on an
        # unqualified part has to be identifiable from its log alone.
        log.warning(
            "MemT stable C4 running on %s, which is outside the qualified set "
            "(%s), because %s permits it. Numerics on this part have not been "
            "measured; do not quote them as validated.",
            _format_capability(capability),
            qualified,
            ARCH_OVERRIDE_VARIABLE,
        )

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
        "MemT C4 selection: native stable exact top-k API v%d on %s; "
        "logical-id ties and output order pinned",
        STABLE_TOPK_API_VERSION,
        _format_capability(capability),
    )
