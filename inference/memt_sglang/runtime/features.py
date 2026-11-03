"""Opt-in graph and slot readers; metadata caching is enabled by default."""

import os
from dataclasses import replace


def enabled(name, *, environment=None):
    environment = os.environ if environment is None else environment
    value = environment.get(
        "MEMT_EXPERIMENT_" + name, "1" if name == "METADATA_CACHE" else "0"
    )
    if value not in ("0", "1"):
        raise ValueError(f"MEMT_EXPERIMENT_{name} must be 0 or 1")
    return value == "1"


def commit_chunk_policy(*, environment=None):
    """Reject requests for an extension that this core package does not install."""
    environment = os.environ if environment is None else environment
    policy = environment.get("MEMT_EXPERIMENT_COMMIT_CHUNK_POLICY", "atomic")
    if policy != "atomic":
        raise ValueError(
            "Continuous commit scheduling is provided by the separate global/tail extension"
        )
    return policy


def candidate_slots_per_transaction():
    """The atomic writer uses one private transaction candidate."""
    return 1


# Numerics tiers, the Flash capture-ownership gate and the stable-C4
# selector require the bundled SGLang patch, so they ship with it.


def validate_native_kv_capacity(actual, requested):
    if (
        enabled("REQUIRE_NATIVE_KV_CAPACITY")
        and requested is not None
        and actual < requested
    ):
        raise MemoryError(
            f"Native KV capacity was clipped from requested={requested} to actual={actual}; "
            "increase --mem-fraction-static or reduce MemT state reservations. "
            "This strict experiment requires the requested capacity before readiness."
        )


def sync_graph_config(args):
    """Use resolved native backends for worker hooks; reject unsupported ones."""
    graph = args.cuda_graph_config
    phases = {phase: getattr(graph, phase) for phase in ("prefill", "decode")}
    for phase, config in phases.items():
        if config.backend not in ("disabled", "breakable"):
            raise ValueError(
                f"MemT {phase} graph requires disabled or breakable backend"
            )
    args.cuda_graph_config = replace(graph, **phases)
    os.environ.update(
        {
            "MEMT_EXPERIMENT_" + phase.upper() + "_GRAPH": "1"
            if config.backend == "breakable"
            else "0"
            for phase, config in phases.items()
        }
    )
    if enabled("MEMORY_DECODE_GRAPH"):
        if phases["decode"].backend != "breakable" or not enabled("SLOT_READ"):
            raise ValueError(
                "Memory decode graph requires breakable decode and SLOT_READ=1"
            )
        if any(overlap_phases()):
            raise ValueError("Memory decode graph requires overlap scheduling disabled")
        if getattr(args, "enable_pdmux", False):
            raise ValueError(
                "Memory decode graph does not support concurrent PDMux streams"
            )


def overlap_phases():
    legacy = enabled("OVERLAP")
    decode_only = enabled("DECODE_ONLY_OVERLAP")
    if decode_only and not legacy:
        raise ValueError("DECODE_ONLY_OVERLAP requires legacy OVERLAP=1")
    phases = (False, True) if decode_only else (legacy, legacy)
    names = ("PREFILL_OVERLAP", "DECODE_OVERLAP")
    if any("MEMT_EXPERIMENT_" + name in os.environ for name in names):
        explicit = tuple(enabled(name) for name in names)
        if (legacy or decode_only) and explicit != phases:
            raise ValueError("Conflicting legacy and phase-specific overlap flags")
        phases = explicit
    return phases


def validate_overlap(args):
    requested = any(overlap_phases())
    if bool(getattr(args, "disable_overlap_schedule", False)) == requested:
        raise ValueError("Resolved native overlap does not match MemT experiment flags")


def validate_graphs(args):
    graph = args.cuda_graph_config
    for phase in ("prefill", "decode"):
        item = (
            graph.get(phase, {}) if isinstance(graph, dict) else getattr(graph, phase)
        )
        backend = item.get("backend") if isinstance(item, dict) else item.backend
        if backend != "disabled" and not (
            backend == "breakable" and enabled(phase.upper() + "_GRAPH")
        ):
            raise ValueError(
                f"MemT {phase} graph requires explicit breakable experiment"
            )


def validate_query_chunks(args):
    """Query budgets follow native page geometry, not memory write-event size."""
    chunk = getattr(args, "chunked_prefill_size", None)
    if type(chunk) is int and chunk == -1:
        return
    if type(chunk) is not int or chunk <= 0 or not enabled("QUERY_CHUNKS"):
        raise ValueError("Query chunking requires QUERY_CHUNKS=1 and a positive budget")
    page = getattr(args, "page_size", None)
    # server_args.py:7639-7645 asserts chunked_prefill_size % page_size == 0, and
    # DP attention divides the operator's value by dp_size first
    # (server_args.py:5616-5620), so an 8192 budget on dp 3 arrives as 2730.
    # Checking only `chunk >= page` let that through to a bare native assert.
    if page is not None and chunk % page:
        raise ValueError(
            f"Query chunk budget {chunk} must be a multiple of native page size "
            f"{page}; note that DP attention divides the configured budget by "
            "dp_size before this check"
        )


def execution_identity(args):
    """Report resolved native settings, including the post-DP chunk budget."""
    graph = args.cuda_graph_config
    backends = {}
    for phase in ("prefill", "decode"):
        item = graph[phase] if isinstance(graph, dict) else getattr(graph, phase)
        backends[phase] = item["backend"] if isinstance(item, dict) else item.backend
    prefill, decode = overlap_phases()
    from dataclasses import asdict

    from .query_coalescing import config_from_env

    coalescing = config_from_env()
    result = dict(
        graph_backends=backends,
        metadata_cache=enabled("METADATA_CACHE"),
        slot_read=enabled("SLOT_READ"),
        nano_compile=enabled("NANO_COMPILE"),
        commit_plan_cache=enabled("COMMIT_PLAN_CACHE"),
        padded_decode_query=enabled("PADDED_DECODE_QUERY"),
        require_native_kv_capacity=enabled("REQUIRE_NATIVE_KV_CAPACITY"),
        commit_chunk_policy=commit_chunk_policy(),
        commit_semantics_version="atomic-fresh-kv-v1",
        prefill_overlap=prefill,
        decode_overlap=decode,
        query_coalescing=asdict(coalescing) if coalescing else None,
        disable_overlap_schedule=args.disable_overlap_schedule,
        query_chunk_per_rank=args.chunked_prefill_size,
    )
    return result
