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
    chunk = getattr(args, "chunked_prefill_size", None)
    if type(chunk) is not int or (
        chunk != -1 and not (enabled("QUERY_CHUNKS") and chunk >= 2048)
    ):
        raise ValueError(
            "Query chunking requires its experiment flag and budget >=2048; otherwise use -1"
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
        prefill_overlap=prefill,
        decode_overlap=decode,
        query_coalescing=asdict(coalescing) if coalescing else None,
        disable_overlap_schedule=args.disable_overlap_schedule,
        query_chunk_per_rank=args.chunked_prefill_size,
    )
    return result
