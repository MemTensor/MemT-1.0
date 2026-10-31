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
    environment = os.environ if environment is None else environment
    value = environment.get("MEMT_EXPERIMENT_COMMIT_CHUNK_POLICY", "atomic")
    if value not in ("atomic", "global", "tail"):
        raise ValueError("COMMIT_CHUNK_POLICY must be atomic, global or tail")
    return value


def candidate_slots_per_transaction():
    # A private replay base protects prior commits/imports in the same transaction.
    return 1 if commit_chunk_policy() == "atomic" else 2


def partial_decode_quanta():
    value = os.environ.get("MEMT_PARTIAL_DECODE_QUANTA", "0")
    if not value.isdecimal() or int(value) > 64:
        raise ValueError("MEMT_PARTIAL_DECODE_QUANTA must be an integer from 0 to 64")
    return int(value)


NUMERICS_TIERS = ("auto", "fast", "strict")
# Switches that exist to keep results independent of batch composition. E1
# measured their cost on Nano (7.7% C1 / 16.6% C32 TPOT) and what they buy
# (round-to-round reproducibility at concurrency 32: 100% on, 88.9% off).
STRICT_NUMERICS_SWITCHES = (
    "SGLANG_DSV4_DETERMINISTIC_PROFILE",
    "SGLANG_DSV4_MHC_FIXED_SPLITS",
    "SGLANG_DSV4_ISOLATE_PREFILL_DECODE",
    "SGLANG_DSV4_MOE_COMBINE_FP32",
)


def numerics_tier(*, environment=None, args=None):
    """Resolved numerics tier.

    `auto` (default) reports what the requested switches actually amount to, so a
    configuration written before this option existed keeps starting and is
    labelled honestly. Only an explicit `fast`/`strict` can fail validation.
    """
    environment = os.environ if environment is None else environment
    value = environment.get("MEMT_NUMERICS", "auto")
    if value not in NUMERICS_TIERS:
        raise ValueError("MEMT_NUMERICS must be one of %s" % (NUMERICS_TIERS,))
    if value != "auto":
        return value
    requested = strict_numerics_pending(environment=environment)
    flag = bool(getattr(args, "enable_deterministic_inference", False)) if args else False
    return "strict" if (requested or flag) else "fast"


def strict_numerics_pending(*, environment=None):
    """Switches the operator asked for that `fast` would otherwise silently ignore.

    Only the DSV4 profile belongs here. PADDED_DECODE_QUERY is a replay-consistency
    requirement (the captured decode geometry), not a numeric-etolerance choice, so
    it is valid -- and required -- in either tier.
    """
    environment = os.environ if environment is None else environment
    return [name for name in STRICT_NUMERICS_SWITCHES
            if environment.get(name) not in (None, "0", "")]


def validate_numerics_tier(args, *, environment=None):
    """Reject a strict-only request made under `fast` instead of dropping it silently."""
    environment = os.environ if environment is None else environment
    requested = strict_numerics_pending(environment=environment)
    flag = bool(getattr(args, "enable_deterministic_inference", False))
    declared = environment.get("MEMT_NUMERICS", "auto")
    tier = numerics_tier(environment=environment, args=args)
    if declared == "auto":
        # Nothing was asserted, so nothing can contradict: report and move on.
        return tier
    if tier == "strict":
        missing = [name for name in ("SGLANG_DSV4_DETERMINISTIC_PROFILE",
                                     "SGLANG_DSV4_MHC_FIXED_SPLITS",
                                     "SGLANG_DSV4_ISOLATE_PREFILL_DECODE",
                                     "SGLANG_DSV4_MOE_COMBINE_FP32")
                   if environment.get(name) in (None, "0", "")]
        if missing or not flag:
            raise ValueError(
                "MEMT_NUMERICS=strict requires --enable-deterministic-inference and %s"
                % ", ".join(missing or ["every DSV4 profile switch"])
            )
        return tier
    if flag or requested:
        raise ValueError(
            "MEMT_NUMERICS=fast does not accept %s; set MEMT_NUMERICS=strict to keep "
            "round-to-round reproducibility, or remove them to run faster"
            % ", ".join(requested + (["--enable-deterministic-inference"] if flag else []))
        )
    return tier


def validate_flash_decode_graph(args):
    """Replay correctness only. Numerics tier is validated separately."""
    if not enabled("MEMORY_DECODE_GRAPH"):
        return
    # Names are listed so a typo produces an error naming the variable: enabled()
    # treats an absent switch as its default, so a misspelt one is otherwise
    # indistinguishable from one the operator never set.
    required = (
        ("MEMT_EXPERIMENT_C4_STABLE_TOPK", "1"),
        ("MEMT_EXPERIMENT_PADDED_DECODE_QUERY", "1"),
        ("MEMT_EXPERIMENT_REQUIRE_NATIVE_KV_CAPACITY", "1"),
        # Owner pinning in the overlay is the AND of these two, not either alone:
        # sglang-optin.patch sets
        #     _pin_decode_graph_owners = DETERMINISTIC_PROFILE and PIN_...
        # so checking only the second leaves a legal configuration that silently
        # skips the retained-owner assertion and reads storage it never kept.
        ("SGLANG_DSV4_PIN_DECODE_INTERMEDIATES", "1"),
        ("SGLANG_DSV4_DETERMINISTIC_PROFILE", "1"),
        ("SGLANG_OPT_USE_TILELANG_MHC_PRE", "1"),
        ("SGLANG_OPT_DEEPGEMM_HC_PRENORM", "1"),
    )
    missing = [
        name
        for name, want in required
        if os.environ.get(name, "1" if name.startswith("SGLANG_OPT_") else None) != want
    ]
    if missing or getattr(args, "speculative_skip_dp_mlp_sync", False):
        raise ValueError(
            "Flash memory decode graph requires these switches to be set exactly; "
            "missing or wrong: %s%s"
            % (
                ", ".join(missing) or "(none)",
                "; and --speculative-skip-dp-mlp-sync must be off"
                if getattr(args, "speculative_skip_dp_mlp_sync", False)
                else "",
            )
        )


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
    if commit_chunk_policy() != "atomic" and phases["prefill"].backend != "disabled":
        raise ValueError(
            "Partial commit requires eager prefill; decode graph remains supported"
        )
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
    chunk = getattr(args, "chunked_prefill_size", None)
    policy = commit_chunk_policy()
    if policy != "atomic":
        if type(chunk) is not int or chunk < (2048 if policy == "tail" else 1):
            raise ValueError("Partial commit requires a positive native token budget")
        if any(overlap_phases()) or not args.disable_overlap_schedule:
            raise ValueError("Partial commit requires serialized scheduler execution")
        if args.enable_dynamic_chunking or args.schedule_policy != "fcfs":
            raise ValueError("MemT partial commit uses its own FCFS chunk admission")
        return
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
        nano_compile=enabled("NANO_COMPILE"),
        commit_plan_cache=enabled("COMMIT_PLAN_CACHE"),
        memory_decode_graph=enabled("MEMORY_DECODE_GRAPH"),
        padded_decode_query=enabled("PADDED_DECODE_QUERY"),
        require_native_kv_capacity=enabled("REQUIRE_NATIVE_KV_CAPACITY"),
        c4_stable_topk=enabled("C4_STABLE_TOPK"),
        dsv4_deterministic_profile=os.environ.get(
            "SGLANG_DSV4_DETERMINISTIC_PROFILE", "0"
        )
        == "1",
        dsv4_mhc_fixed_splits=os.environ.get("SGLANG_DSV4_MHC_FIXED_SPLITS"),
        dsv4_phase_isolation=os.environ.get("SGLANG_DSV4_ISOLATE_PREFILL_DECODE")
        == "1",
        dsv4_moe_combine_fp32=os.environ.get("SGLANG_DSV4_MOE_COMBINE_FP32") == "1",
        dsv4_pin_decode_intermediates=os.environ.get(
            "SGLANG_DSV4_PIN_DECODE_INTERMEDIATES"
        )
        == "1",
        commit_chunk_policy=commit_chunk_policy(),
        commit_semantics_version=(
            "atomic-fresh-kv-v1"
            if commit_chunk_policy() == "atomic"
            else "partial-continuous-kv-private-fragment-read-v1"
        ),
        candidate_slots_per_transaction=candidate_slots_per_transaction(),
        partial_decode_quanta=partial_decode_quanta(),
        partial_query_first=os.environ.get("MEMT_PARTIAL_QUERY_FIRST", "0") == "1",
        partial_native_reference=enabled("PARTIAL_NATIVE_REFERENCE"),
        prefill_overlap=prefill,
        decode_overlap=decode,
        query_coalescing=asdict(coalescing) if coalescing else None,
        disable_overlap_schedule=args.disable_overlap_schedule,
        query_chunk_per_rank=args.chunked_prefill_size,
    )
    return result
