"""Opt-in graph and slot readers; metadata caching is enabled by default."""

import os
from dataclasses import replace
from importlib import import_module


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


NUMERICS_TIERS = ("auto", "fast", "strict")
# These switches select the optional DSV4 reproducibility profile. Tensor
# lifetime and replay buffer geometry are enforced independently of this policy.
STRICT_NUMERICS_SWITCHES = (
    "SGLANG_DSV4_DETERMINISTIC_PROFILE",
    "SGLANG_DSV4_MHC_FIXED_SPLITS",
    "SGLANG_DSV4_ISOLATE_PREFILL_DECODE",
    "SGLANG_DSV4_MOE_COMBINE_FP32",
)


def numerics_tier(*, environment=None, args=None, model=None):
    """Resolved numerics tier.

    `auto` (default) reports what the requested switches actually amount to, so a
    configuration written before this option existed keeps starting and is
    labelled honestly. Partial arithmetic profiles resolve to `custom` and make
    no batch-invariance claim. Only an explicit `fast`/`strict` can fail validation.
    """
    environment = os.environ if environment is None else environment
    value = environment.get("MEMT_NUMERICS", "auto")
    if value not in NUMERICS_TIERS:
        raise ValueError("MEMT_NUMERICS must be one of %s" % (NUMERICS_TIERS,))
    if value != "auto":
        return value
    requested = (
        [] if model == "memt-nano" else strict_numerics_pending(environment=environment)
    )
    flag = (
        bool(getattr(args, "enable_deterministic_inference", False)) if args else False
    )
    if model == "memt-nano" or (model is None and not requested):
        return "strict" if flag else "fast"
    stable_selection = model != "memt-flash" or enabled(
        "C4_STABLE_TOPK", environment=environment
    )
    if flag and len(requested) == len(STRICT_NUMERICS_SWITCHES) and stable_selection:
        return "strict"
    return "custom" if requested or flag else "fast"


def native_flag(name, *, environment=None):
    """Use pinned SGLang's case-insensitive native boolean spelling."""
    environment = os.environ if environment is None else environment
    return str(environment.get(name, "0")).lower() in ("1", "true")


def strict_numerics_pending(*, environment=None):
    """Active DSV4 arithmetic switches, independent of replay buffer safety."""
    environment = os.environ if environment is None else environment
    result = []
    for name in STRICT_NUMERICS_SWITCHES:
        if name == "SGLANG_DSV4_MHC_FIXED_SPLITS":
            try:
                active = int(environment.get(name, "0")) > 0
            except ValueError:
                active = False
        else:
            active = native_flag(name, environment=environment)
        if active:
            result.append(name)
    return result


def validate_numerics_tier(args, *, environment=None, model=None):
    """Reject a strict-only request made under `fast` instead of dropping it silently."""
    environment = os.environ if environment is None else environment
    requested = (
        [] if model == "memt-nano" else strict_numerics_pending(environment=environment)
    )
    flag = bool(getattr(args, "enable_deterministic_inference", False))
    declared = environment.get("MEMT_NUMERICS", "auto")
    tier = numerics_tier(environment=environment, args=args, model=model)
    if declared == "auto":
        # Nothing was asserted, so nothing can contradict: report and move on.
        return tier
    if tier == "strict":
        switches = () if model == "memt-nano" else STRICT_NUMERICS_SWITCHES
        missing = [name for name in switches if name not in requested]
        if model == "memt-flash" and not enabled(
            "C4_STABLE_TOPK", environment=environment
        ):
            missing.append("MEMT_EXPERIMENT_C4_STABLE_TOPK")
        if missing or not flag:
            raise ValueError(
                "MEMT_NUMERICS=strict requires --enable-deterministic-inference"
                + (" and " + ", ".join(missing) if missing else "")
            )
        return tier
    if flag or requested:
        raise ValueError(
            "MEMT_NUMERICS=fast does not accept %s; set MEMT_NUMERICS=strict to keep "
            "round-to-round reproducibility, or remove them to run faster"
            % ", ".join(
                requested + (["--enable-deterministic-inference"] if flag else [])
            )
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
        ("MEMT_EXPERIMENT_PADDED_DECODE_QUERY", "1"),
        # The supplied overlay retains captured tensors independently of the
        # optional deterministic arithmetic profile.
        ("SGLANG_DSV4_PIN_DECODE_INTERMEDIATES", "1"),
    )
    missing = [
        name
        for name, want in required
        if not (
            native_flag(name)
            if name.startswith("SGLANG_")
            else os.environ.get(name) == want
        )
    ]
    if missing or getattr(args, "speculative_skip_dp_mlp_sync", False):
        raise ValueError(
            "Flash memory decode graph requires enabled replay safety features; "
            "missing or wrong: %s%s"
            % (
                ", ".join(missing) or "(none)",
                "; and --speculative-skip-dp-mlp-sync must be off"
                if getattr(args, "speculative_skip_dp_mlp_sync", False)
                else "",
            )
        )


def require_flash_capture_owners():
    """Check the native lifetime API independently of top-k selection policy."""
    native = import_module("sglang.srt.models.deepseek_v4")
    if getattr(native, "DSV4_DECODE_OWNERS_API_VERSION", None) != 1:
        raise ValueError("Flash graph requires captured tensor ownership API v1")


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
    # Native admission rounds an intermediate query chunk down to whole pages.
    # A budget smaller than one page cannot advance a long query.
    if page is not None and chunk < page:
        raise ValueError(
            f"Query chunk budget {chunk} must cover native page size {page}"
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
        dsv4_deterministic_profile=native_flag("SGLANG_DSV4_DETERMINISTIC_PROFILE"),
        dsv4_mhc_fixed_splits=os.environ.get("SGLANG_DSV4_MHC_FIXED_SPLITS"),
        dsv4_phase_isolation=native_flag("SGLANG_DSV4_ISOLATE_PREFILL_DECODE"),
        dsv4_moe_combine_fp32=native_flag("SGLANG_DSV4_MOE_COMBINE_FP32"),
        dsv4_pin_decode_intermediates=native_flag(
            "SGLANG_DSV4_PIN_DECODE_INTERMEDIATES"
        ),
        commit_chunk_policy=commit_chunk_policy(),
        commit_semantics_version="atomic-fresh-kv-v1",
        prefill_overlap=prefill,
        decode_overlap=decode,
        query_coalescing=asdict(coalescing) if coalescing else None,
        disable_overlap_schedule=args.disable_overlap_schedule,
        query_chunk_per_rank=args.chunked_prefill_size,
    )
    return result
