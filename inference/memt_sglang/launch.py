"""Launch the pinned HTTP server with explicit graph opt-ins; dry-run is CPU-only."""

import argparse
import hashlib
import json
import os
import sys
from importlib import metadata
from pathlib import Path

from memt_sglang import SGLANG_REFERENCE_COMMIT, SGLANG_VERSION, require_sglang
from memt_sglang._vendor.fla import __version__ as FLA_VERSION
from memt_sglang.runtime.features import (
    enabled,
    execution_identity,
    overlap_phases,
    sync_graph_config,
)
from memt_sglang.validation import (
    validate_model_config,
    validate_runtime_config,
    validate_server_args,
)


def _flash_prefill_config(rest, graph):
    """Normalize an explicit Flash request before native resolution can change it."""
    from memt_sglang.runtime.flash_prefill import (
        FlashPrefillIntent,
        explicit_capture_config,
    )

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--cuda-graph-config")
    if sum(v.split("=", 1)[0] == "--cuda-graph-config" for v in rest) > 1:
        raise ValueError("Duplicate Flash cuda-graph-config")
    parsed, remaining = parser.parse_known_args(rest)
    override = json.loads(parsed.cuda_graph_config) if parsed.cuda_graph_config else {}
    if not isinstance(override, dict):
        raise ValueError("Flash cuda-graph-config must be an object")
    graph = {**graph, **override}
    phase = graph["prefill"]
    if not isinstance(phase, dict):
        raise ValueError("Flash prefill config must be an object")
    backend = phase.get("backend", "disabled")
    tokens = phase.get("bs", [512, 1024]) if backend == "breakable" else []
    if not isinstance(tokens, list):
        raise ValueError("Flash prefill bs must be a list")
    intent = FlashPrefillIntent(backend, tuple(tokens), 4)
    if backend == "breakable":
        bounded = explicit_capture_config(intent)
        if "max_bs" in phase and phase["max_bs"] != bounded["max_bs"]:
            raise ValueError("Flash prefill max_bs must match requested bs")
        graph = {**graph, "prefill": {**phase, **bounded}}
    return remaining, graph, intent


def _unobserved_flash_actual():
    return {"status": "not_observed", "backend": None,
            "evidence": "worker capture events; replay_counters(runner); optional replay_observer"}


def prepare(argv):
    for feature in (
        "METADATA_CACHE",
        "SLOT_READ",
        "PREFILL_GRAPH",
        "DECODE_GRAPH",
        "QUERY_CHUNKS",
        "NANO_COMPILE",
        "COMMIT_PLAN_CACHE",
        "MEMORY_DECODE_GRAPH",
        "PADDED_DECODE_QUERY",
        "REQUIRE_NATIVE_KV_CAPACITY",
    ):
        enabled(feature)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model-path", required=True)
    for name in (
        "max-sessions-per-rank",
        "max-inflight-commits-per-rank",
        "min-kv-tokens",
    ):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--json-model-override-args", default="{}")
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--deterministic-communication",
        action="store_true",
        help="Use NCCL Tree AllReduce, disable custom/fused collectives; not batch-invariant inference",
    )
    parser.add_argument(
        "--state-directory", help="Server-owned memory snapshot directory"
    )
    options, rest = parser.parse_known_args(argv)
    path = Path(options.model_path).expanduser().resolve()
    raw = (path / "config.json").read_bytes()
    config = json.loads(raw)
    model, memory_config = validate_model_config(config)
    capacity_overrides = {
        key: getattr(options, key)
        for key in (
            "max_sessions_per_rank",
            "max_inflight_commits_per_rank",
            "min_kv_tokens",
        )
    }
    runtime = validate_runtime_config(
        config, json.loads(options.json_model_override_args), capacity_overrides
    )
    graph = {
        phase: {"backend": "breakable" if enabled(phase.upper() + "_GRAPH") else "disabled"}
        for phase in ("prefill", "decode")
    }
    options.flash_prefill_intent = None
    if model == "memt-flash":
        rest, graph, options.flash_prefill_intent = _flash_prefill_config(rest, graph)
    engine = [
        "--model-path",
        str(path),
        "--host",
        "127.0.0.1",
        "--chunked-prefill-size",
        "-1",
        "--disable-radix-cache",
        *([] if any(overlap_phases()) else ["--disable-overlap-schedule"]),
        "--cuda-graph-config",
        json.dumps(graph),
        "--json-model-override-args",
        json.dumps({"memt_runtime": runtime}),
        *rest,
    ]
    if model == "memt-flash":
        engine.append("--enable-dp-attention")
    if options.deterministic_communication:
        if "--disable-custom-all-reduce" not in engine:
            engine.append("--disable-custom-all-reduce")
        engine.append("--enforce-disable-flashinfer-allreduce-fusion")
    identity = {
        "schema_version": 1,
        "model": model,
        "chunk_tokens": memory_config.chunk_tokens,
        "sglang_version": SGLANG_VERSION,
        "sglang_reference_commit": SGLANG_REFERENCE_COMMIT,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "model_revision": config.get("_commit_hash"),
        "memt_checkpoint_id": config.get("memt_checkpoint_id"),
        "memt_runtime": runtime,
        "fla_version": FLA_VERSION,
        "mode": "opt-in"
        if any(
            enabled(n)
            for n in ("PREFILL_GRAPH", "DECODE_GRAPH", "SLOT_READ", "QUERY_CHUNKS")
        )
        or any(overlap_phases())
        else "eager",
    }
    if options.flash_prefill_intent is not None:
        intent = options.flash_prefill_intent
        identity["flash_prefill"] = {
            "requested_backend": intent.requested_backend,
            "requested_capture_tokens": list(intent.capture_tokens),
            "resolved_backend": None, "resolved_capture_tokens": None,
            "status": "not_resolved", "gpu_qualified": False,
            "actual": _unobserved_flash_actual(),
        }
    if options.deterministic_communication:
        identity["communication"] = {
            "NCCL_ALGO": "allreduce:tree",
            "disable_custom_all_reduce": True,
            "enforce_disable_flashinfer_allreduce_fusion": True,
            "moe_a2a_backend": "none",
            "whole_model_determinism": False,
        }
    return options, engine, identity


def redacted_argv(argv):
    result, hide_next = [], False
    for value in argv:
        flag, separator, _ = value.partition("=")
        secret = (
            any(part in flag.lower() for part in ("key", "token", "password", "secret"))
            and flag.startswith("--")
            and flag
            not in {"--min-kv-tokens", "--max-total-tokens", "--max-prefill-tokens"}
        )
        if hide_next:
            result.append("<redacted>")
        elif secret and separator:
            result.append(flag + "=<redacted>")
        else:
            result.append(value)
        hide_next = secret and not separator
    return result


def resolve_flash_prefill(options, args, identity):
    """Validate before publishing the explicit request to spawned workers."""
    from memt_sglang.runtime.flash_prefill import (
        prefill_identity,
        validate_capture_request,
    )

    os.environ.pop("MEMT_FLASH_PREFILL_REQUEST", None)
    intent = options.flash_prefill_intent
    if intent is None:
        return identity
    phase = args.cuda_graph_config.prefill
    receipt = prefill_identity(intent, phase)
    if phase.backend != "disabled":
        if intent.requested_backend != "breakable":
            raise ValueError("Flash prefill resolved enabled without an explicit request")
        receipt = validate_capture_request(args, intent)
        os.environ["MEMT_FLASH_PREFILL_REQUEST"] = json.dumps({
            "requested_backend": intent.requested_backend,
            "capture_tokens": list(intent.capture_tokens), "dp_size": intent.dp_size,
        })
    return {**identity, "flash_prefill": {
        **receipt, "resolved_dp_size": args.dp_size, "actual": _unobserved_flash_actual(),
    }}


def emit_flash_prefill_event(event):
    """Worker stdout evidence, separate from the parent's unresolved actual state."""
    print(json.dumps({**event, "component": "memt_flash_prefill", "pid": os.getpid(),
                      "gpu_qualified": False}), flush=True)


def bootstrap(*, worker=True, replay_observer=None):
    """Shared scheduler entry, including revision_launch's existing bootstrap().

    Parent, tokenizer and detokenizer explicitly pass worker=False. Scheduler
    wrappers must call this before constructing their native ModelRunner.
    An explicit replay_observer(event) may synchronize for diagnosis; default
    production replay has no extra device synchronization or per-batch logging.
    """
    require_sglang()
    model = os.environ["MEMT_SGLANG_MODEL"]
    if model == "memt-nano":
        from memt_sglang.models.qwen.hf_config import register_qwen_config

        register_qwen_config()
    elif model == "memt-flash":
        from memt_sglang.models.deepseek.hf_config import register_deepseek_config

        register_deepseek_config()
    else:
        raise ValueError("Invalid MemT worker model")
    from memt_sglang.models.common.loader import install_model_loader

    install_model_loader()
    request = os.environ.get("MEMT_FLASH_PREFILL_REQUEST")
    if worker and model == "memt-flash" and request:
        from memt_sglang.runtime.flash_prefill import (
            FlashPrefillIntent,
            install_native_worker,
        )

        payload = json.loads(request)
        intent = FlashPrefillIntent(payload["requested_backend"],
                                    tuple(payload["capture_tokens"]), payload["dp_size"])
        install_native_worker(intent, emit=emit_flash_prefill_event,
                              replay_observer=replay_observer)


def run_scheduler(*args, **kwargs):
    bootstrap(worker=True)
    from memt_sglang.runtime.hooks import install_scheduler_hooks

    install_scheduler_hooks()
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


def run_detokenizer(*args, **kwargs):
    bootstrap(worker=False)
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process

    return run_detokenizer_process(*args, **kwargs)


def init_tokenizer(*args, **kwargs):
    bootstrap(worker=False)
    from sglang.srt.entrypoints.http_server import init_tokenizer_manager

    result = init_tokenizer_manager(*args, **kwargs)
    from memt_sglang.runtime.hooks import install_tokenizer_control

    install_tokenizer_control(result[0])
    return result


def main(argv=None):
    options, engine, identity = prepare(sys.argv[1:] if argv is None else argv)
    if options.dry_run:
        print(
            json.dumps({"identity": identity, "argv": redacted_argv(engine)}, indent=2)
        )
        return
    require_sglang()
    if options.deterministic_communication:
        # This is the communication part of the pinned SGLang deterministic
        # policy. Do not enable/bypass its unsupported dsv4 attention guard.
        os.environ["NCCL_ALGO"] = "allreduce:tree"
    # Memory injection uses the native BF16 output projection.
    os.environ["SGLANG_OPT_FP8_WO_A_GEMM"] = "0"
    if identity["model"] == "memt-flash":
        # Pinned 0.5.16 MHC prewarm uses chunked_prefill_size as an allocation
        # size and cannot handle our required -1 (unchunked prefill). Let the
        # unchanged native kernel compile on its actual first forward instead.
        os.environ["SGLANG_DSV4_MHC_PREWARM"] = "0"
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "memt_sglang.models"
    os.environ["MEMT_SGLANG_MODEL"] = identity["model"]
    os.environ["MEMT_MODEL_IDENTITY"] = identity["config_sha256"]
    os.environ["MEMT_STATE_DIRECTORY"] = (
        str(Path(options.state_directory).resolve()) if options.state_directory else ""
    )
    bootstrap(worker=False)
    from sglang.srt.entrypoints import http_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    from memt_sglang.runtime.http import install_http

    args = prepare_server_args(engine)
    sync_graph_config(args)
    validate_server_args(
        args,
        identity["model"],
        chunk_tokens=identity["chunk_tokens"],
        model_path=Path(options.model_path).expanduser().resolve(),
        runtime=identity["memt_runtime"],
        config_file=options.config,
        deterministic_communication=options.deterministic_communication,
    )
    identity = resolve_flash_prefill(options, args, identity)
    from memt_sglang.runtime.features import validate_numerics_tier

    # Enforced here, before any worker starts: the tier decides whether the
    # deterministic switches are allowed to be set at all.
    identity["numerics_tier"] = validate_numerics_tier(args)
    identity["execution"] = execution_identity(args)
    identity["mode"] = (
        "breakable"
        if any(
            backend == "breakable"
            for backend in identity["execution"]["graph_backends"].values()
        )
        else "eager"
    )
    identity["dependencies"] = {
        name: metadata.version(name) for name in ("sglang", "torch", "transformers")
    }
    identity["max_prefill_tokens"] = args.max_prefill_tokens
    identity["max_commit_tokens"] = (
        (args.context_length or args.max_prefill_tokens)
        if identity["execution"]["commit_chunk_policy"] != "atomic"
        else args.max_prefill_tokens
    )
    from transformers import AutoTokenizer

    from memt_sglang.models.common.snapshot_identity import snapshot_identity

    model_path = Path(options.model_path).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        tokenizer_type="deepseek_v4" if identity["model"] == "memt-flash" else None,
    )
    identity["snapshot_identity"] = snapshot_identity(
        json.loads((model_path / "config.json").read_text()), tokenizer
    )
    identity["memory_encoding"] = (
        "qwen-text-raw-v1"
        if identity["model"] == "memt-nano"
        else "deepseek-v4-native-raw-v1"
    )
    os.environ["MEMT_HF_SNAPSHOT_IDENTITY"] = identity["snapshot_identity"]
    os.environ["MEMT_MEMORY_ENCODING"] = identity["memory_encoding"]
    install_http(http_server, identity, dp_size=args.dp_size)
    try:
        # 0.5.16 Engine forwards the scheduler callback through its DP controller.
        http_server.launch_server(
            args,
            init_tokenizer_manager_func=init_tokenizer,
            run_scheduler_process_func=run_scheduler,
            run_detokenizer_process_func=run_detokenizer,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
