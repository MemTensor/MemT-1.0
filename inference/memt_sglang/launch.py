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


def _flash_graph_config(rest, graph):
    """Preserve native decode options; experimental Flash prefill is separate."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--cuda-graph-config")
    if sum(v.split("=", 1)[0] == "--cuda-graph-config" for v in rest) > 1:
        raise ValueError("Duplicate cuda-graph-config")
    parsed, remaining = parser.parse_known_args(rest)
    override = json.loads(parsed.cuda_graph_config) if parsed.cuda_graph_config else {}
    if not isinstance(override, dict):
        raise ValueError("cuda-graph-config must be an object")
    graph = {**graph, **override}
    if not isinstance(graph["prefill"], dict):
        raise ValueError("prefill config must be an object")
    if graph["prefill"].get("backend", "disabled") != "disabled":
        raise ValueError(
            "Flash prefill capture requires the separate experimental adapter; "
            "use eager prefill with breakable decode in this package"
        )
    return remaining, graph


def prepare(argv):
    # A graph request selects its reader and reference geometry by default.
    # Explicit incompatible choices are preserved so validation can explain them.
    if enabled("MEMORY_DECODE_GRAPH"):
        for name in ("SLOT_READ", "PADDED_DECODE_QUERY", "C4_STABLE_TOPK"):
            os.environ.setdefault("MEMT_EXPERIMENT_" + name, "1")
        os.environ.setdefault("SGLANG_DSV4_PIN_DECODE_INTERMEDIATES", "1")
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
        "--control-timeout-seconds",
        type=float,
        help="Tokenizer/worker control timeout (default 180 seconds)",
    )
    parser.add_argument(
        "--deterministic-communication",
        action="store_true",
        help="Use NCCL Tree AllReduce, disable custom/fused collectives; not batch-invariant inference",
    )
    parser.add_argument(
        "--state-directory", help="Server-owned memory snapshot directory"
    )
    options, rest = parser.parse_known_args(argv)
    from memt_sglang.runtime.config import control_timeout_seconds

    if options.control_timeout_seconds is not None:
        os.environ["MEMT_CONTROL_TIMEOUT_SECONDS"] = str(
            options.control_timeout_seconds
        )
    timeout = control_timeout_seconds()
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
        phase: {
            "backend": "breakable" if enabled(phase.upper() + "_GRAPH") else "disabled"
        }
        for phase in ("prefill", "decode")
    }
    if model == "memt-flash":
        rest, graph = _flash_graph_config(rest, graph)
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
    strict_requested = (
        os.environ.get("MEMT_NUMERICS") == "strict"
        or "--enable-deterministic-inference" in engine
    )
    if strict_requested:
        if os.environ.get("MEMT_NUMERICS") == "fast":
            raise ValueError("fast conflicts with --enable-deterministic-inference")
        if "--enable-deterministic-inference" not in engine:
            engine.append("--enable-deterministic-inference")
        if model == "memt-flash":
            os.environ.setdefault("MEMT_EXPERIMENT_C4_STABLE_TOPK", "1")
            for name, value in (
                ("DETERMINISTIC_PROFILE", "1"),
                ("MHC_FIXED_SPLITS", "64"),
                ("ISOLATE_PREFILL_DECODE", "1"),
                ("MOE_COMBINE_FP32", "1"),
            ):
                os.environ.setdefault("SGLANG_DSV4_" + name, value)
    if model == "memt-flash":
        engine.append("--enable-dp-attention")
    if options.deterministic_communication:
        if "--disable-custom-all-reduce" not in engine:
            engine.append("--disable-custom-all-reduce")
        engine.append("--enforce-disable-flashinfer-allreduce-fusion")
    identity = {
        "schema_version": 1,
        "control_timeout_seconds": timeout,
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


def bootstrap(*, worker=True):
    """Register model and loader adapters before native worker construction."""
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


def resolve_mhc_prewarm(args, model):
    """Pinned native warmup allocates the resolved chunk budget as a shape."""
    if model == "memt-flash" and args.chunked_prefill_size == -1:
        # An unchunked sentinel is not a tensor dimension. Positive budgets
        # retain native warmup and the operator's explicit setting.
        os.environ["SGLANG_DSV4_MHC_PREWARM"] = "0"


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
    resolve_mhc_prewarm(args, identity["model"])
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
    from memt_sglang.runtime.features import validate_numerics_tier

    # Enforced here, before any worker starts: the tier decides whether the
    # deterministic switches are allowed to be set at all.
    identity["numerics_tier"] = validate_numerics_tier(args, model=identity["model"])
    identity["execution"] = execution_identity(args)
    if identity["model"] == "memt-flash":
        identity["execution"]["mhc_prewarm"] = os.environ.get(
            "SGLANG_DSV4_MHC_PREWARM", "native-default"
        )
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
    identity["max_commit_tokens"] = args.max_prefill_tokens
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
