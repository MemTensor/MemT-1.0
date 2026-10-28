"""Model configuration validation and post-parse server startup policy."""

import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from memt_sglang.models.common.hf_config import memt_model
from memt_sglang.runtime.config import RuntimeConfig

REQUIRED_SERVER_VALUES = {
    "disable_radix_cache": True,
    "pp_size": 1,
    "attn_cp_size": 1,
    "dcp_size": 1,
    "tokenizer_worker_num": 1,
    "nnodes": 1,
    "quantization": None,
    "speculative_algorithm": None,
    "disaggregation_mode": "null",
    "grpc_port": None,
}
SUPPORTED_SERVER_VALUES = {
    "dtype": ("auto", "bfloat16"),
    "load_format": ("auto", "safetensors"),
    "model_config_parser": ("auto", "hf"),
    "model_impl": ("auto", "sglang"),
}
DISALLOWED_SERVER_FLAGS = (
    "trust_remote_code",
    "use_ray",
    "encoder_only",
    "language_only",
    "is_embedding",
    "smg_grpc_mode",
    "grpc_mode",
    "skip_tokenizer_init",
    "enable_dp_lm_head",
    "enable_return_hidden_states",
    "enable_prefill_cp",
    "enable_dsa_prefill_context_parallel",
    "enable_two_batch_overlap",
    "enable_single_batch_overlap",
    "enable_torch_compile",
    "enable_lora",
    "enable_hierarchical_cache",
)
MODEL_SERVER_VALUES = {
    "memt-nano": {
        "enable_dp_attention": (False,),
        "kv_cache_dtype": ("auto", "bfloat16"),
    },
    "memt-flash": {
        "enable_dp_attention": (True,),
        "kv_cache_dtype": ("auto", "fp8_e4m3"),
    },
}
DETERMINISTIC_SERVER_VALUES = {
    "disable_custom_all_reduce": True,
    "enforce_disable_flashinfer_allreduce_fusion": True,
    "moe_a2a_backend": "none",
    "enable_symm_mem": False,
}


def validate_model_config(config):
    """Return "memt-nano" or "memt-flash" and the validated memory configuration."""
    model = memt_model(config)
    if model == "memt-nano":
        from memt_sglang.models.qwen.config import parse_memory_config
    elif model == "memt-flash":
        from memt_sglang.models.deepseek.config import parse_memory_config
    else:
        raise ValueError(
            "Only published MemT-1.0-Nano and MemT-1.0-Flash configs are supported"
        )
    return model, parse_memory_config(config)


def validate_runtime_config(config, override, capacity_overrides):
    if not isinstance(override, Mapping) or set(override) - {"memt_runtime"}:
        raise ValueError(
            "Model structure must come from config.json; only memt_runtime may be overridden"
        )
    declared = config.get("memt_runtime", {})
    updated = override.get("memt_runtime", {})
    if not isinstance(declared, Mapping) or not isinstance(updated, Mapping):
        raise ValueError("memt_runtime must be an object")
    runtime = {**asdict(RuntimeConfig()), **declared, **updated}
    runtime.update(
        {name: value for name, value in capacity_overrides.items() if value is not None}
    )
    return asdict(RuntimeConfig.from_hf_config({"memt_runtime": runtime}))


def validate_server_args(
    args,
    model,
    *,
    chunk_tokens,
    model_path,
    runtime,
    config_file=None,
    deterministic_communication=False,
    environment=None,
):
    """Validate all policy-relevant arguments once, before server construction."""
    # SGLang consumes --config before constructing ServerArgs; it is not a field.
    if config_file is not None:
        raise ValueError("MemT does not support --config merging")
    from memt_sglang.runtime.features import (
        validate_graphs,
        validate_overlap,
        validate_query_chunks,
    )
    from memt_sglang.runtime.query_coalescing import config_from_env

    validate_graphs(args)
    validate_overlap(args)
    validate_query_chunks(args)
    if config_from_env() is not None:
        layout = (args.tp_size, args.dp_size, args.ep_size)
        if not (
            (model == "memt-nano" and layout == (2, 1, 1))
            or (
                model == "memt-flash"
                and layout in ((4, 4, 4), (8, 8, 8))
                and args.enable_dp_attention
            )
        ):
            raise ValueError(
                "Query coalescing requires Nano TP2/DP1/EP1 or Flash TP=DP=EP4/8"
            )
        if args.enable_prefill_delayer:
            raise ValueError(
                "Query coalescing cannot be combined with native prefill delay"
            )
    for name, expected in REQUIRED_SERVER_VALUES.items():
        if getattr(args, name) != expected:
            raise ValueError(f"MemT requires {name}={expected!r}")
    for name, supported in SUPPORTED_SERVER_VALUES.items():
        if getattr(args, name) not in supported:
            raise ValueError(f"MemT requires {name} in {supported!r}")
    for name in DISALLOWED_SERVER_FLAGS:
        if getattr(args, name):
            raise ValueError(f"MemT does not support {name}")
    for name, supported in MODEL_SERVER_VALUES[model].items():
        if getattr(args, name) not in supported:
            raise ValueError(f"{model} requires {name} in {supported!r}")
    if model == "memt-flash" and args.tp_size != args.dp_size:
        raise ValueError(
            "MemT-1.0-Flash requires attention TP=1 (DP attention with tp_size == dp_size)"
        )
    if args.max_prefill_tokens < chunk_tokens:
        raise ValueError(f"max_prefill_tokens must be at least {chunk_tokens}")
    if Path(args.model_path).expanduser().resolve() != Path(model_path).resolve():
        raise ValueError("Engine model path differs from the validated config.json")
    if json.loads(args.json_model_override_args) != {"memt_runtime": runtime}:
        raise ValueError("Engine overrides differ from the validated runtime config")
    environment = os.environ if environment is None else environment
    if environment.get("SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES", "0").lower() not in (
        "0",
        "false",
        "",
    ):
        raise ValueError("SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES must be disabled")
    if deterministic_communication:
        for name, expected in DETERMINISTIC_SERVER_VALUES.items():
            if getattr(args, name) != expected:
                raise ValueError(
                    f"Deterministic communication requires {name}={expected!r}"
                )
