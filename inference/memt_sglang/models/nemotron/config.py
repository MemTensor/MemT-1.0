"""Published Pro schema and native Nemotron-H configuration translation."""

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

from memt_sglang.models.common.hf_config import memt_model, validate_fixed_fields
from memt_sglang.models.qwen.config import FIXED_QWEN_MEMORY_CONFIG

FIXED_MEMORY_CONFIG = {
    key: value
    for key, value in FIXED_QWEN_MEMORY_CONFIG.items()
    if key != "backbone_gdn_override"
}
FIXED_MEMORY_CONFIG.update(projection="dense_v1", gate_latent_norm=False)
SIZE_FIELDS = (
    "num_memory_heads",
    "num_kv_heads",
    "head_dim",
    "gate_rank",
    "layers",
    "chunk_tokens",
    "norm_eps",
    "output_gate_softcap_limit",
)


@dataclass(frozen=True)
class NemotronMemoryConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    gate_rank: int
    layers: tuple[int, ...]
    rms_norm_eps: float
    chunk_tokens: int
    output_gate_softcap: float

    @property
    def state_shapes(self):
        return {
            index: (1, self.num_heads, self.head_dim, self.head_dim)
            for index in self.layers
        }


def normalize_hf_config(config):
    fields = deepcopy(config if isinstance(config, Mapping) else config.to_dict())
    if memt_model(fields) != "memt-pro":
        raise ValueError("Expected a MemT-1.0-Pro configuration")
    if fields.get("architectures") != ["MemTNemotronHForCausalLM"]:
        raise ValueError("Expected the published MemTNemotronHForCausalLM architecture")
    fields.pop("auto_map", None)
    return fields


def parse_memory_config(config):
    fields = normalize_hf_config(config)
    memory = fields.get("memt")
    backbone = fields.get("backbone_configs")
    validate_fixed_fields(
        memory,
        FIXED_MEMORY_CONFIG,
        section="memt",
        exact=True,
        variable_fields=SIZE_FIELDS,
    )
    validate_fixed_fields(
        backbone,
        {
            "model_type": "nemotron_h",
            "dtype": "bfloat16",
            "quantization_config": None,
            "attention_bias": False,
            "mlp_bias": False,
            "use_bias": False,
            "mamba_proj_bias": False,
            "use_conv_bias": True,
            "mlp_hidden_act": "relu2",
            "mamba_hidden_act": "silu",
            "tie_word_embeddings": False,
            "mamba_ssm_cache_dtype": "float32",
            "residual_in_fp32": False,
        },
        section="backbone_configs",
        variable_fields=(
            "hidden_size",
            "num_attention_heads",
            "head_dim",
            "layers_block_type",
        ),
    )
    for name in (
        "num_memory_heads",
        "num_kv_heads",
        "head_dim",
        "gate_rank",
        "chunk_tokens",
    ):
        if type(memory[name]) is not int or memory[name] < 1:
            raise ValueError(f"memt.{name} must be a positive integer")
    for name in ("norm_eps", "output_gate_softcap_limit"):
        if (
            type(memory[name]) not in (int, float)
            or not math.isfinite(memory[name])
            or memory[name] <= 0
        ):
            raise ValueError(f"memt.{name} must be finite and positive")
    layout = backbone["layers_block_type"]
    if (
        not isinstance(layout, list)
        or not layout
        or any(item not in ("mamba", "attention", "moe", "mlp") for item in layout)
    ):
        raise ValueError("Invalid Nemotron layers_block_type")
    layers = [index for index, kind in enumerate(layout) if kind == "attention"]
    if (
        not layers
        or not isinstance(memory["layers"], list)
        or any(type(index) is not int for index in memory["layers"])
        or memory["layers"] != layers
    ):
        raise ValueError("memt.layers must list all native attention layers in order")
    if (
        memory["num_memory_heads"] != backbone["num_attention_heads"]
        or memory["num_kv_heads"] != memory["num_memory_heads"]
        or memory["head_dim"] != backbone["head_dim"]
    ):
        raise ValueError("Pro memory heads/dimensions must match native attention")
    return NemotronMemoryConfig(
        hidden_size=backbone["hidden_size"],
        num_heads=memory["num_memory_heads"],
        head_dim=memory["head_dim"],
        gate_rank=memory["gate_rank"],
        layers=tuple(layers),
        rms_norm_eps=memory["norm_eps"],
        chunk_tokens=memory["chunk_tokens"],
        output_gate_softcap=memory["output_gate_softcap_limit"],
    )


def native_config_fields(config):
    source = normalize_hf_config(config)
    result = deepcopy(source["backbone_configs"])
    # HF serialization uses these aliases; set the native constructor arguments
    # explicitly, so non-default geometries cannot silently inherit defaults.
    aliases = {
        "n_groups": "mamba_n_groups",
        "conv_kernel": "mamba_d_conv",
        "expand": "mamba_expand",
        "chunk_size": "mamba_chunk_size",
        "time_step_min": "mamba_dt_min",
        "time_step_max": "mamba_dt_max",
        "time_step_floor": "mamba_dt_init_floor",
        "time_step_limit": "mamba_dt_limit",
        "use_conv_bias": "mamba_conv_bias",
        "mamba_ssm_cache_dtype": "mamba_ssm_dtype",
    }
    for old, new in aliases.items():
        if old in result:
            if new in result and result[new] != result[old]:
                raise ValueError(f"Conflicting Nemotron fields: {old}, {new}")
            result[new] = result[old]
    result.update(
        model_type="memt_nemotron_h",
        architectures=["NemotronHForCausalLM"],
        memt=source["memt"],
        backbone_configs=source["backbone_configs"],
        torch_dtype="bfloat16",
        num_nextn_predict_layers=0,
    )
    for name in ("memt_runtime", "memt_checkpoint_id"):
        if name in source:
            result[name] = source[name]
    return result
