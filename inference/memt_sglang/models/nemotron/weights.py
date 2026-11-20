"""Strict published BF16 weight stream, then native Nemotron TP/EP loading."""

import re

_MEMORY = re.compile(r"backbone\.layers\.(0|[1-9][0-9]*)\.mixer\.memt\.(.+)")


def classify_memory(name):
    match = _MEMORY.fullmatch(name)
    if match:
        return int(match[1]), match[2]
    if {"memt", "metis"}.intersection(name.split(".")):
        raise ValueError(f"Malformed Nemotron memory weight key: {name}")
    return None


def native_weight_shapes(config):
    """Validate source completeness; upstream's streaming loader returns no keys.

    Keep the published names. Native SGLang owns QKV stacking, Mamba packing,
    A_log conversion, and expert partitioning. MTP is not used for serving.
    """
    c = config if isinstance(config, dict) else config.backbone_configs
    hidden, vocab = c["hidden_size"], c["vocab_size"]
    result = {
        "backbone.embeddings.weight": (vocab, hidden),
        "backbone.norm_f.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
    }
    for index, kind in enumerate(c["layers_block_type"]):
        prefix = f"backbone.layers.{index}."
        result[prefix + "norm.weight"] = (hidden,)
        if kind == "attention":
            query = c["num_attention_heads"] * c["head_dim"]
            kv = c["num_key_value_heads"] * c["head_dim"]
            shapes = {
                "q_proj.weight": (query, hidden),
                "k_proj.weight": (kv, hidden),
                "v_proj.weight": (kv, hidden),
                "o_proj.weight": (hidden, query),
            }
        elif kind == "mamba":
            heads = c["mamba_num_heads"]
            width = heads * c["mamba_head_dim"]
            conv = width + 2 * c["n_groups"] * c["ssm_state_size"]
            shapes = {
                "in_proj.weight": (width + conv + heads, hidden),
                "out_proj.weight": (hidden, width),
                "norm.weight": (width,),
                "A_log": (heads,),
                "D": (heads,),
                "dt_bias": (heads,),
                "conv1d.weight": (conv, 1, c["conv_kernel"]),
                "conv1d.bias": (conv,),
            }
        elif kind == "moe":
            experts, intermediate = c["n_routed_experts"], c["moe_intermediate_size"]
            latent = c.get("moe_latent_size") or hidden
            shapes = {
                "gate.weight": (experts, hidden),
                "gate.e_score_correction_bias": (experts,),
            }
            for expert in range(experts):
                shapes[f"experts.{expert}.up_proj.weight"] = (intermediate, latent)
                shapes[f"experts.{expert}.down_proj.weight"] = (latent, intermediate)
            if c.get("moe_latent_size"):
                shapes.update(
                    {
                        "fc1_latent_proj.weight": (latent, hidden),
                        "fc2_latent_proj.weight": (hidden, latent),
                    }
                )
            if c["n_shared_experts"]:
                shared = (
                    c["n_shared_experts"] * c["moe_shared_expert_intermediate_size"]
                )
                shapes.update(
                    {
                        "shared_experts.up_proj.weight": (shared, hidden),
                        "shared_experts.down_proj.weight": (hidden, shared),
                    }
                )
        elif kind == "mlp":
            shapes = {
                "up_proj.weight": (c["intermediate_size"], hidden),
                "down_proj.weight": (hidden, c["intermediate_size"]),
            }
        else:
            raise ValueError(f"Unsupported Nemotron block type: {kind}")
        result.update(
            {prefix + "mixer." + name: shape for name, shape in shapes.items()}
        )
    return result


def native_weights(weights, config):
    import torch

    expected = native_weight_shapes(config)
    seen = set()
    for name, tensor in weights:
        if name.startswith("mtp."):
            continue
        if name not in expected or name in seen:
            raise ValueError(f"Unexpected or aliased Nemotron weight: {name}")
        dtype = (
            torch.float32
            if name.endswith(".e_score_correction_bias")
            else torch.bfloat16
        )
        if tuple(tensor.shape) != expected[name] or tensor.dtype != dtype:
            raise ValueError(f"Nemotron weight shape/dtype mismatch: {name}")
        seen.add(name)
        yield name, tensor
    missing = expected.keys() - seen
    if missing:
        raise ValueError(f"Missing Nemotron weights: {sorted(missing)[:8]}")
