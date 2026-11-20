"""Pro configuration and weight-name contracts; no tensors or GPU imports."""

import unittest
from copy import deepcopy

from memt_sglang.models.common.hooks import MemoryHooks
from memt_sglang.models.nemotron.config import (
    FIXED_MEMORY_CONFIG,
    native_config_fields,
    parse_memory_config,
)
from memt_sglang.models.nemotron.weights import classify_memory, native_weight_shapes
from memt_sglang.validation import validate_model_config


def fixture():
    return {
        "model_type": "memt_nemotron_h",
        "architectures": ["MemTNemotronHForCausalLM"],
        "backbone_configs": {
            "model_type": "nemotron_h",
            "dtype": "bfloat16",
            "attention_bias": False,
            "mlp_bias": False,
            "use_bias": False,
            "mamba_proj_bias": False,
            "use_conv_bias": True,
            "mlp_hidden_act": "relu2",
            "mamba_hidden_act": "silu",
            "tie_word_embeddings": False,
            "residual_in_fp32": False,
            "mamba_ssm_cache_dtype": "float32",
            "vocab_size": 1024,
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "layers_block_type": ["mamba", "moe", "attention"],
            "mamba_num_heads": 8,
            "mamba_head_dim": 64,
            "n_groups": 4,
            "ssm_state_size": 128,
            "conv_kernel": 4,
            "chunk_size": 128,
            "n_routed_experts": 8,
            "moe_intermediate_size": 256,
            "moe_latent_size": 128,
            "n_shared_experts": 1,
            "moe_shared_expert_intermediate_size": 512,
        },
        "memt": dict(
            FIXED_MEMORY_CONFIG,
            num_memory_heads=4,
            num_kv_heads=4,
            head_dim=128,
            gate_rank=512,
            layers=[2],
            chunk_tokens=2048,
            norm_eps=1e-5,
            output_gate_softcap_limit=2.0,
        ),
    }


class ConfigTests(unittest.TestCase):
    def test_memory_decode_capture_protocol(self):
        class Graph:
            capture_active = True

            def __init__(self):
                self.injected = set()
                self.reads = []

            def read(self, layer, hidden, block):
                self.reads.append((layer, hidden, block))
                return ["residual"] * len(hidden)

        hooks = object.__new__(MemoryHooks)
        hooks.memory_decode_graph = Graph()
        hidden, block = ["row-0", "row-1"], object()
        residual = hooks.captured_memory(2, hidden, block)
        self.assertEqual(residual, ["residual", "residual"])
        self.assertEqual(hooks.memory_decode_graph.reads, [(2, hidden, block)])
        hooks.note_captured_injection(2, residual)
        self.assertEqual(hooks.memory_decode_graph.injected, {(2, 2)})

        hooks.memory_decode_graph.capture_active = False
        self.assertIsNone(hooks.captured_memory(2, hidden, block))

    def test_geometry_and_native_detection(self):
        config = fixture()
        before = deepcopy(config)
        family, memory = validate_model_config(config)
        self.assertEqual(family, "memt-pro")
        self.assertEqual(memory.state_shapes, {2: (1, 4, 128, 128)})
        self.assertEqual(memory.gate_rank, 512)
        fields = native_config_fields(config)
        self.assertEqual(fields["architectures"], ["NemotronHForCausalLM"])
        self.assertEqual(fields["mamba_ssm_dtype"], "float32")
        self.assertEqual(fields["mamba_n_groups"], 4)
        self.assertEqual(fields["mamba_chunk_size"], 128)
        self.assertEqual(fields["num_nextn_predict_layers"], 0)
        self.assertEqual(config, before)

    def test_invalid_memory_contracts(self):
        for key, value in (
            ("layers", [0]),
            ("layers", [2, 2]),
            ("gate_rank", True),
            ("head_dim", 64),
            ("num_kv_heads", 2),
            ("projection", "factorized"),
            ("gate_latent_norm", True),
            ("chunk_tokens", 0),
            ("norm_eps", float("nan")),
            ("unknown", 1),
        ):
            with self.subTest(key=key, value=value):
                config = fixture()
                config["memt"][key] = value
                with self.assertRaises(ValueError):
                    parse_memory_config(config)

    def test_conflicting_native_alias(self):
        config = fixture()
        config["backbone_configs"]["mamba_chunk_size"] = 256
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            native_config_fields(config)

    def test_memory_names_are_strict(self):
        self.assertEqual(
            classify_memory("backbone.layers.2.mixer.memt.in_proj.weight"),
            (2, "in_proj.weight"),
        )
        self.assertIsNone(classify_memory("backbone.layers.2.mixer.q_proj.weight"))
        for name in (
            "backbone.layers.02.mixer.memt.in_proj.weight",
            "backbone.layers.2.mixer.metis.in_proj.weight",
            "model.layers.2.memt.in_proj.weight",
        ):
            with self.assertRaises(ValueError):
                classify_memory(name)

    def test_backbone_source_shapes(self):
        shapes = native_weight_shapes(fixture()["backbone_configs"])
        self.assertEqual(shapes["backbone.layers.0.mixer.in_proj.weight"], (2056, 256))
        self.assertEqual(
            shapes["backbone.layers.1.mixer.experts.7.up_proj.weight"], (256, 128)
        )
        self.assertEqual(shapes["backbone.layers.2.mixer.q_proj.weight"], (512, 256))
        self.assertFalse(
            any("memt" in name or name.startswith("mtp.") for name in shapes)
        )


if __name__ == "__main__":
    unittest.main()
