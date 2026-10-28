"""Native Qwen text backbone with session memory."""

import torch
from sglang.srt.configs.qwen3_5 import Qwen3_5MoeTextConfig
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import Qwen3_5MoeForCausalLM as NativeQwen35
from sglang.srt.runtime_context import get_server_args
from torch import nn

from memt_sglang.models.common.weights import load_hf_weights
from memt_sglang.models.qwen.integration import QwenMemoryHooks
from memt_sglang.models.qwen.layer import QwenMemoryBlock
from memt_sglang.models.qwen.weights import classify_memory, native_weights


class MemTQwen3_5MoeForCausalLM(nn.Module):
    """Native 3.5 text backbone plus the standard SGLang head/logits boundary.

    In 0.5.16, despite its name, NativeQwen35 returns hidden states and owns no
    lm_head. Keep it nested under model, as native multimodal wrappers do,
    without allocating a vision encoder for this text-only release.
    """

    fall_back_to_pt_during_load = False

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self._memory_config = config.memory_config
        self.memt_state_shapes = self._memory_config.state_shapes
        self._memory_loaded = False
        self.config = config
        fields = config.to_dict()
        fields.pop("model_type", None)
        fields.pop("text_config", None)
        native_config = Qwen3_5MoeTextConfig(**fields)
        self.model = NativeQwen35(
            native_config, quant_config=None, prefix=f"{prefix}.model".lstrip(".")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=f"{prefix}.lm_head".lstrip("."),
            use_attn_tp_group=get_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    def load_weights(self, weights):
        if self._memory_loaded:
            raise ValueError("Qwen memory does not support hot reload")
        self._memory_loaded = True
        blocks = {
            layer: QwenMemoryBlock(
                self._memory_config,
                device=self.model.layers[layer].o_proj.weight.device,
            )
            for layer in self._memory_config.layers
        }
        result = load_hf_weights(weights, self._load_native, blocks, classify_memory)
        missing = set(dict(self.named_parameters())) - result
        if missing:
            raise ValueError(f"Qwen native weights are missing: {sorted(missing)[:8]}")
        from memt_sglang.runtime.features import enabled

        if enabled("PREFILL_GRAPH"):
            from memt_sglang.models.qwen.padding import install_live_gate_projection

            install_live_gate_projection(self.model)
        self._memory_hooks = QwenMemoryHooks(self, blocks)
        return result

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        if kwargs:
            raise ValueError(f"Unsupported Qwen forward arguments: {sorted(kwargs)}")
        if not hasattr(self, "_memory_hooks"):
            raise RuntimeError("Qwen memory weights have not loaded successfully")
        return self._memory_hooks.forward(
            self._native_forward,
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
        )

    def _load_native(self, weights):
        head_loaded = False

        def body():
            nonlocal head_loaded
            for name, tensor in native_weights(weights):
                if name == "lm_head.weight":
                    getattr(
                        self.lm_head.weight, "weight_loader", default_weight_loader
                    )(self.lm_head.weight, tensor)
                    head_loaded = True
                else:
                    yield name, tensor

        loaded = {"model." + name for name in self.model.load_weights(body())}
        if head_loaded:
            loaded.add("lm_head.weight")
        return loaded

    def _native_forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden = self.model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
