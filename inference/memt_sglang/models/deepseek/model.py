"""SGLang 0.5.16 native DeepSeek plus HF-loaded session memory."""

import torch
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM as NativeDeepseekV4

from memt_sglang.models.common.weights import load_hf_weights
from memt_sglang.models.deepseek.integration import (
    DeepSeekMemoryHooks,
)
from memt_sglang.models.deepseek.layer import DeepSeekMemoryBlock
from memt_sglang.models.deepseek.weights import classify_memory, native_weights


class MemTDeepseekV4ForCausalLM(NativeDeepseekV4):
    """Native backbone names plus the published MemT memory parameter names."""

    def __init__(self, config, quant_config=None, prefix=""):
        from memt_sglang.runtime.c4_order import install

        install("memt-flash", {"index_topk": getattr(config, "index_topk", 512)})
        self._memory_config = config.memory_config
        self._memory_loaded = False
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.memt_state_shapes = self._memory_config.state_shapes

    def load_weights(self, weights, is_nextn=False):
        if is_nextn or self._memory_loaded:
            raise ValueError("DeepSeek memory does not support MTP or hot reload")
        self._memory_loaded = True  # A failed native load also requires a new model.
        weights = native_weights(weights, self.config)
        memory_config = self._memory_config
        blocks = {
            layer: DeepSeekMemoryBlock(
                memory_config,
                device=self.model.layers[layer].self_attn.wo_a.weight.device,
                dtype=torch.bfloat16,
            )
            for layer in memory_config.layers
        }
        result = load_hf_weights(
            weights,
            lambda stream: super(MemTDeepseekV4ForCausalLM, self).load_weights(
                stream, is_nextn=False
            ),
            blocks,
            classify_memory,
        )
        # Register memory only after native strict checks and post-load transforms.
        self._memory_hooks = DeepSeekMemoryHooks(self, blocks)
        return result

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        if not hasattr(self, "_memory_hooks"):
            raise RuntimeError("DeepSeek memory weights have not loaded successfully")
        # Native signature: deepseek_v4.py:2490-2497 at fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1.
        return self._memory_hooks.forward(
            super().forward,
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
