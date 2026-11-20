"""Native Nemotron-H Mamba/MoE/KV execution with Pro session memory."""

import torch
from sglang.srt.models.nemotron_h import NemotronHForCausalLM

from memt_sglang.models.common.weights import load_hf_weights
from memt_sglang.models.nemotron.integration import NemotronMemoryHooks
from memt_sglang.models.nemotron.layer import NemotronMemoryBlock
from memt_sglang.models.nemotron.weights import classify_memory, native_weights


class MemTNemotronHForCausalLM(NemotronHForCausalLM):
    fall_back_to_pt_during_load = False

    def __init__(self, *, config, quant_config=None, prefix=""):
        if quant_config is not None:
            raise ValueError("Pro requires the unquantized BF16 checkpoint")
        super().__init__(config=config, quant_config=None, prefix=prefix)
        self._memory_config = config.memory_config
        self.memt_state_shapes = self._memory_config.state_shapes
        self._memory_loaded = False

    def load_weights(self, weights):
        if self._memory_loaded:
            raise ValueError("Nemotron memory does not support hot reload")
        self._memory_loaded = True
        blocks = {
            index: NemotronMemoryBlock(
                self._memory_config,
                device=self.model.layers[index].mixer.o_proj.weight.device,
            )
            for index in self._memory_config.layers
        }
        load_hf_weights(weights, self._load_native, blocks, classify_memory)
        self._memory_hooks = NemotronMemoryHooks(self, blocks)

    def _load_native(self, weights):
        return super().load_weights(native_weights(weights, self.config))

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
            raise RuntimeError("Nemotron memory weights have not loaded successfully")
        return self._memory_hooks.forward(
            super().forward,
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
