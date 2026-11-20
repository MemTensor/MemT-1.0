"""Dense-v1 memory shares Nano's stable-GDN2 math with an explicit gate rank."""

import torch

from memt_sglang.models.qwen.layer import QwenMemoryBlock


class NemotronMemoryBlock(QwenMemoryBlock):
    def __init__(self, memory_config, *, device):
        if torch.device(device).type != "cuda":
            raise ValueError("Nemotron memory requires CUDA")
        # All initialization and subsequent computation stay on the target GPU.
        with torch.device(device):
            super().__init__(memory_config, device=device)
