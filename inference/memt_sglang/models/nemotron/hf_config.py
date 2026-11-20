"""Register Pro while retaining native hybrid-cache/backend detection."""

from sglang.srt.configs.nemotron_h import NemotronHConfig
from transformers import AutoConfig

from memt_sglang.models.nemotron.config import native_config_fields, parse_memory_config


class MemTNemotronHConfig(NemotronHConfig):
    model_type = "memt_nemotron_h"
    has_no_defaults_at_init = True

    def __init__(self, backbone_configs=None, memt=None, **kwargs):
        source = dict(
            kwargs,
            model_type=self.model_type,
            backbone_configs=backbone_configs,
            memt=memt,
        )
        memory = parse_memory_config(source)
        super().__init__(**native_config_fields(source))
        self.memory_config = memory

    def to_dict(self):
        result = super().to_dict()
        result.pop("memory_config", None)
        return result


def register_nemotron_config():
    AutoConfig.register("memt_nemotron_h", MemTNemotronHConfig, exist_ok=True)
