"""HF memory identity, independent of SGLang's adapted configuration."""

import hashlib
import json
from copy import deepcopy

from memt_sglang.models.common.hf_config import memt_model


def snapshot_identity(config, tokenizer):
    """Match HF Session._state_identity without importing checkpoint model code."""
    fields = deepcopy(config if isinstance(config, dict) else config.to_dict())
    model = memt_model(fields)
    if model == "memt-nano":
        backbone = fields["backbone_configs"]
        codec = "qwen-text-raw-v1"
    elif model == "memt-pro":
        backbone = fields["backbone_configs"]
        codec = "nemotron-text-raw-v1"
    elif model == "memt-flash":
        from transformers import DeepseekV4Config

        # HF adds inherited defaults before hashing config.to_dict(). Reproduce
        # that serialization, rather than hashing raw JSON or the engine adapter.
        fields = DeepseekV4Config(**fields).to_dict()
        fields["model_type"] = "memt_deepseek_v4"
        backbone = {
            key: value
            for key, value in fields.items()
            if key
            not in {
                "_name_or_path",
                "transformers_version",
                "auto_map",
                "architectures",
                "memt",
            }
        }
        codec = "deepseek-v4-native-raw-v1"
    else:
        raise ValueError("Unsupported HF memory identity")
    identity = {
        "backbone": backbone,
        "memt": fields["memt"],
        "codec": codec,
        "template": tokenizer.chat_template,
        "vocabulary": tokenizer.get_vocab(),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()
