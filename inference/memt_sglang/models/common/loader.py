"""Select MemT classes while preserving SGLang's native backbone detection."""

from copy import copy
from functools import wraps

MEMT_ARCHITECTURES = {
    "memt_qwen3_5_moe": "MemTQwen3_5MoeForCausalLM",
    "memt_deepseek_v4": "MemTDeepseekV4ForCausalLM",
    "memt_nemotron_h": "MemTNemotronHForCausalLM",
}


def install_model_loader():
    """Adapt the three resolver references in pinned SGLang 0.5.16."""
    from sglang.srt import model_loader
    from sglang.srt.model_loader import loader, utils

    original = utils.get_model_architecture
    if getattr(original, "_memt_loader", False):
        return

    @wraps(original)
    def get_model_architecture(model_config):
        architecture = MEMT_ARCHITECTURES.get(model_config.hf_config.model_type)
        if architecture is None:
            return original(model_config)
        # Native shape derivation and backend selection keep their original HF
        # config. Only class resolution sees the independently registered MemT name.
        selection = copy(model_config)
        selection.hf_config = copy(model_config.hf_config)
        selection.hf_config.architectures = [architecture]
        result = original(selection)
        model_config._resolved_model_arch = selection._resolved_model_arch
        model_config._resolved_model_impl = selection._resolved_model_impl
        return result

    get_model_architecture._memt_loader = True
    utils.get_model_architecture = get_model_architecture
    loader.get_model_architecture = get_model_architecture
    model_loader.get_model_architecture = get_model_architecture
