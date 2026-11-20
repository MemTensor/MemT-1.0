"""SGLang registry entry for MemT-1.0-Pro."""

from memt_sglang.runtime.hooks import install_sglang_hooks

install_sglang_hooks()

from memt_sglang.models.nemotron.model import MemTNemotronHForCausalLM

EntryClass = MemTNemotronHForCausalLM
