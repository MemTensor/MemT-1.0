"""Experiment: enable SGLang's split-KV EAGLE verify kernel on NVIDIA.

The verify step of speculative decoding runs through the prefill
``extend_attention_fwd`` kernel, which loops the whole prefix KV serially per
(sequence, head). With only draft_token_num queries that leaves the memory
system idle: on this target the verify attention kernel is ~4.1 ms of a 26 ms
decode round at concurrency 96, while the plain-decode grouped kernel serving
the same state costs ~0.3 ms.

SGLang already ships a flash-decode-style replacement
(``sglang/kernels/ops/attention/verify_splitkv.py``): split the prefix KV across
parallel programs, merge with a log-sum-exp combine, then handle the small
causal draft-draft block. It is gated to AMD gfx95 only, but the kernel module
is written NV-safe on purpose -- its own comment says keeping it NV-safe "lets
the numerics test run on the CUDA CI lane". ``can_handle()`` falls back for
every shape it cannot serve bit-equivalently, so enabling it can only change
speed, never the answer.

This installs the same enablement rule SGLang uses, minus the AMD-only test.
The flag stays opt-in because the kernel's tile configs were tuned on MI350X;
NVIDIA wants its own measurement before it becomes a default.
"""

import logging


def install_split_kv_verify():
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    if getattr(TritonAttnBackend.__init__, "_memt_split_kv_verify", False):
        return
    if not envs.SGLANG_ENABLE_SPLITKV_VERIFY.get():
        raise ValueError(
            "MEMT_EXPERIMENT_SPLIT_KV_VERIFY needs SGLANG_ENABLE_SPLITKV_VERIFY=1"
        )
    native_init = TritonAttnBackend.__init__

    def __init__(self, *args, **kwargs):
        native_init(self, *args, **kwargs)
        # Same predicate as upstream, without `is_gfx95_supported()`: the caller
        # still gates on topk == 1, which is the only condition the kernel's
        # causal-mask assumption actually requires.
        self.use_verify_splitkv = bool(self.topk == 1)
        logging.getLogger(__name__).info(
            "MemT split-KV verify enabled: topk=%s", self.topk
        )

    __init__._memt_split_kv_verify = True
    TritonAttnBackend.__init__ = __init__
