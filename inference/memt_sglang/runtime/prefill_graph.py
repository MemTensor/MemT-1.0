"""Experiment-only query guard around pinned SGLang breakable prefill."""

from functools import wraps

from .decode_graph import query_graph_allowed


def install_prefill_query_guard(*, exact_tokens=False):
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    original = PrefillCudaGraphRunner.can_run_graph
    if getattr(original, "_memt_query_guard", False):
        return
    original_load = PrefillCudaGraphRunner.load_batch

    @wraps(original_load)
    def load_batch(self, batch, *args, **kwargs):
        static = original_load(self, batch, *args, **kwargs)
        # Native body capture carries token/length metadata, but omits RIDs.
        # The eager outer forward must plan against this replay's live sessions.
        static.rids = list(batch.rids or [])
        return static

    @wraps(original)
    def can_run(self, batch):
        runtime = getattr(self.model_runner.model, "memt_runtime", None)
        if runtime is not None and not query_graph_allowed(runtime, batch, "prefill"):
            return False
        # Padding changes the native backbone's matrix shapes. Nano full-vocab
        # regression found drift even for 511 -> 512 tokens, so use a captured
        # shape only when it preserves the eager token count. Shape metadata
        # is host-visible; this adds no device read or synchronization.
        if runtime is not None and exact_tokens:
            if len(batch.input_ids) not in self.capture_num_tokens:
                return False
        return original(self, batch)

    can_run._memt_query_guard = True
    can_run._memt_exact_tokens = exact_tokens
    PrefillCudaGraphRunner.can_run_graph = can_run
    PrefillCudaGraphRunner.load_batch = load_batch
