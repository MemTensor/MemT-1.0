"""Experiment-only query guard around pinned SGLang breakable prefill."""

from functools import wraps

from .decode_graph import query_graph_allowed


def live_prefill_tokens(batch):
    """Validate host metadata without reading CUDA tensors or changing positions."""
    rids = batch.rids or []
    lengths = batch.extend_seq_lens_cpu
    prefixes = batch.extend_prefix_lens_cpu
    if (
        not rids
        or batch.batch_size != len(rids)
        or lengths is None
        or prefixes is None
        or len(lengths) != len(rids)
        or len(prefixes) != len(rids)
        or any(type(n) is not int or n < 1 for n in lengths)
        or any(type(n) is not int or n < 0 for n in prefixes)
    ):
        raise ValueError("Invalid live prefill request metadata")
    valid = sum(lengths)
    physical = len(batch.input_ids)
    non_padding = getattr(batch, "num_token_non_padded_cpu", None)
    if (
        valid > physical
        or batch.extend_num_tokens != valid
        or (non_padding is not None and non_padding != valid)
        or len(batch.positions) != physical
        or len(batch.out_cache_loc) != physical
    ):
        raise ValueError("Live prefill token metadata disagrees with request spans")
    return valid


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
        runtime = getattr(self.model_runner.model, "memt_runtime", None)
        valid = live_prefill_tokens(batch) if runtime is not None else None
        static = original_load(self, batch, *args, **kwargs)
        # Native body capture carries token/length metadata, but omits RIDs.
        # The eager outer forward must plan against this replay's live sessions.
        static.rids = list(batch.rids or [])
        if valid is not None:
            # Native attention/GDN metadata is initialized using the live batch.
            # Never replace its per-request spans with the capture gear length.
            if (
                static.batch_size != batch.batch_size
                or static.extend_seq_lens_cpu != batch.extend_seq_lens_cpu
                or static.extend_prefix_lens_cpu != batch.extend_prefix_lens_cpu
                or len(static.input_ids) < valid
            ):
                raise ValueError("Native prefill padding changed live request metadata")
            static.extend_seq_lens_cpu = list(batch.extend_seq_lens_cpu)
            static.extend_prefix_lens_cpu = list(batch.extend_prefix_lens_cpu)
            static.num_token_non_padded_cpu = valid
            static._memt_prefill_valid_tokens = valid
            static._memt_prefill_gear_tokens = len(static.input_ids)
        return static

    @wraps(original)
    def can_run(self, batch):
        runtime = getattr(self.model_runner.model, "memt_runtime", None)
        if runtime is not None and not query_graph_allowed(runtime, batch, "prefill"):
            return False
        if runtime is not None:
            try:
                live_prefill_tokens(batch)
            except ValueError:
                return False
        # Retained solely for controlled historical diagnostics. Production
        # uses native gear padding after validating effective request metadata.
        if runtime is not None and exact_tokens:
            if len(batch.input_ids) not in self.capture_num_tokens:
                return False
        return original(self, batch)

    can_run._memt_query_guard = True
    can_run._memt_exact_tokens = exact_tokens
    PrefillCudaGraphRunner.can_run_graph = can_run
    PrefillCudaGraphRunner.load_batch = load_batch
