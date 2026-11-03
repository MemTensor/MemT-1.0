"""Experiment-only query guard around pinned SGLang breakable prefill."""

from functools import wraps

from . import graph_stats
from .decode_graph import query_graph_allowed


def _record(runner, batch, admitted, reasons):
    """Zero-cost label for one admission decision; never changes the outcome.

    Flash batches never reach here: can_run defers them to the DP vote, which
    records the decision itself.
    """
    tokens = len(batch.input_ids)
    gear = None
    if admitted:
        gear = next((size for size in runner.capture_num_tokens if size >= tokens), None)
    graph_stats.record(
        phase="prefill", key=tokens, gear=gear, admitted=admitted, reasons=reasons
    )


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


def _admission_reasons(self, runtime, batch):
    """Classify a fallback for the counters; never decides the outcome.

    Native returns only a boolean, so the two rejection rules it documents are
    re-derived here purely to label why it said no. Order matters:
    the MemT query veto happens before native is consulted.
    """
    if not query_graph_allowed(runtime, batch, "prefill"):
        return ("memt_query_veto",)
    try:
        valid = live_prefill_tokens(batch)
    except ValueError:
        return ("native_veto_other",)
    tokens = self.capture_num_tokens
    if tokens and valid > max(tokens):
        return ("over_max_gear",)
    bucket = next((size for size in tokens if size >= valid), None)
    if bucket is not None and bucket > valid * 2:
        return ("below_padding_factor",)
    return ("native_veto_other",)


def install_prefill_query_guard():
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
        if hasattr(batch, "_memt_flash_prefill_allowed"):
            # Flash: the DP vote already validated every rank's spans, and
            # MAX_LEN rows (shorter or idle ranks) are not live tokens. Native
            # load_batch keeps the live extend spans; only RIDs are missing.
            static = original_load(self, batch, *args, **kwargs)
            static.rids = list(batch.rids or [])
            return static
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
        if hasattr(batch, "_memt_flash_prefill_allowed"):
            # Flash decided cluster-wide in its DP vote. A per-rank veto here
            # (e.g. live-span checks failing on MAX_LEN rows) would split ranks
            # between replay and eager and hang the in-graph collectives.
            return original(self, batch)
        runtime = getattr(self.model_runner.model, "memt_runtime", None)
        if runtime is not None and not query_graph_allowed(runtime, batch, "prefill"):
            _record(self, batch, False, ("memt_query_veto",))
            return False
        if runtime is not None:
            try:
                live_prefill_tokens(batch)
            except ValueError:
                _record(self, batch, False, ("native_veto_other",))
                return False
        allowed = original(self, batch)
        if runtime is not None:
            _record(self, batch, allowed,
                    () if allowed else _admission_reasons(self, runtime, batch))
        return allowed

    can_run._memt_query_guard = True
    PrefillCudaGraphRunner.can_run_graph = can_run
    PrefillCudaGraphRunner.load_batch = load_batch
