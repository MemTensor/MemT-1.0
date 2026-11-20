"""Experimental breakable decode with a fresh eager memory lifecycle per step."""

from contextlib import nullcontext
from dataclasses import fields, replace
from functools import wraps

from . import graph_stats


def _graph_key_value(runner, batch):
    """The batch size native feeds its graph key; must match native exactly.

    Under DP attention Flash syncs per-rank request counts and keys the graph on
    the cross-rank maximum, not the local batch. Re-deriving it differently
    would label buckets that no captured shape corresponds to.
    """
    if getattr(runner, "require_mlp_tp_gather", False):
        counts = getattr(batch, "original_global_num_tokens_cpu", None)
        if counts:
            return max(counts)
    return batch.batch_size


def _decode_reasons(runner, batch):
    """Label a native decode fallback; never decides the outcome."""
    value = _graph_key_value(runner, batch)
    if getattr(runner, "disable_padding", False):
        return ("native_veto_other",)
    if value > getattr(runner, "max_bs", value):
        return ("over_max_gear",)
    return ("native_veto_other",)


def _record(runner, batch, admitted, reasons):
    value = _graph_key_value(runner, batch)
    gear = None
    if admitted:
        # Native replays the smallest captured batch size >= the key
        # (decode load_batch _pad_to_bucket); record that, not the raw key.
        gear = next((bs for bs in sorted(runner.capture_bs) if bs >= value), None)
    graph_stats.record(
        phase="decode",
        key=value,
        gear=gear,
        admitted=admitted,
        reasons=reasons,
    )


def _output_row_count(output):
    """Leading-dim rows of a LogitsProcessorOutput, or None if not derivable."""
    logits = getattr(output, "next_token_logits", None)
    if logits is not None and hasattr(logits, "shape") and len(logits.shape):
        return int(logits.shape[0])
    return None


def install_logits_output_bridge():
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
        BreakableCudaGraphBackend,
    )

    cls = BreakableCudaGraphBackend
    if getattr(cls, "_memt_logits_bridge", False):
        return
    native_alloc, native_slice, native_copy = (
        cls._alloc_full_buffer,
        cls._slice_output,
        cls._copy_output_to_buffer,
    )
    native_rows = cls._output_rows

    def allocate(self, output, size):
        if isinstance(output, LogitsProcessorOutput):
            # ``size`` is the graph KEY, which for a decode/verify graph is the
            # batch size (decode_cuda_graph_runner._capture_graph_size returns
            # ``bs`` unless ragged verify mode is on). The body of a speculative
            # verify produces ``bs * draft_token_num`` logits rows, so allocating
            # by ``size`` and then clamping the copy to it (_output_rows below)
            # truncates the buffer to one row per request -- which is what made
            # eagle_sample fail with "shape '[1, 6]' is invalid for input of
            # size 1". Allocate for whichever is larger: the caller's size or the
            # rows the warmup output actually produced. Buckets are captured
            # largest-first, so the single shared buffer is sized once for the
            # biggest case.
            rows = _output_row_count(output)
            width = size if rows is None else max(size, rows)
            return replace(
                output,
                **{
                    f.name: self._alloc_full_buffer(getattr(output, f.name), width)
                    for f in fields(output)
                },
            )
        return native_alloc(self, output, size)

    def slice_output(self, output, count):
        if isinstance(output, LogitsProcessorOutput):
            return replace(
                output,
                **{
                    f.name: self._slice_output(getattr(output, f.name), count)
                    for f in fields(output)
                },
            )
        return native_slice(self, output, count)

    def copy_output(self, output, buffer, count):
        if isinstance(output, LogitsProcessorOutput) and isinstance(
            buffer, LogitsProcessorOutput
        ):
            for field in fields(output):
                self._copy_output_to_buffer(
                    getattr(output, field.name), getattr(buffer, field.name), count
                )
            return
        return native_copy(self, output, buffer, count)

    cls._alloc_full_buffer, cls._slice_output, cls._copy_output_to_buffer = (
        allocate,
        slice_output,
        copy_output,
    )

    def output_rows(self, output, cap):
        if isinstance(output, LogitsProcessorOutput):
            # Do not clamp a verify's rows to the batch-size key: the body wrote
            # bs * draft_token_num rows and the sampler needs all of them.
            rows = _output_row_count(output)
            if rows is None:
                return native_rows(self, output, cap)
            return rows
        return native_rows(self, output, cap)

    cls._output_rows = output_rows
    cls._memt_logits_bridge = True


def install_decode_query_guard():
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    original_can_run = DecodeCudaGraphRunner.can_run_graph
    if getattr(original_can_run, "_memt_query_guard", False):
        return
    original_execute = DecodeCudaGraphRunner.execute
    original_capture = DecodeCudaGraphRunner.capture_one_shape
    install_logits_output_bridge()

    # Pro needs the native Mamba split-op context around graph capture and
    # replay. SGLang 0.5.16 sets it for prefill but not for breakable decode;
    # Nano/Flash expose no decode_context, so this is a no-op there.
    def model_context(self, batch):
        hooks = getattr(self.model_runner.model, "_memory_hooks", None)
        context = getattr(hooks, "decode_context", None)
        return nullcontext() if context is None else context(batch)

    @wraps(original_capture)
    def capture(self, size, forward, *args, **kwargs):
        @wraps(forward)
        def scoped_forward(input_ids, positions, batch, *fargs, **fkwargs):
            with model_context(self, batch):
                return forward(input_ids, positions, batch, *fargs, **fkwargs)

        return original_capture(self, size, scoped_forward, *args, **kwargs)

    @wraps(original_can_run)
    def can_run(self, batch):
        runtime = getattr(self.model_runner.model, "memt_runtime", None)
        if runtime is None:
            return original_can_run(self, batch)
        if not query_graph_allowed(runtime, batch, "decode"):
            _record(self, batch, False, ("memt_query_veto",))
            return False
        allowed = original_can_run(self, batch)
        _record(self, batch, allowed, () if allowed else _decode_reasons(self, batch))
        return allowed

    @wraps(original_execute)
    def execute(self, batch, *args, **kwargs):
        runtime = getattr(self.model_runner.model, "memt_runtime", None)
        if runtime is None:
            return original_execute(self, batch, *args, **kwargs)
        # load_batch() may already have opened this step: the speculative verify
        # path calls it before execute(). Reuse that plan instead of raising
        # "Previous step has not finished". finish() below closes it either way.
        if runtime.plan is None:
            runtime.begin(batch)
        elif not runtime.owns(batch):
            # A step opened for another batch never executed. Close it and
            # fail closed rather than replay this batch against its plan.
            runtime.finish(success=False)
            raise RuntimeError("Decode graph found a memory step opened for a different batch")
        success = False
        try:
            with model_context(self, batch):
                result = original_execute(self, batch, *args, **kwargs)
            manager = getattr(
                self.model_runner.model._memory_hooks, "memory_decode_graph", None
            )
            if manager is not None:
                captured_layers = {layer for layer, _ in manager.prepared}
                if captured_layers != set(runtime.pool.storage):
                    raise RuntimeError("Memory graph does not cover every memory layer")
                runtime._layers.update(captured_layers)
            if self.model_runner.model._memory_hooks.pending:
                raise RuntimeError("Decode graph left an unconsumed memory residual")
            success = True
            return result
        finally:
            self.model_runner.model._memory_hooks.pending.clear()
            runtime.finish(success=success)

    can_run._memt_query_guard = True
    DecodeCudaGraphRunner.can_run_graph = can_run
    DecodeCudaGraphRunner.capture_one_shape = capture
    DecodeCudaGraphRunner.execute = execute


def install_dp_query_consensus():
    """One CPU vote per Flash backbone forward; memory layers stay TP1-local."""
    import torch
    from sglang.srt.distributed.parallel_state import get_world_group
    from sglang.srt.model_executor.model_runner import ModelRunner

    native = ModelRunner.forward
    if getattr(native, "_memt_dp_graph_consensus", False):
        return

    @wraps(native)
    def forward(self, batch, *args, **kwargs):
        runtime = getattr(self.model, "memt_runtime", None)
        if runtime is not None:
            records = [
                runtime.worker_sessions.requests.get(rid) for rid in (batch.rids or [])
            ]
            allowed = all(
                r is not None and not r.complete and r.request.operation == "query"
                for r in records
            )
            allowed = allowed and getattr(batch, "_memt_graph_override", True)
            mode = batch.forward_mode
            # The same MIN-reduce carries "any rank decodes" and "any rank
            # extends" as negated flags, so the Flash prefill gate can tell a
            # mixed prefill/decode round before native MAX_LEN normalization
            # without adding a collective.
            vote = torch.tensor(
                [int(allowed), -int(mode.is_decode()), -int(mode.is_extend())],
                dtype=torch.int32,
            )
            torch.distributed.all_reduce(
                vote,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            allowed_all, any_decode, any_extend = vote.tolist()
            batch._memt_graph_allowed = bool(allowed_all)
            batch._memt_dp_phase_mixed = bool(any_decode and any_extend)
        return native(self, batch, *args, **kwargs)

    forward._memt_dp_graph_consensus = True
    ModelRunner.forward = forward


def query_graph_allowed(runtime, batch, phase):
    """A world vote can veto a local query, never turn a local write into one."""
    if not getattr(batch, "_memt_graph_allowed", True):
        return False
    rids = batch.rids or []
    if not rids:
        original_mode = (
            getattr(batch, "_original_forward_mode", None) or batch.forward_mode
        )
        return bool(
            getattr(batch, "_memt_graph_allowed", False) and original_mode.is_idle()
        )
    # A speculative step runs as TARGET_VERIFY, not DECODE: one bonus token plus
    # the draft tokens being verified. is_decode() is an identity test against
    # ForwardMode.DECODE, so gating on it rejects every MTP decode step and the
    # captured graph is never replayed -- which is what the traces showed (zero
    # cudaGraphLaunch, kernel count identical to the eager arm, ~89% GPU idle
    # while the step still cost about 90 ms). SGLang's own
    # ForwardMode.is_cuda_graph() treats DECODE and TARGET_VERIFY as one
    # graph-eligible family; match it. Row geometry is handled separately:
    # MemoryDecodeGraph.captured_width is the runner's captured_req_width, which
    # is draft_token_num under a verify.
    mode_ok = (
        (
            batch.forward_mode.is_decode()
            or batch.forward_mode.is_target_verify()
        )
        if phase == "decode"
        else batch.forward_mode.is_extend()
    )
    if not mode_ok:
        return False
    records = (runtime.worker_sessions.requests.get(rid) for rid in rids)
    return all(
        record is not None
        and not record.complete
        and record.request.operation == "query"
        for record in records
    )
