"""Experimental breakable decode with a fresh eager memory lifecycle per step."""

from dataclasses import fields, replace
from functools import wraps


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
            return replace(
                output,
                **{
                    f.name: self._alloc_full_buffer(getattr(output, f.name), size)
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
            return self._output_rows(output.next_token_logits, cap)
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
    install_logits_output_bridge()

    @wraps(original_can_run)
    def can_run(self, batch):
        runtime = getattr(self.model_runner.model, "memt_runtime", None)
        if runtime is not None and not query_graph_allowed(runtime, batch, "decode"):
            return False
        return original_can_run(self, batch)

    @wraps(original_execute)
    def execute(self, batch, *args, **kwargs):
        runtime = getattr(self.model_runner.model, "memt_runtime", None)
        if runtime is None:
            return original_execute(self, batch, *args, **kwargs)
        runtime.begin(batch)
        success = False
        try:
            result = original_execute(self, batch, *args, **kwargs)
            success = True
            return result
        finally:
            self.model_runner.model._memory_hooks.pending.clear()
            runtime.finish(success=success)

    can_run._memt_query_guard = True
    DecodeCudaGraphRunner.can_run_graph = can_run
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
            vote = torch.tensor([int(allowed)], dtype=torch.int32)
            torch.distributed.all_reduce(
                vote,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            batch._memt_graph_allowed = bool(vote.item())
        return native(self, batch, *args, **kwargs)

    forward._memt_dp_graph_consensus = True
    ModelRunner.forward = forward


def query_graph_allowed(runtime, batch, phase):
    """A world vote can veto a local query, never turn a local write into one."""
    if not getattr(batch, "_memt_graph_allowed", True):
        return False
    if not getattr(batch, "_memt_graph_override", True):
        return False
    rids = batch.rids or []
    if not rids:
        original_mode = (
            getattr(batch, "_original_forward_mode", None) or batch.forward_mode
        )
        return bool(
            getattr(batch, "_memt_graph_allowed", False) and original_mode.is_idle()
        )
    mode_ok = (
        batch.forward_mode.is_decode()
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
