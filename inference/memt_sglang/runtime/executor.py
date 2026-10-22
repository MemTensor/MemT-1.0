"""Eager baseline: execute independent samples with one shared step lifecycle."""

from copy import copy

import torch

from memt_sglang.protocol import SessionError
from memt_sglang.runtime.features import enabled
from memt_sglang.runtime.plan import StepPlan
from memt_sglang.runtime.prepared import prepare_group
from memt_sglang.runtime.state import StatePool
from memt_sglang.runtime.worker_sessions import WorkerSessions


class MemoryRuntime:
    def __init__(self, pool: StatePool):
        self.pool = pool
        self.plan: StepPlan | None = None
        self._layers: set[int] = set()
        self.worker_sessions = WorkerSessions(pool)
        self.kv_sessions = set()
        self.metadata_cache = enabled("METADATA_CACHE")
        self.slot_read = enabled("SLOT_READ")
        self._prepared_groups = {}

    def begin(self, forward_batch) -> StepPlan:
        if self.plan is not None:
            raise SessionError("Previous step has not finished")
        batch = copy(forward_batch)
        batch.rids = [
            rid
            if rid in self.worker_sessions.requests
            and not self.worker_sessions.requests[rid].complete
            else "retired"
            for rid in (forward_batch.rids or [])
        ]
        plan = StepPlan.from_batch(batch)
        self.plan = plan
        return plan

    @torch.no_grad()
    def run_layer(
        self, layer: int, hidden: torch.Tensor, output: torch.Tensor, compute
    ) -> torch.Tensor:
        """Execute prepared samples and fill caller-owned residual rows."""
        if self.plan is None or layer in self._layers or layer not in self.pool.storage:
            raise SessionError("Each registered memory layer must run once per step")
        if (
            hidden.ndim != 2
            or output.ndim != 2
            or len(hidden) != len(output)
            or len(hidden) < self.plan.token_count
        ):
            raise ValueError("Expected token-row hidden and residual buffers")
        if hidden.device != output.device or hidden.dtype != output.dtype:
            raise ValueError("Residual buffer must match hidden dtype and device")
        if (
            hidden.numel()
            and output.numel()
            and hidden.untyped_storage().data_ptr()
            == output.untyped_storage().data_ptr()
        ):
            raise ValueError("Residual buffer must not alias hidden states")
        output.zero_()
        self._run_prepared(layer, hidden, output, compute)
        self._layers.add(layer)
        return output

    def _run_prepared(self, layer, hidden, output, compute):
        baseline_layout = not (self.metadata_cache or self.slot_read)
        for commit in (False, True):
            key = (commit, hidden.device)
            if not self.metadata_cache or key not in self._prepared_groups:
                group = prepare_group(
                    self.plan,
                    self.worker_sessions,
                    self.pool.storage[layer].shape[0],
                    hidden.device,
                    commit=commit,
                    slot_read=self.slot_read,
                    contiguous_reads=not baseline_layout,
                )
                if self.metadata_cache:
                    self._prepared_groups[key] = group
            else:
                group = self._prepared_groups[key]
            if group is None:
                continue
            rows = (
                hidden[slice(*group.span)]
                if group.span is not None
                else hidden.index_select(0, group.indices)
            )
            slot_reader = getattr(compute, "packed_query_slots", None)
            if not commit and self.slot_read and callable(slot_reader):
                residual, final = slot_reader(
                    rows,
                    self.pool.storage[layer][:, 0],
                    group.slots,
                    lengths=group.lengths,
                )
            else:
                # Writers read a copy of their private candidate state. Queries
                # may share a single published snapshot, which readers never mutate.
                states = (
                    self.pool.storage[layer][group.slot_ids[0]]
                    if not baseline_layout and not commit and len(group.slot_ids) == 1
                    else torch.cat(
                        [self.pool.storage[layer][slot] for slot in group.slot_ids]
                    )
                )
                residual, final = compute.packed(
                    rows,
                    states,
                    lengths=list(group.lengths) if baseline_layout else group.lengths,
                    commit=commit,
                    write_states=states,
                )
            if (
                residual.shape != (group.token_count, output.shape[1])
                or residual.dtype != output.dtype
            ):
                raise RuntimeError("Packed memory output invariant violated")
            if group.span is not None:
                output[slice(*group.span)].copy_(residual)
            else:
                output.index_copy_(0, group.indices, residual)
            if commit:
                self.pool.storage[layer].index_copy_(0, group.slots, final.unsqueeze(1))

    def finish(self, success: bool) -> None:
        if self.plan is None:
            raise SessionError("No active memory step")
        try:
            if not success:
                return
            if self.plan.samples and self._layers != self.pool.storage.keys():
                raise SessionError("Not all memory layers executed")
            for sample in self.plan.samples:
                if sample.request.operation == "commit":
                    self.worker_sessions.advance(
                        sample.request.rid, sample.stop - sample.start
                    )
        finally:
            self.plan = None
            self._layers.clear()
            self._prepared_groups = {}
