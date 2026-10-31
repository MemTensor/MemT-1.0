"""Phase-selective native overlap, with atomic commit synchronization boundaries."""

import logging
from functools import wraps

logger = logging.getLogger(__name__)


def batch_phase(scheduler, batch):
    if batch is None:
        return None
    if getattr(scheduler, "require_mlp_sync", False):
        return "prefill" if batch.is_extend_in_batch else "decode"
    return "prefill" if batch.forward_mode.is_extend() else "decode"


def has_atomic_write(scheduler, batch, phase):
    if phase != "prefill":
        return False
    from memt_sglang.protocol import MemoryRequest, is_memt_rid

    atomic = any(
        is_memt_rid(req.rid) and MemoryRequest.from_rid(req.rid).operation != "query"
        for req in batch.reqs
    )
    if getattr(scheduler, "require_mlp_sync", False):
        # Only prefill boundaries exchange commit metadata. Memory math is TP1.
        import torch
        from sglang.srt.distributed.parallel_state import get_world_group

        vote = torch.tensor([int(atomic)], dtype=torch.int32)
        torch.distributed.all_reduce(
            vote, op=torch.distributed.ReduceOp.MAX, group=get_world_group().cpu_group
        )
        atomic = bool(vote.item())
    return atomic


def install_phase_overlap(scheduler_class, *, prefill, decode):
    native = scheduler_class.is_disable_overlap_for_batch
    if getattr(native, "_memt_phase_overlap", False):
        if native._memt_phase_flags != (prefill, decode):
            raise RuntimeError(
                "Overlap phases are immutable after scheduler installation"
            )
        return
    native_run = scheduler_class.run_batch
    allowed = {"prefill": prefill, "decode": decode, None: False}

    @wraps(native)
    def disable(self, batch, last_batch):
        phase = batch_phase(self, batch)
        atomic = has_atomic_write(self, batch, phase)
        self._memt_current_phase, self._memt_current_atomic = phase, atomic
        native_disable = native(self, batch, last_batch)
        # Native True pops exactly one result; never drain an empty queue.
        if not self.result_queue or last_batch is None:
            return False
        previous = getattr(self, "_memt_previous_phase", batch_phase(self, last_batch))
        boundary = (
            atomic
            or getattr(self, "_memt_previous_atomic", False)
            or not allowed[phase]
            or not allowed[previous]
        )
        key = (previous, phase, bool(atomic), bool(boundary))
        observed = getattr(self, "_memt_overlap_observed", set())
        if key not in observed:
            observed.add(key)
            self._memt_overlap_observed = observed
            logger.info(
                "MemT overlap prefill=%s decode=%s previous=%s current=%s atomic=%s drain=%s",
                prefill,
                decode,
                previous,
                phase,
                atomic,
                bool(native_disable or boundary),
            )
        return bool(native_disable or boundary)

    @wraps(native_run)
    def run(self, batch, *args, **kwargs):
        atomic = getattr(self, "_memt_current_atomic", False)
        if atomic:
            self.forward_stream.synchronize()
        result = native_run(self, batch, *args, **kwargs)
        if atomic:
            self.forward_stream.synchronize()
        self._memt_previous_phase = self._memt_current_phase
        self._memt_previous_atomic = atomic
        return result

    disable._memt_phase_overlap = True
    disable._memt_phase_flags = (prefill, decode)
    scheduler_class.is_disable_overlap_for_batch = disable
    scheduler_class.run_batch = run


def install_decode_only_overlap(scheduler_class):
    """Compatibility entry point for the frozen Nano experiment."""
    install_phase_overlap(scheduler_class, prefill=False, decode=True)
