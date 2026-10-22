"""Per-forward immutable token/slot metadata shared across memory layers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PreparedGroup:
    slot_ids: tuple[int, ...]
    lengths: tuple[int, ...]
    indices: torch.Tensor | None
    span: tuple[int, int] | None
    slots: torch.Tensor | None
    token_count: int


def prepare_group(
    plan, sessions, capacity, device, *, commit, slot_read, contiguous_reads=True
):
    samples = tuple(
        sample
        for sample in plan.samples
        if (sample.request.operation == "commit") == commit
        and (commit or sessions.memory_read[sample.request.session])
    )
    if not samples:
        return None
    slot_ids = tuple(sessions.requests[sample.request.rid].slot for sample in samples)
    lengths = tuple(sample.stop - sample.start for sample in samples)
    contiguous = (
        contiguous_reads
        and not commit
        and all(a.stop == b.start for a, b in zip(samples, samples[1:]))
    )
    span = (samples[0].start, samples[-1].stop) if contiguous else None
    indices = (
        None
        if span is not None
        else torch.cat(
            [
                torch.arange(sample.start, sample.stop, device=device)
                for sample in samples
            ]
        )
    )
    slots = (
        torch.tensor(slot_ids, device=device, dtype=torch.int64)
        if commit or slot_read
        else None
    )
    return PreparedGroup(slot_ids, lengths, indices, span, slots, sum(lengths))
