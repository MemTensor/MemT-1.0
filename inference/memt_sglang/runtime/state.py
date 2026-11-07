"""Fixed FP32 slots for published memory and transaction candidates."""

from collections.abc import Mapping
from math import prod

import torch

from memt_sglang.protocol import CapacityError, SessionError
from memt_sglang.runtime.features import candidate_slots_per_transaction


def state_bytes_per_session(layer_shapes: Mapping[int, tuple[int, ...]]) -> int:
    """Validate model-provided state geometry and count its FP32 storage."""
    if not layer_shapes or any(
        type(layer) is not int
        or layer < 0
        or not isinstance(shape, (tuple, list))
        or not shape
        or any(type(size) is not int or size < 1 for size in shape)
        for layer, shape in layer_shapes.items()
    ):
        raise ValueError(
            "Each memory layer requires a nonnegative ID and positive state shape"
        )
    return sum(prod(shape) * torch.float32.itemsize for shape in layer_shapes.values())


class StatePool:
    """Worker-local storage managed by the session controller."""

    def __init__(
        self,
        layer_shapes: Mapping[int, tuple[int, ...]],
        *,
        max_sessions_per_rank: int,
        max_inflight_commits_per_rank: int,
        device: torch.device | str,
    ):
        if any(
            type(size) is not int or size < 1
            for size in (max_sessions_per_rank, max_inflight_commits_per_rank)
        ):
            raise ValueError("Session and commit capacities must be positive integers")
        state_bytes_per_session(layer_shapes)
        self.max_sessions = max_sessions_per_rank
        self.max_commits = max_inflight_commits_per_rank
        # One published slot per session and one candidate slot per transaction.
        self.slots_per_transaction = candidate_slots_per_transaction()
        count = self.max_sessions + self.max_commits * self.slots_per_transaction
        self.storage = {
            layer: torch.zeros((count, *shape), dtype=torch.float32, device=device)
            for layer, shape in layer_shapes.items()
        }
        self.sessions: dict[str, int] = {}
        self._free = list(reversed(range(count)))

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() for tensor in self.storage.values()
        )

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def take_slot(self) -> int:
        if not self._free:
            raise CapacityError("No free MemT state slot")
        return self._free.pop()

    def return_slot(self, slot: int) -> None:
        for tensor in self.storage.values():
            tensor[slot].zero_()
        self._free.append(slot)

    def ensure(self, session: str, *, source: str | None = None) -> None:
        if session in self.sessions:
            raise SessionError("Session already exists")
        if len(self.sessions) >= self.max_sessions:
            raise CapacityError("MemT session capacity exhausted")
        if source is not None and source not in self.sessions:
            raise SessionError("Source session does not exist")
        slot = self.take_slot()
        try:
            if source is not None:
                for tensor in self.storage.values():
                    tensor[slot].copy_(tensor[self.sessions[source]])
        except BaseException:
            self.return_slot(slot)
            raise
        self.sessions[session] = slot

    def release(self, session: str) -> None:
        slot = self.sessions.pop(session, None)
        if slot is not None:
            self.return_slot(slot)
