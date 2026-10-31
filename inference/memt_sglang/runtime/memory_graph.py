"""Read-only memory CUDA nodes sharing SGLang's decode capture lifecycle.

Only device slot *values* change at replay. Session planning and publication
remain outside capture. Invalid/padded rows use slot -1, which both readers
mask before loading state. No transaction writes are captured.
"""

import logging
from collections import Counter
from contextlib import contextmanager
from functools import wraps

import torch


class MemoryDecodeGraph:
    def __init__(self, hooks):
        self.hooks = hooks
        self.capture_active = False
        self.slots = None
        self.residuals = {}
        self.constants = {}
        self.prepared = set()
        self.injected = set()
        self.capture_buckets = set()
        self.fallbacks = Counter()
        self.replays = 0
        self.uploads = 0
        self._last_slots = None
        self._host_buffers = []

    def note_fallback(self, reason):
        self.fallbacks[reason] += 1
        if self.fallbacks[reason] == 1:
            logging.getLogger(__name__).info(
                "MemT memory decode graph fallback: %s", reason
            )

    def allocate(self, capacity, device):
        if self.slots is None:
            self.slots = torch.full((capacity,), -1, dtype=torch.int64, device=device)
        elif capacity > self.slots.numel():
            raise RuntimeError("Memory graph buffers cannot grow after capture")

    @contextmanager
    def capture(self, runner):
        if self.capture_active:
            raise RuntimeError("Nested memory capture")
        if runner.captured_req_width != 1:
            raise ValueError("Memory graph supports non-speculative decode only")
        # Immutable query plans and epsilon buffers are cached per stream.
        # Warmup must use the SAME stream as capture, before recording starts.
        with torch.cuda.stream(runner.backend._capture_stream):
            self.allocate(runner.buffers.input_ids.numel(), runner.model_runner.device)
            self.slots.fill_(-1)
            self._last_slots = None
            self.capture_active = True
            try:
                yield
            finally:
                self.capture_active = False

    def read(self, layer, hidden, block):
        rows = len(hidden)
        if self.slots is None or rows > self.slots.numel():
            raise RuntimeError("Memory decode capture exceeds static capacity")
        runtime = self.hooks.model.memt_runtime
        key = (layer, rows)
        owners = block.query_capture_constants((1,) * rows)
        constant_key = (
            layer,
            rows,
            torch.cuda.current_stream(hidden.device),
            tuple(id(value) for value in owners),
        )
        # CUDA nodes contain raw pointers, not Python owners. Retain every
        # captured version, including a recapture after an LRU eviction.
        self.constants[constant_key] = owners
        if key not in self.residuals:
            self.residuals[key] = hidden.new_empty((rows, block.value_dim))
        slots = self.slots[:rows]
        value, final = block.packed_query_slots(
            hidden, runtime.pool.storage[layer][:, 0], slots, lengths=(1,) * rows
        )
        if final is not None:
            raise RuntimeError("Read-only capture produced a state update")
        # Explicitly zero padding, retired, memory-disabled and idle rows.
        self.residuals[key].copy_(torch.where(slots[:, None] >= 0, value, 0))
        self.prepared.add(key)
        self.capture_buckets.add(rows)
        return self.residuals[key]

    def upload(self, runtime):
        if self.slots is None:
            raise RuntimeError("Memory graph has not been captured")
        values = replay_slots(runtime, self.slots.numel())
        if values != self._last_slots:
            # Never rewrite pinned storage while a prior asynchronous upload
            # consumes it. Reuse completed buffers, allocate another otherwise.
            item = next((x for x in self._host_buffers if x[1].query()), None)
            if item is None:
                item = (
                    torch.empty(len(values), dtype=torch.int64, pin_memory=True),
                    torch.cuda.Event(),
                )
                self._host_buffers.append(item)
            host, done = item
            host.copy_(torch.tensor(values, dtype=torch.int64))
            self.slots.copy_(host, non_blocking=True)
            done.record()
            self._last_slots = values
            self.uploads += 1
        self.replays += 1


def replay_slots(runtime, capacity):
    """Resolve live rows on the host; no device reads or cached session IDs."""
    values = [-1] * capacity
    pool_capacity = next(iter(runtime.pool.storage.values())).shape[0]
    for sample in runtime.plan.samples:
        if (
            not sample.decode
            or sample.request.operation != "query"
            or sample.stop != sample.start + 1
        ):
            raise RuntimeError("Non-query entered memory decode replay")
        record = runtime.worker_sessions.requests[sample.request.rid]
        if record.complete:
            raise RuntimeError("Retired request entered memory replay")
        if sample.start < 0 or sample.start >= capacity:
            raise RuntimeError("Live rows exceed memory graph capacity")
        if runtime.worker_sessions.memory_read[sample.request.session]:
            slot = record.slot
            if not 0 <= slot < pool_capacity:
                raise RuntimeError("Invalid live memory slot")
            values[sample.start] = slot
    return tuple(values)


def install_memory_capture():
    """Narrow adapters around pinned SGLang methods; no site-package edits."""
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    if getattr(DecodeCudaGraphRunner.capture_one_shape, "_memt_memory_capture", False):
        return
    native_capture = DecodeCudaGraphRunner.capture_one_shape
    native_load = DecodeCudaGraphRunner.load_batch

    @wraps(native_capture)
    def capture(self, *args, **kwargs):
        hooks = getattr(self.model_runner.model, "_memory_hooks", None)
        manager = getattr(hooks, "memory_decode_graph", None)
        if manager is None:
            return native_capture(self, *args, **kwargs)
        native_model = getattr(self.model_runner.model, "model", None)
        pinned = getattr(native_model, "_pin_decode_graph_owners", False)
        owners = getattr(native_model, "_decode_graph_tensor_owners", ())
        owners_before = len(owners)
        with manager.capture(self):
            result = native_capture(self, *args, **kwargs)
        retained = len(owners) - owners_before
        if pinned and retained < len(native_model.layers):
            raise RuntimeError(
                "Decode capture did not retain every native layer's tensor owners"
            )
        if manager.prepared != manager.injected:
            raise RuntimeError("Memory capture left an unconsumed residual")
        counts = [len(g._segments) for g in self.backend._graphs.values()]
        logging.getLogger(__name__).info(
            "MemT memory decode captured: local buckets=%s, graph segments=%s, native tensor owners=%d",
            sorted(manager.capture_buckets),
            counts,
            retained,
        )
        return result

    @wraps(native_load)
    def load(self, batch, *args, **kwargs):
        result = native_load(self, batch, *args, **kwargs)
        hooks = getattr(self.model_runner.model, "_memory_hooks", None)
        manager = getattr(hooks, "memory_decode_graph", None)
        if manager is not None:
            runtime = self.model_runner.model.memt_runtime
            if runtime.plan is None:
                raise RuntimeError("Memory replay requires an active lifecycle")
            manager.upload(runtime)
        return result

    capture._memt_memory_capture = True
    DecodeCudaGraphRunner.capture_one_shape = capture
    DecodeCudaGraphRunner.load_batch = load
