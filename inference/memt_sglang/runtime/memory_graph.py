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
        self.captured_width = 1
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
        # The runner's capture ladder is fixed for its lifetime: width 1 for a
        # plain decode, draft_token_num for a speculative target-verify. Both are
        # replayable, because a verify row is still one token reading one session
        # slot -- the only difference is that a request owns `width` consecutive
        # rows instead of one. Recording the width here is what replay_slots uses
        # to expand a single session slot across its rows.
        if runner.captured_req_width < 1:
            raise ValueError("Memory graph requires a positive captured row width")
        self.captured_width = runner.captured_req_width
        # Immutable query plans and epsilon buffers are cached per stream.
        # Warmup must use the SAME stream as capture, before recording starts.
        with torch.cuda.stream(runner.backend._capture_stream):
            # The runner resolves its complete capture ladder before this hook.
            # Allocate for its maximum, independent of bucket visitation order.
            self.allocate(runner.max_num_token, runner.model_runner.device)
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
        # Every row is its own one-token sequence. A query read is a stateless
        # per-token lookup into the published snapshot, so the `width` rows a
        # speculative verify gives one request each read the same memory state
        # independently, exactly as `width` consecutive plain decode steps would.
        lengths = (1,) * rows
        value, final = block.packed_query_slots(
            hidden, runtime.pool.storage[layer][:, 0], slots, lengths=lengths
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
        values = replay_slots(runtime, self.slots.numel(), self.captured_width)
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


def replay_slots(runtime, capacity, width=1):
    """Resolve live rows on the host; no device reads or cached session IDs.

    A request occupies `width` consecutive rows: 1 for a plain decode, and
    draft_token_num under a speculative verify. Every one of those rows reads the
    same session slot, because each row is a separate candidate token evaluated
    against the same memory state. The width comes from the runner that captured
    the graph, and each sample's own span must agree with it or the slot buffer
    would be filled against a different geometry than the one captured.
    """
    values = [-1] * capacity
    pool_capacity = next(iter(runtime.pool.storage.values())).shape[0]
    for sample in runtime.plan.samples:
        rows = sample.stop - sample.start
        # `operation` is the only real gate. The row count must equal the width
        # the graph was captured at, otherwise this buffer describes a different
        # geometry than the captured one; a mismatch is a bug, not a fallback.
        if sample.request.operation != "query" or rows != width:
            raise RuntimeError(
                f"Non-query or width mismatch entered memory decode replay "
                f"(rows={rows}, captured_width={width})"
            )
        record = runtime.worker_sessions.requests[sample.request.rid]
        if record.complete:
            raise RuntimeError("Retired request entered memory replay")
        if sample.start < 0 or sample.stop > capacity:
            raise RuntimeError("Live rows exceed memory graph capacity")
        if runtime.worker_sessions.memory_read[sample.request.session]:
            slot = record.slot
            if not 0 <= slot < pool_capacity:
                raise RuntimeError("Invalid live memory slot")
            for row in range(sample.start, sample.stop):
                values[row] = slot
    return tuple(values)


def _graph_segments(runner):
    """Per-bucket segment counts for the capture log, or None if unavailable.

    These are private backend attributes read only to report what was captured.
    A dependency that renames them must not abort a worker over a log line, so
    unlike the ownership checks above this one degrades instead of raising.
    """
    try:
        return [len(graph._segments) for graph in runner.backend._graphs.values()]
    except AttributeError:
        return None


def install_memory_capture():
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
        # DSV4 exposes ownership through its model instance flag. Qwen does not
        # use this API and retains its graph inputs through native ownership.
        if hasattr(native_model, "_pin_decode_graph_owners") and not pinned:
            raise RuntimeError("Native decode tensor ownership is disabled")
        if pinned and retained < len(native_model.layers):
            raise RuntimeError(
                "Decode capture did not retain every native layer's tensor owners"
            )
        if manager.prepared != manager.injected:
            raise RuntimeError("Memory capture left an unconsumed residual")
        logging.getLogger(__name__).info(
            "MemT memory decode captured: local buckets=%s, graph segments=%s, native tensor owners=%d",
            sorted(manager.capture_buckets),
            _graph_segments(self),
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
            # The speculative path reaches load_batch BEFORE execute:
            # eagle_prepare_for_verify() calls can_run_graph() then load_batch()
            # and only later does model_runner.forward() call execute(), which is
            # where MemT normally opens its step. upload() needs the plan, so open
            # the step here when nobody has yet, and let execute() reuse it rather
            # than opening a second one.
            # The open step is bound to the batch that opened it: replaying
            # against a plan built for a different batch would fill the slot
            # buffer from the wrong requests.
            opened = runtime.plan is None
            if opened:
                runtime.begin(batch, preopened=True)
            elif not runtime.owns(batch):
                raise RuntimeError("Memory replay plan belongs to a different batch")
            try:
                manager.upload(runtime)
            except BaseException:
                if opened:
                    runtime.finish(success=False)
                raise
        return result

    capture._memt_memory_capture = True
    DecodeCudaGraphRunner.capture_one_shape = capture
    DecodeCudaGraphRunner.load_batch = load
