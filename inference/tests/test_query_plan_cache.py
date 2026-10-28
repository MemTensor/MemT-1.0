"""Real CPU metadata tests; CUDA stream identity is simulated, never executed."""

import gc
import importlib
import weakref
from contextlib import contextmanager
from dataclasses import fields
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from memt_sglang.models.common.constants import WY_TILE_TOKENS
from memt_sglang.models.common.packed import boundaries
from memt_sglang.models.qwen.ops.layout import build_chunk_plan
from memt_sglang.protocol import SessionError
from memt_sglang.runtime import executor
from memt_sglang.runtime.executor import MemoryRuntime
from memt_sglang.runtime.state import StatePool

plans = importlib.import_module("memt_sglang.models.qwen.ops.query_plan")


def batch(rids, lengths):
    return NS(
        rids=rids,
        forward_mode=NS(
            is_idle=lambda: False, is_decode=lambda: False, is_extend=lambda: True
        ),
        extend_seq_lens_cpu=list(lengths),
        extend_prefix_lens_cpu=[0] * len(rids),
        num_token_non_padded_cpu=sum(lengths),
    )


def runtime(layers=3, *, metadata=True, slots=False):
    pool = StatePool(
        {i: (1, 2, 4, 4) for i in range(layers)},
        max_sessions_per_rank=4,
        max_inflight_commits_per_rank=2,
        device="cpu",
    )
    result = MemoryRuntime(pool)
    result.metadata_cache, result.slot_read = metadata, slots
    for name in ("a", "b", "c"):
        result.worker_sessions.create(name)
    return result


def query(rt, session="a", nonce="q"):
    rid = f"memt|{session}|query|{nonce}"
    rt.worker_sessions.prepare(rid)
    return rid


@pytest.fixture
def cuda_metadata(monkeypatch):
    """Use real CPU tensors to exercise the CUDA cache, not CUDA execution."""

    class Stream:
        def __init__(self, device):
            self.device = torch.device(device)

    first = Stream("cuda:0")
    active = [first]
    creations = []
    tensor = torch.tensor

    @contextmanager
    def stream_context(stream):
        previous = active[0]
        active[0] = stream
        try:
            yield
        finally:
            active[0] = previous

    def allocate(*args, **kwargs):
        requested = torch.device(kwargs.get("device", "cpu"))
        if requested.type == "cuda":
            assert requested == active[0].device
            creations.append(active[0])
            kwargs = {**kwargs, "device": "cpu"}
        return tensor(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: active[0])
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch, "tensor", allocate)
    plans._device_plan.cache_clear()
    yield NS(first=first, active=active, creations=creations, Stream=Stream)
    plans._device_plan.cache_clear()


class PlanReader:
    def __init__(self, device="cpu"):
        self.read_plan = Mock(
            side_effect=lambda lengths, _: plans.query_plan(lengths, device)
        )
        self.observed = []
        self.states = []

    def packed(self, hidden, states, *, lengths, commit, write_states):
        assert not commit
        plan = self.read_plan(lengths, hidden.device)
        self.observed.append(plan)
        self.states.append(states.clone())
        return hidden.clone(), None

    def packed_query_slots(self, hidden, pool, slots, *, lengths):
        return self.packed(
            hidden, pool[slots], lengths=lengths, commit=False, write_states=None
        )


@pytest.mark.parametrize(
    "lengths", [(1,), (1,) * 32, (63, 64, 65), (1, 3), (2, 2), (4097,)]
)
def test_cpu_plan_matches_legacy_reader_fields(lengths, monkeypatch):
    bounds = boundaries(lengths, "cpu")
    expected = build_chunk_plan(
        bounds,
        bounds,
        WY_TILE_TOKENS,
        event_slice=bounds,
        commit_mask=torch.zeros(len(lengths), dtype=torch.bool),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(side_effect=AssertionError))
    actual = plans.query_plan(lengths, "cpu")
    assert actual.num_chunk == expected.num_chunk
    assert actual.wy_tile_tokens == expected.wy_tile_tokens
    for field in ("chunk_tok0", "chunk_len", "chunk_block_ids"):
        torch.testing.assert_close(
            getattr(actual, field), getattr(expected, field), rtol=0, atol=0
        )
        assert getattr(actual, field).dtype == torch.int32


@pytest.mark.parametrize(
    "lengths", [(), (0,), (-1,), (True,), (1.5,), (2**31,), (2**31 - 1, 1)]
)
def test_invalid_lengths_fail_before_allocating(lengths, monkeypatch):
    monkeypatch.setattr(torch, "tensor", Mock(side_effect=AssertionError))
    with pytest.raises(ValueError, match="length|integer|int32"):
        plans.query_plan(lengths, "cpu")


@pytest.mark.parametrize("slots", [False, True])
@pytest.mark.parametrize("metadata", [False, True])
def test_reuse_across_layers_forwards_and_sessions(cuda_metadata, slots, metadata):
    rt, reader = runtime(metadata=metadata, slots=slots), PlanReader("cuda:0")
    for index, session in enumerate(("a", "b")):
        rid = query(rt, session)
        for storage in rt.pool.storage.values():
            storage[rt.pool.sessions[session]].fill_(index + 1)
        rt.begin(batch([rid], [1]))
        for layer in rt.pool.storage:
            rt.run_layer(layer, torch.ones(1, 8), torch.empty(1, 8), reader)
        rt.finish(True)
        assert rt._prepared_groups == {}
    assert all(plan is reader.observed[0] for plan in reader.observed)
    assert len(cuda_metadata.creations) == 3
    assert plans._device_plan.cache_info().hits == 5
    assert reader.states[0].mean() == 1 and reader.states[3].mean() == 2
    assert {field.name for field in fields(reader.observed[0])} == {
        "wy_tile_tokens",
        "num_chunk",
        "chunk_tok0",
        "chunk_len",
        "chunk_block_ids",
    }


def test_stream_partitions_include_slots_and_indices(monkeypatch):
    rt, reader = runtime(slots=True), PlanReader()
    rids = [query(rt, "a"), "ordinary", query(rt, "b")]
    stream_a, stream_b = object(), object()
    active = [stream_a]
    monkeypatch.setattr(executor, "_metadata_stream", lambda device: active[0])
    rt.begin(batch(rids, [1, 2, 3]))
    observed = []
    for layer, stream in enumerate((stream_a, stream_b, stream_a)):
        active[0] = stream
        rt.run_layer(layer, torch.ones(6, 8), torch.empty(6, 8), reader)
        observed.append(rt._prepared_groups[(False, torch.device("cpu"), stream)])
    assert observed[0] is observed[2]
    assert observed[0].slots is not observed[1].slots
    assert observed[0].indices is not observed[1].indices
    assert observed[0].lengths == (1, 3)
    rt.finish(True)


def test_metadata_stream_uses_explicit_device_without_gpu(monkeypatch):
    stream = object()
    current = Mock(return_value=stream)
    monkeypatch.setattr(torch.cuda, "current_stream", current)
    assert executor._metadata_stream(torch.device("cpu")) is None
    assert executor._metadata_stream(torch.device("cuda:1")) is stream
    current.assert_called_once_with(torch.device("cuda:1"))


def test_cache_separates_lengths_streams_and_devices(cuda_metadata):
    env = cuda_metadata
    first = plans.query_plan([1, 3], "cuda:0")
    assert first is plans.query_plan((1, 3), torch.device("cuda:0"))
    assert first is not plans.query_plan((2, 2), "cuda:0")
    for device in ("cuda:0", "cuda:1"):
        env.active[0] = env.Stream(device)
        assert first is not plans.query_plan((1, 3), device)
    env.active[0] = env.first
    assert first is plans.query_plan((1, 3), "cuda:0")
    assert len(env.creations) == 12


def test_lru_is_bounded_and_keeps_live_consumers_on_eviction(cuda_metadata):
    first = plans.query_plan((1,), "cuda:0")
    expected = first.chunk_tok0.clone()
    for length in range(2, 65):
        plans.query_plan((length,), "cuda:0")
    assert plans._device_plan.cache_info().currsize == 64
    assert first is plans.query_plan((1,), "cuda:0")  # Touch to protect MRU.
    evicted = plans.query_plan((2,), "cuda:0")
    # Make 2 least-recently-used again, then force one eviction.
    for length in range(3, 65):
        plans.query_plan((length,), "cuda:0")
    plans.query_plan((1,), "cuda:0")
    plans.query_plan((65,), "cuda:0")
    assert plans._device_plan.cache_info().currsize == 64
    assert evicted is not plans.query_plan((2,), "cuda:0")
    torch.testing.assert_close(first.chunk_tok0, expected, rtol=0, atol=0)
    assert first is plans.query_plan((1,), "cuda:0")


def test_cache_holds_stream_and_buffers_until_evicted(cuda_metadata):
    stream = cuda_metadata.Stream("cuda:0")
    cuda_metadata.active[0] = stream
    plan = plans.query_plan((1,), "cuda:0")
    stream_ref, tensor_ref = weakref.ref(stream), weakref.ref(plan.chunk_tok0)
    cuda_metadata.active[0] = cuda_metadata.first
    cuda_metadata.creations.clear()
    del stream, plan
    gc.collect()
    assert stream_ref() is not None and tensor_ref() is not None
    for length in range(2, 66):
        plans.query_plan((length,), "cuda:0")
    gc.collect()
    assert stream_ref() is None and tensor_ref() is None


@pytest.mark.parametrize("outcome", ["success", "failure", "missing_layer"])
def test_finish_releases_group_buffers_but_keeps_geometry(cuda_metadata, outcome):
    rt = runtime(layers=1 if outcome == "success" else 2)
    rid = query(rt)
    rt.begin(batch([rid], [1]))
    reader = PlanReader("cuda:0")
    rt.slot_read = True
    rt.run_layer(0, torch.ones(1, 8), torch.empty(1, 8), reader)
    geometry = reader.observed[0]
    group = rt._prepared_groups[(False, torch.device("cpu"), None)]
    reference = weakref.ref(group.slots)
    del group
    reader.observed.clear()
    if outcome == "missing_layer":
        with pytest.raises(SessionError, match="Not all"):
            rt.finish(True)
    else:
        rt.finish(outcome == "success")
    gc.collect()
    assert reference() is None
    assert rt.plan is None and not rt._prepared_groups
    rt.begin(batch([], []))
    rt.finish(True)
    assert geometry is plans.query_plan((1,), "cuda:0")
