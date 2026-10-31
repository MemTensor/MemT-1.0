"""Replay metadata must follow live publication, policy and request retirement."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from memt_sglang.protocol import MemoryRequest
from memt_sglang.runtime.executor import MemoryRuntime
from memt_sglang.runtime.features import sync_graph_config
from memt_sglang.runtime.memory_graph import replay_slots
from memt_sglang.runtime.plan import Sample, StepPlan
from memt_sglang.runtime.state import StatePool


def runtime():
    return MemoryRuntime(
        StatePool(
            {0: (1, 1, 2, 2)},
            max_sessions_per_rank=4,
            max_inflight_commits_per_rank=1,
            device="cpu",
        )
    )


def query(rt, session, row=0):
    req = MemoryRequest(session, "query", "query" + session)
    rt.worker_sessions.prepare(req.rid)
    return Sample(req, row, row + 1, 0, True)


def test_replay_tracks_published_slot_and_disable_policy():
    rt = runtime()
    ws = rt.worker_sessions
    ws.create("a")
    ws.create("b", memory_read=False)
    a = query(rt, "a")
    b = query(rt, "b", 1)
    rt.plan = StepPlan((a, b), 2)
    assert replay_slots(rt, 4) == (ws.requests[a.request.rid].slot, -1, -1, -1)
    ws.fence("a")
    ws.fence("b")
    ws.begin("a", "tx")
    tx = ws.transactions["a"]
    tx.has_memory = True
    new_slot = tx.slot
    ws.publish("a", "tx")
    a = query(rt, "a")
    rt.plan = StepPlan((a,), 1)
    assert replay_slots(rt, 4) == (new_slot, -1, -1, -1)


def test_idle_retired_and_slot_reuse():
    rt = runtime()
    ws = rt.worker_sessions
    ws.create("old")
    sample = query(rt, "old")
    old_slot = ws.requests[sample.request.rid].slot
    rt.plan = StepPlan((), 0)
    assert replay_slots(rt, 4) == (-1,) * 4
    ws.fence("old")
    ws.reset("old", close=True)
    ws.create("new")
    assert ws.pool.sessions["new"] == old_slot
    new = query(rt, "new")
    rt.plan = StepPlan((new,), 1)
    assert replay_slots(rt, 4) == (ws.requests[new.request.rid].slot, -1, -1, -1)
    # Reusing a physical slot is legal; retaining the old request identity is not.
    ws.requests[new.request.rid].complete = True
    with pytest.raises(RuntimeError, match="Retired"):
        replay_slots(rt, 4)


def test_invalid_operation_and_capacity_are_not_silent_fallbacks():
    rt = runtime()
    ws = rt.worker_sessions
    ws.create("a")
    a = query(rt, "a", 3)
    rt.plan = StepPlan((a,), 4)
    with pytest.raises(RuntimeError, match="capacity"):
        replay_slots(rt, 2)
    rt.plan = StepPlan((Sample(a.request, 0, 2, 0, False),), 2)
    with pytest.raises(RuntimeError, match="Non-query"):
        replay_slots(rt, 4)


@pytest.mark.parametrize(
    "backend,slot,overlap,pdmux,message",
    [
        ("disabled", "1", "0", False, "breakable"),
        ("breakable", "0", "0", False, "SLOT_READ"),
        ("breakable", "1", "1", False, "overlap"),
        ("breakable", "1", "0", True, "PDMux"),
    ],
)
def test_memory_graph_rejects_unsupported_execution(
    monkeypatch, backend, slot, overlap, pdmux, message
):
    @dataclass
    class Phase:
        backend: str

    @dataclass
    class Graph:
        decode: Phase
        prefill: Phase

    monkeypatch.setenv("MEMT_EXPERIMENT_MEMORY_DECODE_GRAPH", "1")
    monkeypatch.setenv("MEMT_EXPERIMENT_SLOT_READ", slot)
    monkeypatch.setenv("MEMT_EXPERIMENT_PREFILL_OVERLAP", overlap)
    monkeypatch.setenv("MEMT_EXPERIMENT_DECODE_OVERLAP", "0")
    args = SimpleNamespace(
        cuda_graph_config=Graph(Phase(backend), Phase("disabled")), enable_pdmux=pdmux
    )
    with pytest.raises(ValueError, match=message):
        sync_graph_config(args)
