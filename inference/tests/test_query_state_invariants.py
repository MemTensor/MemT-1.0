"""Real CPU runtime/pool integration across query, commit and failure paths."""

from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from test_query_plan_cache import PlanReader, batch, query, runtime
from test_query_readers import blocks as blocks
from test_query_readers import cpu_modules as cpu_modules

from memt_sglang.models.common.hooks import MemoryHooks
from memt_sglang.protocol import SessionError


def snapshot(rt):
    return (
        {layer: state.clone() for layer, state in rt.pool.storage.items()},
        deepcopy(rt.pool.sessions),
        deepcopy(rt.pool._free),
        deepcopy(vars(rt.worker_sessions) | {"pool": None}),
    )


def assert_unchanged(rt, before):
    for layer, state in before[0].items():
        torch.testing.assert_close(rt.pool.storage[layer], state, rtol=0, atol=0)
    assert rt.pool.sessions == before[1] and rt.pool._free == before[2]
    assert vars(rt.worker_sessions) | {"pool": None} == before[3]


@pytest.mark.parametrize("model", ["nano", "flash"])
@pytest.mark.parametrize("slots", [False, True])
@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("populated", [False, True])
def test_queries_preserve_entire_pool_and_session_state(
    blocks, cpu_modules, model, slots, metadata, populated
):
    rt = runtime(layers=2, metadata=metadata, slots=slots)
    rt.worker_sessions.set_memory_read("b", False)
    rt.worker_sessions.begin("c", "untouched")
    if populated:
        for storage in rt.pool.storage.values():
            storage.normal_()
    rids = [query(rt, "a"), query(rt, "b"), "ordinary", query(rt, "c")]
    before = snapshot(rt)
    rt.begin(batch(rids, [1, 2, 1, 3]))
    hidden = torch.randn(9, 24, dtype=torch.bfloat16)[:, ::2]
    for layer in rt.pool.storage:
        output = rt.run_layer(
            layer,
            hidden,
            torch.empty(9, 8, dtype=torch.bfloat16),
            getattr(blocks, model),
        )
        assert output[1:4].count_nonzero() == 0
        assert output[7:].count_nonzero() == 0
    rt.finish(True)
    assert_unchanged(rt, before)
    assert not rt._prepared_groups
    cpu_modules.writer.assert_not_called()


@pytest.mark.parametrize("slots", [False, True])
@pytest.mark.parametrize("metadata", [False, True])
def test_one_sample_alias_and_slot_fallback_do_not_write(
    blocks, cpu_modules, monkeypatch, slots, metadata
):
    rt = runtime(layers=1, metadata=metadata, slots=slots)
    rid = query(rt)
    rt.pool.storage[0].normal_()
    before = snapshot(rt)
    monkeypatch.setattr(cpu_modules.nano, "slot_query_supported", lambda *a: False)
    rt.begin(batch([rid], [1]))
    rt.run_layer(
        0,
        torch.ones(1, 12, dtype=torch.bfloat16),
        torch.empty(1, 8, dtype=torch.bfloat16),
        blocks.nano,
    )
    rt.finish(True)
    assert_unchanged(rt, before)


@pytest.mark.parametrize("model", ["nano", "flash"])
@pytest.mark.parametrize("slots", [False, True])
def test_mixed_commit_only_updates_candidate(blocks, cpu_modules, model, slots):
    rt = runtime(layers=2, slots=slots)
    rt.worker_sessions.begin("b", "tx")
    commit = "memt|b|commit:2|write"
    rt.worker_sessions.prepare(commit, "tx")
    rid = query(rt, "a")
    before = {layer: state.clone() for layer, state in rt.pool.storage.items()}
    published = dict(rt.pool.sessions)
    candidate = rt.worker_sessions.transactions["b"].slot
    cpu_modules.writer.side_effect = lambda *a, **kw: (a[0].clone(), a[6] + 7)
    rt.begin(batch([rid, commit], [1, 2]))
    for layer in rt.pool.storage:
        rt.run_layer(
            layer,
            torch.ones(3, 12, dtype=torch.bfloat16),
            torch.empty(3, 8, dtype=torch.bfloat16),
            getattr(blocks, model),
        )
        expected = before[layer].clone()
        expected[candidate] += 7
        torch.testing.assert_close(rt.pool.storage[layer], expected, rtol=0, atol=0)
    rt.finish(True)
    assert rt.pool.sessions == published
    assert rt.worker_sessions.requests[commit].complete
    assert not rt.worker_sessions.requests[rid].complete
    assert cpu_modules.writer.call_count == 2


@pytest.mark.parametrize("failure", ["reader", "native", "pending", "missing"])
def test_hook_failure_cleans_forward_and_next_forward_works(failure):
    rt, reader = runtime(layers=2), PlanReader()
    rid = query(rt)
    before = snapshot(rt)
    hooks = MemoryHooks()
    hooks.model, hooks.pending = NS(memt_runtime=rt), {}

    def forward(*args):
        rt.run_layer(0, torch.ones(1, 8), torch.empty(1, 8), reader)
        if failure == "reader":
            broken = NS(packed=Mock(side_effect=ValueError("reader failure")))
            rt.run_layer(1, torch.ones(1, 8), torch.empty(1, 8), broken)
        if failure == "native":
            raise ValueError("native failure")
        if failure == "pending":
            hooks.pending[0] = object()

    with pytest.raises((ValueError, RuntimeError, SessionError)):
        hooks.forward(forward, None, None, batch([rid], [1]))
    assert rt.plan is None and not rt._prepared_groups
    assert not hooks.pending
    assert_unchanged(rt, before)
    rt.begin(batch([rid], [1]))
    for layer in rt.pool.storage:
        rt.run_layer(layer, torch.ones(1, 8), torch.empty(1, 8), reader)
    rt.finish(True)
    assert_unchanged(rt, before)


def test_runtime_contract_and_legacy_compute_compatibility():
    rt = runtime(layers=1, slots=True)
    with pytest.raises(SessionError):
        rt.finish(True)
    rid = query(rt)
    rt.begin(batch([rid], [1]))
    with pytest.raises(SessionError, match="Previous"):
        rt.begin(batch([rid], [1]))
    plain = Mock(return_value=(torch.ones(1, 8), None))
    rt.run_layer(0, torch.ones(1, 8), torch.empty(1, 8), NS(packed=plain))
    assert "plan_provider" not in plain.call_args.kwargs
    with pytest.raises(SessionError):
        rt.run_layer(0, torch.ones(1, 8), torch.empty(1, 8), NS(packed=plain))
    rt.finish(True)


def test_empty_and_disabled_groups_never_build_plan():
    rt, reader = runtime(layers=1), PlanReader()
    rt.worker_sessions.set_memory_read("a", False)
    rid = query(rt)
    for rids, lengths in (([], []), ([rid], [1])):
        rt.begin(batch(rids, lengths))
        output = rt.run_layer(0, torch.ones(1, 8), torch.empty(1, 8), reader)
        assert output.count_nonzero() == 0
        rt.finish(True)
    reader.read_plan.assert_not_called()
