"""Scheduler protocol tests: no model execution or numerical tensors."""

from types import SimpleNamespace

import pytest

from memt_sglang.runtime import decode_overlap as overlap
from memt_sglang.runtime.features import (
    execution_identity,
    overlap_phases,
    validate_overlap,
)


def batch(phase, *, global_extend=None):
    return SimpleNamespace(
        reqs=[],
        forward_mode=SimpleNamespace(is_extend=lambda: phase == "prefill"),
        is_extend_in_batch=global_extend,
    )


def scheduler(prefill, decode, *, pending=True, native=False):
    class Scheduler:
        def is_disable_overlap_for_batch(self, batch, last_batch):
            return native

        def run_batch(self, batch):
            self.events.append("forward")
            return "result"

    overlap.install_phase_overlap(Scheduler, prefill=prefill, decode=decode)
    installed = Scheduler.is_disable_overlap_for_batch
    overlap.install_phase_overlap(Scheduler, prefill=prefill, decode=decode)
    assert Scheduler.is_disable_overlap_for_batch is installed
    obj = Scheduler()
    obj.result_queue = [object()] if pending else []
    obj.events = []
    obj.forward_stream = SimpleNamespace(synchronize=lambda: obj.events.append("fence"))
    return obj


@pytest.mark.parametrize(
    "flags", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize(
    "previous,current",
    [
        ("prefill", "prefill"),
        ("prefill", "decode"),
        ("decode", "prefill"),
        ("decode", "decode"),
    ],
)
def test_phase_boundaries(flags, previous, current):
    s = scheduler(*flags)
    allowed = dict(zip(("prefill", "decode"), flags))
    assert s.is_disable_overlap_for_batch(batch(current), batch(previous)) == (
        not (allowed[previous] and allowed[current])
    )


def test_empty_queue_native_sync_and_last_result():
    assert not scheduler(True, True, pending=False).is_disable_overlap_for_batch(
        batch("prefill"), None
    )
    assert scheduler(True, True, native=True).is_disable_overlap_for_batch(
        batch("decode"), batch("decode")
    )
    assert scheduler(True, True).is_disable_overlap_for_batch(None, batch("decode"))


@pytest.mark.parametrize("global_extend", [False, True])
def test_idle_dp_rank_uses_native_global_phase(global_extend):
    s = SimpleNamespace(require_mlp_sync=True)
    active, idle = (
        batch("prefill", global_extend=global_extend),
        batch("idle", global_extend=global_extend),
    )
    assert overlap.batch_phase(s, active) == overlap.batch_phase(s, idle)
    assert overlap.batch_phase(s, idle) == ("prefill" if global_extend else "decode")


def test_commit_drains_and_fences_before_and_after_forward(monkeypatch):
    monkeypatch.setattr(overlap, "has_atomic_write", lambda *args: True)
    s = scheduler(True, True)
    commit = batch("prefill")
    assert s.is_disable_overlap_for_batch(commit, batch("decode"))
    assert s.run_batch(commit) == "result"
    assert s.events == ["fence", "forward", "fence"]
    monkeypatch.setattr(overlap, "has_atomic_write", lambda *args: False)
    assert s.is_disable_overlap_for_batch(batch("decode"), commit)


def test_query_forward_keeps_pipeline(monkeypatch):
    s = scheduler(True, True)
    q = batch("prefill")
    assert not s.is_disable_overlap_for_batch(q, batch("decode"))
    s.run_batch(q)
    assert s.events == ["forward"]


@pytest.mark.parametrize(
    "prefill,decode", [(False, False), (True, False), (False, True), (True, True)]
)
def test_resolved_phase_flags(monkeypatch, prefill, decode):
    for name in ("OVERLAP", "DECODE_ONLY_OVERLAP"):
        monkeypatch.delenv("MEMT_EXPERIMENT_" + name, raising=False)
    monkeypatch.setenv("MEMT_EXPERIMENT_PREFILL_OVERLAP", str(int(prefill)))
    monkeypatch.setenv("MEMT_EXPERIMENT_DECODE_OVERLAP", str(int(decode)))
    assert overlap_phases() == (prefill, decode)
    validate_overlap(SimpleNamespace(disable_overlap_schedule=not (prefill or decode)))
    with pytest.raises(ValueError):
        validate_overlap(SimpleNamespace(disable_overlap_schedule=prefill or decode))


def test_legacy_conflict_is_rejected(monkeypatch):
    monkeypatch.setenv("MEMT_EXPERIMENT_OVERLAP", "1")
    monkeypatch.setenv("MEMT_EXPERIMENT_PREFILL_OVERLAP", "0")
    monkeypatch.setenv("MEMT_EXPERIMENT_DECODE_OVERLAP", "1")
    monkeypatch.delenv("MEMT_EXPERIMENT_DECODE_ONLY_OVERLAP", raising=False)
    with pytest.raises(ValueError, match="Conflicting"):
        overlap_phases()


def test_identity_reports_resolved_dp_chunk_and_phase_settings(monkeypatch):
    for name in ("OVERLAP", "DECODE_ONLY_OVERLAP"):
        monkeypatch.delenv("MEMT_EXPERIMENT_" + name, raising=False)
    monkeypatch.setenv("MEMT_EXPERIMENT_PREFILL_OVERLAP", "0")
    monkeypatch.setenv("MEMT_EXPERIMENT_DECODE_OVERLAP", "1")
    monkeypatch.setenv("MEMT_EXPERIMENT_SLOT_READ", "1")
    args = SimpleNamespace(
        cuda_graph_config={
            "prefill": {"backend": "breakable"},
            "decode": {"backend": "breakable"},
        },
        disable_overlap_schedule=False,
        chunked_prefill_size=8192,
    )
    result = execution_identity(args)
    assert result["query_chunk_per_rank"] == 8192
    assert result["prefill_overlap"] is False and result["decode_overlap"] is True
    assert (
        result["slot_read"] is True
        and result["graph_backends"]["decode"] == "breakable"
    )
