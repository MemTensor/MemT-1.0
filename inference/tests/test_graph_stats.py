"""Graph admission counters: bucket keys, reasons, and side-effect freedom."""

import json
from types import SimpleNamespace as NS

import pytest

from memt_sglang.runtime import decode_graph, graph_stats, prefill_graph


@pytest.fixture(autouse=True)
def clean():
    graph_stats._phases.clear()
    graph_stats._phases.update({name: {} for name in ("prefill", "decode")})
    graph_stats._overflow.update({"prefill": 0, "decode": 0})
    graph_stats._enabled = False
    yield
    graph_stats._enabled = False


def buckets(phase="prefill"):
    return graph_stats.snapshot()["phases"][phase]


def test_record_admitted_keeps_key_and_gear_separate():
    graph_stats.record(phase="prefill", key=511, gear=512, admitted=True)
    assert buckets()["511"] == {
        "gear": 512, "admitted": 1, "fallback": 0, "reasons": {}
    }


def test_record_fallback_counts_each_reason():
    graph_stats.record(phase="prefill", key=511, gear=None, admitted=False,
                       reasons=("over_max_gear", "native_veto_other"))
    graph_stats.record(phase="prefill", key=511, gear=None, admitted=False,
                       reasons=("over_max_gear",))
    assert buckets()["511"]["fallback"] == 2
    assert buckets()["511"]["reasons"] == {"over_max_gear": 2, "native_veto_other": 1}
    assert buckets()["511"]["gear"] is None


def test_fallback_without_reasons_uses_the_catch_all_label():
    graph_stats.record(phase="decode", key=3, gear=None, admitted=False)
    assert buckets("decode")["3"]["reasons"] == {"native_veto_other": 1}


def test_phases_are_counted_separately():
    graph_stats.record(phase="prefill", key=8, gear=8, admitted=True)
    graph_stats.record(phase="decode", key=8, gear=None, admitted=False,
                       reasons=("over_max_gear",))
    assert buckets("prefill")["8"]["admitted"] == 1
    assert buckets("decode")["8"]["fallback"] == 1


def test_bucket_overflow_is_counted_not_recorded():
    for key in range(graph_stats.MAX_BUCKETS + 3):
        graph_stats.record(phase="prefill", key=key + 1, gear=None, admitted=False,
                           reasons=("over_max_gear",))
    result = graph_stats.snapshot()
    assert len(result["phases"]["prefill"]) == graph_stats.MAX_BUCKETS
    assert result["overflow_buckets"]["prefill"] == 3


@pytest.mark.parametrize("key", [-1, "512", None, True])
def test_invalid_keys_rejected(key):
    with pytest.raises(ValueError):
        graph_stats.record(phase="prefill", key=key, gear=None, admitted=False)


def test_unknown_phase_rejected():
    with pytest.raises(ValueError):
        graph_stats.record(phase="both", key=1, gear=None, admitted=False)


def test_summarize_skips_empty_phases_and_emits_json_lines():
    graph_stats.record(phase="prefill", key=512, gear=512, admitted=True)
    events = []
    graph_stats.summarize(model="memt-flash", rank=0, emit=events.append)
    assert [event["phase"] for event in events] == ["prefill"]
    assert json.loads(graph_stats.dumps(events[0]))["buckets"]["512"]["admitted"] == 1


def test_snapshot_is_detached_from_live_counters():
    graph_stats.record(phase="prefill", key=512, gear=512, admitted=True)
    copy = graph_stats.snapshot()
    copy["phases"]["prefill"]["512"]["admitted"] = 99
    assert buckets()["512"]["admitted"] == 1


def test_exit_summary_registers_once_and_never_raises():
    import atexit

    calls = []
    original = atexit.register
    atexit.register = calls.append
    try:
        graph_stats.install_exit_summary(model="memt-nano", rank=0, emit=lambda e: None)
        graph_stats.install_exit_summary(model="memt-nano", rank=0, emit=lambda e: None)
    finally:
        atexit.register = original
        graph_stats._enabled = False
    assert len(calls) == 1


# --- decode bucket keys -----------------------------------------------------

def decode_batch(counts=None, bs=3):
    batch = NS(batch_size=bs)
    if counts is not None:
        batch.original_global_num_tokens_cpu = counts
    return batch


def test_decode_key_uses_cross_rank_max_under_mlp_tp_gather():
    """Flash keys its decode graph on the DP maximum, not the local batch."""
    runner = NS(require_mlp_tp_gather=True)
    assert decode_graph._graph_key_value(runner, decode_batch([2, 7, 4], bs=2)) == 7


def test_decode_key_uses_local_batch_without_mlp_tp_gather():
    runner = NS(require_mlp_tp_gather=False)
    assert decode_graph._graph_key_value(runner, decode_batch([2, 7, 4], bs=2)) == 2


def test_decode_key_falls_back_to_batch_size_when_counts_missing():
    runner = NS(require_mlp_tp_gather=True)
    assert decode_graph._graph_key_value(runner, decode_batch(None, bs=5)) == 5


def test_decode_reason_marks_over_max_bs():
    runner = NS(require_mlp_tp_gather=False, disable_padding=False, max_bs=8)
    assert decode_graph._decode_reasons(runner, decode_batch(None, bs=9)) == ("over_max_gear",)
    assert decode_graph._decode_reasons(runner, decode_batch(None, bs=8)) == (
        "native_veto_other",)


# --- Nano prefill reason classification ------------------------------------

def prefill_runner(gears=(512, 1024)):
    return NS(capture_num_tokens=list(gears))


def batch(n):
    return NS(input_ids=list(range(n)), rids=["r"] * max(n, 1), forward_mode=NS())


def test_prefill_reason_below_padding_factor(monkeypatch):
    monkeypatch.setattr(prefill_graph, "query_graph_allowed", lambda *a: True)
    monkeypatch.setattr(prefill_graph, "live_prefill_tokens", lambda b: len(b.input_ids))
    # 255 -> bucket 512, and 512 > 255 * 2, so the native 2x rule rejects it.
    reasons = prefill_graph._admission_reasons(prefill_runner(), object(), batch(255))
    assert reasons == ("below_padding_factor",)


def test_prefill_reason_over_max_gear(monkeypatch):
    monkeypatch.setattr(prefill_graph, "query_graph_allowed", lambda *a: True)
    monkeypatch.setattr(prefill_graph, "live_prefill_tokens", lambda b: len(b.input_ids))
    reasons = prefill_graph._admission_reasons(prefill_runner(), object(), batch(2000))
    assert reasons == ("over_max_gear",)


def test_prefill_reason_query_veto_wins_first(monkeypatch):
    monkeypatch.setattr(prefill_graph, "query_graph_allowed", lambda *a: False)
    reasons = prefill_graph._admission_reasons(prefill_runner(), object(), batch(600))
    assert reasons == ("memt_query_veto",)


def test_prefill_reason_unknown_metadata_is_catch_all(monkeypatch):
    monkeypatch.setattr(prefill_graph, "query_graph_allowed", lambda *a: True)

    def invalid(_batch):
        raise ValueError("bad")

    monkeypatch.setattr(prefill_graph, "live_prefill_tokens", invalid)
    reasons = prefill_graph._admission_reasons(prefill_runner(), object(), batch(600))
    assert reasons == ("native_veto_other",)


@pytest.mark.parametrize("bs,gear", [(1, 1), (3, 4), (4, 4), (5, 8), (8, 8)])
def test_decode_records_the_replayed_gear_not_the_raw_batch(bs, gear):
    runner = NS(require_mlp_tp_gather=False, capture_bs=[1, 2, 4, 8])
    decode_graph._record(runner, decode_batch(None, bs=bs), True, ())
    assert buckets("decode")[str(bs)]["gear"] == gear
