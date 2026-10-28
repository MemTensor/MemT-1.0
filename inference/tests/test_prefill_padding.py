"""Live token metadata must survive native capture-gear padding."""

import sys
from types import SimpleNamespace

import pytest
import torch

from memt_sglang.runtime import prefill_graph


def batch(lengths, prefixes=None):
    prefixes = prefixes or [0] * len(lengths)
    n = sum(lengths)
    return SimpleNamespace(
        batch_size=len(lengths),
        rids=[f"request-{i}" for i in range(len(lengths))],
        input_ids=torch.arange(n),
        positions=torch.cat([torch.arange(p, p + n) for p, n in zip(prefixes, lengths)]),
        out_cache_loc=torch.arange(1, n + 1),
        extend_seq_lens_cpu=list(lengths),
        extend_prefix_lens_cpu=list(prefixes),
        extend_num_tokens=n,
        num_token_non_padded_cpu=n,
    )


@pytest.fixture
def runner(monkeypatch):
    class Runner:
        capture_num_tokens = [256, 512, 1024, 2048]

        def __init__(self):
            self.model_runner = SimpleNamespace(model=SimpleNamespace(memt_runtime=object()))
            self.allowed = True
            self.corrupt = False

        def can_run_graph(self, live):
            return self.allowed

        def load_batch(self, live):
            n = len(live.input_ids)
            gear = next(g for g in self.capture_num_tokens if g >= n)
            result = SimpleNamespace(**vars(live))
            result.input_ids = torch.nn.functional.pad(live.input_ids, (0, gear - n))
            result.positions = torch.nn.functional.pad(live.positions, (0, gear - n))
            result.out_cache_loc = torch.nn.functional.pad(live.out_cache_loc, (0, gear - n))
            result.rids = None
            if self.corrupt:
                result.extend_seq_lens_cpu = [gear]
            return result

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner",
        SimpleNamespace(PrefillCudaGraphRunner=Runner),
    )
    monkeypatch.setattr(prefill_graph, "query_graph_allowed", lambda *args: True)
    prefill_graph.install_prefill_query_guard()
    return Runner()


@pytest.mark.parametrize("lengths", [[511], [512], [513], [511, 1], [512, 1], [511, 512], [511, 513], [511, 512, 513]])
def test_gear_padding_preserves_live_rows_and_positions(runner, lengths):
    live = batch(lengths)
    assert runner.can_run_graph(live)
    static = runner.load_batch(live)
    n = sum(lengths)
    assert static.rids == live.rids and static.rids is not live.rids
    assert static.num_token_non_padded_cpu == n
    assert static.extend_seq_lens_cpu == lengths
    assert static._memt_prefill_valid_tokens == n
    assert static._memt_prefill_gear_tokens == len(static.input_ids)
    assert torch.equal(static.positions[:n], live.positions)
    assert torch.equal(static.out_cache_loc[:n], live.out_cache_loc)
    assert torch.count_nonzero(static.out_cache_loc[n:]) == 0
    assert len(live.input_ids) == n


def test_prefix_positions_and_replay_metadata_are_live(runner):
    a, b = batch([511, 1], [17, 40]), batch([1, 511], [8, 9])
    first, second = runner.load_batch(a), runner.load_batch(b)
    assert first.extend_prefix_lens_cpu == [17, 40]
    assert second.extend_prefix_lens_cpu == [8, 9]
    assert torch.equal(second.positions[:512], b.positions)
    b.rids[0] = "later-request"
    assert second.rids[0] != b.rids[0]


@pytest.mark.parametrize("field,value", [
    ("extend_seq_lens_cpu", [512]),
    ("extend_prefix_lens_cpu", [-1]),
    ("num_token_non_padded_cpu", 510),
    ("extend_num_tokens", 512),
    ("batch_size", 2),
])
def test_invalid_live_metadata_rejected_before_native_load(runner, field, value):
    live = batch([511])
    setattr(live, field, value)
    assert not runner.can_run_graph(live)
    with pytest.raises(ValueError):
        runner.load_batch(live)


def test_native_must_not_relabel_padding_as_request_tokens(runner):
    runner.corrupt = True
    with pytest.raises(ValueError, match="metadata"):
        runner.load_batch(batch([511]))


def test_native_veto_and_query_guard_remain_effective(runner, monkeypatch):
    runner.allowed = False
    assert not runner.can_run_graph(batch([512]))
    runner.allowed = True
    monkeypatch.setattr(prefill_graph, "query_graph_allowed", lambda *args: False)
    assert not runner.can_run_graph(batch([512]))


def test_missing_valid_count_is_derived_from_live_lengths(runner):
    live = batch([511])
    live.num_token_non_padded_cpu = None
    assert runner.load_batch(live).num_token_non_padded_cpu == 511
