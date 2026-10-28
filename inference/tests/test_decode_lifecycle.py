"""Graph lifecycle checks without constructing a numerical model."""

import sys
from types import SimpleNamespace

import pytest

from memt_sglang.runtime import decode_graph


@pytest.mark.parametrize("outcome", ["consumed", "unconsumed", "failed"])
def test_decode_residual_must_be_consumed_and_step_always_finishes(
    monkeypatch, outcome
):
    events = []
    pending = {3: object()}

    class Runner:
        def can_run_graph(self, batch):
            return True

        def execute(self, batch):
            if outcome == "failed":
                raise ValueError("native failure")
            if outcome == "consumed":
                pending.clear()
            return "output"

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner",
        SimpleNamespace(DecodeCudaGraphRunner=Runner),
    )
    monkeypatch.setattr(decode_graph, "install_logits_output_bridge", lambda: None)
    decode_graph.install_decode_query_guard()
    runtime = SimpleNamespace(
        begin=lambda batch: events.append("begin"),
        finish=lambda *, success: events.append(success),
    )
    runner = Runner()
    runner.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            memt_runtime=runtime, _memory_hooks=SimpleNamespace(pending=pending)
        )
    )
    if outcome == "consumed":
        assert runner.execute(object()) == "output"
    else:
        error = RuntimeError if outcome == "unconsumed" else ValueError
        message = (
            "unconsumed memory residual"
            if outcome == "unconsumed"
            else "native failure"
        )
        with pytest.raises(error, match=message):
            runner.execute(object())
    assert pending == {}
    assert events == ["begin", outcome == "consumed"]
