"""Production replay instrumentation must not synchronize CUDA or log each step."""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from memt_sglang.runtime import flash_prefill as fp


def setup_runner(monkeypatch, *, observer=None):
    calls, emitted = [], []
    def synchronize():
        calls.append("sync")
    class Key:
        size = 512
    class Backend:
        def replay(self, shape_key, static_forward_batch, **kwargs):
            calls.append("backend")
            if getattr(static_forward_batch, "backend_error", False):
                raise ValueError("backend failure")
            return "hidden"
    class Attention:
        def prepare_forward_metadata_for_breakable_cuda_graph_replay(self, *a, **kw):
            pass
    class Runner:
        def capture(self):
            self.backend._graphs = {Key(): NS(_segments=[object()])}
            self.attn_metadata_buffers = {512: object()}
        def can_run_graph(self, batch):
            return True
        def execute(self, batch, **kwargs):
            calls.append("execute")
            if not getattr(batch, "skip_backend", False):
                shape = NS(size=getattr(batch, "replay_size", len(batch.input_ids)))
                self.backend.replay(shape, batch, **kwargs)
            if getattr(batch, "tail_error", False):
                raise ValueError("tail failure")
            return "logits"
    class ModelRunner:
        def _prepare_eager_forward_batch(self, batch):
            pass
    monkeypatch.setattr(fp, "validate_capture_request", lambda *a: {})
    monkeypatch.setattr(fp, "_worker_preflight", lambda *a: {})
    fp.install_worker_hooks(fp.FlashPrefillIntent("breakable", (512,), 4),
                            Runner, Attention, ModelRunner,
                            gather=lambda v: [v] * 4, rank=lambda: 0,
                            emit=emitted.append, synchronize=synchronize,
                            memory=lambda: {}, replay_observer=observer)
    runner = Runner()
    runner.backend = Backend()
    runner.model_runner = NS(server_args=NS(), attn_backend=Attention(),
                            model_config=NS(hf_config=NS(model_type="memt_deepseek_v4")))
    runner.capture()
    assert calls == ["sync"]  # Capture qualification still synchronizes once.
    calls.clear()
    emitted.clear()
    batch = NS(input_ids=[0] * 512, _memt_flash_prefill_allowed=True)
    return runner, batch, calls, emitted


def test_production_replay_never_syncs_logs_or_replaces_backend_method(monkeypatch):
    runner, batch, calls, emitted = setup_runner(monkeypatch)
    delegate = runner.backend.replay
    for _ in range(3):
        assert runner.execute(batch) == "logits"
        assert runner.backend.replay is delegate
    assert calls == ["execute", "backend"] * 3
    assert emitted == []
    counters = fp.replay_counters(runner)
    assert counters["backend_submissions"] == counters["execute_returns"] == 3
    assert counters["execution_errors"] == 0
    assert counters["cuda_completion_verified"] is False
    assert counters["numeric_qualified"] is False
    counters["backend_submissions"] = 999
    assert fp.replay_counters(runner)["backend_submissions"] == 3


def test_external_observer_can_sync_only_after_native_return(monkeypatch):
    events, ordering = [], []
    def diagnostic_sync():
        assert calls == ["execute", "backend"]
        ordering.append("external_sync")
    def diagnostic(event):
        if event["event"] == "replay_return":
            diagnostic_sync()
            event = {**event, "cuda_synchronized": True,
                     "actual": {"backend": "breakable", "status": "replayed_synchronized"}}
        events.append(event)
    runner, batch, calls, _ = setup_runner(monkeypatch, observer=diagnostic)
    assert runner.execute(batch) == "logits"
    assert ordering == ["external_sync"]
    assert [e["event"] for e in events] == ["replay_enter", "replay_return"]
    assert events[-1]["backend_replay_count"] == 1
    assert events[-1]["cuda_synchronized"] is True
    assert events[-1]["numeric_qualified"] is False
    # An external diagnostic cannot upgrade the production counter's meaning.
    assert fp.replay_counters(runner)["cuda_completion_verified"] is False


def test_observer_without_sync_reports_submission_only(monkeypatch):
    events = []
    runner, batch, calls, _ = setup_runner(monkeypatch, observer=events.append)
    runner.execute(batch)
    assert calls == ["execute", "backend"]
    assert events[-1]["actual"]["status"] == "replay_submitted"
    assert events[-1]["cuda_synchronized"] is False


@pytest.mark.parametrize("failure,submitted", [("backend_error", 0), ("tail_error", 1), ("skip_backend", 0), ("wrong_gear", 0)])
def test_failures_are_not_successful_execute_returns(monkeypatch, failure, submitted):
    runner, batch, calls, emitted = setup_runner(monkeypatch)
    delegate = runner.backend.replay
    if failure == "wrong_gear":
        batch.replay_size = 1024
    else:
        setattr(batch, failure, True)
    with pytest.raises((ValueError, RuntimeError)):
        runner.execute(batch)
    counters = fp.replay_counters(runner)
    assert counters["backend_submissions"] == submitted
    assert counters["execute_returns"] == 0
    assert counters["execution_errors"] == 1
    assert runner.backend.replay is delegate
    assert "sync" not in calls
    assert emitted[-1]["event"] == "replay_error"
    assert runner._memt_flash_expected_tokens is None


def test_observer_sync_failure_propagates_without_reclassifying_submission(monkeypatch):
    def diagnostic(event):
        if event["event"] == "replay_return":
            raise RuntimeError("async CUDA error observed externally")
    runner, batch, _, emitted = setup_runner(monkeypatch, observer=diagnostic)
    with pytest.raises(RuntimeError, match="externally"):
        runner.execute(batch)
    assert fp.replay_counters(runner)["backend_submissions"] == 1
    assert emitted[-1]["event"] == "replay_error"
    assert runner._memt_flash_expected_tokens is None


def test_no_observer_keeps_all_rank_admission_and_fallback_silent():
    from test_flash_prefill import runner_and_batch
    class Runner:
        def can_run_graph(self, batch):
            return True
    class ModelRunner:
        def _prepare_eager_forward_batch(self, batch):
            pass
    votes = []
    def gather(vote):
        votes.append(vote)
        return [replace(vote, rank=i) for i in range(4)]
    fp._install_replay_consensus(fp.FlashPrefillIntent("breakable", (512,), 4),
                                 Runner, ModelRunner, gather, lambda: 0,
                                 lambda e: pytest.fail("per-batch production log"))
    runner, batch = runner_and_batch(counts=(512,) * 4)
    runner._memt_flash_capture_complete = True
    mr = ModelRunner()
    mr.model_config = NS(hf_config=NS(model_type="memt_deepseek_v4"))
    mr.prefill_cuda_graph_runner = runner
    mr._prepare_eager_forward_batch(batch)
    assert batch._memt_flash_prefill_allowed is True
    batch.original_global_num_tokens_cpu = [512, 0, 512, 512]
    mr._prepare_eager_forward_batch(batch)
    assert batch._memt_flash_prefill_allowed is False
    assert len(votes) == 2
    assert fp.replay_counters(runner)["admissions"] == 1
    assert fp.replay_counters(runner)["fallbacks"] == 1


def test_large_gear_scratch_estimate_is_not_a_capture_pool_bound():
    scratch = fp.scratch_observation(4096, context_capacity=32768, free_bytes=2**30)
    assert scratch["single_c4_logits_bytes"] == 128 * 1024**2
    assert scratch["total_capture_pool_bytes"] is None
    with pytest.raises(ValueError, match="scratch"):
        fp.scratch_observation(4096, context_capacity=32768,
                               free_bytes=scratch["single_c4_logits_bytes"])


def test_observer_type_is_validated_before_any_hooks_are_installed(monkeypatch):
    def untouched(*a, **kw):
        pytest.fail("invalid observer must fail before modifying classes")
    monkeypatch.setattr(fp, "_install_metadata_hook", untouched)
    with pytest.raises(TypeError, match="observer"):
        fp.install_worker_hooks(fp.FlashPrefillIntent("breakable", (512,), 4),
                                type("Runner", (), {}), object, object,
                                gather=untouched, rank=untouched, emit=untouched,
                                synchronize=untouched, memory=untouched, replay_observer=False)
    with pytest.raises(TypeError, match="observer"):
        fp.install_native_worker(fp.FlashPrefillIntent("breakable", (512,), 4),
                                 emit=untouched, replay_observer=False)


def test_duplicate_counter_and_out_of_scope_backend_replay_are_rejected(monkeypatch):
    runner, batch, calls, _ = setup_runner(monkeypatch)
    with pytest.raises(RuntimeError, match="already"):
        fp._attach_replay_counter(runner)
    with pytest.raises(RuntimeError, match="admitted"):
        runner.backend.replay(NS(size=512), batch)
    assert calls == []
    assert fp.replay_counters(runner)["backend_submissions"] == 0


def test_observer_cannot_reenter_execute_or_damage_next_replay(monkeypatch):
    invoked = False
    def diagnostic(event):
        nonlocal invoked
        if event["event"] == "replay_enter" and not invoked:
            invoked = True
            runner.execute(batch)
    runner, batch, calls, _ = setup_runner(monkeypatch, observer=diagnostic)
    with pytest.raises(RuntimeError, match="reentered"):
        runner.execute(batch)
    assert runner._memt_flash_expected_tokens is None
    assert calls == []
    assert runner.execute(batch) == "logits"
    assert fp.replay_counters(runner)["backend_submissions"] == 1
