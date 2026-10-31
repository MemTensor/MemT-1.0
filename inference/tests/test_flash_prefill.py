"""CPU admission tests; these do not qualify CUDA capture or scratch memory."""

import ast
import importlib.util
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from memt_sglang.runtime import flash_prefill as fp
from memt_sglang.runtime.flash_prefill import (
    FlashPrefillIntent,
    agree_replay,
    prefill_identity,
    replay_vote,
    restrict_batch,
)


def mode(name="extend"):
    return NS(
        is_extend=lambda: name == "extend",
        is_idle=lambda: name == "idle",
        is_target_verify=lambda: name == "verify",
    )


def runner_and_batch(rank=0, counts=(512, 512)):
    n = max(counts)
    record = NS(complete=False, request=NS(operation="query"))
    backend = NS(
        use_captured_forward_metadata_for_breakable_cuda_graph=True,
        init_forward_metadata_for_breakable_cuda_graph_capture=lambda b: None,
        prepare_forward_metadata_for_breakable_cuda_graph_replay=lambda *a: None,
    )
    runner = NS(
        dp_size=len(counts),
        capture_num_tokens=[512, 1024],
        max_num_tokens=1024,
        prefill_backend_name="breakable",
        use_captured_attn_metadata=True,
        model_runner=NS(
            attn_backend=backend,
            model=NS(memt_runtime=NS(worker_sessions=NS(requests={"q": record}))),
        ),
        _is_full_backend=False,
        _capture_req_slots=1,
        capture_hidden_mode="null",
        _has_unsupported_mha_prefix=lambda b: False,
        _has_inactive_dp_rank=lambda b: any(n == 0 for n in b.global_num_tokens_cpu),
        _uses_eager_prefill_tail=lambda: True,
        _pad_to_bucket=lambda n, sizes: next(s for s in sizes if s >= n),
    )
    batch = NS(
        input_ids=[7] * n,
        original_global_num_tokens_cpu=list(counts),
        global_num_tokens_cpu=[n] * len(counts),
        global_dp_buffer_len=n * len(counts),
        dp_padding_mode=NS(is_max_len=lambda: True),
        forward_mode=mode(),
        _original_forward_mode=None,
        rids=["q"],
        batch_size=1,
        extend_seq_lens_cpu=[counts[rank]],
        extend_prefix_lens_cpu=[0],
        num_token_non_padded_cpu=counts[rank],
        can_run_dp_breakable_cuda_graph=True,
        is_extend_in_batch=True,
        input_embeds=None,
        replace_embeds=None,
        capture_hidden_mode="null",
        return_logprob=False,
    )
    return runner, batch


def vote(rank=0, counts=(512, 512), native=lambda r, b: True):
    runner, batch = runner_and_batch(rank, counts)
    return replay_vote(runner, batch, rank=rank, native_can_run=native)


def native_source_path(relative):
    """Locate real SGLang source without importing CUDA-dependent modules.

    MEMT_TEST_SGLANG_SOURCE optionally names a package source root containing
    srt/ (including a CPU-only source archive). Otherwise use installed SGLang.
    Missing optional SGLang skips native integration tests; an explicitly
    configured or installed but incomplete source tree is a test failure.
    Native source hashes are still checked by the unmodified production gate.
    """
    configured = os.environ.get("MEMT_TEST_SGLANG_SOURCE")
    if configured is not None:
        if not configured.strip():
            pytest.fail("MEMT_TEST_SGLANG_SOURCE must name a source root containing srt/")
        roots = [Path(configured).expanduser()]
    else:
        spec = importlib.util.find_spec("sglang")
        if spec is None:
            pytest.skip("Native integration requires SGLang or MEMT_TEST_SGLANG_SOURCE")
        roots = [Path(root) for root in (spec.submodule_search_locations or ())]
    for root in roots:
        path = root / relative
        if path.is_file():
            return path.resolve()
    pytest.fail(f"SGLang source is missing {relative}; searched roots: {roots}")


def native_eligibility():
    # Execute the actual native method, rather than a hand-copied veto list.
    source = native_source_path("srt/model_executor/runner/prefill_cuda_graph_runner.py")
    tree = ast.parse(source.read_text())
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "can_run_graph"
    )
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    namespace = {"_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR": 2}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["can_run_graph"]


def test_intent_survives_native_downgrade_without_modifying_arguments():
    intent = FlashPrefillIntent("breakable", (512, 1024), 2)
    config = NS(backend="disabled", bs=[512, 1024])
    result = prefill_identity(intent, config)
    assert result["requested_backend"] == "breakable"
    assert result["resolved_backend"] == "disabled"
    assert result["status"] == "disabled_by_native"
    assert result["gpu_qualified"] is False
    assert config.backend == "disabled"
    with pytest.raises(FrozenInstanceError):
        intent.dp_size = 4
    config.backend = "breakable"
    assert prefill_identity(intent, config)["status"] == "resolved_unqualified"
    assert "capture_pool_bound" in prefill_identity(intent, config)["pending"]


def test_disabled_intent_and_resolved_shape_identity_are_distinct():
    intent = FlashPrefillIntent("disabled", (), 4)
    assert prefill_identity(intent, NS(backend="disabled", bs=None))["status"] == "not_requested"
    intent = FlashPrefillIntent("breakable", (512,), 4)
    result = prefill_identity(intent, NS(backend="breakable", bs=[1024]))
    assert result["requested_capture_tokens"] == [512]
    assert result["resolved_capture_tokens"] == [1024]
    assert "resolved_shapes_changed" in result["pending"]


@pytest.mark.parametrize("backend,buckets,dp", [
    ("full", (512,), 2), ("breakable", (), 2),
    ("breakable", (0,), 2), ("breakable", (1024, 512), 2),
    ("breakable", (512, 512), 2), ("breakable", (True,), 2),
    ("breakable", (512,), True), ("breakable", (512,), 0),
])
def test_bad_intent_rejected(backend, buckets, dp):
    with pytest.raises(ValueError):
        FlashPrefillIntent(backend, buckets, dp)


@pytest.mark.parametrize("counts", [(0, 512), (511, 512), (512, 513), (512, 0, 512, 512)])
def test_original_idle_or_uneven_counts_cannot_be_hidden_by_max_len(counts):
    votes = [vote(rank, counts) for rank in range(len(counts))]
    assert all(not item.eligible for item in votes)
    assert not agree_replay(votes, dp_size=len(counts)).allowed


def test_native_inactive_guard_alone_misses_max_len_converted_idle():
    runner, batch = runner_and_batch(0, (0, 512))
    batch._original_forward_mode = mode("idle")
    native = native_eligibility()
    assert native(runner, batch) is True
    assert not replay_vote(runner, batch, rank=0, native_can_run=native).eligible


@pytest.mark.parametrize("size", [511, 513])
def test_dp_bucket_expansion_needs_a_separate_collective_shape_adapter(size):
    runner, batch = runner_and_batch(counts=(size, size))
    result = replay_vote(runner, batch, rank=0, native_can_run=native_eligibility())
    assert "dp_bucket_padding_unverified" in result.reasons
    assert result.bucket == (512 if size == 511 else 1024)


@pytest.mark.parametrize("field,value", [
    ("original_global_num_tokens_cpu", None),
    ("original_global_num_tokens_cpu", [512]),
    ("original_global_num_tokens_cpu", [True, 512]),
    ("global_num_tokens_cpu", [512, 1024]),
    ("global_dp_buffer_len", 512),
    ("can_run_dp_breakable_cuda_graph", False),
    ("extend_seq_lens_cpu", [511]),
    ("extend_prefix_lens_cpu", [-1]),
    ("num_token_non_padded_cpu", 511),
    ("forward_mode", mode("idle")),
    ("_original_forward_mode", mode("idle")),
    ("dp_padding_mode", NS(is_max_len=lambda: False)),
])
def test_missing_or_inconsistent_metadata_fails_closed(field, value):
    runner, batch = runner_and_batch()
    setattr(batch, field, value)
    assert not replay_vote(runner, batch, rank=0, native_can_run=lambda *a: True).eligible


@pytest.mark.parametrize("failure", ["runtime", "query", "metadata", "capture", "backend", "shape"])
def test_required_live_protocols(failure):
    runner, batch = runner_and_batch()
    if failure == "runtime":
        runner.model_runner.model.memt_runtime = None
    elif failure == "query":
        runner.model_runner.model.memt_runtime.worker_sessions.requests["q"].complete = True
    elif failure == "metadata":
        runner.model_runner.attn_backend.prepare_forward_metadata_for_breakable_cuda_graph_replay = None
    elif failure == "capture":
        runner.use_captured_attn_metadata = False
    elif failure == "backend":
        runner.prefill_backend_name = "full"
    else:
        runner.capture_num_tokens = [1024, 512]
    assert not replay_vote(runner, batch, rank=0, native_can_run=lambda *a: True).eligible


@pytest.mark.parametrize("veto", ["embeds", "replace", "prefix", "verify", "hidden", "slots", "dp", "logprob", "max", "factor"])
def test_all_archived_native_vetoes_are_retained(veto):
    runner, batch = runner_and_batch()
    if veto == "embeds":
        batch.input_embeds = object()
    elif veto == "replace":
        batch.replace_embeds = object()
    elif veto == "prefix":
        runner._has_unsupported_mha_prefix = lambda b: True
    elif veto == "verify":
        batch.forward_mode.is_target_verify = lambda: True
    elif veto == "hidden":
        batch.capture_hidden_mode = "full"
    elif veto == "slots":
        runner._is_full_backend = True
        batch.batch_size = 2
    elif veto == "dp":
        batch.can_run_dp_breakable_cuda_graph = False
    elif veto == "logprob":
        batch.return_logprob = True
        runner._uses_eager_prefill_tail = lambda: False
    elif veto == "max":
        runner.max_num_tokens = 511
    elif veto == "factor":
        runner._pad_to_bucket = lambda n, sizes: 2048
    native = native_eligibility()
    assert not native(runner, batch)
    assert not replay_vote(runner, batch, rank=0, native_can_run=native).eligible


def test_native_delegated_once_and_exceptions_are_not_swallowed():
    runner, batch = runner_and_batch()
    calls = []
    def native(r, b):
        calls.append((r, b))
        return False
    result = replay_vote(runner, batch, rank=0, native_can_run=native)
    assert result.reasons == ("native_veto",)
    assert calls == [(runner, batch)]
    def broken(r, b):
        raise RuntimeError("native fault")
    with pytest.raises(RuntimeError, match="native fault"):
        replay_vote(runner, batch, rank=0, native_can_run=broken)


def test_all_rank_agreement_is_required_even_for_locally_valid_shapes():
    first, second = vote(0), vote(1)
    assert first.eligible and second.eligible
    agreed = agree_replay([first, second], dp_size=2)
    assert agreed.allowed and agreed.bucket == 512
    for votes in ([], [first], [first, first], [first, replace(second, bucket=1024)],
                  [first, replace(second, reasons=("native_veto",))],
                  [first, replace(second, original_counts=(511, 511))],
                  [first, replace(second, capture_tokens=(512, 2048))]):
        assert not agree_replay(votes, dp_size=2).allowed


def test_restriction_is_a_copy_and_never_overrides_an_existing_veto():
    _, batch = runner_and_batch()
    allowed = agree_replay([vote(0), vote(1)], dp_size=2)
    denied = agree_replay([], dp_size=2)
    assert restrict_batch(batch, allowed)._memt_graph_allowed is True
    assert not hasattr(batch, "_memt_graph_allowed")
    assert restrict_batch(batch, denied)._memt_graph_allowed is False
    batch._memt_graph_allowed = False
    assert restrict_batch(batch, allowed)._memt_graph_allowed is False


def test_balanced_multi_request_prefill_keeps_lengths_and_prefixes():
    runner, batch = runner_and_batch()
    runner.model_runner.model.memt_runtime.worker_sessions.requests["q2"] = NS(
        complete=False, request=NS(operation="query")
    )
    batch.rids = ["q", "q2"]
    batch.batch_size = 2
    batch.extend_seq_lens_cpu = [255, 257]
    batch.extend_prefix_lens_cpu = [10, 20]
    assert replay_vote(runner, batch, rank=0, native_can_run=native_eligibility()).eligible
    assert batch.extend_seq_lens_cpu == [255, 257]
    assert batch.extend_prefix_lens_cpu == [10, 20]


@pytest.mark.parametrize("rank,dp_size", [(True, 2), (-1, 2), (2, 2), (0, False), (0, 0)])
def test_invalid_rank_topology_raises_before_native(rank, dp_size):
    runner, batch = runner_and_batch()
    runner.dp_size = dp_size
    with pytest.raises(ValueError, match="Invalid DP"):
        replay_vote(runner, batch, rank=rank, native_can_run=lambda *a: True)


def test_consensus_rejects_invalid_topology_and_missing_shape():
    with pytest.raises(ValueError, match="Invalid DP"):
        agree_replay([], dp_size=0)
    no_shape = [replace(vote(rank), bucket=None) for rank in range(2)]
    assert not agree_replay(no_shape, dp_size=2).allowed
    assert not agree_replay([object(), object()], dp_size=2).allowed


def test_metadata_probe_never_copies_device_counts_to_host():
    class DeviceCounts:
        def cpu(self):
            raise AssertionError("device read")

        def __iter__(self):
            raise AssertionError("device read")

    runner, batch = runner_and_batch()
    batch.original_global_num_tokens_cpu = DeviceCounts()
    result = replay_vote(runner, batch, rank=0, native_can_run=lambda *a: True)
    assert result.reasons == ("original_dp_counts_missing",)


def test_explicit_config_uses_native_lock_and_locks_max_bs_too():
    intent = FlashPrefillIntent("breakable", (512, 1024), 4)
    assert fp.explicit_capture_config(intent) == {
        "backend": "breakable", "bs": [512, 1024], "max_bs": 1024
    }
    for invalid in (
        FlashPrefillIntent("disabled", (), 4),
        FlashPrefillIntent("breakable", (256, 512), 4),
        FlashPrefillIntent("breakable", (512,), 2),
    ):
        with pytest.raises(ValueError):
            fp.explicit_capture_config(invalid)


def native_server_method(name):
    source = native_source_path("srt/server_args.py")
    tree = ast.parse(source.read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def native_rule_source():
    source = native_source_path("srt/server_args.py")
    node = native_server_method("_disable_breakable_cudagraph_if_incompatible")
    return "\n".join(source.read_text().splitlines()[node.lineno - 1:node.end_lineno]) + "\n"


def test_native_explicit_backend_skips_blanket_checks():
    node = native_server_method("_apply_cuda_graph_compatibility")
    namespace = {"Phase": NS(PREFILL="prefill")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "native", "exec"), namespace)
    args = NS(_cuda_graph_config_locked={("prefill", "backend")})
    # A blanket-veto function is not needed: native returns before calling it.
    namespace[node.name](args)


def config_args():
    values = dict(
        enable_dp_attention=True, attn_cp_size=1, moe_a2a_backend="none",
        quantization=None,
    )
    args = NS(
        cuda_graph_config=NS(prefill=NS(backend="breakable", bs=[512, 1024], max_bs=1024)),
        _cuda_graph_config_locked={("prefill", k) for k in ("backend", "bs", "max_bs")},
        _resolved=lambda: NS(**values),
        get_model_config=lambda: NS(hf_config=NS(model_type="memt_deepseek_v4"), is_multimodal=False),
        use_mla_backend=lambda: False,
        device="cuda", tp_size=4, dp_size=4, ep_size=4, pp_size=1, dcp_size=1,
        lora_paths=None, enable_lora=False, load_format="safetensors",
        disable_overlap_schedule=True, enable_two_batch_overlap=False,
        enable_memory_saver=False, speculative_algorithm=None,
        disaggregation_mode="null", enable_torch_compile=False,
        enable_hierarchical_cache=False, cpu_offload_gb=0, enable_symm_mem=False,
        enable_dsa_prefill_context_parallel=False, enable_prefill_cp=False,
        enable_deepseek_v4_fp4_indexer=False, enable_eplb=False,
        expert_distribution_recorder_mode=None, debug_cuda_graph=False,
    )
    return args, values


def test_config_gate_rechecks_native_vetoes_despite_explicit_lock(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "sglang.srt.arg_groups.overrides", NS(resolved_view=lambda a: a._resolved()))
    monkeypatch.setitem(sys.modules, "sglang.srt.configs.model_config", NS(is_deepseek_v4=lambda c: True))
    source = native_rule_source()
    validator = fp.audited_native_validator(source, {
        "Backend": NS(DISABLED="disabled"),
        "logger": NS(warning=lambda *a: None),
    })
    intent = FlashPrefillIntent("breakable", (512, 1024), 4)
    args, values = config_args()
    receipt = fp.validate_capture_request(args, intent, native_validator=validator)
    assert receipt["status"] == "capture_attempt_permitted"
    assert receipt["gpu_qualified"] is False
    for name, value in (("attn_cp_size", 2), ("moe_a2a_backend", "deepep")):
        values[name] = value
        with pytest.raises(ValueError, match="native|scope"):
            fp.validate_capture_request(args, intent, native_validator=validator)
        values[name] = 1 if name == "attn_cp_size" else "none"
    args.enable_lora = True
    with pytest.raises(ValueError, match="native"):
        fp.validate_capture_request(args, intent, native_validator=validator)
    assert args.cuda_graph_config.prefill.backend == "breakable"
    with pytest.raises(ValueError, match="source"):
        fp.audited_native_validator(source.replace("LoRA", "changed"), {})


@pytest.mark.parametrize("field,value", [
    ("device", "cpu"), ("tp_size", 2), ("dp_size", 8), ("pp_size", 2),
    ("disable_overlap_schedule", False), ("enable_two_batch_overlap", True),
    ("enable_memory_saver", True), ("speculative_algorithm", "EAGLE"),
    ("disaggregation_mode", "prefill"), ("enable_torch_compile", True),
])
def test_capture_scope_rejects_unqualified_paths(field, value):
    args, _ = config_args()
    setattr(args, field, value)
    with pytest.raises(ValueError):
        fp.validate_capture_request(args, FlashPrefillIntent("breakable", (512, 1024), 4), native_validator=lambda a: None)


def metadata():
    import torch
    page = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    core = NS(page_table=page)
    for name in fp.CORE_STABLE_FIELDS:
        setattr(core, name, torch.zeros(2, dtype=torch.int32))
    indexer = NS(page_table=page, c4_seq_lens=core.c4_topk_lengths_raw,
                 deep_gemm_metadata=torch.zeros(2), topk_metadata=torch.empty(0))
    compress = NS(write_loc=torch.zeros(2), extra_data=None,
                  plan=NS(indices=torch.arange(2)))
    return NS(core_attn_metadata=core, indexer_metadata=indexer,
              c4_compress_metadata=compress, sparse_prefill_cache=None)


def test_refresh_restores_page_table_alias_and_preserves_all_stable_addresses():
    old = metadata()
    original = old.indexer_metadata.page_table
    addresses = fp.metadata_addresses(old)
    def native_refresh():
        fresh = metadata()
        fresh.core_attn_metadata.page_table.fill_(9)
        old.core_attn_metadata.page_table = fresh.core_attn_metadata.page_table
        old.indexer_metadata.page_table.copy_(fresh.indexer_metadata.page_table)
        assert old.core_attn_metadata.page_table is not old.indexer_metadata.page_table
    fp.refresh_metadata(old, native_refresh)
    assert old.core_attn_metadata.page_table is original
    assert old.indexer_metadata.page_table is original
    assert original.tolist() == [[9] * 4] * 2
    assert fp.metadata_addresses(old) == addresses


def test_refresh_rejects_reallocated_captured_tensor():
    old = metadata()
    def replace_storage():
        old.indexer_metadata.page_table = old.indexer_metadata.page_table.clone()
    with pytest.raises(RuntimeError, match="address"):
        fp.refresh_metadata(old, replace_storage)


def test_capture_measurement_is_not_a_total_pool_estimate():
    result = fp.scratch_observation(512, context_capacity=32768, free_bytes=2**30)
    assert result["single_c4_logits_bytes"] == 512 * 8192 * 4
    assert result["total_capture_pool_bytes"] is None
    with pytest.raises(ValueError, match="scratch"):
        fp.scratch_observation(1024, context_capacity=32768, free_bytes=1)


def test_capture_event_retains_failure_and_requires_real_graphs():
    intent = FlashPrefillIntent("breakable", (512,), 4)
    runner = NS(backend=NS(_graphs={}), attn_metadata_buffers={512: object()})
    events = []
    calls = []
    with pytest.raises(RuntimeError, match="capture registry"):
        fp.capture_attempt(runner, intent, lambda: calls.append("native"), emit=events.append,
                           synchronize=lambda: calls.append("sync"), memory=lambda: {"free_bytes": 100})
    assert calls == ["native", "sync"]
    assert [e["event"] for e in events] == ["capture_enter", "capture_error"]
    assert events[-1]["error_type"] == "RuntimeError"


def test_capture_receipt_reports_segments_and_propagates_oom():
    class Key:
        size = 512
    runner = NS(backend=NS(_graphs={Key(): NS(_segments=[object(), object()])}),
                attn_metadata_buffers={512: object()})
    intent = FlashPrefillIntent("breakable", (512,), 4)
    events = []
    assert fp.capture_attempt(runner, intent, lambda: 42, emit=events.append,
                              synchronize=lambda: None, memory=lambda: {}) == 42
    assert events[-1]["event"] == "capture_return"
    assert events[-1]["segments"] == {512: 2}
    assert events[-1]["gpu_qualified"] is False
    assert events[-1]["actual"]["status"] == "captured"
    def oom():
        raise MemoryError("capture allocation")
    with pytest.raises(MemoryError):
        fp.capture_attempt(runner, intent, oom, emit=events.append,
                           synchronize=lambda: None, memory=lambda: {})
    assert events[-1]["event"] == "capture_error"


def test_installed_worker_gate_reaches_capture_and_refresh(monkeypatch):
    intent = FlashPrefillIntent("breakable", (512,), 4)
    events = []
    class Attention:
        def prepare_forward_metadata_for_breakable_cuda_graph_replay(self, meta, live, **kwargs):
            fresh = metadata()
            fresh.core_attn_metadata.page_table.fill_(17)
            meta.core_attn_metadata.page_table = fresh.core_attn_metadata.page_table
            meta.indexer_metadata.page_table.copy_(fresh.indexer_metadata.page_table)
            self.forward_metadata = meta
    class Runner:
        def capture(self):
            self.backend._graphs = {NSKey(512): NS(_segments=[object()])}
            self.attn_metadata_buffers = {512: metadata()}
            return "captured"
        def can_run_graph(self, batch):
            return getattr(batch, "_memt_graph_allowed", True)
        def execute(self, batch):
            if getattr(batch, "fail", False):
                raise RuntimeError("native replay failure")
            self.backend.replay(NSKey(len(batch.input_ids)), batch)
            return "replayed"
    class ModelRunner:
        def _prepare_eager_forward_batch(self, batch):
            batch.normalized = True
    class NSKey:
        def __init__(self, size): self.size = size
    def gather(value):
        if isinstance(value, fp.ReplayVote):
            return [replace(value, rank=i) for i in range(4)]
        return [value] * 4
    monkeypatch.setattr(fp, "validate_capture_request", lambda *a: {"status": "capture_attempt_permitted"})
    monkeypatch.setattr(fp, "_worker_preflight", lambda *a: {"single_c4_logits_bytes": 1})
    fp.install_worker_hooks(intent, Runner, Attention, ModelRunner,
                            gather=gather, rank=lambda: 0, emit=events.append,
                            synchronize=lambda: None, memory=lambda: {},
                            replay_observer=events.append)
    runner = Runner()
    runner.backend = NS(_graphs={}, replay=lambda *a: "backend")
    runner.model_runner = NS(server_args=NS(), attn_backend=Attention(),
                            model_config=NS(hf_config=NS(model_type="memt_deepseek_v4")))
    runner.capture_num_tokens = [512]
    assert runner.capture() == "captured"
    meta = runner.attn_metadata_buffers[512]
    runner.model_runner.attn_backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(meta, object())
    assert meta.core_attn_metadata.page_table is meta.indexer_metadata.page_table
    assert meta.indexer_metadata.page_table[0, 0].item() == 17
    assert events[-1]["event"] == "capture_return"
    batch = NS(input_ids=[1] * 512, _memt_flash_prefill_allowed=True)
    assert runner.execute(batch) == "replayed"
    assert events[-1]["event"] == "replay_return"
    assert events[-1]["backend_replay_count"] == 1
    assert events[-1]["actual"]["status"] == "replay_submitted"
    assert events[-1]["cuda_synchronized"] is False
    batch.fail = True
    with pytest.raises(RuntimeError, match="native replay failure"):
        runner.execute(batch)
    assert events[-1]["event"] == "replay_error"
    batch._memt_flash_prefill_allowed = False
    assert not runner.can_run_graph(batch)
    with pytest.raises(RuntimeError, match="evidence"):
        runner.execute(batch)
    with pytest.raises(RuntimeError, match="already"):
        fp.install_worker_hooks(intent, Runner, Attention, ModelRunner,
                                gather=gather, rank=lambda: 0, emit=events.append,
                                synchronize=lambda: None, memory=lambda: {})


def test_worker_consensus_restricts_real_normalized_batch(monkeypatch):
    intent = FlashPrefillIntent("breakable", (512,), 4)
    class Runner:
        def capture(self): pass
        def can_run_graph(self, batch): return True
        def execute(self, batch): return "native"
    class Attention:
        def prepare_forward_metadata_for_breakable_cuda_graph_replay(self, *a, **k): pass
    class ModelRunner:
        def _prepare_eager_forward_batch(self, batch):
            batch.global_num_tokens_cpu = [512] * 4
            batch.global_dp_buffer_len = 2048
    def gather(value):
        return [replace(value, rank=i) for i in range(4)]
    fp.install_worker_hooks(intent, Runner, Attention, ModelRunner,
                            gather=gather, rank=lambda: 0, emit=lambda e: None,
                            synchronize=lambda: None, memory=lambda: {})
    mr = ModelRunner()
    runner, batch = runner_and_batch(counts=(512, 512, 512, 512))
    runner.capture_num_tokens = [512]
    runner._memt_flash_capture_complete = True
    mr.model_config = NS(hf_config=NS(model_type="memt_deepseek_v4"))
    mr.prefill_cuda_graph_runner = runner
    mr._prepare_eager_forward_batch(batch)
    assert batch._memt_graph_allowed is True
    batch.original_global_num_tokens_cpu = [512, 0, 512, 512]
    mr._prepare_eager_forward_batch(batch)
    assert batch._memt_graph_allowed is False


def test_bs32_at_512_tokens_falls_back_without_explicit_4096_gear():
    runner, batch = runner_and_batch(counts=(4096,) * 4)
    batch.rids = ["q"] * 8
    batch.batch_size = 8
    batch.extend_seq_lens_cpu = [512] * 8
    batch.extend_prefix_lens_cpu = [0] * 8
    result = fp.replay_vote(runner, batch, rank=0, native_can_run=lambda *a: True)
    assert not result.eligible
    assert "capture_shape_unavailable" in result.reasons


def test_ast_validator_retains_every_other_native_rule_and_is_nonmutating(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "sglang.srt.arg_groups.overrides", NS(resolved_view=lambda a: a._resolved()))
    monkeypatch.setitem(sys.modules, "sglang.srt.configs.model_config", NS(is_deepseek_v4=lambda c: True))
    source = native_rule_source()
    before = source
    warnings = []
    validator = fp.audited_native_validator(source, {
        "Backend": NS(DISABLED="disabled"),
        "logger": NS(warning=lambda *args: warnings.append(args)),
    })
    for kind in ("mla", "cp", "dcp", "lora_paths", "lora", "a2a", "multimodal"):
        args, values = config_args()
        if kind == "mla":
            args.use_mla_backend = lambda: True
        elif kind == "cp":
            values["attn_cp_size"] = 2
        elif kind == "dcp":
            args.dcp_size = 2
        elif kind == "lora_paths":
            args.lora_paths = ["adapter"]
        elif kind == "lora":
            args.enable_lora = True
        elif kind == "a2a":
            values["moe_a2a_backend"] = "deepep"
        elif kind == "multimodal":
            args.get_model_config = lambda: NS(hf_config=NS(), is_multimodal=True,
                                               is_multimodal_breakable_cuda_graph_supported=False)
        validator(args)
        assert args.cuda_graph_config.prefill.backend == "disabled", kind
    assert len(warnings) == 7
    assert source == before
    # Only surrounding whitespace is normalized; source edits require review.
    fp.audited_native_validator("\n" + source + "\n", {})
    with pytest.raises(ValueError, match="source"):
        fp.audited_native_validator("# comment\n" + source, {})
    with pytest.raises(ValueError, match="source"):
        fp.audited_native_validator(source.replace("self.dcp_size > 1", "self.dcp_size > 2"), {})


def test_rule_source_identity_never_depends_on_python_ast_serialization(monkeypatch):
    source = native_rule_source()
    monkeypatch.setattr(fp.ast, "dump", lambda *a, **kw: pytest.fail("version-dependent AST hash"))
    fp.audited_native_validator(source, {})


def test_config_mismatch_and_missing_explicit_locks_fail():
    intent = FlashPrefillIntent("breakable", (512, 1024), 4)
    for kind in ("backend", "bs", "max_bs", "locks"):
        args, _ = config_args()
        if kind == "locks":
            args._cuda_graph_config_locked = set()
        else:
            setattr(args.cuda_graph_config.prefill, kind, "disabled" if kind == "backend" else [512] if kind == "bs" else 512)
        with pytest.raises(ValueError, match="resolved"):
            fp.validate_capture_request(args, intent, native_validator=lambda a: None)


def test_worker_preflight_source_geometry_and_headroom(monkeypatch):
    import torch
    intent = FlashPrefillIntent("breakable", (512, 1024), 4)
    runner, _ = runner_and_batch(counts=(512,) * 4)
    runner.require_mlp_tp_gather = True
    runner.backend = NS(_debug_eager=False)
    attn = runner.model_runner.attn_backend
    attn.page_size, attn.c4_topk = 256, 512
    attn.req_to_token = NS(shape=(32, 32768))
    attn.MAX_SEQ_LEN_FOR_CAPTURE = 32768
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(fp.inspect, "getfile", lambda cls: __file__)
    monkeypatch.setattr(fp, "_DSV4_SOURCE_SHA", fp.hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def memory():
        return {"free_bytes": 2**30}

    assert fp._worker_preflight(runner, intent, memory)["single_c4_logits_bytes"] == 33554432
    attn.c4_topk = 1024
    with pytest.raises(ValueError, match="layout"):
        fp._worker_preflight(runner, intent, memory)
    attn.c4_topk = 512
    attn.MAX_SEQ_LEN_FOR_CAPTURE = 16384
    with pytest.raises(ValueError, match="capacity"):
        fp._worker_preflight(runner, intent, memory)
    attn.MAX_SEQ_LEN_FOR_CAPTURE = 32768
    monkeypatch.setattr(fp, "_DSV4_SOURCE_SHA", "changed")
    with pytest.raises(ValueError, match="source"):
        fp._worker_preflight(runner, intent, memory)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        fp._worker_preflight(runner, intent, memory)


def test_one_rank_preflight_rejection_prevents_native_capture(monkeypatch):
    class Runner:
        def capture(self):
            raise AssertionError("native capture must not start")
    intent = FlashPrefillIntent("breakable", (512,), 4)
    events = []
    monkeypatch.setattr(fp, "validate_capture_request", lambda *a: {})
    def reject(*a): raise ValueError("insufficient headroom")
    monkeypatch.setattr(fp, "_worker_preflight", reject)
    fp._install_capture_hook(intent, Runner, lambda v: [v] * 4, events.append,
                             lambda: None, lambda: {})
    runner = Runner()
    runner.model_runner = NS(server_args=NS(), model_config=NS(hf_config=NS(model_type="memt_deepseek_v4")))
    with pytest.raises(RuntimeError, match="preflight"):
        runner.capture()
    assert events[-1]["event"] == "capture_preflight_rejected"


def test_rank_eligibility_exception_is_collected_before_abort(monkeypatch):
    class Runner:
        def can_run_graph(self, batch): raise ValueError("native veto crashed")
    class ModelRunner:
        def _prepare_eager_forward_batch(self, batch): pass
    runner, batch = runner_and_batch(counts=(512,) * 4)
    runner._memt_flash_capture_complete = True
    mr = ModelRunner()
    mr.model_config = NS(hf_config=NS(model_type="memt_deepseek_v4"))
    mr.prefill_cuda_graph_runner = runner
    events, gathered = [], []
    def gather(v):
        gathered.append(v)
        return [replace(v, rank=i) for i in range(4)]
    fp._install_replay_consensus(FlashPrefillIntent("breakable", (512, 1024), 4),
                                 Runner, ModelRunner, gather, lambda: 0, events.append)
    with pytest.raises(RuntimeError, match="eligibility"):
        mr._prepare_eager_forward_batch(batch)
    assert len(gathered) == 1
    assert events[-1]["event"] == "replay_admission_error"


def test_capture_records_sampled_headroom_without_claiming_peak_bound():
    class Key:
        size = 512
    runner = NS(backend=NS(_graphs={Key(): NS(_segments=[object()])}),
                attn_metadata_buffers={512: object()})
    observations = iter([{"free_bytes": 1000}, {"free_bytes": 800}])
    events = []
    fp.capture_attempt(runner, FlashPrefillIntent("breakable", (512,), 4),
                       lambda: None, emit=events.append, synchronize=lambda: None,
                       memory=lambda: next(observations))
    assert events[-1]["sampled_min_free_bytes"] == 800
    assert events[-1]["free_delta_bytes"] == 200
    assert events[-1]["peak_scope"] == "process_lifetime"


def test_runner_return_without_backend_replay_is_not_success():
    class Runner:
        def execute(self, batch):
            return "eager-only"
    events = []
    fp._install_execute_evidence(Runner, events.append)
    runner = Runner()
    runner.model_runner = NS(model_config=NS(hf_config=NS(model_type="memt_deepseek_v4")))
    runner._memt_flash_capture_complete = True
    def original(*a):
        return None

    runner.backend = NS(replay=original)
    fp._attach_replay_counter(runner)
    delegate = runner.backend.replay
    batch = NS(input_ids=[0] * 512, _memt_flash_prefill_allowed=True)
    with pytest.raises(RuntimeError, match="backend replay"):
        runner.execute(batch)
    assert runner.backend.replay is delegate
    assert events[-1]["event"] == "replay_error"
    assert not any(e["event"] == "replay_return" for e in events)
