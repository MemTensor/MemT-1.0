"""CPU launcher/worker integration; CUDA qualification remains runtime evidence."""

import importlib.util
import json
import sys
from types import SimpleNamespace as NS

import pytest

from memt_sglang import launch
from memt_sglang.runtime import flash_prefill as fp


@pytest.fixture
def model(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(launch, "validate_model_config",
                        lambda c: ("memt-flash", NS(chunk_tokens=2048)))
    for key in ("MEMT_EXPERIMENT_PREFILL_GRAPH", "MEMT_EXPERIMENT_DECODE_GRAPH",
                "MEMT_FLASH_PREFILL_REQUEST"):
        monkeypatch.delenv(key, raising=False)
    return ["--model-path", str(tmp_path)]


def graph(engine):
    assert engine.count("--cuda-graph-config") == 1
    return json.loads(engine[engine.index("--cuda-graph-config") + 1])


def test_flash_default_is_disabled_and_dry_run_does_not_claim_resolution(model):
    options, engine, identity = launch.prepare(model)
    assert graph(engine)["prefill"] == {"backend": "disabled"}
    assert options.flash_prefill_intent.requested_backend == "disabled"
    assert identity["flash_prefill"]["requested_backend"] == "disabled"
    assert identity["flash_prefill"]["resolved_backend"] is None
    assert identity["flash_prefill"]["actual"]["status"] == "not_observed"
    assert identity["flash_prefill"]["gpu_qualified"] is False


@pytest.mark.parametrize("request_kind", ["environment", "json", "equals_json"])
def test_explicit_request_locks_small_gears_and_preserves_decode(model, monkeypatch, request_kind):
    argv = list(model)
    if request_kind == "environment":
        monkeypatch.setenv("MEMT_EXPERIMENT_PREFILL_GRAPH", "1")
    else:
        config = json.dumps({"prefill": {"backend": "breakable"},
                             "decode": {"backend": "disabled"}})
        argv += (["--cuda-graph-config", config] if request_kind == "json"
                 else ["--cuda-graph-config=" + config])
    options, engine, identity = launch.prepare(argv)
    assert graph(engine)["prefill"] == {"backend": "breakable", "bs": [512, 1024], "max_bs": 1024}
    assert graph(engine)["decode"]["backend"] == "disabled"
    assert options.flash_prefill_intent.capture_tokens == (512, 1024)
    assert identity["flash_prefill"]["requested_capture_tokens"] == [512, 1024]


@pytest.mark.parametrize("phase", [
    {"backend": "breakable", "bs": [2048]},
    {"backend": "breakable", "bs": [512], "max_bs": 1024},
    {"backend": "breakable", "bs": "512"},
    {"backend": "full"},
])
def test_unsupported_flash_requests_are_not_silently_changed(model, phase):
    with pytest.raises(ValueError):
        launch.prepare(model + ["--cuda-graph-config", json.dumps({"prefill": phase})])


def test_explicit_4096_preserves_decode_and_does_not_change_default(model):
    config = {"prefill": {"backend": "breakable", "bs": [512, 1024, 4096], "max_bs": 4096},
              "decode": {"backend": "breakable", "bs": [1, 8], "max_bs": 8}}
    options, engine, identity = launch.prepare(model + ["--cuda-graph-config", json.dumps(config)])
    assert graph(engine) == config
    assert options.flash_prefill_intent.capture_tokens == (512, 1024, 4096)
    assert identity["flash_prefill"]["gpu_qualified"] is False
    assert identity["flash_prefill"]["resolved_backend"] is None
    _, engine, _ = launch.prepare(model + ["--cuda-graph-config", '{"prefill":{"backend":"breakable"}}'])
    assert graph(engine)["prefill"]["bs"] == [512, 1024]
    assert graph(engine)["prefill"]["max_bs"] == 1024


def test_explicit_disable_overrides_env_and_decode_only_stays_prefill_off(model, monkeypatch):
    monkeypatch.setenv("MEMT_EXPERIMENT_PREFILL_GRAPH", "1")
    _, engine, _ = launch.prepare(model + ["--cuda-graph-config", '{"prefill":{"backend":"disabled"}}'])
    assert graph(engine)["prefill"]["backend"] == "disabled"
    monkeypatch.delenv("MEMT_EXPERIMENT_PREFILL_GRAPH")
    _, engine, _ = launch.prepare(model + ["--cuda-graph-config", '{"decode":{"backend":"breakable"}}'])
    assert graph(engine)["prefill"]["backend"] == "disabled"
    assert graph(engine)["decode"]["backend"] == "breakable"


def test_resolve_runs_gate_before_publishing_request_and_records_native_identity(model, monkeypatch):
    monkeypatch.setenv("MEMT_EXPERIMENT_PREFILL_GRAPH", "1")
    options, _, identity = launch.prepare(model)
    args = NS(cuda_graph_config=NS(prefill=NS(backend="breakable", bs=[512, 1024])), dp_size=4)
    calls = []
    def validate(actual, intent):
        assert "MEMT_FLASH_PREFILL_REQUEST" not in launch.os.environ
        calls.append((actual, intent))
        return {**fp.prefill_identity(intent, actual.cuda_graph_config.prefill),
                "status": "capture_attempt_permitted"}
    monkeypatch.setattr(fp, "validate_capture_request", validate)
    resolved = launch.resolve_flash_prefill(options, args, identity)
    assert len(calls) == 1
    assert identity["flash_prefill"]["resolved_backend"] is None
    assert resolved["flash_prefill"]["resolved_backend"] == "breakable"
    assert resolved["flash_prefill"]["actual"]["status"] == "not_observed"
    assert json.loads(launch.os.environ["MEMT_FLASH_PREFILL_REQUEST"])["capture_tokens"] == [512, 1024]


def test_native_disable_is_reported_and_never_reenabled(model, monkeypatch):
    monkeypatch.setenv("MEMT_EXPERIMENT_PREFILL_GRAPH", "1")
    options, _, identity = launch.prepare(model)
    args = NS(cuda_graph_config=NS(prefill=NS(backend="disabled", bs=[512, 1024])), dp_size=4)
    monkeypatch.setattr(fp, "validate_capture_request", lambda *a: pytest.fail("must retain native disable"))
    result = launch.resolve_flash_prefill(options, args, identity)
    assert result["flash_prefill"]["status"] == "disabled_by_native"
    assert "MEMT_FLASH_PREFILL_REQUEST" not in launch.os.environ


def test_gate_rejection_does_not_publish_worker_request(model, monkeypatch):
    monkeypatch.setenv("MEMT_EXPERIMENT_PREFILL_GRAPH", "1")
    options, _, identity = launch.prepare(model)
    args = NS(cuda_graph_config=NS(prefill=NS(backend="breakable", bs=[512, 1024])), dp_size=4)
    def reject(*a):
        raise ValueError("native veto")
    monkeypatch.setattr(fp, "validate_capture_request", reject)
    with pytest.raises(ValueError, match="native veto"):
        launch.resolve_flash_prefill(options, args, identity)
    assert "MEMT_FLASH_PREFILL_REQUEST" not in launch.os.environ


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize("model_name,requested", [("memt-flash", False), ("memt-flash", True), ("memt-nano", False)])
def test_same_bootstrap_entry_installs_only_explicit_flash_worker(monkeypatch, model_name, requested, observed):
    monkeypatch.setenv("MEMT_SGLANG_MODEL", model_name)
    monkeypatch.delenv("MEMT_FLASH_PREFILL_REQUEST", raising=False)
    if requested:
        monkeypatch.setenv("MEMT_FLASH_PREFILL_REQUEST", json.dumps({
            "requested_backend": "breakable", "capture_tokens": [512], "dp_size": 4}))
    monkeypatch.setattr(launch, "require_sglang", lambda: None)
    monkeypatch.setitem(sys.modules, "memt_sglang.models.deepseek.hf_config", NS(register_deepseek_config=lambda: None))
    monkeypatch.setitem(sys.modules, "memt_sglang.models.qwen.hf_config", NS(register_qwen_config=lambda: None))
    monkeypatch.setitem(sys.modules, "memt_sglang.models.common.loader", NS(install_model_loader=lambda: None))
    installs = []
    monkeypatch.setattr(fp, "install_native_worker", lambda intent, **kw: installs.append((intent, kw)))
    launch.bootstrap(worker=False)
    assert not installs  # Parent/tokenizer/detokenizer never install capture hooks.
    observer = (lambda event: None) if observed else None
    if observed:
        launch.bootstrap(replay_observer=observer)
    else:
        launch.bootstrap()  # Existing revision_launch wrapper uses this same entry.
    assert len(installs) == int(requested)
    if requested:
        assert installs[0][0].capture_tokens == (512,)
        assert callable(installs[0][1]["emit"])
        assert installs[0][1]["replay_observer"] is observer


def test_worker_evidence_is_json_and_does_not_claim_numeric_qualification(capsys):
    launch.emit_flash_prefill_event({"event": "replay_return", "dp_rank": 2,
                                   "cuda_synchronized": True, "numeric_qualified": False})
    record = json.loads(capsys.readouterr().out)
    assert record["component"] == "memt_flash_prefill"
    assert record["event"] == "replay_return"
    assert record["pid"] > 0
    assert record["numeric_qualified"] is False


def test_scheduler_uses_worker_bootstrap_before_native_scheduler(monkeypatch):
    calls = []
    monkeypatch.setattr(launch, "bootstrap", lambda **kw: calls.append(("bootstrap", kw)))
    monkeypatch.setitem(sys.modules, "memt_sglang.runtime.hooks", NS(install_scheduler_hooks=lambda: calls.append("hooks")))
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", NS(run_scheduler_process=lambda *a, **kw: calls.append("native")))
    launch.run_scheduler()
    assert calls == [("bootstrap", {"worker": True}), "hooks", "native"]


@pytest.mark.parametrize("gate_rejects", [False, True])
def test_main_gates_before_launch_and_preserves_wrapper_scheduler(model, monkeypatch, gate_rejects):
    monkeypatch.setattr(launch.os, "environ", dict(launch.os.environ))
    monkeypatch.setenv("MEMT_EXPERIMENT_PREFILL_GRAPH", "1")
    calls, identities = [], []
    args = NS(cuda_graph_config=NS(prefill=NS(backend="breakable", bs=[512, 1024])),
              dp_size=4, max_prefill_tokens=16384)
    def parse(engine):
        assert graph(engine)["prefill"]["max_bs"] == 1024
        calls.append("parse")
        return args
    def gate(actual, intent):
        calls.append("gate")
        if gate_rejects:
            raise ValueError("native veto")
        return {**fp.prefill_identity(intent, actual.cuda_graph_config.prefill),
                "status": "capture_attempt_permitted"}
    def wrapper(*a, **kw):
        pass
    def serve(actual, **callbacks):
        calls.append("serve")
        assert callbacks["run_scheduler_process_func"] is wrapper
        assert "MEMT_FLASH_PREFILL_REQUEST" in launch.os.environ
    http = NS(launch_server=serve)
    monkeypatch.setattr(launch, "require_sglang", lambda: None)
    def bootstrap(**kw):
        assert kw == {"worker": False}
        calls.append("parent_bootstrap")
    monkeypatch.setattr(launch, "bootstrap", bootstrap)
    monkeypatch.setattr(launch, "run_scheduler", wrapper)
    monkeypatch.setattr(launch, "sync_graph_config", lambda a: calls.append("sync"))
    monkeypatch.setattr(launch, "validate_server_args", lambda *a, **kw: calls.append("policy"))
    monkeypatch.setattr(fp, "validate_capture_request", gate)
    monkeypatch.setattr(launch, "execution_identity", lambda a: {"graph_backends": {"prefill": "breakable"}})
    monkeypatch.setattr(launch.metadata, "version", lambda name: "test-only")
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints", NS(http_server=http))
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", NS(prepare_server_args=parse))
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", NS(kill_process_tree=lambda *a, **kw: calls.append("cleanup")))
    monkeypatch.setitem(sys.modules, "memt_sglang.runtime.http", NS(install_http=lambda h, identity, **kw: identities.append(identity)))
    monkeypatch.setitem(sys.modules, "transformers", NS(AutoTokenizer=NS(from_pretrained=lambda *a, **kw: object())))
    monkeypatch.setitem(sys.modules, "memt_sglang.models.common.snapshot_identity", NS(snapshot_identity=lambda *a: "snapshot"))
    if gate_rejects:
        with pytest.raises(ValueError, match="native veto"):
            launch.main(model)
        assert "serve" not in calls
        assert not identities
    else:
        launch.main(model)
        assert calls == ["parent_bootstrap", "parse", "sync", "policy", "gate", "serve", "cleanup"]
        assert identities[0]["flash_prefill"]["actual"]["status"] == "not_observed"
        assert identities[0]["flash_prefill"]["gpu_qualified"] is False


def test_dry_run_never_requires_native(model, monkeypatch, capsys):
    monkeypatch.setattr(launch, "require_sglang", lambda: pytest.fail("native import"))
    launch.main(model + ["--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert output["identity"]["flash_prefill"]["resolved_backend"] is None


def test_unrequested_resolution_and_stale_worker_request_rejected(model, monkeypatch):
    options, _, identity = launch.prepare(model)
    monkeypatch.setenv("MEMT_FLASH_PREFILL_REQUEST", "stale")
    args = NS(cuda_graph_config=NS(prefill=NS(backend="breakable", bs=[512])), dp_size=4)
    with pytest.raises(ValueError, match="explicit request"):
        launch.resolve_flash_prefill(options, args, identity)
    assert "MEMT_FLASH_PREFILL_REQUEST" not in launch.os.environ


@pytest.mark.parametrize("config", ["[]", '{"prefill":[]}', '{"prefill":{"backend":"breakable","bs":[]}}'])
def test_bad_flash_json_rejected(model, config):
    with pytest.raises(ValueError):
        launch.prepare(model + ["--cuda-graph-config", config])


def test_duplicate_graph_json_rejected(model):
    with pytest.raises(ValueError, match="Duplicate"):
        launch.prepare(model + ["--cuda-graph-config", "{}", "--cuda-graph-config={}"])


@pytest.mark.parametrize("observed", [False, True])
def test_native_worker_adapter_defers_cuda_until_callbacks_and_uses_world_cpu_group(monkeypatch, observed):
    callbacks, events = {}, []
    cpu_group = object()
    initialized = False
    def world():
        assert initialized, "process group accessed during installation"
        return NS(cpu_group=cpu_group)
    def parallel():
        assert initialized, "rank accessed during installation"
        return NS(attn_dp_rank=2)
    def mem_info():
        assert initialized, "CUDA accessed during installation"
        return (800, 1000)
    def gather(output, value, *, group):
        assert group is cpu_group
        output[:] = [value] * 4
    cuda = NS(mem_get_info=mem_info, memory_allocated=lambda: 100,
              memory_reserved=lambda: 200, max_memory_allocated=lambda: 300,
              max_memory_reserved=lambda: 400, synchronize=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", NS(cuda=cuda, distributed=NS(all_gather_object=gather)))
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed.parallel_state", NS(get_world_group=world))
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.attention.deepseek_v4_backend", NS(DeepseekV4AttnBackend=object))
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.model_runner", NS(ModelRunner=object))
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.runner.prefill_cuda_graph_runner", NS(PrefillCudaGraphRunner=object))
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", NS(get_parallel=parallel))
    monkeypatch.setattr(fp, "install_worker_hooks", lambda *a, **kw: callbacks.update(kw))
    observer_events = []
    fp.install_native_worker(fp.FlashPrefillIntent("breakable", (512,), 4), emit=events.append,
                             replay_observer=observer_events.append if observed else None)
    initialized = True
    assert callbacks["gather"]("vote") == ["vote"] * 4
    assert callbacks["rank"]() == 2
    assert callbacks["memory"]() == {
        "free_bytes": 800, "total_bytes": 1000, "allocated_bytes": 100,
        "reserved_bytes": 200, "peak_allocated_bytes": 300, "peak_reserved_bytes": 400,
    }
    callbacks["emit"]({"event": "capture_enter"})
    assert events == [{"event": "capture_enter", "dp_rank": 2}]
    if observed:
        callbacks["replay_observer"]({"event": "replay_return"})
        assert observer_events == [{"event": "replay_return", "dp_rank": 2}]
    else:
        assert callbacks["replay_observer"] is None


def test_native_scheduler_sync_cannot_be_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_SCHEDULER_SKIP_ALL_GATHER", "1")
    runner = NS(model_runner=NS(attn_backend=NS()))
    with pytest.raises(ValueError, match="all-rank"):
        fp._worker_preflight(runner, fp.FlashPrefillIntent("breakable", (512,), 4), lambda: {})


def test_tokenizer_and_detokenizer_use_nonworker_bootstrap(monkeypatch):
    calls = []
    monkeypatch.setattr(launch, "bootstrap", lambda **kw: calls.append(kw))
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.detokenizer_manager", NS(run_detokenizer_process=lambda: "detokenizer"))
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.http_server", NS(init_tokenizer_manager=lambda: ("manager",)))
    monkeypatch.setitem(sys.modules, "memt_sglang.runtime.hooks", NS(install_tokenizer_control=lambda manager: calls.append(manager)))
    assert launch.run_detokenizer() == "detokenizer"
    assert launch.init_tokenizer() == ("manager",)
    assert calls == [{"worker": False}, {"worker": False}, "manager"]


def test_install_hooks_never_calls_process_group_cuda_or_model():
    def unavailable(*args, **kwargs):
        pytest.fail("installation invoked runtime callback before initialization")
    class Runner:
        capture = unavailable
        can_run_graph = unavailable
        execute = unavailable
    class Attention:
        prepare_forward_metadata_for_breakable_cuda_graph_replay = unavailable
    class ModelRunner:
        _prepare_eager_forward_batch = unavailable
    fp.install_worker_hooks(fp.FlashPrefillIntent("breakable", (512,), 4),
                            Runner, Attention, ModelRunner, gather=unavailable,
                            rank=unavailable, emit=unavailable, synchronize=unavailable,
                            memory=unavailable)
    assert Runner._memt_flash_capture_gate is True


def test_metadata_adapter_forwards_native_keyword_only_static_batch():
    observed = []
    class Attention:
        def prepare_forward_metadata_for_breakable_cuda_graph_replay(
            self, capture_metadata, forward_batch, *, static_forward_batch=None
        ):
            observed.append((capture_metadata, forward_batch, static_forward_batch))
    fp._install_metadata_hook(Attention)
    # The same signature must work before tagging (native passthrough).
    attention = Attention()
    objects = [object(), object(), object()]
    attention.prepare_forward_metadata_for_breakable_cuda_graph_replay(
        objects[0], objects[1], static_forward_batch=objects[2])
    assert observed == [tuple(objects)]


def test_default_capture_gate_reads_real_inspect_source_without_validator_injection(monkeypatch):
    import ast
    import inspect

    from test_flash_prefill import config_args, native_source_path

    path = native_source_path("srt/server_args.py")
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == "_disable_breakable_cudagraph_if_incompatible")
    namespace = {"Backend": NS(DISABLED="disabled"), "logger": NS(warning=lambda *a: None)}
    # Preserve native filename and line locations: inspect.getsource reads the
    # actual native method text, including its class indentation and comments.
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    Args = type("Args", (), {node.name: namespace[node.name]})
    args = Args()
    args.__dict__.update(vars(config_args()[0]))
    monkeypatch.setitem(sys.modules, "sglang.srt.arg_groups.overrides", NS(resolved_view=lambda a: a._resolved()))
    monkeypatch.setitem(sys.modules, "sglang.srt.configs.model_config", NS(is_deepseek_v4=lambda c: True))
    intent = fp.FlashPrefillIntent("breakable", (512, 1024), 4)
    assert inspect.getsource(Args._disable_breakable_cudagraph_if_incompatible).startswith("    def ")
    result = fp.validate_capture_request(args, intent)
    assert result["status"] == "capture_attempt_permitted"
    assert result["native_rule_source_sha256"] == fp._RULE_SOURCE_SHA
    args.enable_lora = True
    with pytest.raises(ValueError, match="native prefill"):
        fp.validate_capture_request(args, intent)
    assert args.cuda_graph_config.prefill.backend == "breakable"


def test_native_source_locator_explicit_root_needs_no_installed_package(tmp_path, monkeypatch):
    from test_flash_prefill import native_source_path

    root = tmp_path / "archive"
    (root / "srt").mkdir(parents=True)
    source = root / "srt/server_args.py"
    source.write_text("# Source location fixture, not a native validator fixture.\n")
    monkeypatch.setenv("MEMT_TEST_SGLANG_SOURCE", str(root))

    def unexpected_lookup(name):
        pytest.fail(f"Explicit source must not probe installed modules: {name}")

    monkeypatch.setattr(importlib.util, "find_spec", unexpected_lookup)
    assert native_source_path("srt/server_args.py") == source.resolve()


def test_native_source_locator_uses_installed_package_without_importing_it(tmp_path, monkeypatch):
    from test_flash_prefill import native_source_path

    root = tmp_path / "sglang"
    (root / "srt").mkdir(parents=True)
    source = root / "srt/server_args.py"
    source.write_text("# Source location fixture, not a native validator fixture.\n")
    # Importing this fake package would fail; find_spec only returns its location.
    (root / "__init__.py").write_text("raise AssertionError('must not import SGLang')\n")
    monkeypatch.delenv("MEMT_TEST_SGLANG_SOURCE", raising=False)
    lookups = []

    def find_spec(name):
        lookups.append(name)
        return NS(submodule_search_locations=[str(root)])

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    assert native_source_path("srt/server_args.py") == source.resolve()
    assert lookups == ["sglang"]


def test_native_source_locator_skips_only_when_optional_package_is_absent(monkeypatch):
    from test_flash_prefill import native_source_path

    monkeypatch.delenv("MEMT_TEST_SGLANG_SOURCE", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(pytest.skip.Exception, match="Native integration requires"):
        native_source_path("srt/server_args.py")


@pytest.mark.parametrize("explicit", [False, True])
def test_native_source_locator_fails_for_incomplete_source(tmp_path, monkeypatch, explicit):
    from test_flash_prefill import native_source_path

    if explicit:
        monkeypatch.setenv("MEMT_TEST_SGLANG_SOURCE", str(tmp_path))
    else:
        monkeypatch.delenv("MEMT_TEST_SGLANG_SOURCE", raising=False)
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: NS(
            submodule_search_locations=[str(tmp_path)]
        ))
    with pytest.raises(pytest.fail.Exception, match="source is missing"):
        native_source_path("srt/server_args.py")


def test_native_source_locator_rejects_empty_explicit_root(monkeypatch):
    from test_flash_prefill import native_source_path

    monkeypatch.setenv("MEMT_TEST_SGLANG_SOURCE", " ")
    with pytest.raises(pytest.fail.Exception, match="must name a source root"):
        native_source_path("srt/server_args.py")
