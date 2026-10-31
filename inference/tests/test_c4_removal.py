"""CPU removal contracts; native model/kernel boundaries use isolated doubles."""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

INFERENCE = Path(__file__).resolve().parents[1]
PACKAGE = INFERENCE / "memt_sglang"
C4_FLAG = "MEMT_EXPERIMENT_C4_CANONICAL_ORDER"


@pytest.fixture(autouse=True)
def isolated_experiments(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("MEMT_EXPERIMENT_"):
            monkeypatch.delenv(name)


@pytest.mark.parametrize(
    "relative_path",
    [
        "runtime/c4_order.py",
        "models/deepseek/ops/c4_order.py",
        "models/deepseek/ops/_triton/c4_order.py",
    ],
)
def test_c4_implementation_is_deleted(relative_path):
    assert not (PACKAGE / relative_path).exists()


@pytest.mark.parametrize("relative_path", ["models/deepseek/model.py", "launch.py"])
def test_model_and_launcher_have_no_c4_install_or_flag(relative_path):
    source = (PACKAGE / relative_path).read_text()
    assert "c4_order" not in source
    assert "C4_CANONICAL_ORDER" not in source


@pytest.mark.parametrize("readme", ["README.md", "README_zh.md"])
def test_product_docs_remove_c4_and_keep_communication_option(readme):
    source = (INFERENCE / readme).read_text()
    assert C4_FLAG not in source
    assert "--deterministic-communication" in source


@pytest.fixture
def native_model_boundary(monkeypatch):
    def topk_v1(*args, **kwargs):
        return "native-v1", args, kwargs

    def topk_v2(*args, **kwargs):
        return "native-v2", args, kwargs

    native = SimpleNamespace(
        topk_transform_512=topk_v1, topk_transform_512_v2=topk_v2
    )
    indexer = SimpleNamespace(
        topk_transform_512=topk_v1, topk_transform_512_v2=topk_v2
    )

    class NativeModel:
        def __init__(self, config, quant_config=None, prefix=""):
            self.config = config
            self.indexer = indexer
            self.index_topk = config.index_topk
            self.quant_config = quant_config
            self.prefix = prefix
            self.topk_at_init = (
                indexer.topk_transform_512, indexer.topk_transform_512_v2
            )

    # Import the real MemT model, replacing only GPU/model-loading boundaries.
    # A private module spec avoids leaving a cached model with a fake base class.
    modules = {
        "sglang.srt.models.deepseek_v4": SimpleNamespace(
            DeepseekV4ForCausalLM=NativeModel
        ),
        "sglang.jit_kernel.dsv4.topk": native,
        "sglang.srt.layers.attention.dsv4.indexer": indexer,
        "memt_sglang.models.deepseek.integration": SimpleNamespace(
            DeepSeekMemoryHooks=object
        ),
        "memt_sglang.models.deepseek.layer": SimpleNamespace(
            DeepSeekMemoryBlock=object
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        native=native, indexer=indexer, originals=(topk_v1, topk_v2)
    )


@pytest.mark.parametrize("legacy_flag", [None, "0", "1", "true"])
@pytest.mark.parametrize("index_topk", [256, 512, 1024])
def test_model_keeps_native_topk_identity_and_config(
    monkeypatch, native_model_boundary, legacy_flag, index_topk
):
    if legacy_flag is not None:
        monkeypatch.setenv(C4_FLAG, legacy_flag)
    boundary = native_model_boundary
    spec = importlib.util.spec_from_file_location(
        "_memt_c4_removal_model", PACKAGE / "models/deepseek/model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = SimpleNamespace(
        index_topk=index_topk,
        memory_config=SimpleNamespace(state_shapes={1: (1, 2, 4, 4)}),
    )
    quant_config = object()
    for _ in range(2):
        model = module.MemTDeepseekV4ForCausalLM(
            config, quant_config=quant_config, prefix="native-prefix"
        )
        assert model.config is config
        assert model.indexer is boundary.indexer
        assert model.index_topk == config.index_topk == index_topk
        assert model.quant_config is quant_config
        assert model.prefix == "native-prefix"
        assert model.memt_state_shapes is config.memory_config.state_shapes
        for position, name in enumerate(
            ("topk_transform_512", "topk_transform_512_v2")
        ):
            original = boundary.originals[position]
            assert model.topk_at_init[position] is original
            assert getattr(boundary.indexer, name) is original
            assert getattr(boundary.native, name) is original
        assert not hasattr(boundary.indexer, "_memt_c4_logical_order")


@pytest.mark.parametrize("legacy_flag", ["0", "1", "true", ""])
@pytest.mark.parametrize("deterministic", [False, True])
def test_launcher_ignores_removed_flag_and_preserves_communication(
    monkeypatch, tmp_path, legacy_flag, deterministic
):
    from memt_sglang import launch

    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(
        launch,
        "validate_model_config",
        lambda config: ("memt-flash", SimpleNamespace(chunk_tokens=2048)),
    )
    argv = ["--model-path", str(tmp_path), "--dry-run"]
    if deterministic:
        argv.append("--deterministic-communication")
    _, baseline_engine, baseline_identity = launch.prepare(argv)
    monkeypatch.setenv(C4_FLAG, legacy_flag)
    options, engine, identity = launch.prepare(argv)
    assert engine == baseline_engine
    assert identity == baseline_identity
    assert options.deterministic_communication is deterministic
    assert "--enable-dp-attention" in engine
    assert ("--disable-custom-all-reduce" in engine) is deterministic
    assert ("--enforce-disable-flashinfer-allreduce-fusion" in engine) is deterministic
    if deterministic:
        assert identity["communication"] == {
            "NCCL_ALGO": "allreduce:tree",
            "disable_custom_all_reduce": True,
            "enforce_disable_flashinfer_allreduce_fusion": True,
            "moe_a2a_backend": "none",
            "whole_model_determinism": False,
        }
    else:
        assert "communication" not in identity
