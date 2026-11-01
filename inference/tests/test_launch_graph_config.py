"""Core launch options preserve native decode shapes without experiment gates."""

import json
from types import SimpleNamespace

import pytest

from memt_sglang import launch


@pytest.mark.parametrize("buckets", [[1, 8, 32], [1, 16, 64, 96]])
def test_decode_shapes_are_operator_configuration(buckets):
    config = {
        "prefill": {"backend": "disabled"},
        "decode": {"backend": "breakable", "bs": buckets, "max_bs": max(buckets)},
    }
    remaining, resolved = launch._flash_graph_config(
        ["--cuda-graph-config", json.dumps(config), "--tp-size", "4"], config
    )
    assert remaining == ["--tp-size", "4"]
    assert resolved == config


def test_removed_prefill_adapter_is_explicit():
    with pytest.raises(ValueError, match="separate experimental adapter"):
        launch._flash_graph_config([], {"prefill": {"backend": "breakable"}})


def test_graph_request_resolves_safe_defaults_without_strict(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(
        launch,
        "validate_model_config",
        lambda c: ("memt-flash", SimpleNamespace(chunk_tokens=2048)),
    )
    for name in ("SLOT_READ", "C4_STABLE_TOPK", "PADDED_DECODE_QUERY"):
        monkeypatch.delenv("MEMT_EXPERIMENT_" + name, raising=False)
    monkeypatch.delenv("SGLANG_DSV4_PIN_DECODE_INTERMEDIATES", raising=False)
    monkeypatch.delenv("SGLANG_DSV4_DETERMINISTIC_PROFILE", raising=False)
    monkeypatch.setenv("MEMT_EXPERIMENT_MEMORY_DECODE_GRAPH", "1")
    launch.prepare(["--model-path", str(tmp_path)])
    import os

    assert os.environ["SGLANG_DSV4_PIN_DECODE_INTERMEDIATES"] == "1"
    assert "SGLANG_DSV4_DETERMINISTIC_PROFILE" not in os.environ
