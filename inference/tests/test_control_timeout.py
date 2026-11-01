"""Control timeouts are deployment settings and must be checked before startup."""

from types import SimpleNamespace

import pytest

from memt_sglang import launch
from memt_sglang.runtime.config import control_timeout_seconds
from memt_sglang.runtime.control import ControlChannel


def test_existing_agent_default_is_preserved(monkeypatch):
    monkeypatch.delenv("MEMT_CONTROL_TIMEOUT_SECONDS", raising=False)
    assert control_timeout_seconds() == 180
    assert ControlChannel(None).timeout == 180


def test_control_channel_resolves_operator_setting(monkeypatch):
    monkeypatch.setenv("MEMT_CONTROL_TIMEOUT_SECONDS", "45.5")
    assert ControlChannel(None).timeout == 45.5
    assert ControlChannel(None, timeout=1).timeout == 1


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "broken"])
def test_invalid_timeout_rejected_before_weight_loading(monkeypatch, value):
    monkeypatch.setenv("MEMT_CONTROL_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError):
        launch.prepare(["--model-path", "/not-loaded"])


def test_launch_identity_records_cli_timeout(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(
        launch,
        "validate_model_config",
        lambda c: ("memt-nano", SimpleNamespace(chunk_tokens=2048)),
    )
    monkeypatch.delenv("MEMT_CONTROL_TIMEOUT_SECONDS", raising=False)
    options, engine, identity = launch.prepare(
        ["--model-path", str(tmp_path), "--control-timeout-seconds", "72"]
    )
    assert identity["control_timeout_seconds"] == 72
    assert ControlChannel(None).timeout == 72
    assert "--control-timeout-seconds" not in engine
