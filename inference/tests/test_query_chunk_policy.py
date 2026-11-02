"""Admission budgets must permit progress without imposing writer-event sizes."""

from types import SimpleNamespace

import pytest

from memt_sglang.launch import resolve_mhc_prewarm
from memt_sglang.runtime.features import validate_query_chunks


@pytest.mark.parametrize("budget", [512, 1024, 1536, 2048, 4096])
def test_sub_write_event_query_budgets(monkeypatch, budget):
    monkeypatch.setenv("MEMT_EXPERIMENT_QUERY_CHUNKS", "1")
    validate_query_chunks(SimpleNamespace(chunked_prefill_size=budget, page_size=256))


def test_native_page_budget_must_make_progress(monkeypatch):
    monkeypatch.setenv("MEMT_EXPERIMENT_QUERY_CHUNKS", "1")
    with pytest.raises(ValueError, match="page size"):
        validate_query_chunks(SimpleNamespace(chunked_prefill_size=128, page_size=256))


@pytest.mark.parametrize("budget", [0, -2, True, 2.5, -1.0])
def test_invalid_query_budget(monkeypatch, budget):
    monkeypatch.setenv("MEMT_EXPERIMENT_QUERY_CHUNKS", "1")
    with pytest.raises(ValueError, match="positive budget"):
        validate_query_chunks(SimpleNamespace(chunked_prefill_size=budget))


@pytest.mark.parametrize("model", ["memt-nano", "memt-flash"])
@pytest.mark.parametrize("budget", [-1, 1024])
def test_prewarm_only_disabled_for_flash_unchunked(monkeypatch, model, budget):
    import os

    monkeypatch.setenv("SGLANG_DSV4_MHC_PREWARM", "1")
    resolve_mhc_prewarm(SimpleNamespace(chunked_prefill_size=budget), model)
    assert os.environ["SGLANG_DSV4_MHC_PREWARM"] == (
        "0" if model == "memt-flash" and budget == -1 else "1"
    )
