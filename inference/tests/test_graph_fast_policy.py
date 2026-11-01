"""Graph safety must remain available with deterministic arithmetic disabled."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from memt_sglang.runtime.features import (
    validate_flash_decode_graph,
    validate_numerics_tier,
)
from memt_sglang.runtime.memory_graph import MemoryDecodeGraph


@pytest.mark.parametrize("strict", [False, True])
def test_graph_and_numerics_are_compatible(monkeypatch, strict):
    monkeypatch.setenv("MEMT_NUMERICS", "strict" if strict else "fast")
    for name in (
        "MEMORY_DECODE_GRAPH",
        "SLOT_READ",
        "C4_STABLE_TOPK",
        "PADDED_DECODE_QUERY",
    ):
        monkeypatch.setenv("MEMT_EXPERIMENT_" + name, "1")
    monkeypatch.setenv("SGLANG_DSV4_PIN_DECODE_INTERMEDIATES", "1")
    for name in (
        "DETERMINISTIC_PROFILE",
        "MHC_FIXED_SPLITS",
        "ISOLATE_PREFILL_DECODE",
        "MOE_COMBINE_FP32",
    ):
        monkeypatch.setenv(
            "SGLANG_DSV4_" + name,
            "64" if strict and name == "MHC_FIXED_SPLITS" else str(int(strict)),
        )
    args = SimpleNamespace(
        enable_deterministic_inference=strict, speculative_skip_dp_mlp_sync=False
    )
    validate_numerics_tier(args)
    validate_flash_decode_graph(args)


@pytest.mark.parametrize("order", [(1, 8, 4), (8, 4, 1)])
def test_capture_uses_resolved_maximum_in_any_order(monkeypatch, order):
    import torch

    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    graph = MemoryDecodeGraph(None)
    runner = SimpleNamespace(
        captured_req_width=1,
        max_num_token=8,
        backend=SimpleNamespace(_capture_stream=None),
        model_runner=SimpleNamespace(device="cpu"),
    )
    address = None
    for bucket in order:
        runner.current_bucket = bucket
        with graph.capture(runner):
            assert graph.slots.numel() == 8
            if address is None:
                address = graph.slots.data_ptr()
            assert graph.slots.data_ptr() == address
    assert not graph.capture_active


def test_native_kernel_choice_is_not_a_benchmark_allowlist(monkeypatch):
    for name in ("MEMORY_DECODE_GRAPH", "C4_STABLE_TOPK", "PADDED_DECODE_QUERY"):
        monkeypatch.setenv("MEMT_EXPERIMENT_" + name, "1")
    monkeypatch.setenv("SGLANG_DSV4_PIN_DECODE_INTERMEDIATES", "1")
    monkeypatch.setenv("SGLANG_OPT_USE_TILELANG_MHC_PRE", "0")
    monkeypatch.setenv("SGLANG_OPT_DEEPGEMM_HC_PRENORM", "0")
    validate_flash_decode_graph(SimpleNamespace(speculative_skip_dp_mlp_sync=False))


def test_native_true_spelling_is_not_rejected(monkeypatch):
    for name in ("MEMORY_DECODE_GRAPH", "C4_STABLE_TOPK", "PADDED_DECODE_QUERY"):
        monkeypatch.setenv("MEMT_EXPERIMENT_" + name, "1")
    monkeypatch.setenv("SGLANG_DSV4_PIN_DECODE_INTERMEDIATES", "TRUE")
    validate_flash_decode_graph(SimpleNamespace(speculative_skip_dp_mlp_sync=False))
