import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from memt_sglang.protocol import MemoryRequest
from memt_sglang.runtime.features import enabled, sync_graph_config
from memt_sglang.runtime.query_chunks import install_atomic_commit_guard


@pytest.mark.parametrize("value", ["true", "yes", "2", "", "-1"])
def test_invalid_flags_are_rejected(value):
    with pytest.raises(ValueError):
        enabled("SLOT_READ", environment={"MEMT_EXPERIMENT_SLOT_READ": value})


@pytest.mark.parametrize("backend", ["full", "piecewise", "auto"])
def test_unsupported_graph_backend_is_not_silently_disabled(backend):
    @dataclass
    class Phase:
        backend: str

    @dataclass
    class Graph:
        prefill: Phase
        decode: Phase

    args = SimpleNamespace(cuda_graph_config=Graph(Phase(backend), Phase("disabled")))
    with pytest.raises(ValueError, match="requires disabled or breakable"):
        sync_graph_config(args)
    assert args.cuda_graph_config.prefill.backend == backend


@pytest.mark.parametrize("occupied", [True, False])
@pytest.mark.parametrize("accept", [True, False])
def test_large_commit_deferral_and_budget_accounting(monkeypatch, occupied, accept):
    class Adder:
        def __init__(self):
            self.rem_chunk_tokens = 2048
            self.can_run_list = [object()] if occupied else []
            self.new_chunked_req = None
            self.calls = 0

        def ceil_paged_tokens(self, n):
            return (n + 255) // 256 * 256

        def add_one_req(self, req):
            self.calls += 1
            assert self.rem_chunk_tokens is None
            if not accept:
                return "capacity"
            self.can_run_list.append(req)
            req.extend_range = SimpleNamespace(
                length=len(req.full_untruncated_fill_ids)
            )
            return "accepted"

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.schedule_policy",
        SimpleNamespace(
            PrefillAdder=Adder, AddReqResult=SimpleNamespace(OTHER="deferred")
        ),
    )
    install_atomic_commit_guard()
    req = SimpleNamespace(
        rid=MemoryRequest("session", "commit", "nonce", length=4097).rid,
        prefix_indices=[],
        full_untruncated_fill_ids=[1] * 4097,
    )
    adder = Adder()
    answer = adder.add_one_req(req)
    assert adder.calls == int(not occupied)
    assert answer == ("deferred" if occupied else "accepted" if accept else "capacity")
    assert adder.rem_chunk_tokens == (2048 - 4352 if accept and not occupied else 2048)
    assert adder.new_chunked_req is None
