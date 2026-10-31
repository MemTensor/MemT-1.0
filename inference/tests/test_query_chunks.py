"""Admission protocol tests; commits must not enter native truncation."""

import sys
from types import SimpleNamespace

import pytest

from memt_sglang.protocol import MemoryRequest
from memt_sglang.runtime.query_chunks import install_atomic_commit_guard


@pytest.mark.parametrize(
    "budget,accepted",
    [(0, True), (1536, True), (2048, True), (4096, True), (None, True)],
)
def test_empty_prefill_admits_one_whole_release_commit(monkeypatch, budget, accepted):
    class Adder:
        def __init__(self):
            self.rem_chunk_tokens = budget
            self.can_run_list = []
            self.new_chunked_req = None
            self.calls = 0

        def ceil_paged_tokens(self, n):
            return (n + 63) // 64 * 64

        def add_one_req(self, req, *args, **kwargs):
            self.calls += 1
            req.extend_range = SimpleNamespace(
                length=len(req.full_untruncated_fill_ids)
            )
            self.can_run_list.append(req)
            return "accepted"

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.schedule_policy",
        SimpleNamespace(
            PrefillAdder=Adder, AddReqResult=SimpleNamespace(OTHER="deferred")
        ),
    )
    install_atomic_commit_guard()
    installed = Adder.add_one_req
    install_atomic_commit_guard()
    assert Adder.add_one_req is installed
    req = SimpleNamespace(
        rid=MemoryRequest("session", "commit", "nonce", length=2048).rid,
        prefix_indices=[],
        full_untruncated_fill_ids=[1] * 2048,
    )
    a = Adder()
    result = a.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
    assert (result == "accepted") is accepted
    assert a.calls == int(accepted)
    assert len(req.full_untruncated_fill_ids) == 2048
    assert a.new_chunked_req is None
