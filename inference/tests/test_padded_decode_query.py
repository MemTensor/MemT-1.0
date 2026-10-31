"""Native padding, disabled reads and next-step state changes stay isolated."""

import torch

from memt_sglang.protocol import MemoryRequest
from memt_sglang.runtime.executor import MemoryRuntime
from memt_sglang.runtime.plan import Sample, StepPlan
from memt_sglang.runtime.state import StatePool


def test_padded_reader_preserves_rows_and_state_across_steps(monkeypatch):
    monkeypatch.setenv("MEMT_EXPERIMENT_PADDED_DECODE_QUERY", "1")
    monkeypatch.setenv("MEMT_EXPERIMENT_SLOT_READ", "1")
    rt = MemoryRuntime(
        StatePool(
            {0: (1, 1, 2, 2)},
            max_sessions_per_rank=4,
            max_inflight_commits_per_rank=1,
            device="cpu",
        )
    )
    ws = rt.worker_sessions
    for name in ("a", "disabled", "b"):
        ws.create(name, memory_read=name != "disabled")
    observed = []

    class Reader:
        def packed_query_slots(self, hidden, pool, slots, *, lengths):
            assert len(hidden) == 4 and lengths == (1,) * 4
            observed.append(slots.clone())
            # Invalid rows intentionally return NaN: the runtime must mask them.
            values = hidden.new_full(hidden.shape, float("nan"))
            for index, slot in enumerate(slots.tolist()):
                if slot >= 0:
                    values[index].fill_(float(pool[slot].sum()))
            return values, None

    def plan():
        samples = []
        for index, name in enumerate(("a", "disabled", "b")):
            request = MemoryRequest(name, "query", "q" + name)
            ws.prepare(request.rid)
            samples.append(Sample(request, index, index + 1, 0, True))
        rt.plan = StepPlan(tuple(samples), 3)

    rt.pool.storage[0].fill_(2.0)
    before = rt.pool.storage[0].clone()
    for value in (2.0, 3.0):
        rt.pool.storage[0][rt.pool.sessions["b"]].fill_(value)
        plan()
        output = torch.empty(4, 2)
        rt.run_layer(0, torch.zeros(4, 2), output, Reader())
        assert torch.equal(
            output,
            torch.tensor([[8.0, 8.0], [0.0, 0.0], [4 * value, 4 * value], [0.0, 0.0]]),
        )
        expected = before.clone()
        expected[rt.pool.sessions["b"]].fill_(value)
        assert torch.equal(rt.pool.storage[0], expected)
        rt.finish(True)
        for name in ("a", "disabled", "b"):
            ws.fence(name)
    assert observed[0].tolist() == observed[1].tolist()
    assert observed[0][1].item() == observed[0][3].item() == -1
