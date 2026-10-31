"""Logical ownership and publication tests; no CUDA/model dependency."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from memt_sglang.protocol import MemoryRequest, ReceiptStatus, SessionError
from memt_sglang.runtime.features import commit_chunk_policy
from memt_sglang.runtime.plan import StepPlan, validate_commit_span
from memt_sglang.runtime.worker_sessions import WorkerSessions


class Cell:
    def __init__(self, value=0):
        self.value = value

    def copy_(self, other):
        self.value = other.value

    def zero_(self):
        self.value = 0


class Pool:
    def __init__(self):
        self.sessions = {}
        self.max_commits = 2
        self.free = [3, 2, 1, 0]
        self.storage = {4: [Cell() for _ in range(4)]}

    def take_slot(self):
        return self.free.pop()

    def return_slot(self, slot):
        self.free.append(slot)

    def ensure(self, session, source=None):
        self.sessions[session] = self.take_slot()
        if source is not None:
            self.storage[4][self.sessions[session]].copy_(
                self.storage[4][self.sessions[source]]
            )

    def release(self, session):
        self.return_slot(self.sessions.pop(session))


class PartialProgress(unittest.TestCase):
    def setup_request(self, policy, length):
        self.env = patch.dict(
            os.environ, {"MEMT_EXPERIMENT_COMMIT_CHUNK_POLICY": policy}
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        pool = Pool()
        w = WorkerSessions(pool)
        w.create("s")
        pool.storage[4][pool.sessions["s"]].value = 41
        w.has_memory["s"] = True
        w.begin("s", "tx")
        r = MemoryRequest("s", "commit", "n", length=length)
        w.prepare(r.rid, "tx")
        return pool, w, r

    def apply(self, pool, w, r, prefix, count):
        w.validate_span(r.rid, prefix, count)
        pool.storage[4][w.requests[r.rid].slot].value += count
        w.advance(r.rid, count, prefix=prefix)

    def test_global_fragments_publish_once(self):
        pool, w, r = self.setup_request("global", 4097)
        offset = 0
        for count in (777, 2048, 1272):
            self.apply(pool, w, r, offset, count)
            offset += count
            self.assertEqual(pool.storage[4][pool.sessions["s"]].value, 41)
            self.assertEqual(
                w.transactions["s"].receipts["n"],
                ReceiptStatus.APPLIED if offset == 4097 else ReceiptStatus.PENDING,
            )
            with self.assertRaises(SessionError):
                w.publish("s", "tx")
        w.fence("s", r.rid)
        w.publish("s", "tx")
        self.assertEqual(pool.storage[4][pool.sessions["s"]].value, 41 + 4097)

    def test_duplicate_and_gap_rejected_before_mutation(self):
        pool, w, r = self.setup_request("global", 4096)
        self.apply(pool, w, r, 0, 777)
        value = pool.storage[4][w.requests[r.rid].slot].value
        for prefix in (0, 776, 778, 2048):
            with self.assertRaises(SessionError):
                self.apply(pool, w, r, prefix, 10)
        self.assertEqual(pool.storage[4][w.requests[r.rid].slot].value, value)
        self.assertEqual(w.requests[r.rid].progress, 777)

    def test_cancelled_partial_cannot_publish_and_slot_reuse_starts_zero(self):
        pool, w, r = self.setup_request("global", 4096)
        self.apply(pool, w, r, 0, 777)
        slot = w.requests[r.rid].slot
        w.fence("s", r.rid)
        self.assertEqual(w.transactions["s"].receipts["n"], ReceiptStatus.ABORTED)
        with self.assertRaises(SessionError):
            w.publish("s", "tx")
        w.abort("s", "tx")
        self.assertEqual(pool.storage[4][pool.sessions["s"]].value, 41)
        w.begin("s", "tx2")
        r2 = MemoryRequest("s", "commit", "n2", length=2048)
        w.prepare(r2.rid, "tx2")
        self.assertEqual(w.requests[r2.rid].slot, slot)
        self.assertEqual(w.requests[r2.rid].progress, 0)
        self.apply(pool, w, r2, 0, 2048)
        w.fence("s")
        w.publish("s", "tx2")
        self.assertEqual(pool.storage[4][pool.sessions["s"]].value, 41 + 2048)

    def test_tail_policy(self):
        pool, w, r = self.setup_request("tail", 4096 + 777)
        with self.assertRaises(ValueError):
            self.apply(pool, w, r, 0, 1024)
        self.apply(pool, w, r, 0, 2048)
        self.apply(pool, w, r, 2048, 2048)
        self.apply(pool, w, r, 4096, 127)
        self.apply(pool, w, r, 4223, 650)
        self.assertTrue(w.requests[r.rid].complete)

    def test_legacy_atomic_remains_atomic(self):
        pool, w, r = self.setup_request("atomic", 4096)
        with self.assertRaises(ValueError):
            self.apply(pool, w, r, 0, 2048)
        self.apply(pool, w, r, 0, 4096)
        self.assertTrue(w.requests[r.rid].complete)
        with self.assertRaises(SessionError):
            w.advance(r.rid, 4096)

    def test_native_retraction_restores_base_not_already_written_prefix(self):
        pool, w, r = self.setup_request("global", 4096)
        self.apply(pool, w, r, 0, 777)
        w.mark_retracted(r.rid)
        with self.assertRaises(SessionError):
            w.validate_span(r.rid, 0, 2048)
        w.prepare_span(r.rid, 0, 1024)
        self.assertEqual(pool.storage[4][w.requests[r.rid].slot].value, 41)
        self.assertEqual(w.requests[r.rid].rewind_count, 1)
        self.apply(pool, w, r, 0, 1024)
        self.apply(pool, w, r, 1024, 3072)
        w.fence("s")
        w.publish("s", "tx")
        self.assertEqual(pool.storage[4][pool.sessions["s"]].value, 41 + 4096)

    def test_replay_base_keeps_prior_commits_in_same_transaction(self):
        pool, w, r = self.setup_request("global", 2048)
        self.apply(pool, w, r, 0, 2048)
        w.fence("s")
        r2 = MemoryRequest("s", "commit", "n2", length=4096)
        w.prepare(r2.rid, "tx")
        self.apply(pool, w, r2, 0, 777)
        w.mark_retracted(r2.rid)
        w.prepare_span(r2.rid, 0, 4096)
        self.assertEqual(pool.storage[4][w.requests[r2.rid].slot].value, 41 + 2048)
        self.apply(pool, w, r2, 0, 4096)
        w.fence("s")
        w.publish("s", "tx")
        self.assertEqual(pool.storage[4][pool.sessions["s"]].value, 41 + 2048 + 4096)
        self.assertEqual(len(pool.free), 3)

    def test_invalid_replay_does_not_consume_reset(self):
        pool, w, r = self.setup_request("tail", 4096 + 777)
        self.apply(pool, w, r, 0, 2048)
        w.mark_retracted(r.rid)
        with self.assertRaises(ValueError):
            w.prepare_span(r.rid, 0, 777)
        self.assertTrue(w.requests[r.rid].rewind_pending)
        self.assertEqual(w.requests[r.rid].progress, 2048)
        w.prepare_span(r.rid, 0, 2048)
        self.assertEqual(w.requests[r.rid].progress, 0)

    def test_invalid_spans_and_policy(self):
        r = MemoryRequest("s", "commit", "n", length=2048)
        for prefix, count in ((0, 0), (-1, 1), (0, True), (0, 2049), (2048, 1)):
            with self.assertRaises(ValueError):
                validate_commit_span(r, prefix, count, "global")
        with self.assertRaises(ValueError):
            commit_chunk_policy(
                environment={"MEMT_EXPERIMENT_COMMIT_CHUNK_POLICY": "typo"}
            )

    def test_planner_partial_prefix(self):
        mode = SimpleNamespace(
            is_idle=lambda: False, is_decode=lambda: False, is_extend=lambda: True
        )
        r = MemoryRequest("s", "commit", "n", length=4097)
        batch = SimpleNamespace(
            forward_mode=mode,
            rids=[r.rid],
            extend_seq_lens_cpu=[777],
            extend_prefix_lens_cpu=[2048],
            num_token_non_padded_cpu=777,
        )
        with patch.dict(os.environ, {"MEMT_EXPERIMENT_COMMIT_CHUNK_POLICY": "global"}):
            plan = StepPlan.from_batch(batch)
            self.assertEqual(plan.samples[0].prefix_length, 2048)
        with patch.dict(os.environ, {"MEMT_EXPERIMENT_COMMIT_CHUNK_POLICY": "atomic"}):
            with self.assertRaises(ValueError):
                StepPlan.from_batch(batch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
