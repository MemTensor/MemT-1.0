"""Integration of queue re-entry, exact-fit admission, cancellation and replay."""

import os
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from memt_sglang.protocol import MemoryRequest
from memt_sglang.runtime.partial_scheduler import (
    fragment_tokens,
    install_partial_prefill,
)


class SchedulerContract(unittest.TestCase):
    def install(
        self, policy, budget=3072, maximum=2048, capacity=100000, swa_capacity=None
    ):
        self.env = patch.dict(
            os.environ,
            {
                "MEMT_EXPERIMENT_COMMIT_CHUNK_POLICY": policy,
                "MEMT_PARTIAL_MAX_CHUNK_TOKENS": str(maximum),
                "MEMT_PARTIAL_QUERY_TOKEN_RESERVE": "0",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        freed = []
        retracted = []

        class Req:
            def __init__(self, name, total):
                self.rid = MemoryRequest(name, "commit", "n", length=total).rid
                self.full_untruncated_fill_ids = list(range(total))
                self.prefix_indices = []
                self.req_pool_idx = None
                self.kv = None
                self.inflight_middle_chunks = 0
                self._done = False

            def finished(self):
                return self._done

            def init_next_round_input(self, tree_cache=None):
                if tree_cache is not None:
                    self.prefix_indices = []

            def set_extend_range(self, start, end):
                self.extend_range = NS(start=start, end=end, length=end - start)

            def reset_for_retract(self):
                self.prefix_indices = []
                self.kv = None
                self.req_pool_idx = None

        class Batch:
            def __init__(self, reqs):
                self.reqs = reqs
                self.forward_mode = NS(is_extend=lambda: True)

            def filter_batch(self, *, chunked_req_to_exclude):
                self.reqs = [r for r in self.reqs if r not in chunked_req_to_exclude]

            def _evict_swa(self, req, end):
                req.swa_evicted_to = end

        class Adder:
            def __init__(self, scheduler):
                self.rem_input_tokens = budget
                self.rem_chunk_tokens = budget
                self.rem_total_token_offset = 0
                self.can_run_list = []
                self.is_hybrid_swa = swa_capacity is not None
                self.page_size = 64
                self.rem_swa_tokens = swa_capacity

            def ceil_paged_tokens(self, n):
                return (n + 63) // 64 * 64

            def budget_state(self):
                return (
                    "continue"
                    if self.rem_chunk_tokens is None or self.rem_chunk_tokens > 0
                    else "other"
                )

            def add_one_req(self, req, *args, **kwargs):
                count = len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
                cost = self.ceil_paged_tokens(count)
                if self.can_run_list and cost >= self.rem_input_tokens:
                    return "other"
                req.set_extend_range(
                    len(req.prefix_indices), len(req.full_untruncated_fill_ids)
                )
                self.can_run_list.append(req)
                self.rem_input_tokens -= cost
                return self.budget_state()

        class Scheduler:
            def __init__(self):
                self.waiting_queue = []
                self.tree_cache = NS(
                    supports_swa=lambda: swa_capacity is not None,
                    is_chunk_cache=lambda: True,
                    sliding_window_size=512,
                )
                self.max_running_requests = 32
                self.swa_tokens_per_layer = swa_capacity
                self.chunked_prefill_size = budget
                self.page_size = 64
                self.req_to_token_pool = NS(available_size=lambda: 32)

                def available():
                    return capacity - sum(
                        r.kv.kv_allocated_len
                        for r, _ in getattr(self, "_memt_partial_records", {}).values()
                        if r.kv is not None
                    )

                self.token_to_kv_pool_allocator = NS(
                    available_size=available, full_available_size=available
                )
                sessions = NS(mark_retracted=lambda rid: retracted.append(rid))
                self.tp_worker = NS(
                    model_runner=NS(model=NS(memt_runtime=NS(worker_sessions=sessions)))
                )

            def _get_new_batch_prefill_raw(self, *, running_batch=None):
                adder = Adder(self)
                for r in self.waiting_queue:
                    r.init_next_round_input(self.tree_cache)
                    result = adder.add_one_req(r)
                    if result != "continue":
                        break
                self.waiting_queue = [
                    r for r in self.waiting_queue if r not in adder.can_run_list
                ]
                return Batch(adder.can_run_list)

            def get_next_batch_to_run(self, running_batch, last_batch):
                return self._get_new_batch_prefill_raw()

            def stash_chunked_request(self, req):
                req.prefix_indices = list(range(req.extend_range.end))

            def get_num_allocatable_reqs(self, running_bs):
                return 32 - running_bs

            def abort_request(self, message):
                self.waiting_queue = [
                    r
                    for r in self.waiting_queue
                    if not (message.abort_all or r.rid.startswith(message.rid))
                ]

        def release(req, *args, **kwargs):
            freed.append(req.rid)
            req.kv = None
            req.req_pool_idx = None

        mocks = {
            "sglang.srt.managers.scheduler": NS(Scheduler=Scheduler),
            "sglang.srt.managers.schedule_policy": NS(
                PrefillAdder=Adder, AddReqResult=NS(OTHER="other", NO_TOKEN="no-token")
            ),
            "sglang.srt.managers.schedule_batch": NS(Req=Req),
            "sglang.srt.mem_cache.common": NS(release_kv_cache=release),
        }
        self.modules = patch.dict(sys.modules, mocks)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        install_partial_prefill()
        return Req, Scheduler(), NS(batch_is_full=False), freed, retracted

    def finish_step(self, batch, spans):
        for r in batch.reqs:
            spans.setdefault(r.rid, []).append(
                (r.extend_range.start, r.extend_range.length)
            )
            r.req_pool_idx = 1
            r.kv = NS(kv_allocated_len=r.extend_range.end)
            if r.inflight_middle_chunks:
                r.inflight_middle_chunks -= 1
            else:
                r._done = True
                r.kv = None
                r.req_pool_idx = None

    def test_global_uses_budget_and_all_requests_progress(self):
        Req, s, running, _, _ = self.install("global")
        reqs = [Req("s" + str(i), 4873) for i in range(3)]
        s.waiting_queue = list(reqs)
        last = None
        spans = {}
        for _ in range(40):
            last = s.get_next_batch_to_run(running, last)
            self.assertLessEqual(sum(r.extend_range.length for r in last.reqs), 3072)
            self.finish_step(last, spans)
            if all(r.finished() for r in reqs):
                break
        self.assertTrue(all(r.finished() for r in reqs))
        for r in reqs:
            cursor = 0
            for prefix, count in spans[r.rid]:
                self.assertEqual(prefix, cursor)
                cursor += count
            self.assertEqual(cursor, 4873)
        self.assertEqual(spans[reqs[1].rid][0], (0, 1024))

    def test_tail_keeps_ordinary_blocks_and_batches_tails(self):
        Req, s, running, _, _ = self.install("tail")
        reqs = [Req("s" + str(i), 4873) for i in range(3)]
        s.waiting_queue = list(reqs)
        last = None
        spans = {}
        for _ in range(40):
            last = s.get_next_batch_to_run(running, last)
            self.finish_step(last, spans)
            if all(r.finished() for r in reqs):
                break
        self.assertTrue(all(r.finished() for r in reqs))
        for items in spans.values():
            self.assertEqual(items[:2], [(0, 2048), (2048, 2048)])
            self.assertEqual(sum(n for p, n in items if p >= 4096), 777)

    def test_parked_cancel_releases_kv_and_retraction_marks_owner(self):
        Req, s, running, freed, retracted = self.install("global")
        r = Req("s", 8192)
        s.waiting_queue = [r]
        first = s.get_next_batch_to_run(running, None)
        self.finish_step(first, {})
        # Stash it, then temporarily remove token budget by retaining the queue.
        s.stash_chunked_request(r)
        r._memt_resume = True
        s.waiting_queue = [r]
        r.reset_for_retract()
        self.assertEqual(retracted, [r.rid])
        self.assertFalse(r._memt_resume)
        r.req_pool_idx = 1
        r.kv = NS(kv_allocated_len=2048)
        r._memt_resume = True
        s.abort_request(NS(abort_all=False, rid=r.rid))
        self.assertEqual(freed, [r.rid])
        self.assertEqual(s.waiting_queue, [])

    def test_cancel_between_result_and_requeue_releases_once(self):
        Req, s, running, freed, _ = self.install("global")
        r = Req("s", 8192)
        s.waiting_queue = [r]
        first = s.get_next_batch_to_run(running, None)
        self.finish_step(first, {})
        s.last_batch = first
        s.running_batch = NS(reqs=[])
        s.abort_request(NS(abort_all=False, rid=r.rid))
        self.assertEqual(freed, [r.rid])
        self.assertEqual(first.reqs, [])
        self.assertEqual(s.waiting_queue, [])

    def test_full_context_credit_does_not_deadlock_behind_fresh_request(self):
        Req, s, running, _, _ = self.install(
            "global", budget=512, maximum=512, capacity=1600
        )
        reqs = [Req("s" + str(i), 1024) for i in range(2)]
        s.waiting_queue = list(reqs)
        last = None
        spans = {}
        for _ in range(10):
            last = s.get_next_batch_to_run(running, last)
            self.finish_step(last, spans)
            if all(r.finished() for r in reqs):
                break
        self.assertTrue(all(r.finished() for r in reqs))
        self.assertEqual(spans[reqs[0].rid], [(0, 512), (512, 512)])

    def test_arbitrary_final_tail_and_invalid_limits(self):
        self.assertEqual(fragment_tokens("tail", 4097, 4096, 1), 1)
        self.assertEqual(fragment_tokens("global", 4097, 777, 113), 113)
        self.assertEqual(fragment_tokens("tail", 4097, 0, 2047), 0)
        with self.assertRaises(ValueError):
            fragment_tokens("tail", 4097, 1, 2048)

    def test_swa_holds_windows_not_the_entire_logical_document(self):
        Req, s, running, _, _ = self.install(
            "global", budget=2048, maximum=2048, capacity=300000, swa_capacity=4096
        )
        r = Req("long", 65536)
        s.waiting_queue = [r]
        first = s.get_next_batch_to_run(running, None)
        self.assertEqual(first.reqs, [r])
        self.finish_step(first, {})
        second = s.get_next_batch_to_run(running, first)
        self.assertEqual(r.swa_evicted_to, 2048)
        self.assertEqual(second.reqs, [r])
        self.assertEqual(r.extend_range.start, 2048)

    def test_decode_credit_preserves_queue_and_clears_after_query_finishes(self):
        Req, s, _, _, _ = self.install("global")
        commit = Req("writer", 8192)
        query = Req("reader", 1)
        query.rid = MemoryRequest("reader", "query", "n").rid
        running = NS(reqs=[query])
        s.waiting_queue = [commit]
        s._memt_partial_decode_credit = 2
        self.assertEqual(
            s._get_new_batch_prefill_raw(running_batch=running), (None, running)
        )
        self.assertEqual(s.waiting_queue, [commit])
        self.assertEqual(s._memt_partial_decode_credit, 1)
        query._done = True
        admitted = s._get_new_batch_prefill_raw(running_batch=running)
        self.assertEqual(s._memt_partial_decode_credit, 0)
        self.assertEqual(admitted.reqs, [commit])

    def test_query_can_bypass_a_fresh_commit_blocked_on_context_credit(self):
        Req, s, running, _, _ = self.install(
            "global", budget=512, maximum=512, capacity=1600
        )
        with patch.dict(os.environ, {"MEMT_PARTIAL_QUERY_FIRST": "1"}):
            # Reinstall against a fresh class so the opt-in policy is captured.
            Req, s, running, _, _ = self.install(
                "global", budget=512, maximum=512, capacity=1600
            )
        first, blocked = Req("held", 1024), Req("blocked", 1024)
        s.waiting_queue = [first, blocked]
        last = s.get_next_batch_to_run(running, None)
        self.finish_step(last, {})
        reader = Req("reader", 1)
        reader.rid = MemoryRequest("reader", "query", "n").rid
        s.waiting_queue.append(reader)
        admitted = s.get_next_batch_to_run(running, last)
        self.assertIs(admitted.reqs[0], reader)
        self.assertIn(blocked, s.waiting_queue)

    def test_queries_charge_pending_native_request_slots(self):
        Req, s, running, _, _ = self.install("global")
        s.req_to_token_pool.available_size = lambda: 1
        queries = [Req("q" + str(i), 1) for i in range(2)]
        for i, req in enumerate(queries):
            req.rid = MemoryRequest("q" + str(i), "query", "n").rid
        s.waiting_queue = list(queries)
        admitted = s.get_next_batch_to_run(running, None)
        self.assertEqual(admitted.reqs, queries[:1])
        self.assertEqual(s.waiting_queue, queries[1:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
