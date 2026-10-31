"""Scheduling protocol tests, without model execution or GPU imports."""

import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from memt_sglang.runtime import query_coalescing as m


class CoalescingTest(unittest.TestCase):
    def test_new_arrivals_do_not_extend_deadline(self):
        policy = m.QueryCoalescer(m.CoalescingConfig(5))
        first = policy.decide([("a", 512)], 10)
        later = policy.decide([("a", 512), ("b", 512)], 10.004)
        self.assertEqual(first["deadline"], later["deadline"])
        self.assertTrue(later["wait"])
        self.assertEqual(
            policy.decide([("a", 512), ("b", 512)], 10.006)["reason"], "deadline"
        )

    def test_cancelled_request_removed_without_resetting_survivor(self):
        policy = m.QueryCoalescer(m.CoalescingConfig(5))
        policy.decide([("a", 512)], 10)
        policy.decide([("a", 512), ("b", 512)], 10.002)
        decision = policy.decide([("b", 512), ("c", 512)], 10.004)
        self.assertAlmostEqual(decision["deadline"], 10.007)
        self.assertNotIn("a", policy.first_seen)
        policy.clear()
        self.assertEqual(policy.first_seen, {})

    def test_release_thresholds_do_not_cap_native_batch(self):
        policy = m.QueryCoalescer(m.CoalescingConfig(5))
        decision = policy.decide([(str(i), 512) for i in range(32)], 10)
        self.assertEqual(
            (decision["reason"], decision["requests"]), ("request_target", 32)
        )
        self.assertEqual(policy.decide([("long", 8192)], 11)["reason"], "token_target")

    def scheduler(self, rank=0):
        class Scheduler:
            def _get_new_batch_prefill_raw(
                self, prefill_delayer_single_pass, running_batch
            ):
                self.native_calls += 1
                return self.waiting_queue, running_batch

            def on_idle(self):
                self.native_idle_calls += 1

        m.install_query_coalescing(Scheduler, m.CoalescingConfig(5))
        first = Scheduler.on_idle
        m.install_query_coalescing(Scheduler, m.CoalescingConfig(5))
        self.assertIs(first, Scheduler.on_idle)
        s = Scheduler()
        s.native_calls = s.native_idle_calls = 0
        s.waiting_queue = [NS(rid="a", origin_input_ids=[0] * 512)]
        records = {"a": NS(complete=False, request=NS(operation="query"))}
        s.tp_worker = NS(
            model_runner=NS(
                model=NS(memt_runtime=NS(worker_sessions=NS(requests=records)))
            )
        )
        s.running_batch = NS(is_empty=lambda: True, batch_is_full=False)
        s.chunked_req = None
        s.result_queue = []
        s.last_batch = None
        s.ps = NS(attn_tp_size=2, attn_tp_rank=rank)
        s.require_mlp_sync = False
        return s, records

    def call(self, s):
        return s._get_new_batch_prefill_raw(None, s.running_batch)

    def test_idle_wait_uses_event_poll_once_and_keeps_queue(self):
        s, _ = self.scheduler()
        original_queue = s.waiting_queue
        with (
            patch.object(m, "_broadcast_decision", side_effect=lambda _, d: d),
            patch.object(m.time, "monotonic", return_value=10),
            patch.object(m, "_wait_for_input") as wait,
        ):
            self.assertEqual(self.call(s), (None, s.running_batch))
            self.assertEqual(s.native_calls, 0)
            s.on_idle()
            self.assertEqual(wait.call_count, 1)
            self.assertIn(wait.call_args.args[1], (5, 6))
            s.on_idle()
            self.assertEqual(wait.call_count, 1)
        self.assertIs(s.waiting_queue, original_queue)

    def test_expired_deadline_never_sleeps(self):
        s, _ = self.scheduler()
        s._memt_query_wait_deadline = 10
        with (
            patch.object(m.time, "monotonic", return_value=11),
            patch.object(m, "_wait_for_input") as wait,
        ):
            s.on_idle()
            wait.assert_not_called()

    def test_follower_uses_leader_decision_without_local_clock_or_poll(self):
        s, _ = self.scheduler(rank=1)
        with (
            patch.object(
                m,
                "_broadcast_decision",
                return_value={"wait": True, "deadline": 10.005},
            ) as broadcast,
            patch.object(
                m.time, "monotonic", side_effect=AssertionError("Follower read clock")
            ),
            patch.object(m, "_wait_for_input") as wait,
        ):
            self.assertEqual(self.call(s), (None, s.running_batch))
            broadcast.assert_called_once_with(s, None)
            s.on_idle()
            wait.assert_not_called()

    def test_active_work_and_control_requests_bypass_wait(self):
        for condition in (
            "decode",
            "pending_result",
            "chunk",
            "full",
            "commit",
            "unprepared",
            "complete",
            "empty",
        ):
            with self.subTest(condition=condition):
                s, records = self.scheduler()
                if condition == "decode":
                    s.running_batch.is_empty = lambda: False
                if condition == "pending_result":
                    s.result_queue.append(object())
                if condition == "chunk":
                    s.chunked_req = object()
                if condition == "full":
                    s.running_batch.batch_is_full = True
                if condition == "commit":
                    records["a"].request.operation = "commit"
                if condition == "complete":
                    records["a"].complete = True
                if condition == "unprepared":
                    records.clear()
                if condition == "empty":
                    s.waiting_queue.clear()
                with (
                    patch.object(m, "_broadcast_decision") as broadcast,
                    patch.object(m, "_wait_for_input") as wait,
                ):
                    self.assertEqual(self.call(s), (s.waiting_queue, s.running_batch))
                    self.assertEqual(s.native_calls, 1)
                    s.on_idle()
                    broadcast.assert_not_called()
                    wait.assert_not_called()

    def test_deadline_releases_through_native_admission(self):
        s, _ = self.scheduler()
        with (
            patch.object(m, "_broadcast_decision", side_effect=lambda _, d: d),
            patch.object(m.time, "monotonic", side_effect=[10, 10.006]),
        ):
            self.call(s)
            self.assertEqual(self.call(s), (s.waiting_queue, s.running_batch))
            self.assertEqual(s.native_calls, 1)
            self.assertIsNone(s._memt_query_wait_deadline)

    def test_poll_observes_both_ingress_sockets_without_receiving(self):
        events = []

        class Poller:
            def register(self, socket, flag):
                events.append(("register", socket, flag))

            def poll(self, timeout):
                events.append(("poll", timeout))

        s = NS(
            request_receiver=NS(recv_from_tokenizer="tokenizer", recv_from_rpc="rpc")
        )
        with patch.dict(sys.modules, {"zmq": NS(Poller=Poller, POLLIN=1)}):
            m._wait_for_input(s, 5)
        self.assertEqual(
            events, [("register", "tokenizer", 1), ("register", "rpc", 1), ("poll", 5)]
        )

    def test_invalid_configuration_is_not_silently_ignored(self):
        for value in (-1, 0, float("nan"), float("inf"), 11):
            with self.assertRaises(ValueError):
                m.CoalescingConfig(value)
        with patch.dict(
            m.os.environ, {m.PREFIX + "MS": "0", m.PREFIX + "REQUESTS": "4"}, clear=True
        ):
            with self.assertRaises(ValueError):
                m.config_from_env()

    def test_dp_global_release_and_busy_rank(self):
        cfg = m.CoalescingConfig(5)
        empty = [0, 0, 0, 0]
        self.assertTrue(
            m.dp_decision([[0, 1, 512, 3], empty, empty, empty], cfg)["wait"]
        )
        self.assertEqual(
            m.dp_decision([[0, 1, 512, 3]] * 4, cfg)["reason"], "request_target"
        )
        self.assertEqual(
            m.dp_decision([[0, 1, 8192, 3], empty, empty, empty], cfg)["reason"],
            "token_target",
        )
        self.assertEqual(
            m.dp_decision([[0, 1, 512, 0], empty, empty, empty], cfg)["reason"],
            "deadline",
        )
        self.assertEqual(
            m.dp_decision([[0, 1, 512, 3], [1, 0, 0, 0], empty, empty], cfg)["reason"],
            "bypass",
        )
        self.assertEqual(m.dp_decision([empty] * 4, cfg)["reason"], "bypass")

    def test_dp_empty_rank_still_votes_and_waits(self):
        s, _ = self.scheduler()
        s.require_mlp_sync = True
        s.ps = NS(attn_tp_size=1, attn_tp_rank=0, dp_rank=2)
        s.server_args = NS(dp_size=4)
        s.waiting_queue.clear()
        with (
            patch.object(
                m,
                "_exchange_dp_observation",
                return_value=[[0, 1, 512, 3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ) as exchange,
            patch.object(m.time, "monotonic", return_value=10),
            patch.object(m, "_wait_for_input") as poll,
        ):
            self.assertEqual(self.call(s), (None, s.running_batch))
            exchange.assert_called_once_with(s, [0, 0, 0, 0])
            self.assertAlmostEqual(s._memt_query_wait_deadline, 10.003)
            s.on_idle()
            poll.assert_called_once()

    def test_dp_commit_on_any_rank_bypasses_wait(self):
        s, records = self.scheduler()
        s.require_mlp_sync = True
        s.ps = NS(attn_tp_size=1, attn_tp_rank=0, dp_rank=0)
        s.server_args = NS(dp_size=4)
        records["a"].request.operation = "commit"

        def exchange(_, local):
            self.assertEqual(local, [1, 0, 0, 0])
            return [local, [0, 1, 512, 5], [0, 0, 0, 0], [0, 0, 0, 0]]

        with patch.object(m, "_exchange_dp_observation", side_effect=exchange):
            self.assertEqual(self.call(s), (s.waiting_queue, s.running_batch))
            self.assertIsNone(s._memt_query_wait_deadline)
            self.assertEqual(s.native_calls, 1)

    def test_dp_previous_global_work_skips_extra_collective(self):
        s, _ = self.scheduler()
        s.require_mlp_sync = True
        s.ps = NS(attn_tp_size=1, attn_tp_rank=0, dp_rank=2)
        s.server_args = NS(dp_size=4)
        # This can be a native idle placeholder on a locally empty rank.
        s.last_batch = object()
        s.waiting_queue.clear()
        with patch.object(m, "_exchange_dp_observation") as exchange:
            self.assertEqual(self.call(s), (s.waiting_queue, s.running_batch))
            exchange.assert_not_called()
            self.assertIsNone(s._memt_query_wait_deadline)

    def test_dp8_empty_rank_joins_global_deadline(self):
        s, _ = self.scheduler()
        s.require_mlp_sync = True
        s.ps = NS(attn_tp_size=1, attn_tp_rank=0, dp_rank=7)
        s.server_args = NS(dp_size=8)
        s.waiting_queue.clear()
        observations = [[0, 1, 512, 2]] + [[0, 0, 0, 0]] * 7
        with (
            patch.object(
                m, "_exchange_dp_observation", return_value=observations
            ) as exchange,
            patch.object(m.time, "monotonic", return_value=10),
        ):
            self.assertEqual(self.call(s), (None, s.running_batch))
            exchange.assert_called_once_with(s, [0, 0, 0, 0])
            self.assertAlmostEqual(s._memt_query_wait_deadline, 10.002)

    def test_dp8_exchange_includes_rank_seven(self):
        class Values:
            def __init__(self, shape):
                self.rows = [[0] * shape[1] for _ in range(shape[0])]

            def __setitem__(self, index, value):
                self.rows[index] = value

            def tolist(self):
                return self.rows

        group = NS(world_size=8, rank_in_group=7, cpu_group=object())
        reduced = []
        torch = NS(
            float64="float64",
            zeros=lambda shape, **kw: Values(shape),
            tensor=lambda data, **kw: list(data),
            distributed=NS(all_reduce=lambda values, **kw: reduced.append(values)),
        )
        with patch.dict(
            sys.modules,
            {
                "torch": torch,
                "sglang.srt.distributed.parallel_state": NS(
                    get_world_group=lambda: group
                ),
            },
        ):
            rows = m._exchange_dp_observation(
                NS(server_args=NS(dp_size=8)), [0, 1, 512, 2]
            )
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[7], [0, 1, 512, 2])
            self.assertEqual(len(reduced), 1)
            with self.assertRaisesRegex(RuntimeError, "one attention-DP rank"):
                m._exchange_dp_observation(NS(server_args=NS(dp_size=4)), [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
