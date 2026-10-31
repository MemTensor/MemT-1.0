"""Bounded query coalescing at idle prefill boundaries, Nano TP2 / Flash DP4.

The request/token targets release a batch; they do not cap the native batch.
Native admission, cancellation, state ownership, and commit execution stay intact.
"""

import logging
import math
import os
import time
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)
PREFIX = "MEMT_EXPERIMENT_QUERY_COALESCE_"


@dataclass(frozen=True)
class CoalescingConfig:
    wait_ms: float
    target_requests: int = 4
    target_tokens: int = 4096

    def __post_init__(self):
        if not math.isfinite(self.wait_ms) or not 0 < self.wait_ms <= 10:
            raise ValueError("Query coalescing wait must be finite and in (0, 10] ms")
        if type(self.target_requests) is not int or not 2 <= self.target_requests <= 32:
            raise ValueError("Query coalescing request target must be in [2, 32]")
        if type(self.target_tokens) is not int or not 1 <= self.target_tokens <= 32768:
            raise ValueError("Query coalescing token target must be in [1, 32768]")


def config_from_env():
    wait_ms = float(os.environ.get(PREFIX + "MS", "0"))
    if wait_ms == 0:
        if any(PREFIX + name in os.environ for name in ("REQUESTS", "TOKENS")):
            raise ValueError("Query coalescing targets require a positive wait")
        return None
    return CoalescingConfig(
        wait_ms,
        int(os.environ.get(PREFIX + "REQUESTS", "4")),
        int(os.environ.get(PREFIX + "TOKENS", "4096")),
    )


class QueryCoalescer:
    def __init__(self, config):
        self.config = config
        self.first_seen = {}
        self.last_release = None
        self.poll_calls = 0
        self.poll_wall_ms = self.poll_cpu_ms = 0.0

    def clear(self):
        self.first_seen.clear()
        self.last_release = None
        self.poll_calls = 0
        self.poll_wall_ms = self.poll_cpu_ms = 0.0

    def decide(self, requests, now):
        # Prune removed/cancelled IDs; arrivals cannot extend an older deadline.
        self.first_seen = {rid: self.first_seen.get(rid, now) for rid, _ in requests}
        count, tokens = len(requests), sum(length for _, length in requests)
        elapsed_ms = (now - min(self.first_seen.values())) * 1000
        if count >= self.config.target_requests:
            reason = "request_target"
        elif tokens >= self.config.target_tokens:
            reason = "token_target"
        elif elapsed_ms >= self.config.wait_ms:
            reason = "deadline"
        else:
            reason = "wait"
        return dict(
            wait=reason == "wait",
            reason=reason,
            requests=count,
            deadline=min(self.first_seen.values()) + self.config.wait_ms / 1000,
            tokens=tokens,
            elapsed_ms=elapsed_ms,
        )


def _broadcast_decision(scheduler, decision):
    import torch

    values = [decision]
    # This experiment is guarded to TP2/DP1/PP1. Rank 0 owns the wall clock;
    # using independent rank clocks can diverge before model collectives.
    torch.distributed.broadcast_object_list(
        values, src=0, group=scheduler.attn_tp_cpu_group
    )
    return values[0]


def _wait_for_input(scheduler, timeout_ms):
    import zmq

    # Watch both native ingress sockets, including abort and control RPCs.
    # Poll only reports readiness; the native receiver still owns all reads.
    poller = zmq.Poller()
    receiver = scheduler.request_receiver
    for sock in (receiver.recv_from_tokenizer, receiver.recv_from_rpc):
        if sock is not None:
            poller.register(sock, zmq.POLLIN)
    poller.poll(timeout_ms)


def dp_decision(observations, config):
    """One decision from [busy, query count, input tokens, remaining ms] per rank."""
    count = sum(row[1] for row in observations)
    tokens = sum(row[2] for row in observations)
    remaining = min((row[3] for row in observations if row[1]), default=0)
    if any(row[0] for row in observations) or not count:
        reason = "bypass"
    elif count >= config.target_requests:
        reason = "request_target"
    elif tokens >= config.target_tokens:
        reason = "token_target"
    elif remaining <= 0:
        reason = "deadline"
    else:
        reason = "wait"
    return dict(
        wait=reason == "wait",
        reason=reason,
        requests=int(count),
        tokens=int(tokens),
        remaining_ms=max(0, remaining),
    )


def _exchange_dp_observation(scheduler, observation):
    import torch
    from sglang.srt.distributed.parallel_state import get_world_group

    group = get_world_group()
    # Every DP rank enters once per scheduling iteration, including empty ranks.
    # This is CPU scheduling metadata, independent of TP1 memory computation.
    world = scheduler.server_args.dp_size
    if group.world_size != world:
        raise RuntimeError(
            "Flash coalescing requires one attention-DP rank per world rank"
        )
    values = torch.zeros((world, 4), dtype=torch.float64)
    values[group.rank_in_group] = torch.tensor(observation, dtype=torch.float64)
    torch.distributed.all_reduce(values, group=group.cpu_group)
    return values.tolist()


def _dp_prefill(scheduler, native, config, prefill_delayer_single_pass, running_batch):
    if scheduler.ps.attn_tp_size != 1 or scheduler.server_args.dp_size not in (4, 8):
        raise RuntimeError(
            "DP query coalescing requires Flash attention TP1 / DP4 or DP8"
        )
    policy = getattr(scheduler, "_memt_query_coalescer", None)
    if policy is None:
        policy = scheduler._memt_query_coalescer = QueryCoalescer(config)
    # Native DP attention gives every rank an idle placeholder when any rank
    # executes a forward. Therefore last_batch is None on all ranks together.
    # Reuse that global fact while busy instead of adding a CPU collective to
    # every decode token; coalescing only starts after a globally idle iteration.
    if scheduler.last_batch is not None:
        policy.clear()
        return native(scheduler, prefill_delayer_single_pass, running_batch)
    busy = (
        not running_batch.is_empty()
        or running_batch.batch_is_full
        or scheduler.chunked_req is not None
        or bool(getattr(scheduler, "result_queue", ()))
    )
    requests = []
    records = (
        scheduler.tp_worker.model_runner.model.memt_runtime.worker_sessions.requests
    )
    for req in scheduler.waiting_queue:
        record = records.get(req.rid)
        if record is None or record.complete or record.request.operation != "query":
            busy = True
            break
        requests.append((req.rid, len(req.origin_input_ids)))
    now = time.monotonic()
    if busy or not requests:
        policy.clear()
        observation = [int(busy), 0, 0, 0]
    else:
        local = policy.decide(requests, now)
        observation = [
            0,
            len(requests),
            local["tokens"],
            (local["deadline"] - now) * 1000,
        ]
    decision = dp_decision(_exchange_dp_observation(scheduler, observation), config)
    if decision["wait"]:
        # Count collective time against the budget. Each rank polls its own
        # ingress; the next metadata exchange reunites ranks before admission.
        scheduler._memt_query_wait_deadline = now + decision["remaining_ms"] / 1000
        return None, running_batch
    if decision["reason"] == "bypass":
        policy.clear()
    elif scheduler.ps.dp_rank == 0:
        logger.info(
            "MemT DP query coalescing: reason=%s requests=%d tokens=%d max_wait_ms=%.3f",
            decision["reason"],
            decision["requests"],
            decision["tokens"],
            config.wait_ms,
        )
    return native(scheduler, prefill_delayer_single_pass, running_batch)


def install_query_coalescing(scheduler_class, config):
    native = scheduler_class._get_new_batch_prefill_raw
    if getattr(native, "_memt_query_coalescing", False):
        return
    native_idle = scheduler_class.on_idle

    @wraps(native)
    def prefill(self, prefill_delayer_single_pass, running_batch):
        self._memt_query_wait_deadline = None
        if self.require_mlp_sync:
            return _dp_prefill(
                self, native, config, prefill_delayer_single_pass, running_batch
            )
        policy = getattr(self, "_memt_query_coalescer", None)
        if policy is None:
            policy = self._memt_query_coalescer = QueryCoalescer(config)
        queue = self.waiting_queue
        # Coalesce only at idle boundaries. During decode, admit prefills without
        # an extra scheduler iteration (an iteration can exceed the wait budget).
        bypass = (
            not queue
            or not running_batch.is_empty()
            or running_batch.batch_is_full
            or self.chunked_req is not None
            or bool(getattr(self, "result_queue", ()))
        )
        requests = []
        if not bypass:
            records = (
                self.tp_worker.model_runner.model.memt_runtime.worker_sessions.requests
            )
            for req in queue:
                record = records.get(req.rid)
                if (
                    record is None
                    or record.complete
                    or record.request.operation != "query"
                ):
                    bypass = True
                    break
                requests.append((req.rid, len(req.origin_input_ids)))
        if bypass:
            policy.clear()
        else:
            if self.ps.attn_tp_size != 2 or self.require_mlp_sync:
                raise RuntimeError(
                    "Experimental query coalescing requires Nano TP2/DP1"
                )
            leader = self.ps.attn_tp_rank == 0
            decision = policy.decide(requests, time.monotonic()) if leader else None
            decision = _broadcast_decision(self, decision)
            if decision["wait"]:
                if leader:
                    self._memt_query_wait_deadline = decision["deadline"]
                return None, running_batch
            release = tuple(rid for rid, _ in requests)
            if leader and policy.last_release != release:
                policy.last_release = release
                logger.info(
                    "MemT query coalescing: reason=%s requests=%d tokens=%d wait_ms=%.3f poll_calls=%d poll_wall_ms=%.3f poll_cpu_ms=%.3f",
                    decision["reason"],
                    decision["requests"],
                    decision["tokens"],
                    decision["elapsed_ms"],
                    policy.poll_calls,
                    policy.poll_wall_ms,
                    policy.poll_cpu_ms,
                )
        return native(self, prefill_delayer_single_pass, running_batch)

    @wraps(native_idle)
    def on_idle(self):
        deadline = getattr(self, "_memt_query_wait_deadline", None)
        self._memt_query_wait_deadline = None
        if (
            deadline is not None
            and self.running_batch.is_empty()
            and not getattr(self, "result_queue", ())
        ):
            remaining_ms = (deadline - time.monotonic()) * 1000
            if remaining_ms > 0:
                # ZMQ poll has integer-ms resolution. Ceil avoids a sub-ms
                # busy loop; actual release can overshoot by rounding and OS
                # scheduling. Peers block in the existing ingress broadcast.
                policy = self._memt_query_coalescer
                wall, cpu = time.perf_counter(), time.process_time()
                _wait_for_input(self, math.ceil(remaining_ms))
                policy.poll_calls += 1
                policy.poll_wall_ms += (time.perf_counter() - wall) * 1000
                policy.poll_cpu_ms += (time.process_time() - cpu) * 1000
        return native_idle(self)

    prefill._memt_query_coalescing = True
    scheduler_class._get_new_batch_prefill_raw = prefill
    scheduler_class.on_idle = on_idle
