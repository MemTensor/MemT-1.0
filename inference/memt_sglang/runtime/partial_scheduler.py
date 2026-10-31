"""Opt-in multi-request partial prefill on pinned native scheduling primitives.

KV/positions continue across fragments. Memory reads the current private
candidate at each accepted fragment. Full logical-input KV credits prevent
parked partial requests from overcommitting the non-evictable chunk cache.
All hooks require serialized, FCFS scheduling; normal mode remains untouched.
"""

import os
from functools import wraps

from memt_sglang.protocol import MemoryRequest, is_memt_rid
from memt_sglang.runtime.features import commit_chunk_policy, partial_decode_quanta


def fragment_tokens(policy, total, prefix, budget, *, maximum=0):
    if policy not in ("global", "tail"):
        raise ValueError("Partial policy must be global or tail")
    if any(type(x) is not int for x in (total, prefix, budget, maximum)):
        raise ValueError("Fragment limits must be integers")
    if total < 1 or not 0 <= prefix < total or maximum < 0:
        raise ValueError("Invalid logical fragment range")
    if budget <= 0:
        return 0
    remaining = total - prefix
    if policy == "tail" and prefix < total // 2048 * 2048:
        if prefix % 2048:
            raise ValueError("Ordinary block starts must be aligned")
        return 2048 if budget >= 2048 else 0
    return min(remaining, budget, maximum or remaining)


def _nonnegative_env(name, default):
    text = os.environ.get(name, str(default))
    if not text.isdecimal():
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(text)


def partial_swa_floor(policy, requests, window, page, query_reserve):
    """Enough retained windows and one forward-progress fragment, in tokens."""
    quantum = 2048 if policy == "tail" else page
    return requests * (window + 2 * page) + quantum + query_reserve


def install_partial_prefill():
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.common import release_kv_cache

    if getattr(Scheduler, "_memt_partial_installed", False):
        return
    policy = commit_chunk_policy()
    if policy == "atomic":
        return
    maximum = _nonnegative_env("MEMT_PARTIAL_MAX_CHUNK_TOKENS", 0)
    query_reserve = _nonnegative_env("MEMT_PARTIAL_QUERY_TOKEN_RESERVE", 4096)
    decode_quanta = partial_decode_quanta()
    query_first = os.environ.get("MEMT_PARTIAL_QUERY_FIRST", "0")
    if query_first not in ("0", "1"):
        raise ValueError("MEMT_PARTIAL_QUERY_FIRST must be 0 or 1")
    query_first = query_first == "1"
    context = {}

    def records(scheduler):
        if not hasattr(scheduler, "_memt_partial_records"):
            scheduler._memt_partial_records = {}
        return scheduler._memt_partial_records

    def debt(scheduler):
        result = 0
        for req, credit in records(scheduler).values():
            if req.finished():
                continue
            allocated = req.kv.kv_allocated_len if req.kv is not None else 0
            result += max(0, credit - allocated)
        return result

    def swa_floor(scheduler, extra=0):
        """Retained prefixes need a window, not a full-document SWA cache.

        Completed fragments are evicted to the window before queue re-entry.
        Leave a minimum progress fragment, plus the query reserve. The actual
        step shrinks to available SWA rather than requiring the configured
        token ceiling to fit in the pool.
        Full/compressed KV still receives full logical-context credits.
        """
        window = scheduler.tree_cache.sliding_window_size
        page = scheduler.page_size
        retained = sum(not r.finished() for r, _ in records(scheduler).values())
        return partial_swa_floor(policy, retained + extra, window, page, query_reserve)

    old_raw = Scheduler._get_new_batch_prefill_raw

    @wraps(old_raw)
    def raw(self, *args, **kwargs):
        running = kwargs.get("running_batch")
        credit = getattr(self, "_memt_partial_decode_credit", 0)
        if credit and running is not None:
            has_query = any(
                not req.finished() and "|query|" in req.rid for req in running.reqs
            )
            if has_query:
                self._memt_partial_decode_credit = credit - 1
                return None, running
            # Completion/cancellation must not strand parked commits when no
            # decode work remains to consume the credit.
            self._memt_partial_decode_credit = 0
        context["scheduler"] = self
        # Retained fragments reuse their native request slots. When no new
        # slots remain, do not let a fresh request block ready continuations.
        if records(self):
            free_slots = self.req_to_token_pool.available_size()

            def priority(req):
                if query_first and free_slots > 0 and "|query|" in req.rid:
                    return 0
                return 1 if req.req_pool_idx is not None else 2

            self.waiting_queue.sort(key=priority)
        try:
            return old_raw(self, *args, **kwargs)
        finally:
            context.pop("scheduler", None)

    Scheduler._get_new_batch_prefill_raw = raw

    old_add = PrefillAdder.add_one_req

    @wraps(old_add)
    def add(self, req, *args, **kwargs):
        scheduler = context.get("scheduler")
        if scheduler is None:
            return old_add(self, req, *args, **kwargs)
        request = MemoryRequest.from_rid(req.rid) if is_memt_rid(req.rid) else None
        if request is None or request.operation != "commit":
            # New queries may use unreserved KV only. Decode has additional
            # explicit headroom; retraction recovery remains required.
            credit = debt(scheduler)
            new_slot = req.req_pool_idx is None
            planned = getattr(self, "_memt_new_slots", 0)
            if new_slot and planned >= scheduler.req_to_token_pool.available_size():
                return AddReqResult.NO_TOKEN
            self.rem_total_token_offset += credit
            try:
                result = old_add(self, req, *args, **kwargs)
            finally:
                self.rem_total_token_offset -= credit
            if new_slot and any(r is req for r in self.can_run_list):
                self._memt_new_slots = planned + 1
            return result
        sessions = scheduler.tp_worker.model_runner.model.memt_runtime.worker_sessions
        req._memt_worker_sessions = sessions
        full = req.full_untruncated_fill_ids
        prefix = len(req.prefix_indices)
        if len(full) != request.length:
            raise ValueError("Partial commit native input differs from logical length")
        room = self.rem_input_tokens
        if self.rem_chunk_tokens is not None:
            room = min(room, self.rem_chunk_tokens)
        if self.is_hybrid_swa:
            room = min(
                room,
                max(
                    0,
                    int(self.rem_swa_tokens)
                    - query_reserve
                    - scheduler.tree_cache.sliding_window_size
                    - 2 * self.page_size,
                ),
            )
        count = fragment_tokens(
            policy, request.length, prefix, int(room), maximum=maximum
        )
        if count <= 0:
            return AddReqResult.OTHER
        recs = records(scheduler)
        new = req.rid not in recs
        if new:
            allocator = scheduler.token_to_kv_pool_allocator
            available = (
                allocator.full_available_size()
                if self.is_hybrid_swa
                else allocator.available_size()
            )
            credit = self.ceil_paged_tokens(request.length) + self.page_size
            if available - debt(scheduler) - query_reserve < credit:
                return AddReqResult.NO_TOKEN
            if self.is_hybrid_swa:
                swa_capacity = scheduler.swa_tokens_per_layer
                if swa_capacity < swa_floor(scheduler, extra=1):
                    return AddReqResult.NO_TOKEN
            recs[req.rid] = (req, credit)
        new_slot = req.req_pool_idx is None
        planned = getattr(self, "_memt_new_slots", 0)
        if new_slot and planned >= scheduler.req_to_token_pool.available_size():
            if new:
                recs.pop(req.rid, None)
            return AddReqResult.NO_TOKEN
        old_chunk = self.rem_chunk_tokens
        # Bound the native operation to this one fragment. The full input is
        # restored before batch construction; extend_range selects actual rows.
        req.full_untruncated_fill_ids = full[: prefix + count]
        self.rem_chunk_tokens = None
        self.rem_input_tokens += 1  # Native exact-fit check is >= rather than >.
        try:
            result = old_add(self, req, *args, **kwargs)
        finally:
            self.rem_input_tokens -= 1
            consumed = old_chunk
            req.full_untruncated_fill_ids = full
            self.rem_chunk_tokens = consumed
        accepted = any(r is req for r in self.can_run_list)
        if not accepted:
            if new:
                recs.pop(req.rid, None)
            return result
        if req.extend_range.start != prefix or req.extend_range.length != count:
            raise RuntimeError("Native admission changed a validated commit fragment")
        if old_chunk is not None:
            self.rem_chunk_tokens = old_chunk - count
        self._memt_new_slots = planned + int(new_slot)
        req._memt_partial = req.extend_range.end < request.length
        if req._memt_partial:
            req.inflight_middle_chunks += 1
        # The original result may have observed chunk=None; honor the real
        # remaining chunk budget after charging this fragment.
        return self.budget_state()

    PrefillAdder.add_one_req = add

    old_init = Req.init_next_round_input

    @wraps(old_init)
    def init(self, tree_cache=None, *args, **kwargs):
        if getattr(self, "_memt_resume", False):
            tree_cache = None  # ChunkCache.match_prefix intentionally returns0.
        return old_init(self, tree_cache, *args, **kwargs)

    Req.init_next_round_input = init

    old_retract = Req.reset_for_retract

    @wraps(old_retract)
    def retract(self):
        result = old_retract(self)
        sessions = getattr(self, "_memt_worker_sessions", None)
        if sessions is not None:
            sessions.mark_retracted(self.rid)
        self._memt_resume = False
        return result

    Req.reset_for_retract = retract

    old_next = Scheduler.get_next_batch_to_run

    @wraps(old_next)
    def next_batch(self, running_batch, last_batch):
        recs = records(self)
        for rid, (req, _credit) in list(recs.items()):
            if req.finished():
                recs.pop(rid, None)
        if last_batch is not None and last_batch.forward_mode.is_extend():
            if (
                decode_quanta
                and any("|commit:" in r.rid for r in last_batch.reqs)
                and any("|query|" in r.rid for r in last_batch.reqs)
            ):
                self._memt_partial_decode_credit = decode_quanta
            partials = [
                r
                for r in last_batch.reqs
                if getattr(r, "_memt_partial", False) and not r.finished()
            ]
            for req in partials:
                if req.extend_range.end > len(req.prefix_indices):
                    self.stash_chunked_request(req)
                if self.tree_cache.supports_swa() and self.tree_cache.is_chunk_cache():
                    # Serialized native extend normally performs this just
                    # before its next allocation. Do it when parking instead,
                    # so admission sees the actually reusable SWA capacity.
                    last_batch._evict_swa(req, req.extend_range.end)
                req._memt_resume = True
                if not any(r is req for r in self.waiting_queue):
                    self.waiting_queue.append(req)
            if partials:
                last_batch.filter_batch(chunked_req_to_exclude=partials)
                running_batch.batch_is_full = False
        return old_next(self, running_batch, last_batch)

    Scheduler.get_next_batch_to_run = next_batch

    old_slots = Scheduler.get_num_allocatable_reqs

    @wraps(old_slots)
    def slots(self, running_bs):
        value = old_slots(self, running_bs)
        retained = sum(
            r.req_pool_idx is not None and not r.finished()
            for r, _ in records(self).values()
        )
        return max(
            value,
            min(
                self.max_running_requests - running_bs,
                retained + self.req_to_token_pool.available_size(),
            ),
        )

    Scheduler.get_num_allocatable_reqs = slots

    old_abort = Scheduler.abort_request

    @wraps(old_abort)
    def abort(self, recv_req):
        # Native waiting-queue abort assumes that no KV has been allocated.
        # Parked fragments violate that assumption, so release their KV first.
        recs = records(self)
        for rid, (req, _credit) in list(recs.items()):
            if not (recv_req.abort_all or rid.startswith(recv_req.rid)):
                continue
            if not getattr(req, "_memt_partial", False):
                continue
            # A fence can arrive after result processing but before the next
            # scheduler call has parked this fragment in waiting_queue.
            for batch in (
                getattr(self, "running_batch", None),
                getattr(self, "last_batch", None),
            ):
                if batch is not None and any(r is req for r in batch.reqs):
                    batch.filter_batch(chunked_req_to_exclude=[req])
            if not any(r is req for r in self.waiting_queue):
                self.waiting_queue.append(req)
            req._memt_resume = True
        for req in self.waiting_queue:
            if (recv_req.abort_all or req.rid.startswith(recv_req.rid)) and getattr(
                req, "_memt_resume", False
            ):
                if req.kv is not None and getattr(req, "mamba_pool_idx", None) is None:
                    release_kv_cache(req, self.tree_cache, is_insert=False)
                req._memt_resume = False
                recs.pop(req.rid, None)
        return old_abort(self, recv_req)

    Scheduler.abort_request = abort
    Scheduler._memt_partial_installed = True
