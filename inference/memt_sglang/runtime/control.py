"""Correlated control messages on SGLang's existing typed control channel.

GetInternalStateReq.rid carries the versioned JSON envelope. Ordinary messages
retain native handling. There is no new pickle message, socket or global flush.
"""

import asyncio
import json
import os
from uuid import uuid4

from memt_sglang.protocol import (
    SNAPSHOT_FORMAT,
    CapacityError,
    SessionError,
    validate_boolean,
)
from memt_sglang.runtime.config import CONTROL_TIMEOUT_SECONDS
from memt_sglang.runtime.kv import close_kv, kv_status, open_kv, reset_kv
from memt_sglang.runtime.snapshots import load_snapshot, save_snapshot

PREFIX = "memt-control-v1:"


class ControlChannel:
    def __init__(self, manager, *, timeout=CONTROL_TIMEOUT_SECONDS):
        self.manager = manager
        self.timeout = timeout
        self.pending = {}

    async def call(self, rank, action, **payload):
        from sglang.srt.managers.io_struct import GetInternalStateReq

        rid = PREFIX + json.dumps(
            {"id": uuid4().hex, "rank": rank, "action": action, **payload},
            separators=(",", ":"),
        )
        future = asyncio.get_running_loop().create_future()
        self.pending[rid] = (rank, future)
        try:
            self.manager.auto_create_handle_loop()
            self.manager._dispatch_to_scheduler(GetInternalStateReq(rid=rid))
            async with asyncio.timeout(self.timeout):
                result = await future
            if not result["ok"]:
                error_type = {
                    "ValueError": ValueError,
                    "CapacityError": CapacityError,
                }.get(result.get("error_type"), SessionError)
                raise error_type(result["error"])
            return result["result"]
        finally:
            self.pending.pop(rid, None)

    def receive(self, message):
        pending = self.pending.get(message.rid)
        value = message.internal_state
        if pending and value["rank"] == pending[0] and not pending[1].done():
            pending[1].set_result(value)


class WorkerControl:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.runtime = scheduler.tp_worker.model_runner.model.memt_runtime
        self.sessions = self.runtime.worker_sessions

    def dispatch(self, envelope):
        handlers = {
            "create": "_create",
            "begin": "_begin",
            "prepare": "_prepare",
            "fence": "_fence",
            "publish": "_publish",
            "abort": "_abort",
            "status": "_status",
            "memory_read": "_memory_read",
            "kv_reset": "_kv_reset",
            "reset": "_reset",
            "close": "_close",
            "export": "_export",
            "import": "_import",
        }
        action = envelope["action"]
        if action not in handlers:
            raise ValueError("Unknown MemT control action")
        result = getattr(self, handlers[action])(envelope)
        if result is not None:
            return result
        if action == "close":
            return {"status": "closed"}
        session = envelope["session"]
        return {
            **self.sessions.status(session),
            "memory_read": self.sessions.memory_read[session],
        }

    def _create(self, envelope):
        session = envelope["session"]
        self.sessions.create(
            session,
            envelope.get("source"),
            memory_read=envelope.get("memory_read", True),
        )
        if envelope.get("retain_kv"):
            try:
                open_kv(self.scheduler, session)
            except BaseException:
                self.sessions.reset(session, close=True)
                raise
            self.runtime.kv_sessions.add(session)

    def _begin(self, envelope):
        session = envelope["session"]
        self.sessions.begin(session, envelope["transaction"])

    def _prepare(self, envelope):
        from memt_sglang.protocol import MemoryRequest
        from memt_sglang.runtime.features import commit_chunk_policy
        from memt_sglang.runtime.partial_scheduler import partial_swa_floor

        request = MemoryRequest.from_rid(envelope["request_id"])
        if request.operation == "commit" and commit_chunk_policy() != "atomic":
            reserve = int(os.environ.get("MEMT_PARTIAL_QUERY_TOKEN_RESERVE", "4096"))
            needed = request.length + self.scheduler.page_size + reserve
            full = self.scheduler.max_total_num_tokens
            swa = getattr(self.scheduler, "swa_tokens_per_layer", None)
            swa_needed = (
                partial_swa_floor(
                    commit_chunk_policy(),
                    1,
                    self.scheduler.tree_cache.sliding_window_size,
                    self.scheduler.page_size,
                    reserve,
                )
                if swa is not None and swa > 0
                else 0
            )
            if needed > full or (swa_needed and swa_needed > swa):
                raise CapacityError(
                    "Logical commit cannot fit full KV credits or SWA window/step "
                    "headroom; increase capacity or lower context/chunk budget"
                )
        self.sessions.prepare(
            envelope["request_id"],
            envelope.get("transaction"),
            receipt=envelope.get("receipt"),
        )

    def _fence(self, envelope):
        session = envelope["session"]
        from sglang.srt.managers.io_struct import AbortReq

        rid = envelope.get("request_id")
        ids = [
            key
            for key, record in self.sessions.requests.items()
            if record.request.session == session and (rid is None or key == rid)
        ]
        self.sessions.fence(session, rid)
        for key in ids:
            self.scheduler.abort_request(AbortReq(rid=key))

    def _publish(self, envelope):
        session = envelope["session"]
        memory_read = envelope.get("memory_read")
        if session in self.runtime.kv_sessions:
            self.sessions.transaction(session, envelope["transaction"])
            reset_kv(self.scheduler, session)
        self.sessions.publish(session, envelope["transaction"])
        if memory_read is not None:
            self.sessions.set_memory_read(session, memory_read)

    def _abort(self, envelope):
        session = envelope["session"]
        self.sessions.abort(session, envelope["transaction"])

    def _status(self, envelope):
        session = envelope["session"]
        self.sessions.require(session)
        result = {
            **self.sessions.status(session),
            "memory_read": self.sessions.memory_read[session],
        }
        actual = getattr(self.scheduler, "max_total_num_tokens", None)
        if actual is not None:
            allocator = self.scheduler.token_to_kv_pool_allocator
            result["native_kv_capacity"] = {
                "full_tokens": actual,
                "swa_tokens": getattr(allocator, "_size_swa", None),
                "requested_full_tokens": self.scheduler.server_args.max_total_tokens,
            }
        if session in self.runtime.kv_sessions:
            result = {**result, "kv": kv_status(self.scheduler, session)}
        return result

    def _memory_read(self, envelope):
        session = envelope["session"]
        self.sessions.require(session)
        enabled = envelope["enabled"]
        if session in self.sessions.transactions or self.sessions.has_active_request(
            session
        ):
            raise SessionError("Reading policy requires an idle session")
        if session in self.runtime.kv_sessions:
            reset_kv(self.scheduler, session)
        self.sessions.set_memory_read(session, enabled)
        return {"memory_read": enabled}

    def _kv_reset(self, envelope):
        session = envelope["session"]
        if session not in self.runtime.kv_sessions:
            raise SessionError("Session does not retain native KV")
        reset_kv(self.scheduler, session)

    def _reset(self, envelope):
        session = envelope["session"]
        if session in self.runtime.kv_sessions:
            reset_kv(self.scheduler, session)
        self.sessions.reset(session)

    def _close(self, envelope):
        session = envelope["session"]
        if session in self.runtime.kv_sessions:
            close_kv(self.scheduler, session)
            self.runtime.kv_sessions.remove(session)
        self.sessions.reset(session, close=True)

    def _snapshot_slot(self, envelope, *, require_transaction):
        session = envelope["session"]
        self.sessions.require(session)
        transaction = envelope.get("transaction")
        if transaction is not None:
            slot = self.sessions.transaction(session, transaction).slot
        elif require_transaction or session in self.sessions.transactions:
            raise SessionError("Import or pending writes require a transaction")
        else:
            slot = self.sessions.pool.sessions[session]
        if self.sessions.has_active_request(session):
            raise SessionError("Snapshot requires no active request")
        return slot

    @staticmethod
    def _snapshot_options(envelope):
        if envelope["encoding"] != os.environ["MEMT_MEMORY_ENCODING"]:
            raise ValueError("Snapshot codec does not match the server's HF identity")
        return {
            "directory": os.environ.get("MEMT_STATE_DIRECTORY"),
            "identity": os.environ["MEMT_HF_SNAPSHOT_IDENTITY"],
        }

    def _import(self, envelope):
        slot = self._snapshot_slot(envelope, require_transaction=True)
        result = load_snapshot(
            self.sessions.pool,
            slot,
            envelope["file"],
            **self._snapshot_options(envelope),
        )
        self.sessions.transaction(
            envelope["session"], envelope["transaction"]
        ).has_memory = result["has_memory"]
        return result

    def _export(self, envelope):
        slot = self._snapshot_slot(envelope, require_transaction=False)
        transaction = envelope.get("transaction")
        has_memory = (
            self.sessions.transaction(envelope["session"], transaction).has_memory
            if transaction is not None
            else self.sessions.has_memory[envelope["session"]]
        )
        if self.scheduler.ps.attn_tp_rank == 0:
            return save_snapshot(
                self.sessions.pool,
                slot,
                envelope["file"],
                overwrite=validate_boolean(
                    envelope.get("overwrite", False), "overwrite"
                ),
                has_memory=has_memory,
                **self._snapshot_options(envelope),
            )
        return {
            "snapshot_id": envelope["file"],
            "file": envelope["file"],
            "format": SNAPSHOT_FORMAT,
            "has_memory": has_memory,
        }


def worker_control(scheduler, envelope):
    return WorkerControl(scheduler).dispatch(envelope)
