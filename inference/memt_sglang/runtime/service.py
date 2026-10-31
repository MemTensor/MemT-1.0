"""Tokenizer-owned leases and worker-confirmed transactions on one server."""

import asyncio
from dataclasses import dataclass, field
from time import monotonic
from uuid import uuid4

from memt_sglang.protocol import (
    DEFAULT_LEASE_SECONDS,
    HTTP_PROTOCOL,
    MAX_LEASE_SECONDS,
    SNAPSHOT_FORMAT,
    CapacityError,
    MemoryRequest,
    ReceiptStatus,
    SessionError,
    TransactionStatus,
    validate_boolean,
    validate_payload,
    validate_session_id,
)
from memt_sglang.runtime.config import (
    MAX_CONTROL_ARGUMENT_LENGTH,
    RETIRE_POLL_SECONDS,
    RETIRE_TIMEOUT_SECONDS,
    RuntimeConfig,
)


@dataclass
class Session:
    rank: int
    token: str | None = None
    expires: float = 0
    busy: bool = False
    uncertain: bool = False
    retain_kv: bool = False
    memory_read: bool = True
    context_ids: list = field(default_factory=list)
    pending_input: list = field(default_factory=list)
    active_query: str | None = None
    active_internal: str | None = None
    last_query: str | None = None
    cancelled_query: str | None = None


class SessionService:
    def __init__(
        self,
        channel,
        identity,
        *,
        dp_size,
        clock=monotonic,
        native_stream_incremental=None,
    ):
        config = RuntimeConfig.from_hf_config(identity)
        self.channel = channel
        self.dp_size = dp_size
        self.max_sessions = config.max_sessions_per_rank
        self.max_prefill_tokens = identity["max_prefill_tokens"]
        self.commit_policy = identity.get("execution", {}).get(
            "commit_chunk_policy", "atomic"
        )
        self.commit_semantics_version = identity.get("execution", {}).get(
            "commit_semantics_version", "atomic-fresh-kv-v1"
        )
        self.max_commit_tokens = identity.get(
            "max_commit_tokens", self.max_prefill_tokens
        )
        self.chunk_tokens = identity["chunk_tokens"]
        self.sessions = {}
        self.clock = clock
        self.native_stream_incremental = native_stream_incremental or (lambda: False)
        self.epoch = uuid4().hex
        self.snapshot_identity = identity.get("snapshot_identity")
        self.memory_encoding = identity.get("memory_encoding")

    def capabilities(self):
        return {
            "protocol": HTTP_PROTOCOL,
            "epoch": self.epoch,
            "memory_snapshot": SNAPSHOT_FORMAT,
            "chunk_tokens": self.chunk_tokens,
            "max_prefill_tokens": self.max_prefill_tokens,
            "max_commit_tokens": self.max_commit_tokens,
            "commit_chunk_policy": self.commit_policy,
            "commit_semantics_version": self.commit_semantics_version,
            "snapshot_identity": self.snapshot_identity,
            "memory_encoding": self.memory_encoding,
        }

    def _check_epoch(self, epoch):
        if epoch is not None and epoch != self.epoch:
            raise SessionError("Engine epoch changed; create a new session")

    async def _call(self, state, action, session, **payload):
        try:
            return await self.channel().call(
                state.rank, action, session=session, **payload
            )
        except BaseException as error:
            # A correlated negative reply is known. Transport loss/cancellation
            # leaves only this session uncertain until a new fence is acknowledged.
            if not isinstance(error, (ValueError, SessionError)):
                state.uncertain = True
            raise

    def _state(self, session):
        if session not in self.sessions:
            raise SessionError("Unknown session; create_session first")
        return self.sessions[session]

    def _owned(self, session, token, *, recovery=False):
        state = self._state(session)
        if not token or token != state.token:
            raise SessionError("A valid session lease is required")
        if state.busy:
            raise SessionError("Session has an active operation")
        if not recovery and (state.uncertain or self.clock() >= state.expires):
            raise SessionError(
                "Lease expired or result uncertain; recover before reuse"
            )
        return state

    async def _recover(self, state, session):
        result = await self._call(state, "fence", session)
        if result["status"] == TransactionStatus.OPEN:
            result = await self._call(
                state, "abort", session, transaction=result["transaction"]
            )
        state.uncertain = False
        if type(result.get("memory_read")) is bool:
            state.memory_read = result["memory_read"]
        return result

    async def control(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Expected a control object")
        self._check_epoch(payload.get("epoch"))
        action = payload.get("action")
        session = payload.get("session")
        validate_session_id(session)
        match action:
            case "create":
                return await self._create(payload, session)
            case "inspect":
                return await self._inspect(session, self._state(session))
            case "cancel":
                return await self._cancel(payload, session, self._state(session))
            case (
                "acquire"
                | "renew"
                | "release"
                | "begin"
                | "publish"
                | "abort"
                | "status"
                | "reset"
                | "close"
                | "kv_reset"
                | "export"
                | "import"
                | "memory_read"
                | "recover"
            ):
                state = self._state(session)
            case _:
                raise ValueError("Unknown control action")
        ttl = payload.get("ttl", DEFAULT_LEASE_SECONDS)
        if type(ttl) not in (int, float) or not 1 <= ttl <= MAX_LEASE_SECONDS:
            raise ValueError(
                f"Lease ttl must be between 1 and {MAX_LEASE_SECONDS} seconds"
            )
        if action == "acquire":
            return await self._acquire(session, state, ttl)
        state = self._owned(
            session,
            payload.get("lease"),
            recovery=action in {"recover", "status", "release"},
        )
        state.busy = True
        try:
            match action:
                case "renew":
                    return await self._renew(state, ttl)
                case "release":
                    return await self._release(session, state)
                case "recover":
                    return await self._recover_control(session, state, ttl)
                case "begin":
                    return await self._begin(payload, session, state)
                case "publish":
                    return await self._publish(payload, session, state)
                case "abort":
                    return await self._abort(payload, session, state)
                case "status":
                    return await self._status(session, state)
                case "reset":
                    return await self._reset(session, state)
                case "close":
                    return await self._close(session, state)
                case "kv_reset":
                    return await self._kv_reset(session, state)
                case "export":
                    return await self._export(payload, session, state)
                case "import":
                    return await self._import(payload, session, state)
                case "memory_read":
                    return await self._memory_read(payload, session, state)
        finally:
            state.busy = False

    async def _create(self, payload, session):
        if session in self.sessions:
            raise SessionError("Session already exists")
        source = payload.get("source")
        if source is not None:
            validate_session_id(source)
        source_state = (
            self._owned(source, payload.get("source_lease")) if source else None
        )
        counts = [
            sum(session_state.rank == rank for session_state in self.sessions.values())
            for rank in range(self.dp_size)
        ]
        candidates = [
            rank for rank, count in enumerate(counts) if count < self.max_sessions
        ]
        if source_state:
            candidates = (
                [source_state.rank]
                if counts[source_state.rank] < self.max_sessions
                else []
            )
        if not candidates:
            raise CapacityError("Session capacity exhausted")
        retain_kv = payload.get("retain_kv", False)
        validate_boolean(retain_kv, "retain_kv")
        memory_read = payload.get("memory_read", True)
        validate_boolean(memory_read, "memory_read")
        state = Session(
            min(candidates, key=lambda rank: (counts[rank], rank)),
            busy=True,
            retain_kv=retain_kv,
            memory_read=memory_read,
        )
        self.sessions[session] = state
        if source_state:
            source_state.busy = True
        try:
            result = await self._call(
                state,
                "create",
                session,
                source=source,
                retain_kv=retain_kv,
                memory_read=memory_read,
            )
            return {"session": session, "epoch": self.epoch, **result}
        except (SessionError, ValueError):
            del self.sessions[session]
            raise
        finally:
            state.busy = False
            if source_state:
                source_state.busy = False

    async def _inspect(self, session, state):
        return {
            "session": session,
            "epoch": self.epoch,
            "busy": state.busy,
            "uncertain": state.uncertain,
            "memory_read": state.memory_read,
            "retain_kv": state.retain_kv,
            "lease_held": state.token is not None,
        }

    async def _cancel(self, payload, session, state):
        if not payload.get("lease") or payload["lease"] != state.token:
            raise SessionError("A valid session lease is required")
        rid = payload.get("request_id")
        if not isinstance(rid, str):
            raise ValueError("cancel requires request_id")
        if state.busy:
            if rid != state.active_query or state.active_internal is None:
                raise SessionError("Cancel ID does not match the active request")
            state.cancelled_query = rid
            await self._call(state, "fence", session, request_id=state.active_internal)
            deadline = monotonic() + RETIRE_TIMEOUT_SECONDS
            while state.busy:
                if monotonic() >= deadline:
                    state.uncertain = True
                    raise TimeoutError("Cancelled request has not retired")
                await asyncio.sleep(RETIRE_POLL_SECONDS)
        elif rid not in {state.last_query, state.cancelled_query}:
            raise SessionError("Cancel ID does not match the latest query")
        # Also handle stop arriving just after natural completion. The
        # caller keeps authoritative pre-turn context and replays it next.
        if state.uncertain:
            raise SessionError("Recover the uncertain cancellation before reuse")
        state.busy = True
        try:
            if state.retain_kv:
                await self._call(state, "kv_reset", session)
                state.context_ids.clear()
            state.cancelled_query = rid
            return {"status": "cancelled", "request_id": rid}
        finally:
            state.busy = False

    async def _acquire(self, session, state, ttl):
        if state.busy or (state.token and self.clock() < state.expires):
            raise SessionError("Session lease is held by another operation")
        state.busy = True
        try:
            # Expiration is reclaimed lazily, always after a worker fence.
            if state.token or state.uncertain:
                await self._recover(state, session)
            state.token = uuid4().hex
            state.expires = self.clock() + ttl
            return {"lease": state.token, "ttl": ttl}
        finally:
            state.busy = False

    async def _renew(self, state, ttl):
        state.expires = self.clock() + ttl
        return {"ttl": ttl}

    async def _release(self, session, state):
        result = await self._recover(state, session)
        state.token = None
        return result

    async def _recover_control(self, session, state, ttl):
        result = await self._recover(state, session)
        state.expires = self.clock() + ttl
        return result

    @staticmethod
    def _string_arguments(**arguments):
        if any(
            not isinstance(value, str)
            or not value
            or len(value) > MAX_CONTROL_ARGUMENT_LENGTH
            for value in arguments.values()
        ):
            raise ValueError("Control arguments must be nonempty bounded strings")
        return arguments

    @staticmethod
    def _clear_context(state):
        # Even a lost acknowledgement may follow native KV invalidation.
        state.context_ids.clear()
        state.last_query = None
        state.cancelled_query = None

    async def _begin(self, payload, session, state):
        arguments = self._string_arguments(transaction=payload.get("transaction"))
        return await self._call(state, "begin", session, **arguments)

    async def _publish(self, payload, session, state):
        arguments = self._string_arguments(transaction=payload.get("transaction"))
        if "memory_read" in payload:
            arguments["memory_read"] = validate_boolean(
                payload["memory_read"], "memory_read"
            )
        self._clear_context(state)
        result = await self._call(state, "publish", session, **arguments)
        if "memory_read" in arguments:
            state.memory_read = arguments["memory_read"]
        return result

    async def _abort(self, payload, session, state):
        arguments = self._string_arguments(transaction=payload.get("transaction"))
        return await self._call(state, "abort", session, **arguments)

    async def _status(self, session, state):
        return await self._call(state, "status", session)

    async def _reset(self, session, state):
        self._clear_context(state)
        return await self._call(state, "reset", session)

    async def _close(self, session, state):
        result = await self._call(state, "close", session)
        del self.sessions[session]
        return result

    async def _kv_reset(self, session, state):
        self._clear_context(state)
        return await self._call(state, "kv_reset", session)

    async def _export(self, payload, session, state):
        arguments = {
            "file": uuid4().hex + ".safetensors"
            if payload.get("file") is None
            else payload["file"],
            "encoding": payload.get("encoding"),
        }
        if payload.get("transaction") is not None:
            arguments["transaction"] = payload["transaction"]
        arguments = self._string_arguments(**arguments)
        overwrite = validate_boolean(payload.get("overwrite", False), "overwrite")
        return await self._call(
            state,
            "export",
            session,
            **arguments,
            overwrite=overwrite,
        )

    async def _import(self, payload, session, state):
        arguments = self._string_arguments(
            **{key: payload.get(key) for key in ("transaction", "file", "encoding")}
        )
        return await self._call(state, "import", session, **arguments)

    async def _memory_read(self, payload, session, state):
        enabled = validate_boolean(payload.get("enabled"), "enabled")
        self._clear_context(state)
        result = await self._call(state, "memory_read", session, enabled=enabled)
        state.memory_read = enabled
        return result

    async def prepare(self, payload):
        self._check_epoch(payload.get("memt_epoch"))
        operation = MemoryRequest.from_rid(payload.get("rid")).operation
        limit = (
            self.max_commit_tokens if operation == "commit" else self.max_prefill_tokens
        )
        request = validate_payload(payload, max_prefill_tokens=limit)
        state = self._owned(request.session, payload.get("memt_lease"))
        if payload.get("session_params") and not state.retain_kv:
            raise SessionError("Session was not created with retain_kv")
        ids = payload["input_ids"]
        if request.operation == "query" and state.retain_kv:
            prefix = state.context_ids
            if ids[: len(prefix)] != prefix or len(ids) <= len(prefix):
                raise SessionError(
                    "KV continuation requires the exact acknowledged token prefix"
                )
        state.busy = True
        # Never let reused public RIDs authorize an old, still-retiring request.
        internal = MemoryRequest(
            request.session, request.operation, uuid4().hex, length=request.length
        )
        if request.operation == "query" or self.commit_policy != "atomic":
            state.active_query = request.rid
            state.active_internal = internal.rid
            state.cancelled_query = None
        try:
            await self._call(
                state,
                "prepare",
                request.session,
                request_id=internal.rid,
                transaction=payload.get("memt_transaction"),
                receipt=request.nonce,
            )
        except BaseException:
            state.busy = False
            state.active_query = state.active_internal = None
            raise
        forwarded = dict(payload)
        forwarded.pop("memt_lease", None)
        forwarded.pop("memt_transaction", None)
        forwarded.pop("memt_epoch", None)
        forwarded.update(rid=internal.rid, routed_dp_rank=state.rank)
        if request.operation == "query" and state.retain_kv:
            state.pending_input = list(ids)
            forwarded["input_ids"] = ids[len(state.context_ids) :]
            forwarded["session_params"] = {"id": request.session}
        return request, internal, state, forwarded

    async def finish(self, request, internal, state, *, success, result=None):
        try:
            status = await self._call(
                state, "fence", request.session, request_id=internal.rid
            )
            if (
                success
                and request.operation == "commit"
                and status.get("receipts", {}).get(request.nonce)
                != ReceiptStatus.APPLIED
            ):
                raise SessionError("Commit did not finish; abort this transaction")
            if request.operation == "query" and state.retain_kv:
                # A memory fence retires authorization; native abort cleanup may
                # need another scheduler iteration before releasing KV/Mamba.
                deadline = self.clock() + RETIRE_TIMEOUT_SECONDS
                while True:
                    observation = await self._call(state, "status", request.session)
                    if not observation["kv"]["inflight"]:
                        break
                    if self.clock() >= deadline:
                        state.uncertain = True
                        raise TimeoutError("Native KV retirement timed out")
                    await asyncio.sleep(RETIRE_POLL_SECONDS)
                if success and state.cancelled_query != request.rid:
                    output = (
                        result.get("output_ids") if isinstance(result, dict) else None
                    )
                    if not isinstance(output, list) or any(
                        type(output_token) is not int for output_token in output
                    ):
                        state.uncertain = True
                        raise SessionError("Missing authoritative output token IDs")
                    state.context_ids = state.pending_input + output
                    result["memt_kv"] = observation["kv"]
                else:
                    # Native abort drops KV and retains old token arrays. Reset
                    # that internal cache context explicitly before full replay.
                    await self._call(state, "kv_reset", request.session)
                    state.context_ids.clear()
                state.pending_input.clear()
            if request.operation == "query":
                state.last_query = request.rid
                if state.cancelled_query == request.rid:
                    raise SessionError("Query cancelled by its lease owner")
            return status
        finally:
            state.active_query = state.active_internal = None
            state.busy = False
