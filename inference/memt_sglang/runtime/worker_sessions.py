"""Worker control operations run between forwards, after the forward stream fence.

Tokenizer admission validates external input. Forward consumes prepared records;
unknown/retired request IDs are inert, including requests still being aborted by
the native scheduler. Published state is never a writer's destination: a commit
writes its final state back into its transaction's private candidate slot.
"""

from dataclasses import dataclass, field

from memt_sglang.protocol import (
    CapacityError,
    MemoryRequest,
    ReceiptStatus,
    SessionError,
    TransactionStatus,
)
from memt_sglang.runtime.config import MAX_TRANSACTION_RECEIPTS
from memt_sglang.runtime.features import commit_chunk_policy
from memt_sglang.runtime.plan import validate_commit_span


@dataclass
class Transaction:
    id: str
    slot: int
    receipts: dict = field(default_factory=dict)
    has_memory: bool = False


@dataclass
class PreparedRequest:
    request: MemoryRequest
    slot: int  # Query: the published slot. Commit: the transaction slot.
    complete: bool = False
    receipt: str | None = None
    progress: int = 0
    commit_policy: str = "atomic"
    replay_base_slot: int | None = None
    base_has_memory: bool = False
    rewind_pending: bool = False
    rewind_count: int = 0


class WorkerSessions:
    def __init__(self, pool):
        self.pool = pool
        self.transactions = {}
        self.requests = {}
        self.latest = {}
        self.memory_read = {}
        self.has_memory = {}

    def create(self, session, source=None, *, memory_read=True):
        if session in self.pool.sessions:
            raise SessionError("Session already exists")
        self.pool.ensure(session, source=source)
        self.memory_read[session] = memory_read
        self.has_memory[session] = (
            self.has_memory[source] if source is not None else False
        )

    def require(self, session):
        if session not in self.pool.sessions:
            raise SessionError("Unknown session; create_session first")

    def has_active_request(self, session):
        return any(
            record.request.session == session for record in self.requests.values()
        )

    def set_memory_read(self, session, enabled):
        self.require(session)
        if session in self.transactions or self.has_active_request(session):
            raise SessionError("Reading policy requires an idle session")
        self.memory_read[session] = enabled

    def _copy(self, source, destination):
        for storage in self.pool.storage.values():
            storage[destination].copy_(storage[source])

    def begin(self, session, transaction):
        self.require(session)
        if session in self.transactions:
            if self.transactions[session].id == transaction:
                return
            raise SessionError("Session already has a transaction")
        if len(self.transactions) >= self.pool.max_commits:
            raise CapacityError("Transaction capacity exhausted")
        slot = self.pool.take_slot()
        self._copy(self.pool.sessions[session], slot)
        self.transactions[session] = Transaction(
            transaction, slot, has_memory=self.has_memory[session]
        )

    def transaction(self, session, transaction):
        self.require(session)
        value = self.transactions.get(session)
        if value is None or value.id != transaction:
            raise SessionError("Unknown transaction")
        return value

    def prepare(self, rid, transaction=None, *, receipt=None):
        request = MemoryRequest.from_rid(rid)
        policy = commit_chunk_policy() if request.operation == "commit" else "atomic"
        receipt = receipt or request.nonce
        self.require(request.session)
        if self.has_active_request(request.session):
            raise SessionError("Session has an active request")
        slot = self.pool.sessions[request.session]
        replay_base = None
        base_has_memory = False
        if request.operation == "commit":
            transaction_state = self.transaction(request.session, transaction)
            if len(transaction_state.receipts) >= MAX_TRANSACTION_RECEIPTS:
                raise CapacityError(
                    f"Publish or abort after {MAX_TRANSACTION_RECEIPTS} requests"
                )
            if receipt in transaction_state.receipts:
                raise SessionError(
                    "Commit nonce already used; inspect transaction status"
                )
            slot = transaction_state.slot
            base_has_memory = transaction_state.has_memory
            if policy != "atomic":
                replay_base = self.pool.take_slot()
                try:
                    self._copy(slot, replay_base)
                except BaseException:
                    self.pool.return_slot(replay_base)
                    raise
            transaction_state.receipts[receipt] = ReceiptStatus.PENDING
        self.requests[rid] = PreparedRequest(
            request,
            slot,
            receipt=receipt,
            commit_policy=policy,
            replay_base_slot=replay_base,
            base_has_memory=base_has_memory,
        )

    def mark_retracted(self, rid):
        """Native KV reset is authoritative; defer GPU restoration to begin()."""
        record = self.requests.get(rid)
        if record is None or record.request.operation != "commit":
            return
        if record.commit_policy == "atomic":
            return
        if record.complete:
            raise SessionError("A completed commit cannot be retracted")
        record.rewind_pending = True

    def prepare_span(self, rid, prefix, count):
        record = self.requests[rid]
        if record.rewind_pending:
            validate_commit_span(record.request, prefix, count, record.commit_policy)
            if prefix != 0 or record.replay_base_slot is None:
                raise SessionError("Retraction must restart from the complete input")
            self._copy(record.replay_base_slot, record.slot)
            record.progress = 0
            record.rewind_pending = False
            record.rewind_count += 1
            self.transactions[
                record.request.session
            ].has_memory = record.base_has_memory
        self.validate_span(rid, prefix, count)

    def validate_span(self, rid, prefix, count):
        """Reject replay/gaps before writes; recovery must roll back explicitly."""
        record = self.requests[rid]
        if record.request.operation != "commit" or record.complete:
            raise SessionError("Commit is not active")
        if record.rewind_pending:
            raise SessionError("Pending retraction requires state restoration")
        if prefix != record.progress:
            raise SessionError("Commit prefix replay or gap requires explicit recovery")
        validate_commit_span(record.request, prefix, count, record.commit_policy)

    def advance(self, rid, count, *, prefix=0):
        """A fragment has written candidate state; only the last earns a receipt."""
        self.validate_span(rid, prefix, count)
        record = self.requests[rid]
        record.progress += count
        record.complete = record.progress == record.request.length
        transaction_state = self.transactions[record.request.session]
        if record.complete:
            transaction_state.receipts[record.receipt] = ReceiptStatus.APPLIED
        transaction_state.has_memory = True

    def fence(self, session, rid=None):
        """Retire exact IDs; late native steps become no-ops.

        Partial forwards may already have written a private candidate. Aborting
        its receipt prevents publication; the published slot is never modified.
        """
        for key, record in list(self.requests.items()):
            if record.request.session != session or (rid is not None and key != rid):
                continue
            if record.replay_base_slot is not None:
                self.pool.return_slot(record.replay_base_slot)
            del self.requests[key]
            if record.request.operation == "commit" and not record.complete:
                self.transactions[session].receipts[record.receipt] = (
                    ReceiptStatus.ABORTED
                )

    def publish(self, session, transaction):
        transaction_state = self.transaction(session, transaction)
        if self.has_active_request(session):
            raise SessionError("Fence completed requests before publishing")
        if any(
            value != ReceiptStatus.APPLIED
            for value in transaction_state.receipts.values()
        ):
            raise SessionError("Transaction has incomplete/aborted requests")
        old = self.pool.sessions[session]
        self.pool.sessions[session] = transaction_state.slot
        self.has_memory[session] = transaction_state.has_memory
        self.pool.return_slot(old)
        self.latest[session] = {
            "transaction": transaction_state.id,
            "status": TransactionStatus.PUBLISHED,
            "receipts": dict(transaction_state.receipts),
        }
        del self.transactions[session]

    def abort(self, session, transaction):
        transaction_state = self.transaction(session, transaction)
        self.fence(session)
        self.pool.return_slot(transaction_state.slot)
        self.latest[session] = {
            "transaction": transaction_state.id,
            "status": TransactionStatus.ABORTED,
            "receipts": dict(transaction_state.receipts),
        }
        del self.transactions[session]

    def status(self, session):
        self.require(session)
        transaction_state = self.transactions.get(session)
        if transaction_state is not None:
            return {
                "transaction": transaction_state.id,
                "status": TransactionStatus.OPEN,
                "receipts": dict(transaction_state.receipts),
                "has_memory": self.has_memory[session],
            }
        return {
            **self.latest.get(session, {"status": TransactionStatus.IDLE}),
            "has_memory": self.has_memory[session],
        }

    def reset(self, session, *, close=False):
        self.require(session)
        self.fence(session)
        if session in self.transactions:
            self.abort(session, self.transactions[session].id)
        for storage in self.pool.storage.values():
            storage[self.pool.sessions[session]].zero_()
        self.latest.pop(session, None)
        self.has_memory[session] = False
        if close:
            self.pool.release(session)
            self.memory_read.pop(session, None)
            self.has_memory.pop(session, None)
