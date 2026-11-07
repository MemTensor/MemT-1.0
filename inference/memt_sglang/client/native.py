"""Native HTTP transport, leases, and acknowledged memory transactions."""

import json
import urllib.error
import urllib.request
from contextlib import contextmanager
from uuid import uuid4

from memt_sglang.client.chat_format import chunks, token_ids
from memt_sglang.protocol import (
    HTTP_PROTOCOL,
    MAX_LEASE_SECONDS,
    MemoryRequest,
    TransactionStatus,
    validate_payload,
)

# Client-side defaults; wire-format values are in memt_sglang.protocol.
DEFAULT_HTTP_TIMEOUT_SECONDS = 300
DEFAULT_MAX_NEW_TOKENS = 512


class MemTClientError(RuntimeError):
    """A failed operation and, when available, its transaction receipt.

    uncertain means the server may have applied the write but its acknowledgement
    was lost. Reconcile the receipt before another write; replay could duplicate it.
    answer preserves a completed generation if the later memory commit failed.
    """

    def __init__(
        self, message, *, stage=None, transaction=None, uncertain=False, answer=None
    ):
        super().__init__(message)
        self.stage = stage
        self.transaction = transaction
        self.uncertain = uncertain
        self.answer = answer


class NativeClient:
    def __init__(
        self,
        base_url,
        *,
        api_key=None,
        timeout=DEFAULT_HTTP_TIMEOUT_SECONDS,
        transport=None,
        control_transport=None,
        capabilities_transport=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._unreleased = {}
        self._epochs = {}
        self._capabilities = None
        self.transport = transport or (lambda payload: self._http("/generate", payload))
        self.control_transport = control_transport or (
            lambda payload: self._http("/memt/control", payload)
        )
        self.capabilities_transport = capabilities_transport or (
            lambda: self._http("/memt/capabilities", None)
        )

    def _http(self, path, payload):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise MemTClientError(
                error.read().decode(errors="replace"), uncertain=error.code >= 500
            ) from error
        except (OSError, ValueError) as error:
            raise MemTClientError(str(error), uncertain=True) from error

    def control(self, session, action, *, lease=None, **kwargs):
        if session in self._epochs:
            kwargs["epoch"] = self._epochs[session]
        result = self.control_transport(
            dict(session=session, action=action, lease=lease, **kwargs)
        )
        if not isinstance(result, dict) or result.get("error"):
            raise MemTClientError(
                f"Invalid control response: {result!r}", stage=action, uncertain=True
            )
        return result

    def capabilities(self, *, refresh=False):
        """Reuse validated limits; refresh when opening a new server session."""
        if self._capabilities is not None and not refresh:
            return dict(self._capabilities)
        result = self.capabilities_transport()
        if (
            not isinstance(result, dict)
            or result.get("protocol") != HTTP_PROTOCOL
            or not isinstance(result.get("epoch"), str)
            or not result["epoch"]
        ):
            raise MemTClientError("Invalid engine capabilities", stage="capabilities")
        for field in ("chunk_tokens", "max_prefill_tokens"):
            if type(result.get(field)) is not int or result[field] < 1:
                raise MemTClientError(f"Invalid {field}", stage="capabilities")
        policy = result.get("commit_chunk_policy", "atomic")
        if policy not in ("atomic", "global", "tail"):
            raise MemTClientError("Invalid commit chunk policy", stage="capabilities")
        if policy != "atomic" and (
            type(result.get("max_commit_tokens")) is not int
            or result["max_commit_tokens"] < 1
        ):
            raise MemTClientError("Invalid max_commit_tokens", stage="capabilities")
        if policy == "atomic" and result["chunk_tokens"] > result["max_prefill_tokens"]:
            raise MemTClientError(
                "Memory chunk exceeds prefill capacity", stage="capabilities"
            )
        self._capabilities = dict(result)
        return dict(self._capabilities)

    def create_session(
        self,
        session=None,
        *,
        source=None,
        source_lease=None,
        memory_read=None,
        epoch=None,
    ):
        session = session or uuid4().hex
        args = {"source": source, "source_lease": source_lease}
        if memory_read is not None:
            args["memory_read"] = memory_read
        if epoch is not None:
            args["epoch"] = epoch
        result = self.control(session, "create", **args)
        if epoch is not None:
            if result.get("epoch") != epoch:
                raise MemTClientError(
                    "Creation epoch was not confirmed", uncertain=True
                )
            self._epochs[session] = epoch
        return session

    @contextmanager
    def lease(self, session):
        if session in self._unreleased:
            raise MemTClientError(
                "Recover the previous lease before reuse", uncertain=True
            )
        token = self.control(session, "acquire", ttl=MAX_LEASE_SECONDS)["lease"]
        try:
            yield token
        except BaseException as error:
            try:
                self.control(session, "release", lease=token)
            except Exception as cleanup:
                self._unreleased[session] = token
                error.add_note(f"Lease cleanup also failed: {cleanup}")
            raise
        else:
            try:
                self.control(session, "release", lease=token)
            except Exception:
                self._unreleased[session] = token
                raise

    def has_unreleased_lease(self, session):
        return session in self._unreleased

    def recover(self, session):
        token = self._unreleased.get(session)
        if token is not None:
            result = self.control(session, "recover", lease=token)
            self.control(session, "release", lease=token)
            del self._unreleased[session]
            return result
        with self.lease(session) as lease:
            return self.control(session, "recover", lease=lease)

    @contextmanager
    def transaction(self, session, *, lease, memory_read=None):
        transaction = uuid4().hex
        self.control(session, "begin", lease=lease, transaction=transaction)
        try:
            yield transaction
        except BaseException:
            self._rollback_transaction(session, lease, transaction)
            raise
        self._publish_transaction(session, lease, transaction, memory_read)

    def _rollback_transaction(self, session, lease, transaction):
        try:
            self.control(session, "recover", lease=lease)
        except Exception as rollback_error:
            raise MemTClientError(
                "Rollback acknowledgement unavailable",
                stage="rollback",
                transaction=transaction,
                uncertain=True,
            ) from rollback_error

    def _publish_transaction(self, session, lease, transaction, memory_read):
        try:
            options = {} if memory_read is None else {"memory_read": memory_read}
            self.control(
                session, "publish", lease=lease, transaction=transaction, **options
            )
        except Exception as error:
            # Resolve a lost publish acknowledgement using its transaction ID.
            try:
                status = self.control(session, "status", lease=lease)
            except Exception:
                status = None
            if (
                status
                and status.get("transaction") == transaction
                and status.get("status") == TransactionStatus.PUBLISHED
            ):
                return
            if (
                status
                and status.get("transaction") == transaction
                and status.get("status") == TransactionStatus.OPEN
            ):
                self._rollback_transaction(session, lease, transaction)
                raise MemTClientError(
                    "Publication failed; transaction rolled back",
                    stage="publish",
                    transaction=transaction,
                ) from error
            raise MemTClientError(
                "Publication outcome is unknown; inspect session status",
                stage="publish",
                transaction=transaction,
                uncertain=True,
            ) from error

    def request(
        self,
        session,
        operation,
        ids,
        *,
        max_new_tokens,
        lease,
        transaction=None,
        nonce=None,
        sampling=None,
        require_stop=False,
    ):
        tokens = token_ids(ids)
        memory = MemoryRequest(
            session,
            operation,
            nonce or uuid4().hex,
            length=len(tokens) if operation == "commit" else None,
        )
        params = dict(sampling or {})
        if any(key in params for key in ("max_new_tokens", "n", "ignore_eos")):
            raise ValueError("max_new_tokens/n/ignore_eos are managed by the client")
        params.update(max_new_tokens=max_new_tokens, n=1, ignore_eos=False)
        params.setdefault("skip_special_tokens", False)
        payload = {
            "rid": memory.rid,
            "input_ids": list(tokens),
            "sampling_params": params,
            "stream": False,
            "memt_lease": lease,
            "memt_transaction": transaction,
        }
        if session in self._epochs:
            payload["memt_epoch"] = self._epochs[session]
        validate_payload(payload)
        result = self.transport(payload)
        meta = result.get("meta_info") if isinstance(result, dict) else None
        reason = meta.get("finish_reason") if isinstance(meta, dict) else None
        if (
            not isinstance(reason, dict)
            or reason.get("type") not in ("stop", "length")
            or meta.get("id") != memory.rid
            or result.get("error")
        ):
            raise MemTClientError(
                f"MemT request failed or returned an incomplete result: {memory.rid}",
                stage=operation,
                transaction=transaction,
                uncertain=True,
            )
        if require_stop and reason["type"] != "stop":
            raise MemTClientError(
                "Query reached the token limit; response is incomplete",
                stage="query",
                answer=result,
            )
        return result

    def commit(self, session, ids, *, lease=None, transaction=None):
        """Explicit document/token ingestion; chunking is not used for QA turns."""
        capabilities = self.capabilities()
        if capabilities.get("commit_chunk_policy", "atomic") != "atomic":
            tokens = token_ids(ids)
            if len(tokens) > capabilities["max_commit_tokens"]:
                raise ValueError("Logical commit exceeds the server context limit")
            return self._commit_encoded(
                session, [tokens], lease=lease, transaction=transaction
            )
        chunk_tokens = capabilities["chunk_tokens"]
        return self._commit_encoded(
            session,
            chunks(ids, chunk_tokens=chunk_tokens),
            lease=lease,
            transaction=transaction,
        )

    def commit_turns(self, session, turns, *, lease=None, transaction=None):
        encoded = [token_ids(turn) for turn in turns]
        if not encoded:
            raise ValueError("Expected at least one encoded turn")
        capabilities = self.capabilities()
        prefill_limit = capabilities.get(
            "max_commit_tokens", capabilities["max_prefill_tokens"]
        )
        if any(len(turn) > prefill_limit for turn in encoded):
            raise ValueError(
                f"A complete QA turn exceeds the {prefill_limit}-token prefill capacity"
            )
        return self._commit_encoded(
            session, encoded, lease=lease, transaction=transaction
        )

    def _commit_encoded(self, session, encoded, *, lease, transaction):
        if lease is None:
            if transaction is not None:
                raise ValueError("A transaction requires its owning lease")
            with self.lease(session) as token:
                return self._commit_encoded(
                    session, encoded, lease=token, transaction=None
                )
        if transaction is None:
            with self.transaction(session, lease=lease) as transaction_id:
                return self._commit_encoded(
                    session, encoded, lease=lease, transaction=transaction_id
                )
        return [
            self.request(
                session,
                "commit",
                turn,
                max_new_tokens=0,
                lease=lease,
                transaction=transaction,
            )
            for turn in encoded
        ]

    def query(
        self,
        session,
        ids,
        *,
        lease=None,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        sampling=None,
        require_stop=False,
    ):
        if lease is None:
            with self.lease(session) as token:
                return self.query(
                    session,
                    ids,
                    lease=token,
                    max_new_tokens=max_new_tokens,
                    sampling=sampling,
                    require_stop=require_stop,
                )
        return self.request(
            session,
            "query",
            ids,
            max_new_tokens=max_new_tokens,
            lease=lease,
            sampling=sampling,
            require_stop=require_stop,
        )

    def reset(self, session):
        with self.lease(session) as lease:
            return self.control(session, "reset", lease=lease)

    def close(self, session):
        lease = self._unreleased.get(session)
        if lease is not None:
            self.control(session, "recover", lease=lease)
        else:
            lease = self.control(session, "acquire", ttl=MAX_LEASE_SECONDS)["lease"]
        try:
            result = self.control(session, "close", lease=lease)
        except BaseException:
            self._unreleased[session] = lease
            raise
        self._unreleased.pop(session, None)
        self._epochs.pop(session, None)
        return result
