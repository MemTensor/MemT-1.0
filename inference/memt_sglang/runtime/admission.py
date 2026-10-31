"""Validate before scheduling; acknowledge only after a correlated worker fence."""

import asyncio
import json
import logging
from functools import partial

from memt_sglang.protocol import (
    CapacityError,
    SessionError,
    is_memt_rid,
)
from memt_sglang.runtime.sse import SSEDecoder, event

logger = logging.getLogger(__name__)


class MemoryRequestMiddleware:
    def __init__(self, app, *, service):
        self.app = app
        self.service = service
        self._cleanup_tasks = set()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "").rstrip("/")
        if path in {
            "/flush_cache",
            "/release_memory_occupation",
            "/resume_memory_occupation",
            "/init_weights_update_group",
            "/destroy_weights_update_group",
            "/update_weight_version",
        } or path.startswith(
            ("/update_weights", "/update_weight", "/load_lora", "/unload_lora")
        ):
            return await self._error(
                send,
                409,
                "Live weight/cache administration is disabled with MemT sessions",
            )
        if scope.get("method") not in {"POST", "PUT"}:
            return await self.app(scope, receive, send)
        messages = []
        body = []
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] != "http.request":
                return
            body.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        try:
            payload = json.loads(b"".join(body))
        except (ValueError, UnicodeDecodeError):
            payload = None

        async def replay():
            return messages.pop(0) if messages else await receive()

        try:
            if path == "/memt/control":
                result = await self.service.control(payload)
                return await self._json(send, 200, result)
            rid = payload.get("rid") if isinstance(payload, dict) else None
            request_ids = rid if isinstance(rid, list) else [rid]
            uses_memory = any(is_memt_rid(request_id) for request_id in request_ids)
            if uses_memory and path != "/generate":
                raise ValueError("MemT RIDs require /generate")
            if not uses_memory:
                return await self.app(scope, replay, send)
            request, internal, state, forwarded = await self.service.prepare(payload)
        except (ValueError, SessionError, TimeoutError, OSError) as error:
            return await self._exception(send, error)

        rewritten = json.dumps(forwarded).encode()
        messages[:] = [{"type": "http.request", "body": rewritten, "more_body": False}]
        scope = dict(scope)
        scope["headers"] = [
            (k, v)
            for k, v in scope.get("headers", [])
            if k.lower() not in {b"content-length", b"transfer-encoding"}
        ] + [(b"content-length", str(len(rewritten)).encode())]
        if forwarded.get("stream"):
            return await self._stream(scope, replay, send, request, internal, state)
        status = 500
        complete = False
        response = []

        async def capture(message):
            nonlocal status, complete
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body":
                response.append(message.get("body", b""))
                complete = not message.get("more_body", False)

        success = False
        result = None
        failure = None
        try:
            await self.app(scope, replay, capture)
            try:
                result = json.loads(b"".join(response))
            except (ValueError, UnicodeDecodeError):
                result = None
            if complete and 200 <= status < 300 and isinstance(result, dict):
                meta = result.get("meta_info")
                reason = meta.get("finish_reason") if isinstance(meta, dict) else None
                success = (
                    not result.get("error")
                    and isinstance(reason, dict)
                    and reason.get("type") in {"stop", "length"}
                    and meta.get("id") == internal.rid
                )
        finally:
            try:
                await self._finish(
                    request, internal, state, success=success, result=result
                )
            except (ValueError, SessionError, TimeoutError, OSError) as error:
                failure = error
        if failure:
            return await self._exception(send, failure)
        if not success:
            return await self._error(
                send,
                status if status >= 400 else 502,
                "Native request failed or returned an incomplete result",
            )
        result["meta_info"]["id"] = request.rid
        await self._json(send, status, result)

    async def _stream(self, scope, receive, send, request, internal, state):
        """Forward real increments, but hold success until worker retirement."""
        decoder = SSEDecoder()
        started = False
        complete = False
        terminal = None
        done = False
        failure = None
        cumulative_text = ""
        cumulative_ids = []
        incremental = self.service.native_stream_incremental()

        async def capture(message):
            nonlocal started, complete, terminal, done, cumulative_text, cumulative_ids
            if message["type"] == "http.response.start":
                if message["status"] != 200:
                    raise ValueError(f"Native stream HTTP {message['status']}")
                started = True
                # SSE headers open the stream; later failures use an error event.
                # HTTP status cannot change after the first response frame.
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            (b"content-type", b"text/event-stream"),
                            (b"cache-control", b"no-cache"),
                            (b"x-accel-buffering", b"no"),
                        ],
                    }
                )
            elif message["type"] == "http.response.body":
                for data in decoder.feed(message.get("body", b"")):
                    if data == "[DONE]":
                        if terminal is None:
                            raise ValueError(
                                "Native stream ended without terminal metadata"
                            )
                        done = True
                        continue
                    if done or terminal is not None:
                        raise ValueError("Unexpected data after native terminal result")
                    result = json.loads(data)
                    meta = result.get("meta_info", {})
                    if result.get("error") or meta.get("id") != internal.rid:
                        raise ValueError("Invalid native stream result or RID")
                    text, ids = result.get("text", ""), result.get("output_ids", [])
                    if not isinstance(text, str) or not isinstance(ids, list):
                        raise ValueError("Native stream requires text and token IDs")
                    if incremental:
                        cumulative_text += text
                        cumulative_ids += ids
                    else:
                        if (
                            not text.startswith(cumulative_text)
                            or ids[: len(cumulative_ids)] != cumulative_ids
                        ):
                            raise ValueError(
                                "Native cumulative output revised an emitted prefix"
                            )
                        cumulative_text, cumulative_ids = text, list(ids)
                    result.update(text=cumulative_text, output_ids=list(cumulative_ids))
                    meta["id"] = request.rid
                    if meta.get("finish_reason") is not None:
                        if meta["finish_reason"].get("type") not in {"stop", "length"}:
                            raise ValueError("Native generation aborted")
                        terminal = result
                    else:
                        await send(
                            {
                                "type": "http.response.body",
                                "body": event(result),
                                "more_body": True,
                            }
                        )
                if not message.get("more_body", False):
                    decoder.finish()
                    complete = True
                    ended.set()

        # SGLang's background abort sleeps after the final body. Wait for the
        # body, then fence its RID; observe the unrelated background tail later.
        ended = asyncio.Event()
        native = asyncio.create_task(self.app(scope, receive, capture))
        try:
            waiter = asyncio.create_task(ended.wait())
            try:
                await asyncio.wait(
                    {native, waiter}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                waiter.cancel()
            if native.done():
                native.result()
            if not complete or not done or terminal is None:
                raise ValueError("Incomplete native stream")
        except Exception as error:
            failure = error
        except BaseException:
            # Unwind native generation before fencing an interrupted request.
            native.cancel()
            await asyncio.wait({native})
            raise
        finally:
            if native.done():
                if not native.cancelled():
                    native.exception()
            else:
                self._cleanup_tasks.add(native)
                native.add_done_callback(
                    partial(self._native_done, internal_id=internal.rid)
                )
            try:
                await self._finish(
                    request,
                    internal,
                    state,
                    success=failure is None
                    and complete
                    and done
                    and (terminal is not None),
                    result=terminal,
                )
            except (ValueError, SessionError, TimeoutError, OSError) as error:
                failure = error
        if failure is not None:
            if not started:
                return await self._exception(send, failure)
            await send(
                {
                    "type": "http.response.body",
                    "body": event(
                        {
                            "error": {
                                "code": "native_stream_failed",
                                "message": str(failure),
                                "generation_status": "failed",
                                "retryable": False,
                            }
                        }
                    ),
                    "more_body": False,
                }
            )
            return
        await send(
            {
                "type": "http.response.body",
                "body": event(terminal) + event("[DONE]"),
                "more_body": False,
            }
        )

    async def _finish(self, request, internal, state, *, success, result):
        task = asyncio.create_task(
            self.service.finish(
                request, internal, state, success=success, result=result
            )
        )
        self._cleanup_tasks.add(task)
        task.add_done_callback(
            partial(
                self._done,
                session=request.session,
                request_id=request.rid,
                internal_id=internal.rid,
            )
        )
        await asyncio.shield(task)

    def _done(self, task, *, session, request_id, internal_id):
        self._cleanup_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            # This callback also runs after the HTTP task has been cancelled.
            # Log once here, whether or not a caller can still receive the error.
            logger.error(
                "Memory cleanup failed: session=%s request_id=%s internal_id=%s",
                session,
                request_id,
                internal_id,
                exc_info=(type(error), error, error.__traceback__),
            )

    def _native_done(self, task, *, internal_id):
        self._cleanup_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.warning(
                "Native stream tail failed after completion: internal_id=%s",
                internal_id,
                exc_info=(type(error), error, error.__traceback__),
            )

    @classmethod
    async def _exception(cls, send, error):
        # Check capacity first because CapacityError subclasses SessionError.
        if isinstance(error, CapacityError):
            # No free session or in-flight commit slot.
            status = 429
        elif isinstance(error, SessionError):
            # Lease, busy-session or epoch conflict.
            status = 409
        elif isinstance(error, ValueError):
            # Malformed or unsupported request parameters.
            status = 400
        else:
            # Worker failure or missing acknowledgement; outcome may be unknown.
            status = 503
        await cls._error(
            send,
            status,
            str(error)
            or "Worker acknowledgement unavailable; recover session before reuse",
        )

    @classmethod
    async def _error(cls, send, status, message):
        await cls._json(
            send,
            status,
            {"error": {"type": "invalid_memory_request", "message": message}},
        )

    @staticmethod
    async def _json(send, status, result):
        body = json.dumps(result).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
