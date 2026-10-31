"""Stream success follows the worker fence, without waiting for late aborts."""

import asyncio
import json
import unittest
from types import SimpleNamespace

from memt_sglang.protocol import MemoryRequest, validate_payload
from memt_sglang.runtime.admission import MemoryRequestMiddleware
from memt_sglang.runtime.sse import event

PAYLOAD = dict(
    rid="memt|u|query|n",
    input_ids=[1],
    stream=True,
    sampling_params={"max_new_tokens": 5},
    memt_lease="lease",
)


class FakeService:
    max_prefill_tokens = 100

    def __init__(self):
        self.order = []
        self.native_stream_incremental = lambda: False

    async def prepare(self, payload):
        request = validate_payload(payload)
        internal = MemoryRequest("u", "query", "internal")
        return request, internal, SimpleNamespace(), {**payload, "rid": internal.rid}

    async def finish(self, *args, success, result=None):
        self.order.append(("fence", success))
        if getattr(self, "fail", False):
            raise TimeoutError("fence unavailable")


async def exercise(native, service):
    messages = [{"type": "http.request", "body": json.dumps(PAYLOAD).encode()}]

    async def receive():
        if messages:
            return messages.pop(0)
        await asyncio.Future()

    async def send(message):
        service.order.append(("send", message))

    await MemoryRequestMiddleware(native, service=service)(
        {"type": "http", "method": "POST", "path": "/generate", "headers": []},
        receive,
        send,
    )
    return b"".join(x[1].get("body", b"") for x in service.order if x[0] == "send")


def output(text, ids, reason=None):
    return dict(
        text=text,
        output_ids=ids,
        meta_info={"id": "memt|u|query|internal", "finish_reason": reason},
    )


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_incremental_forward_terminal_after_fence(self):
        s = FakeService()

        async def native(scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200})
            await send(
                {
                    "type": "http.response.body",
                    "body": event(output("A", [2])),
                    "more_body": True,
                }
            )
            self.assertIn(b'"text": "A"', s.order[-1][1]["body"])
            self.assertNotIn(("fence", True), s.order)
            terminal = event(output("AB", [2, 3], {"type": "stop"})) + event("[DONE]")
            # Split arbitrary bytes to check the SSE boundary parser.
            await send(
                {"type": "http.response.body", "body": terminal[:9], "more_body": True}
            )
            await send(
                {"type": "http.response.body", "body": terminal[9:], "more_body": False}
            )

        body = await exercise(native, s)
        self.assertEqual(s.order[-2], ("fence", True))
        self.assertTrue(body.endswith(event("[DONE]")))
        self.assertNotIn(b"|internal", body)

    async def test_fence_failure_never_emits_success(self):
        s = FakeService()
        s.fail = True

        async def native(scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200})
            await send(
                {
                    "type": "http.response.body",
                    "body": event(output("A", [2], {"type": "length"}))
                    + event("[DONE]"),
                }
            )

        body = await exercise(native, s)
        self.assertIn(b"native_stream_failed", body)
        self.assertNotIn(b"[DONE]", body)
        self.assertNotIn(b'"finish_reason": {', body)

    async def test_incomplete_stream_is_not_success(self):
        s = FakeService()

        async def native(scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": event(output("A", [2]))})

        body = await exercise(native, s)
        self.assertIn(("fence", False), s.order)
        self.assertNotIn(b"[DONE]", body)

    async def test_incremental_native_normalized_to_cumulative(self):
        s = FakeService()
        s.native_stream_incremental = lambda: True

        async def native(scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200})
            await send(
                {
                    "type": "http.response.body",
                    "body": event(output("A", [2])),
                    "more_body": True,
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": event(output("B", [3], {"type": "stop"})) + event("[DONE]"),
                }
            )

        body = await exercise(native, s)
        self.assertIn(b'"text": "AB"', body)
        self.assertIn(b'"output_ids": [2, 3]', body)

    async def test_terminal_does_not_wait_for_native_background_tail(self):
        # SGLang's StreamingResponse runs a 2 s abort task after the final body.
        s = FakeService()
        tail = asyncio.Event()
        finished = asyncio.Event()

        async def native(scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200})
            await send(
                {
                    "type": "http.response.body",
                    "body": event(output("A", [2], {"type": "stop"})) + event("[DONE]"),
                }
            )
            await tail.wait()
            finished.set()

        middleware = MemoryRequestMiddleware(native, service=s)
        messages = [{"type": "http.request", "body": json.dumps(PAYLOAD).encode()}]

        async def receive():
            if messages:
                return messages.pop(0)
            await asyncio.Future()

        async def send(message):
            s.order.append(("send", message))

        await asyncio.wait_for(
            middleware(
                {"type": "http", "method": "POST", "path": "/generate", "headers": []},
                receive,
                send,
            ),
            1,
        )
        body = b"".join(x[1].get("body", b"") for x in s.order if x[0] == "send")
        self.assertEqual(s.order[-2], ("fence", True))
        self.assertTrue(body.endswith(event("[DONE]")))
        self.assertFalse(finished.is_set())
        self.assertTrue(middleware._cleanup_tasks)
        tail.set()
        await asyncio.wait_for(finished.wait(), 1)
        await asyncio.sleep(0)
        self.assertFalse(middleware._cleanup_tasks)

    async def test_native_tail_failure_after_completion_keeps_success(self):
        s = FakeService()

        async def native(scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200})
            await send(
                {
                    "type": "http.response.body",
                    "body": event(output("A", [2], {"type": "stop"})) + event("[DONE]"),
                }
            )
            await asyncio.sleep(0.01)
            raise RuntimeError("late abort failed")

        with self.assertLogs("memt_sglang.runtime.admission", "WARNING"):
            body = await exercise(native, s)
            await asyncio.sleep(0.05)
        self.assertEqual(s.order[-2], ("fence", True))
        self.assertTrue(body.endswith(event("[DONE]")))

    async def test_cancellation_still_fences(self):
        s = FakeService()
        entered = asyncio.Event()

        async def native(scope, receive, send):
            await receive()
            entered.set()
            await asyncio.Future()

        t = asyncio.create_task(exercise(native, s))
        await entered.wait()
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t
        self.assertIn(("fence", False), s.order)

    async def test_pending_background_never_bypasses_worker_fence(self):
        service = FakeService()
        entered, release, tail = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def finish(*args, success, result=None):
            entered.set()
            await release.wait()
            await FakeService.finish(service, *args, success=success, result=result)

        service.finish = finish

        async def native(scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200})
            await send(
                {
                    "type": "http.response.body",
                    "body": event(output("A", [2], {"type": "stop"})) + event("[DONE]"),
                }
            )
            await tail.wait()

        task = asyncio.create_task(exercise(native, service))
        await asyncio.wait_for(entered.wait(), 1)
        self.assertFalse(task.done())
        self.assertFalse(
            any(
                b"[DONE]" in row[1].get("body", b"")
                for row in service.order
                if row[0] == "send"
            )
        )
        release.set()
        body = await asyncio.wait_for(task, 1)
        self.assertTrue(body.endswith(event("[DONE]")))
        self.assertEqual(service.order[-2], ("fence", True))
        tail.set()
        await asyncio.sleep(0)

    async def test_cancellation_during_fence_keeps_cleanup_alive(self):
        service = FakeService()
        entered, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def finish(*args, success, result=None):
            entered.set()
            await release.wait()
            await FakeService.finish(service, *args, success=success, result=result)
            settled.set()

        service.finish = finish

        async def native(scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200})
            await send(
                {
                    "type": "http.response.body",
                    "body": event(output("A", [2], {"type": "stop"})) + event("[DONE]"),
                }
            )

        task = asyncio.create_task(exercise(native, service))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(settled.is_set())
        release.set()
        await asyncio.wait_for(settled.wait(), 1)
        self.assertIn(("fence", True), service.order)
        self.assertFalse(
            any(
                b"[DONE]" in row[1].get("body", b"")
                for row in service.order
                if row[0] == "send"
            )
        )
