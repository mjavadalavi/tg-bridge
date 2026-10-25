"""JsonRpcPeer (tg_bridge/agents/codex/rpc.py) over an in-memory pipe pair.

Two peers are wired back to back. Each write is JSON-encoded, queued, and parsed by the other side's reader thread,
so the tests exercise the same line-by-line path the real stdio pipe uses. Scenarios: response correlation, timeout
(and a late reply after it), server request routing, -32601 for unknown server requests, notification order,
emittedAtMs dropped, the reader-thread guard, close failing pending calls, and at-most-once answers.
"""

import json
import queue
import threading
import time
import unittest

from tg_bridge.agents.codex.rpc import (
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    JsonRpcPeer,
    PeerClosed,
    RpcError,
)

WAIT = 10.0
SHORT = 0.5


class Endpoint:
    """One side of the pair: a peer, its inbox, and the reader thread that feeds the peer."""

    def __init__(self, name: str, reports: list[str]) -> None:
        self.name = name
        self.inbox: queue.Queue[str | None] = queue.Queue()
        self.sent: list[dict] = []
        self.refuse_writes = False
        self.other: "Endpoint | None" = None
        self.reader: threading.Thread | None = None
        self.peer = JsonRpcPeer(self._write, self._on_reader_thread, report=reports.append)

    def _write(self, frame: dict) -> bool:
        if self.refuse_writes:
            return False
        self.sent.append(frame)
        assert self.other is not None
        self.other.inbox.put(json.dumps(frame, ensure_ascii=False))
        return True

    def _on_reader_thread(self) -> bool:
        return threading.current_thread() is self.reader

    def _run(self) -> None:
        while True:
            line = self.inbox.get()
            if line is None:
                break
            self.peer.handle_line(json.loads(line))
        self.peer.close()

    def start(self) -> None:
        self.reader = threading.Thread(target=self._run, name=f"reader-{self.name}", daemon=True)
        self.reader.start()

    def stop(self) -> None:
        self.inbox.put(None)
        if self.reader is not None:
            self.reader.join(WAIT)


class PipePair(unittest.TestCase):
    def setUp(self) -> None:
        self.reports: list[str] = []
        self.a = Endpoint("a", self.reports)
        self.b = Endpoint("b", self.reports)
        self.a.other = self.b
        self.b.other = self.a
        self.a.start()
        self.b.start()

    def tearDown(self) -> None:
        self.a.stop()
        self.b.stop()


class ResponseCorrelationTest(PipePair):
    def test_responses_return_to_the_matching_caller(self) -> None:
        self.b.peer.on_request("echo", lambda rid, params: self.b.peer.respond(rid, {"echo": params}))
        first = self.a.peer.request("echo", {"x": 1}, timeout=WAIT)
        second = self.a.peer.request("echo", {"x": 2}, timeout=WAIT)
        self.assertEqual(first, {"echo": {"x": 1}})
        self.assertEqual(second, {"echo": {"x": 2}})

    def test_request_ids_are_increasing_ints_from_one(self) -> None:
        self.b.peer.on_request("echo", lambda rid, params: self.b.peer.respond(rid, None))
        self.a.peer.request("echo", timeout=WAIT)
        self.a.peer.request("echo", timeout=WAIT)
        ids = [frame["id"] for frame in self.a.sent]
        self.assertEqual(ids, [1, 2])
        self.assertTrue(all(type(i) is int for i in ids))

    def test_error_reply_raises_rpc_error_with_code(self) -> None:
        self.b.peer.on_request("fail", lambda rid, params: self.b.peer.error(rid, -32000, "boom"))
        with self.assertRaises(RpcError) as ctx:
            self.a.peer.request("fail", timeout=WAIT)
        self.assertEqual(ctx.exception.code, -32000)
        self.assertEqual(ctx.exception.message, "boom")

    def test_request_without_params_sends_an_empty_object(self) -> None:
        self.b.peer.on_request("ping", lambda rid, params: self.b.peer.respond(rid, "pong"))
        self.assertEqual(self.a.peer.request("ping", timeout=WAIT), "pong")
        self.assertEqual(self.a.sent[0]["params"], {})

    def test_reply_with_unknown_id_is_ignored(self) -> None:
        self.a.peer.handle_line({"id": 999, "result": {"stray": True}})
        self.b.peer.on_request("echo", lambda rid, params: self.b.peer.respond(rid, "ok"))
        self.assertEqual(self.a.peer.request("echo", timeout=WAIT), "ok")


class TimeoutTest(PipePair):
    def test_timeout_raises_then_late_reply_is_ignored(self) -> None:
        release = threading.Event()
        self.b.peer.on_request("slow", lambda rid, params: release.wait(WAIT) and self.b.peer.respond(rid, "late"))
        with self.assertRaises(TimeoutError):
            self.a.peer.request("slow", timeout=SHORT)
        release.set()
        self.b.peer.on_request("fast", lambda rid, params: self.b.peer.respond(rid, "fresh"))
        self.assertEqual(self.a.peer.request("fast", timeout=WAIT), "fresh")
        self.assertEqual(self.a.peer.request("fast", timeout=WAIT), "fresh")


class ServerRequestTest(PipePair):
    def test_server_request_is_routed_and_answered_by_handler(self) -> None:
        seen: list[tuple[int, dict]] = []

        def approve(rid: int, params: dict) -> None:
            seen.append((rid, params))
            self.a.peer.respond(rid, {"decision": "accept"})

        self.a.peer.on_request("approve", approve)
        answer = self.b.peer.request("approve", {"command": "ls"}, timeout=WAIT)
        self.assertEqual(answer, {"decision": "accept"})
        self.assertEqual(seen, [(1, {"command": "ls"})])

    def test_handler_may_answer_later(self) -> None:
        held: list[int] = []
        self.a.peer.on_request("approve", lambda rid, params: held.append(rid))
        result: list[object] = []
        caller = threading.Thread(target=lambda: result.append(self.b.peer.request("approve", timeout=WAIT)))
        caller.start()
        while not held:
            time.sleep(0.01)
        self.assertTrue(self.a.peer.respond(held[0], "later"))
        caller.join(WAIT)
        self.assertEqual(result, ["later"])

    def test_unknown_server_request_gets_minus_32601(self) -> None:
        with self.assertRaises(RpcError) as ctx:
            self.b.peer.request("item/tool/requestUserInput", {}, timeout=WAIT)
        self.assertEqual(ctx.exception.code, METHOD_NOT_FOUND)
        self.assertEqual(ctx.exception.message, "unsupported")

    def test_handler_that_raises_gets_internal_error_once(self) -> None:
        def broken(rid: int, params: dict) -> None:
            raise ValueError("bad params")

        self.a.peer.on_request("broken", broken)
        with self.assertRaises(RpcError) as ctx:
            self.b.peer.request("broken", timeout=WAIT)
        self.assertEqual(ctx.exception.code, INTERNAL_ERROR)
        self.assertTrue(any("bad params" in r for r in self.reports))

    def test_second_answer_to_the_same_server_request_is_refused(self) -> None:
        results: list[tuple[bool, bool]] = []

        def answer_twice(rid: int, params: dict) -> None:
            first = self.a.peer.respond(rid, "one")
            second = self.a.peer.respond(rid, "two")
            results.append((first, second))

        self.a.peer.on_request("twice", answer_twice)
        self.assertEqual(self.b.peer.request("twice", timeout=WAIT), "one")
        self.assertEqual(results, [(True, False)])
        self.assertEqual(sum(1 for f in self.a.sent if f.get("id") == 1), 1)

    def test_answer_for_unseen_id_is_refused(self) -> None:
        self.assertFalse(self.a.peer.respond(42, "nobody asked"))
        self.assertFalse(self.a.peer.error(42, -1, "nobody asked"))
        self.assertEqual(self.a.sent, [])

    def test_server_request_with_string_id_is_ignored_and_reported(self) -> None:
        self.a.peer.handle_line({"id": "x1", "method": "approve", "params": {}})
        self.assertEqual(self.a.sent, [])
        self.assertTrue(any("non-integer id" in r for r in self.reports))


class NotificationTest(PipePair):
    def test_notification_order_is_preserved(self) -> None:
        got: list[int] = []
        self.a.peer.on_notification(lambda method, params: got.append(params["i"]))
        self.a.peer.on_request("sync", lambda rid, params: self.a.peer.respond(rid, "synced"))
        for i in range(50):
            self.b.peer.notify("item/agentMessage/delta", {"i": i})
        self.assertEqual(self.b.peer.request("sync", timeout=WAIT), "synced")
        self.assertEqual(got, list(range(50)))

    def test_method_name_is_passed_to_the_handler(self) -> None:
        got: list[tuple[str, dict]] = []
        self.a.peer.on_notification(lambda method, params: got.append((method, params)))
        self.b.peer.notify("turn/completed", {"status": "completed"})
        self.b.peer.on_request("sync", lambda rid, params: self.b.peer.respond(rid, None))
        self.a.peer.request("sync", timeout=WAIT)
        self.assertEqual(got, [("turn/completed", {"status": "completed"})])

    def test_emitted_at_ms_is_dropped_from_params(self) -> None:
        got: list[dict] = []
        self.a.peer.on_notification(lambda method, params: got.append(params))
        self.b.peer.notify("thread/tokenUsage/updated", {"emittedAtMs": 1700, "total": 5})
        self.b.peer.on_request("sync", lambda rid, params: self.b.peer.respond(rid, None))
        self.a.peer.request("sync", timeout=WAIT)
        self.assertEqual(got, [{"total": 5}])

    def test_notification_without_handler_is_dropped_quietly(self) -> None:
        self.b.peer.notify("warning", {"message": "unused"})
        self.b.peer.on_request("sync", lambda rid, params: self.b.peer.respond(rid, "still alive"))
        self.assertEqual(self.a.peer.request("sync", timeout=WAIT), "still alive")

    def test_notify_without_params_omits_the_key(self) -> None:
        self.assertTrue(self.a.peer.notify("initialized"))
        self.assertEqual(self.a.sent, [{"method": "initialized"}])

    def test_failing_notification_handler_does_not_kill_the_reader(self) -> None:
        def boom(method: str, params: dict) -> None:
            raise RuntimeError("handler bug")

        self.a.peer.on_notification(boom)
        self.b.peer.notify("anything", {})
        self.b.peer.on_request("sync", lambda rid, params: self.b.peer.respond(rid, "alive"))
        self.assertEqual(self.a.peer.request("sync", timeout=WAIT), "alive")
        self.assertTrue(any("handler bug" in r for r in self.reports))


class ReaderThreadGuardTest(PipePair):
    def test_request_from_the_reader_thread_raises(self) -> None:
        outcome: list[BaseException | str] = []

        def try_request(method: str, params: dict) -> None:
            try:
                self.a.peer.request("inner", {}, timeout=SHORT)
                outcome.append("returned")
            except BaseException as exc:
                outcome.append(exc)

        self.a.peer.on_notification(try_request)
        self.b.peer.notify("trigger", {})
        self.b.peer.on_request("sync", lambda rid, params: self.b.peer.respond(rid, "done"))
        self.assertEqual(self.a.peer.request("sync", timeout=WAIT), "done")
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], RuntimeError)
        self.assertIn("reader thread", str(outcome[0]))

    def test_request_from_another_thread_is_allowed(self) -> None:
        self.b.peer.on_request("echo", lambda rid, params: self.b.peer.respond(rid, "from main"))
        self.assertEqual(self.a.peer.request("echo", timeout=WAIT), "from main")


class CloseTest(PipePair):
    def test_close_fails_pending_request(self) -> None:
        arrived = threading.Event()
        self.b.peer.on_request("hang", lambda rid, params: arrived.set())
        errors: list[BaseException] = []

        def call() -> None:
            try:
                self.a.peer.request("hang", timeout=WAIT)
            except BaseException as exc:
                errors.append(exc)

        caller = threading.Thread(target=call)
        caller.start()
        self.assertTrue(arrived.wait(WAIT))
        self.a.peer.close()
        caller.join(WAIT)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], PeerClosed)

    def test_after_close_calls_fail_fast(self) -> None:
        self.a.peer.close()
        self.a.peer.close()
        self.assertTrue(self.a.peer.closed)
        with self.assertRaises(PeerClosed):
            self.a.peer.request("anything", timeout=WAIT)
        self.assertFalse(self.a.peer.notify("anything"))
        self.assertFalse(self.a.peer.respond(1, None))

    def test_stream_end_closes_the_peer(self) -> None:
        self.b.stop()
        self.b.reader.join(WAIT)
        self.assertTrue(self.b.peer.closed)

    def test_refused_write_raises_peer_closed(self) -> None:
        self.a.refuse_writes = True
        with self.assertRaises(PeerClosed):
            self.a.peer.request("anything", timeout=WAIT)
        self.assertFalse(self.a.peer.notify("anything"))


if __name__ == "__main__":
    unittest.main()
