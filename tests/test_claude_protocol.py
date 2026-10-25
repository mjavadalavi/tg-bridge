"""Claude protocol: frame builders and ControlChannel.

Port of the control-protocol checks of fakecli/test_topic_session.py and test_permissions.py: initialize, interrupt,
set_permission_mode, set_model, user turns, allow/deny answers, the unhandled-request reply, correlation by request_id,
timeouts, stream end, and the reader-thread guard.
"""

import json
import os
import sys
import threading
import time
import unittest

from tests.fakes.backend import RecordingSink
from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.claude import protocol
from tg_bridge.agents.claude.protocol import ControlChannel
from tg_bridge.agents.process import ProcessSession
from tg_bridge.logging_ import Logger

WAIT = 15.0
def _python(code: str) -> list[str]:
    return [sys.executable, "-I", "-c", code]


class _Wire:
    """A stdin stand-in for ControlChannel: records frames, and can refuse writes like a closed stdin."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.accept = True
        self._lock = threading.Lock()

    def write(self, frame: dict) -> bool:
        with self._lock:
            self.frames.append(frame)
        return self.accept

    def request_ids(self) -> list[str]:
        with self._lock:
            return [f["request_id"] for f in self.frames if f.get("type") == "control_request"]


def _reply(request_id: str, response: dict | None = None, *, error: str | None = None) -> dict:
    """A control_response event as the CLI sends it."""
    inner = {"subtype": "error", "request_id": request_id, "error": error} if error is not None else {
        "subtype": "success", "request_id": request_id, "response": response or {}}
    return {"type": "control_response", "response": inner}


def _call_in_thread(fn):
    """Run fn on a thread; returns (thread, box) where box holds 'value' or 'error' once it finishes."""
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # the test inspects the exception type
            box["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


def _wait_until(predicate, timeout: float = WAIT) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True




class FrameShapeTests(unittest.TestCase):
    """The shapes of the builders on their own."""

    def test_control_request_puts_subtype_first_then_fields(self) -> None:
        frame = protocol.control_request("req_1_x", "set_model", model="opus")
        self.assertEqual(list(frame), ["type", "request_id", "request"])
        self.assertEqual(list(frame["request"]), ["subtype", "model"])

    def test_user_message_shape(self) -> None:
        self.assertEqual(protocol.user_message("hi", "s-1"), {
            "type": "user", "message": {"role": "user", "content": "hi"},
            "parent_tool_use_id": None, "session_id": "s-1"})

    def test_permission_allow_drops_empty_permissions_but_keeps_empty_input(self) -> None:
        frame = protocol.permission_allow("p", None, [])
        self.assertEqual(frame["response"]["response"], {"behavior": "allow", "updatedInput": {}})

    def test_control_error_shape(self) -> None:
        self.assertEqual(protocol.control_error("h", "nope"), {
            "type": "control_response", "response": {"subtype": "error", "request_id": "h", "error": "nope"}})


class ControlChannelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wire = _Wire()
        self.channel = ControlChannel(self.wire.write, lambda: False, suffix=lambda: "abcd")

    def test_reply_is_matched_to_its_request_and_inner_response_returned(self) -> None:
        thread, box = _call_in_thread(lambda: self.channel.request("interrupt", timeout=WAIT))
        self.assertTrue(_wait_until(lambda: self.wire.request_ids()))
        rid = self.wire.request_ids()[0]
        self.assertEqual(self.wire.frames[0], protocol.interrupt_request(rid))
        self.assertTrue(self.channel.deliver(_reply(rid, {"still_queued": []})))
        thread.join(WAIT)
        self.assertEqual(box.get("value"), {"still_queued": []})
        self.assertEqual(self.channel.pending_count, 0)

    def test_replies_are_matched_by_request_id_not_by_order(self) -> None:
        first_thread, first = _call_in_thread(lambda: self.channel.request("set_model", timeout=WAIT, model="a"))
        second_thread, second = _call_in_thread(lambda: self.channel.request("set_model", timeout=WAIT, model="b"))
        self.assertTrue(_wait_until(lambda: len(self.wire.request_ids()) == 2))
        models = {f["request"]["model"]: f["request_id"] for f in self.wire.frames}
        rid_a, rid_b = models["a"], models["b"]  # the threads race, so map each request by its payload
        self.assertEqual(self.channel.pending_count, 2)
        self.assertTrue(self.channel.deliver(_reply(rid_b, {"from": "b"})))  # answered out of order
        self.assertTrue(self.channel.deliver(_reply(rid_a, {"from": "a"})))
        second_thread.join(WAIT)
        first_thread.join(WAIT)
        self.assertEqual(second.get("value"), {"from": "b"})
        self.assertEqual(first.get("value"), {"from": "a"})

    def test_second_reply_for_the_same_request_is_ignored(self) -> None:
        thread, box = _call_in_thread(lambda: self.channel.request("interrupt", timeout=WAIT))
        self.assertTrue(_wait_until(lambda: self.wire.request_ids()))
        rid = self.wire.request_ids()[0]
        self.assertTrue(self.channel.deliver(_reply(rid, {"n": 1})))
        self.assertFalse(self.channel.deliver(_reply(rid, {"n": 2})))
        thread.join(WAIT)
        self.assertEqual(box.get("value"), {"n": 1})

    def test_error_reply_raises_runtime_error_with_the_cli_text(self) -> None:
        thread, box = _call_in_thread(lambda: self.channel.request("set_permission_mode", timeout=WAIT, mode="x"))
        self.assertTrue(_wait_until(lambda: self.wire.request_ids()))
        self.channel.deliver(_reply(self.wire.request_ids()[0], error="bad mode"))
        thread.join(WAIT)
        self.assertIsInstance(box.get("error"), RuntimeError)
        self.assertEqual(str(box["error"]), "bad mode")

    def test_error_reply_without_text_uses_a_generic_message(self) -> None:
        thread, box = _call_in_thread(lambda: self.channel.request("set_model", timeout=WAIT, model="x"))
        self.assertTrue(_wait_until(lambda: self.wire.request_ids()))
        self.channel.deliver(_reply(self.wire.request_ids()[0], error=""))
        thread.join(WAIT)
        self.assertEqual(str(box.get("error")), "control error")

    def test_timeout_raises_and_clears_the_pending_request(self) -> None:
        with self.assertRaises(TimeoutError) as caught:
            self.channel.request("interrupt", timeout=0.05)
        self.assertEqual(str(caught.exception), "interrupt timed out after 0.05s")
        self.assertEqual(self.channel.pending_count, 0)
        self.assertFalse(self.channel.deliver(_reply("req_1_abcd", {})))  # a late reply finds nothing

    def test_closed_stdin_raises_at_once_and_clears_the_pending_request(self) -> None:
        self.wire.accept = False
        started = time.monotonic()
        with self.assertRaises(RuntimeError) as caught:
            self.channel.request("set_model", timeout=30, model="x")
        self.assertLess(time.monotonic() - started, 5)  # it did not wait for the 30 s timeout
        self.assertEqual(str(caught.exception), "set_model: stdin closed")
        self.assertEqual(self.channel.pending_count, 0)

    def test_stream_end_wakes_waiters_with_session_ended(self) -> None:
        thread, box = _call_in_thread(lambda: self.channel.request("interrupt", timeout=30))
        self.assertTrue(_wait_until(lambda: self.channel.pending_count == 1))
        self.channel.fail_all()
        thread.join(WAIT)
        self.assertIsInstance(box.get("error"), RuntimeError)
        self.assertEqual(str(box["error"]), "interrupt: session ended before reply")
        self.assertEqual(self.channel.pending_count, 0)

    def test_deliver_ignores_unknown_and_malformed_events(self) -> None:
        self.assertFalse(self.channel.deliver({"type": "control_response", "response": {"request_id": "nope"}}))
        self.assertFalse(self.channel.deliver({"type": "control_response", "response": "text"}))
        self.assertFalse(self.channel.deliver({}))

    def test_request_ids_are_numbered_and_unique(self) -> None:
        for _ in range(3):
            with self.assertRaises(TimeoutError):
                self.channel.request("interrupt", timeout=0.01)
        self.assertEqual(self.wire.request_ids(), ["req_1_abcd", "req_2_abcd", "req_3_abcd"])

    def test_initialize_sends_hooks_none_and_raises_on_timeout(self) -> None:
        with self.assertRaises(TimeoutError) as caught:
            self.channel.initialize(timeout=0.01)
        self.assertEqual(str(caught.exception), "initialize timed out after 0.01s")
        self.assertEqual(self.wire.frames[0], protocol.initialize_request("req_1_abcd"))

    def test_reader_thread_guard_refuses_before_writing(self) -> None:
        channel = ControlChannel(self.wire.write, lambda: True, suffix=lambda: "abcd")
        with self.assertRaises(RuntimeError) as caught:
            channel.request("interrupt", timeout=30)
        self.assertIn("reader thread", str(caught.exception))
        self.assertEqual(self.wire.frames, [])
        self.assertEqual(channel.pending_count, 0)


class JsonChannelSession(ProcessSession):
    """A real child process with a ControlChannel wired the way the session will wire it."""

    def __init__(self, argv: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._argv_value = list(argv)
        self.channel = ControlChannel(self.write, self.on_reader_thread, suffix=lambda: "live")
        self.guard_errors: list[tuple[str, str]] = []

    def _argv(self) -> list[str]:
        return list(self._argv_value)

    def _on_line(self, line: str) -> None:
        if line == "probe":  # the probe child asks from inside the reader thread
            try:
                self.channel.request("interrupt", timeout=1)
            except RuntimeError as e:  # a TimeoutError is not a RuntimeError, so it would escape: that means the guard failed
                self.guard_errors.append((type(e).__name__, str(e)))
            return
        try:
            event = json.loads(line)
        except ValueError:
            return
        if isinstance(event, dict) and event.get("type") == "control_response":
            self.channel.deliver(event)


ECHO_CHILD = (
    "import sys, json\n"
    "print('ready', flush=True)\n"
    "for line in sys.stdin:\n"
    "    req = json.loads(line)\n"
    "    print(json.dumps({'type': 'control_response', 'response': {'subtype': 'success', 'request_id': "
    "req['request_id'], 'response': {'echo': req['request']['subtype']}}}), flush=True)\n"
)
PROBE_CHILD = (
    "import sys\n"
    "print('probe', flush=True)\n"
    "sys.stdin.read()\n"
)


class ChannelOverRealProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.sessions: list[JsonChannelSession] = []
        self.addCleanup(self._stop_all)

    def _stop_all(self) -> None:
        for session in self.sessions:
            session.stop(grace=1)

    def _session(self, code: str) -> tuple[JsonChannelSession, RecordingSink]:
        sink = RecordingSink()
        session = JsonChannelSession(
            _python(code), name="probe", cwd=self.home.home, env=dict(os.environ),
            errfile_dir=os.path.join(self.home.state, "turns"), sink=sink, logger=self.logger, clock=FakeClock(),
        )
        self.sessions.append(session)
        return session, sink

    def test_request_round_trips_through_a_real_child_and_its_reader(self) -> None:
        session, sink = self._session(ECHO_CHILD)
        session.start()
        result = session.channel.request("set_model", timeout=WAIT, model="sonnet")
        self.assertEqual(result, {"echo": "set_model"})
        self.assertEqual(session.channel.pending_count, 0)

    def test_request_from_the_reader_thread_is_refused_not_deadlocked(self) -> None:
        session, sink = self._session(PROBE_CHILD)
        session.start()
        self.assertTrue(_wait_until(lambda: bool(session.guard_errors), timeout=WAIT))
        self.assertTrue(session.guard_errors, "the probe line never reached _on_line")
        error_type, text = session.guard_errors[0]
        self.assertEqual(error_type, "RuntimeError")
        self.assertIn("reader thread", text)
        self.assertEqual(session.channel.pending_count, 0)


if __name__ == "__main__":
    unittest.main()
