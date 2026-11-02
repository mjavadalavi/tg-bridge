"""FakeBackend / FakeSession: scripted events arrive in order, approvals block until answer(), interrupt ends a turn.

These tests pin the fake that every turn-level test (TurnRunner, permissions, feed) relies on, so a failure here
means the fake is wrong, not the service under test.
"""

import threading
import time
import unittest
from unittest import mock

from tests.fakes import backend as fake_backend
from tests.fakes.backend import FakeBackend, Later, RecordingSink, WaitForInterrupt, fake_spec
from tg_bridge.agents.events import (
    ApprovalRequest,
    Decision,
    SessionStarted,
    TextEvent,
    ToolEnd,
    ToolStart,
    TurnBegan,
    TurnResult,
    TurnStatus,
)

APPROVAL = ApprovalRequest(
    approval_id="req-1",
    kind="command",
    tool="Bash",
    input={"command": "git status"},
    summary="$ git status",
    choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY}),
    always_hint="Bash(git status *)",
)


def _started_session(backend: FakeBackend, sink: RecordingSink, **overrides):
    session = backend.open_session(fake_spec(**overrides), sink)
    session.start()
    return session


class ScriptPlaybackTests(unittest.TestCase):
    def test_script_emits_events_in_order(self) -> None:
        turn = [
            TextEvent(text="looking"),
            ToolStart(tool_id="t1", name="Bash", label="$ git status"),
            ToolEnd(tool_id="t1", ok=True),
            TextEvent(text="done", final=True),
            TurnResult(TurnStatus.OK, answer="done"),
        ]
        backend = FakeBackend([turn])
        sink = RecordingSink()
        session = _started_session(backend, sink)
        session.send_user("hello")
        self.assertTrue(session.wait_idle())
        self.assertIsInstance(sink.events[0], SessionStarted)
        self.assertEqual(sink.events[1:], turn)
        self.assertEqual(session.calls, [("start",), ("send_user", "hello", ())])
        self.assertEqual(backend.turns_left(), 0)

    def test_session_started_comes_with_the_first_turn_not_from_start(self) -> None:
        """The real Claude session announces its id on the first user message (system/init), so start() emits nothing."""
        backend = FakeBackend([[TurnResult(TurnStatus.OK, answer="ok")]])
        sink = RecordingSink()
        session = _started_session(backend, sink)
        self.assertEqual(sink.events, [])
        session.send_user("hi")
        self.assertTrue(session.wait_idle())
        self.assertIsInstance(sink.events[0], SessionStarted)

    def test_session_started_carries_spec_fields(self) -> None:
        backend = FakeBackend()
        sink = RecordingSink()
        session = backend.open_session(fake_spec(cwd="/work", mode="plan", model="m1", resume_id="abc"), sink)
        session.start()
        session.send_user("hi")  # like Claude, SessionStarted comes with the first turn, not from start()
        self.assertTrue(session.wait_idle())
        started = sink.events[0]
        self.assertEqual(started.session_id, "abc")
        self.assertEqual((started.cwd, started.mode, started.model, started.resumed),
                         ("/work", "plan", "m1", True))

    def test_new_session_gets_generated_id_or_new_id(self) -> None:
        backend = FakeBackend()
        generated = _started_session(backend, RecordingSink())
        preassigned = _started_session(backend, RecordingSink(), new_id="pre-1")
        self.assertEqual(generated.session_id, "fake-session-1")
        self.assertEqual(preassigned.session_id, "pre-1")

    def test_exhausted_script_reports_error(self) -> None:
        backend = FakeBackend()
        sink = RecordingSink()
        session = _started_session(backend, sink)
        session.send_user("hi")
        self.assertTrue(session.wait_idle())
        last = sink.events[-1]
        self.assertIsInstance(last, TurnResult)
        self.assertEqual(last.status, TurnStatus.ERROR)
        self.assertEqual(last.error, "fake script exhausted")

    def test_turns_are_consumed_in_order_across_sessions(self) -> None:
        backend = FakeBackend([[TurnResult(TurnStatus.OK, answer="first")],
                               [TurnResult(TurnStatus.OK, answer="second")]])
        first_sink, second_sink = RecordingSink(), RecordingSink()
        first = _started_session(backend, first_sink)
        second = _started_session(backend, second_sink)
        first.send_user("a")
        self.assertTrue(first.wait_idle())
        second.send_user("b")
        self.assertTrue(second.wait_idle())
        self.assertEqual(first_sink.events[-1].answer, "first")
        self.assertEqual(second_sink.events[-1].answer, "second")

    def test_send_user_while_turn_runs_is_refused_and_keeps_the_turn(self) -> None:
        backend = FakeBackend([[APPROVAL, TurnResult(TurnStatus.OK, answer="x")]])
        sink = RecordingSink()
        session = _started_session(backend, sink)
        session.send_user("one")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest)))
        with self.assertRaises(RuntimeError):
            session.send_user("two")
        self.assertEqual(backend.turns_left(), 0)
        session.answer(APPROVAL, Decision.ALLOW_ONCE)
        self.assertTrue(session.wait_idle())

    def test_send_user_before_start_is_refused(self) -> None:
        backend = FakeBackend([[TurnResult(TurnStatus.OK, answer="x")]])
        session = backend.open_session(fake_spec(), RecordingSink())
        with self.assertRaises(RuntimeError):
            session.send_user("too early")
        self.assertEqual(backend.turns_left(), 1)


class ApprovalTests(unittest.TestCase):
    def test_answer_unblocks_the_approval_and_the_turn_finishes(self) -> None:
        backend = FakeBackend([[APPROVAL, TurnResult(TurnStatus.OK, answer="ran it")]])
        sink = RecordingSink()
        session = _started_session(backend, sink)
        session.send_user("status please")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest)))
        self.assertFalse(session.wait_idle(timeout=0.2), "the turn must wait for the approval")
        self.assertFalse(any(isinstance(e, TurnResult) for e in sink.events))
        session.answer(APPROVAL, Decision.ALLOW_ALWAYS, updated_input={"command": "git status"})
        self.assertTrue(session.wait_idle())
        self.assertEqual(sink.events[-1], TurnResult(TurnStatus.OK, answer="ran it"))
        self.assertIn(("answer", "req-1", Decision.ALLOW_ALWAYS, {"command": "git status"}, None), session.calls)

    def test_answer_for_another_approval_does_not_unblock(self) -> None:
        other = ApprovalRequest(approval_id="req-2", kind="tool", tool="Edit", input={}, summary="edit",
                                choices=frozenset({Decision.DENY}))
        backend = FakeBackend([[APPROVAL, TurnResult(TurnStatus.OK, answer="x")]])
        sink = RecordingSink()
        session = _started_session(backend, sink)
        session.send_user("go")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest)))
        session.answer(other, Decision.DENY)
        self.assertFalse(session.wait_idle(timeout=0.2))
        session.answer(APPROVAL, Decision.ALLOW_ONCE)
        self.assertTrue(session.wait_idle())

    def test_interrupt_during_approval_yields_interrupted(self) -> None:
        backend = FakeBackend([[APPROVAL, TurnResult(TurnStatus.OK, answer="x")]])
        sink = RecordingSink()
        session = _started_session(backend, sink)
        session.send_user("go")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ApprovalRequest)))
        self.assertTrue(session.interrupt())
        self.assertTrue(session.wait_idle())
        self.assertEqual(sink.events[-1].status, TurnStatus.INTERRUPTED)


class InterruptAndStopTests(unittest.TestCase):
    def test_interrupt_yields_turn_result_interrupted(self) -> None:
        backend = FakeBackend([[TextEvent(text="working"), WaitForInterrupt()]])
        sink = RecordingSink()
        session = _started_session(backend, sink)
        session.send_user("sleep 120")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TextEvent)))
        self.assertTrue(session.interrupt())
        self.assertTrue(session.wait_idle())
        last = sink.events[-1]
        self.assertIsInstance(last, TurnResult)
        self.assertEqual(last.status, TurnStatus.INTERRUPTED)
        self.assertIsNone(last.answer)
        self.assertTrue(session.alive())

    def test_interrupt_when_idle_returns_false(self) -> None:
        session = _started_session(FakeBackend(), RecordingSink())
        self.assertFalse(session.interrupt())
        self.assertEqual(session.calls[-1], ("interrupt",))

    def test_interrupt_after_result_emits_nothing_more(self) -> None:
        backend = FakeBackend([[TurnResult(TurnStatus.OK, answer="done")]])
        sink = RecordingSink()
        session = _started_session(backend, sink)
        session.send_user("hi")
        self.assertTrue(session.wait_idle())
        self.assertFalse(session.interrupt(), "no turn is running, so there is nothing to interrupt")
        self.assertEqual(len([e for e in sink.events if isinstance(e, TurnResult)]), 1)

    def test_stop_ends_the_turn_without_a_result(self) -> None:
        backend = FakeBackend([[WaitForInterrupt(), TurnResult(TurnStatus.OK, answer="never")]])
        sink = RecordingSink()
        session = _started_session(backend, sink)
        session.send_user("hang")
        session.stop(grace=2.0)
        self.assertTrue(session.wait_idle())
        self.assertFalse(session.alive())
        self.assertFalse(any(isinstance(e, TurnResult) for e in sink.events))
        self.assertEqual(session.calls[-1], ("stop", 2.0))

    def test_set_mode_and_model_report_live_capability(self) -> None:
        session = _started_session(FakeBackend(), RecordingSink())
        self.assertTrue(session.set_mode("plan"))
        self.assertTrue(session.set_model("m2"))
        self.assertEqual(session.calls[-2:], [("set_mode", "plan"), ("set_model", "m2")])


class LaterTests(unittest.TestCase):
    def test_later_waits_for_its_gate_then_emits_turn_began_and_items(self) -> None:
        gate = threading.Event()
        later = Later(gate=gate, items=(TextEvent(text="background done"), TurnResult(TurnStatus.OK, answer="bg")))
        backend = FakeBackend([[TextEvent(text="user"), TurnResult(TurnStatus.OK, answer="user answer"), later]])
        sink = RecordingSink()
        session = _started_session(backend, sink)
        session.send_user("start")
        self.assertTrue(session.wait_idle())
        self.assertFalse(sink.wait_for(lambda e: isinstance(e, TurnBegan), timeout=0.2),
                         "the gate is not set, so the self turn must not begin")
        gate.set()
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnBegan)))
        self.assertTrue(session.wait_idle())
        self.assertEqual(sink.events[1:], [
            TextEvent(text="user"),
            TurnResult(TurnStatus.OK, answer="user answer"),
            TurnBegan(origin="task-notification"),
            TextEvent(text="background done"),
            TurnResult(TurnStatus.OK, answer="bg"),
        ])

    def test_stop_before_the_gate_plays_nothing(self) -> None:
        for ending in ("stop", "end_gracefully"):
            with self.subTest(ending=ending):
                gate = threading.Event()
                later = Later(gate=gate, items=(TextEvent(text="late"), TurnResult(TurnStatus.OK, answer="late")))
                backend = FakeBackend([[TurnResult(TurnStatus.OK, answer="user"), later]])
                sink = RecordingSink()
                session = _started_session(backend, sink)
                session.send_user("start")
                self.assertTrue(session.wait_idle())
                if ending == "stop":
                    session.stop(grace=2.0)
                else:
                    session.end_gracefully(1.0)
                gate.set()
                # wait_idle() is true while the Later waits, so give the Later time to notice the gate first
                self.assertFalse(sink.wait_for(lambda e: isinstance(e, TurnBegan), timeout=0.3),
                                 "a closed session must not start the Later's turn")
                self.assertTrue(session.wait_idle())
                self.assertEqual(sink.events[1:], [TurnResult(TurnStatus.OK, answer="user")])

    def test_a_timed_out_later_leaves_a_user_turn_busy(self) -> None:
        # the Later's wait ends (WAIT_SECONDS) while a user turn started after it still runs: that turn keeps its busy flag
        with mock.patch.object(fake_backend, "WAIT_SECONDS", 3.0):
            later = Later(gate=threading.Event(), items=(TurnResult(TurnStatus.OK, answer="never"),))
            backend = FakeBackend([[TurnResult(TurnStatus.OK, answer="user one"), later],
                                   [WaitForInterrupt(), TurnResult(TurnStatus.OK, answer="never")]])
            sink = RecordingSink()
            session = _started_session(backend, sink)
            session.send_user("one")
            self.assertTrue(session.wait_idle())
            time.sleep(1.0)  # the Later's deadline (3 s from the wait's start) falls after this line
            session.send_user("two")
            self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult) and e.status is TurnStatus.ERROR,
                                          timeout=6.0), "the Later must time out")
            with self.assertRaisesRegex(RuntimeError, "already has a turn running"):
                session.send_user("three")
            self.assertTrue(session.interrupt())
            self.assertTrue(session.wait_idle())
        self.assertEqual(sink.events[-1].status, TurnStatus.INTERRUPTED)

    def test_send_user_during_a_later_turn_raises(self) -> None:
        gate = threading.Event()
        later = Later(gate=gate, items=(WaitForInterrupt(), TurnResult(TurnStatus.OK, answer="never")))
        backend = FakeBackend([[TurnResult(TurnStatus.OK, answer="user"), later]])
        sink = RecordingSink()
        session = _started_session(backend, sink)
        session.send_user("start")
        self.assertTrue(session.wait_idle())
        gate.set()
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnBegan)))
        with self.assertRaises(RuntimeError):
            session.send_user("while the self turn runs")
        self.assertTrue(session.interrupt())
        self.assertTrue(session.wait_idle())
        self.assertEqual(sink.events[-1].status, TurnStatus.INTERRUPTED)

    def test_session_is_idle_between_the_turn_and_its_later(self) -> None:
        gate = threading.Event()
        later = Later(gate=gate, items=(TurnResult(TurnStatus.OK, answer="later"),))
        backend = FakeBackend([[TurnResult(TurnStatus.OK, answer="user one"), later],
                               [TurnResult(TurnStatus.OK, answer="user two")]])
        sink = RecordingSink()
        session = _started_session(backend, sink)
        session.send_user("one")
        self.assertTrue(session.wait_idle())
        self.assertEqual(backend.turns_left(), 1, "the Later is not a scripted turn of its own")
        session.send_user("two")  # idle, so a user turn may start while the Later waits for its gate
        self.assertTrue(session.wait_idle())
        gate.set()
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnBegan)))
        self.assertTrue(session.wait_idle())
        self.assertEqual(sink.events[1:], [
            TurnResult(TurnStatus.OK, answer="user one"),
            TurnResult(TurnStatus.OK, answer="user two"),
            TurnBegan(origin="task-notification"),
            TurnResult(TurnStatus.OK, answer="later"),
        ])

    def test_chained_later_waits_for_each_gate(self) -> None:
        first_gate, second_gate = threading.Event(), threading.Event()
        second = Later(gate=second_gate, items=(TurnResult(TurnStatus.OK, answer="second"),), origin="scheduled")
        first = Later(gate=first_gate, items=(TurnResult(TurnStatus.OK, answer="first"), second))
        backend = FakeBackend([[TurnResult(TurnStatus.OK, answer="user"), first]])
        sink = RecordingSink()
        session = _started_session(backend, sink)
        session.send_user("start")
        self.assertTrue(session.wait_idle())
        first_gate.set()
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnResult) and e.answer == "first"))
        self.assertTrue(session.wait_idle())
        self.assertFalse(sink.wait_for(lambda e: isinstance(e, TurnBegan) and e.origin == "scheduled", timeout=0.2))
        second_gate.set()
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnBegan) and e.origin == "scheduled"))
        self.assertTrue(session.wait_idle())
        self.assertEqual(sink.events[-1], TurnResult(TurnStatus.OK, answer="second"))

    def test_later_must_be_the_last_item_of_its_turn(self) -> None:
        gate = threading.Event()
        with self.assertRaises(ValueError):
            FakeBackend([[Later(gate=gate, items=()), TurnResult(TurnStatus.OK, answer="after")]])
        with self.assertRaises(ValueError):
            FakeBackend([[Later(gate=gate, items=(Later(gate=gate, items=()), TextEvent(text="after")))]])


if __name__ == "__main__":
    unittest.main()
