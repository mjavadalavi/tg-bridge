"""SessionRouter (core/session_router.py): the sink of a Claude session for its whole life. The cases follow the routing
rules in the module docstring: gate delivery and its end, idle session events, self turn buffering, the buffer cap, the
listener errors and the close. Plain doubles only: no process, no thread other than the ones a case starts itself.
"""

import os
import threading
import unittest

from tests.support import TempHome
from tg_bridge.agents.events import (
    ApprovalRequest,
    ApprovalWithdrawn,
    BackgroundWork,
    Decision,
    ProcessExited,
    SessionStarted,
    TextEvent,
    ToolStart,
    TurnBegan,
    TurnResult,
    TurnStatus,
    UsageEvent,
)
from tg_bridge.core.session_router import MAX_BUFFERED_EVENTS, Attach, SessionRouter
from tg_bridge.logging_ import Logger

OK = TurnResult(TurnStatus.OK, answer="ok", duration_ms=5, num_turns=1, subtype="success")
APPROVAL = ApprovalRequest(approval_id="r1", kind="command", tool="Bash", input={"command": "git status"},
                           summary="Bash", choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY}))


class RouterCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.log_path = os.path.join(self.home.home, "bridge.log")
        self.router = SessionRouter(Logger(self.log_path))
        self.gate_seen: list[object] = []
        self.session_seen: list[object] = []
        self.activity: list[bool] = []

    def log_text(self) -> str:
        if not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, encoding="utf-8") as handle:
            return handle.read()

    def hold(self) -> None:
        self.router.hold(self.session_seen.append, lambda: self.activity.append(True))


class SessionRouterTests(RouterCase):
    def test_events_reach_the_attached_gate_in_order(self) -> None:
        self.assertTrue(self.router.attach(self.gate_seen.append))
        events = [TextEvent(text="a"), ToolStart(tool_id="t1", name="Bash", label="ls"),
                  TextEvent(text="b", final=True), OK]
        for event in events:
            self.router(event)
        self.assertEqual(self.gate_seen, events)

    def test_after_the_turn_end_events_are_dropped_like_end_gate(self) -> None:
        self.router.attach(self.gate_seen.append)
        self.router(OK)
        self.router(TurnResult(TurnStatus.ERROR, answer=None, subtype="error_during_execution"))
        self.router(UsageEvent())
        self.assertEqual(self.gate_seen, [OK])
        self.assertEqual(self.log_text().count("turn ended: ignoring a later TurnResult from the process"), 1)
        self.assertNotIn("UsageEvent", self.log_text(), "one log line per idle stretch")

    def test_attach_refused_when_closed_or_already_attached(self) -> None:
        self.assertTrue(self.router.attach(self.gate_seen.append))
        self.assertFalse(self.router.attach(self.session_seen.append), "a turn is attached already")
        self.router(OK)
        self.assertTrue(self.router.attach(self.gate_seen.append), "the gate was detached by the turn's end")
        self.router.close()
        self.assertFalse(self.router.attach(self.gate_seen.append), "a closed router takes no gate")

    def test_close_drops_everything_and_logs_once(self) -> None:
        self.router.attach(self.gate_seen.append)
        self.router.close()
        self.router(TextEvent(text="x"))
        self.router(OK)
        self.router(BackgroundWork(pending=2))
        self.assertEqual(self.gate_seen, [])
        self.assertEqual(self.router.pending, 0)
        self.assertEqual(self.log_text().count("session router closed"), 1)

    def test_background_work_sets_pending_and_activity_and_never_reaches_a_gate(self) -> None:
        self.router.attach(self.gate_seen.append)
        self.router(BackgroundWork(pending=2))
        self.assertEqual(self.router.pending, 2)
        self.assertTrue(self.router.activity)
        self.assertEqual(self.gate_seen, [])
        self.router(OK)
        self.assertTrue(self.router.attach(self.gate_seen.append))
        self.assertFalse(self.router.activity, "attach() clears the activity flag")
        self.assertEqual(self.router.pending, 2, "attach() keeps the pending count")
        self.assertEqual(self.gate_seen, [OK])

    def test_turn_began_is_ignored_unless_held_or_armed_and_no_gate_is_attached(self) -> None:
        self.router(TurnBegan(origin="scheduled"))
        self.assertFalse(self.router.self_pending, "neither held nor armed: ignored")
        self.hold()
        self.router(TurnBegan(origin="task-notification"))
        self.assertTrue(self.router.self_pending)
        self.assertEqual(len(self.activity), 1, "the self turn's activity reaches the session's hook")
        self.assertTrue(self.router.attach(self.gate_seen.append))
        self.assertFalse(self.router.self_pending, "attach() takes the one buffered self turn")
        self.router(TurnBegan(origin="scheduled"))
        self.assertFalse(self.router.self_pending, "a gate is attached: ignored")
        self.assertEqual(self.gate_seen, [])

    def test_self_turn_complete_needs_the_result_of_the_first_waiting_turn(self) -> None:
        self.hold()
        self.assertFalse(self.router.self_turn_complete, "no self turn waits")
        self.router(TurnBegan(origin="task-notification"))
        self.router(TextEvent(text="a"))
        self.assertFalse(self.router.self_turn_complete, "its result has not arrived")
        self.router(OK)
        self.assertTrue(self.router.self_turn_complete)
        self.router(TurnBegan(origin="task-notification"))
        self.assertTrue(self.router.self_turn_complete, "the first waiting turn is the complete one")

    def test_second_self_turn_waits_behind_the_first_and_each_attach_takes_one(self) -> None:
        self.hold()
        first = [TextEvent(text="a"), OK]
        second = [TextEvent(text="b"), OK]
        self.router(TurnBegan(origin="task-notification"))
        for event in first:
            self.router(event)
        self.router(TurnBegan(origin="task-notification"))  # the runner is busy: the second turn starts meanwhile
        for event in second:
            self.router(event)
        self.assertTrue(self.router.self_pending)
        self.assertTrue(self.router.attach(self.gate_seen.append))
        self.assertEqual(self.gate_seen, first, "one attach delivers one self turn, up to its result")
        self.assertTrue(self.router.self_pending, "the second self turn still waits for its gate")
        second_seen: list[object] = []
        self.assertTrue(self.router.attach(second_seen.append))
        self.assertEqual(second_seen, second)
        self.assertFalse(self.router.self_pending)

    def test_armed_router_buffers_a_self_turn_that_starts_after_the_turn_result(self) -> None:
        self.router.arm(self.session_seen.append)
        self.assertTrue(self.router.attach(self.gate_seen.append))
        self.router(OK)  # the user turn's result detaches its gate; the session is not held yet
        self.router(TurnBegan(origin="task-notification"))
        self.assertTrue(self.router.self_pending, "armed: the self turn is buffered, not ignored")
        events = [TextEvent(text="a"), OK]
        for event in events:
            self.router(event)
        self.assertEqual(self.gate_seen, [OK], "nothing of the self turn reaches the old gate")
        self.assertTrue(self.router.attach(self.gate_seen.append))
        self.assertEqual(self.gate_seen, [OK, *events])
        self.assertNotIn("turn ended", self.log_text())

    def test_turn_began_after_the_turn_result_is_ignored_when_not_armed(self) -> None:
        self.router.attach(self.gate_seen.append)
        self.router(OK)
        self.router(TurnBegan(origin="task-notification"))
        self.assertFalse(self.router.self_pending)
        self.router(TextEvent(text="a"))
        self.assertEqual(self.gate_seen, [OK])
        self.assertIn("turn ended: ignoring a later TextEvent", self.log_text())

    def test_close_drops_a_buffered_self_turn(self) -> None:
        self.router.arm(self.session_seen.append)
        self.router.attach(self.gate_seen.append)
        self.router(OK)
        self.router(TurnBegan(origin="task-notification"))
        self.router(TextEvent(text="a"))
        self.router.close()
        self.assertFalse(self.router.self_pending)
        self.assertFalse(self.router.attach(self.gate_seen.append), "a closed router takes no gate")
        self.assertEqual(self.gate_seen, [OK])

    def test_held_idle_sends_approvals_and_session_started_to_the_session_sink(self) -> None:
        self.hold()
        withdrawn = ApprovalWithdrawn(approval_id="r1")
        started = SessionStarted(session_id="s1")
        for event in (APPROVAL, withdrawn, started, TextEvent(text="x"), ToolStart(tool_id="t", name="Bash", label="l"),
                      OK):
            self.router(event)
        self.assertEqual(self.session_seen, [APPROVAL, withdrawn, started])
        self.assertEqual(self.gate_seen, [])
        self.assertIn("turn ended: ignoring a later TextEvent", self.log_text())

    def test_armed_idle_approvals_and_session_started_reach_the_session_sink_without_a_hold(self) -> None:
        # the window between the turn's result and hold(): the router is armed, no gate and no hold
        self.router.arm(self.session_seen.append)
        self.assertTrue(self.router.attach(self.gate_seen.append))
        self.router(OK)
        withdrawn = ApprovalWithdrawn(approval_id="r1")
        started = SessionStarted(session_id="s1")
        for event in (APPROVAL, withdrawn, started, TextEvent(text="x")):
            self.router(event)
        self.assertEqual(self.session_seen, [APPROVAL, withdrawn, started])
        self.assertEqual(self.gate_seen, [OK], "the turn's gate got its end and nothing after it")
        self.assertIn("turn ended: ignoring a later TextEvent", self.log_text())

    def test_not_armed_and_not_held_idle_approval_is_dropped_as_before(self) -> None:
        self.router.attach(self.gate_seen.append)
        self.router(OK)
        self.router(APPROVAL)
        self.assertEqual(self.session_seen, [])
        self.assertEqual(self.gate_seen, [OK])
        self.assertIn("turn ended: ignoring a later ApprovalRequest", self.log_text())

    def test_an_approval_during_the_attached_gate_still_goes_to_the_gate_when_armed(self) -> None:
        self.router.arm(self.session_seen.append)
        self.router.attach(self.gate_seen.append)
        self.router(APPROVAL)
        self.assertEqual(self.gate_seen, [APPROVAL])
        self.assertEqual(self.session_seen, [], "a turn that runs gets its own events, not the session sink")

    def test_close_drops_the_armed_session_sink(self) -> None:
        self.router.arm(self.session_seen.append)
        self.router.close()
        self.router(APPROVAL)
        self.assertEqual(self.session_seen, [])

    def test_self_turn_events_are_buffered_until_attach_then_flushed_in_order(self) -> None:
        self.hold()
        self.router(TurnBegan(origin="task-notification"))
        events = [TextEvent(text="a"), ToolStart(tool_id="t1", name="Bash", label="ls"), OK]
        for event in events:
            self.router(event)
        self.assertEqual(self.gate_seen, [], "nothing is delivered before the runner thread attaches")
        self.assertTrue(self.router.self_pending)
        self.assertTrue(self.router.attach(self.gate_seen.append))
        self.assertEqual(self.gate_seen, events)

    def test_self_turn_events_flushed_while_another_thread_emits_keep_one_strict_order(self) -> None:
        self.hold()
        self.router(TurnBegan(origin="task-notification"))
        first = TextEvent(text="a")
        self.router(first)
        self.router(TextEvent(text="b"))
        emitter = threading.Thread(target=lambda: (self.router(TextEvent(text="c")), self.router(TextEvent(text="d"))))

        def gate(event: object) -> None:
            self.gate_seen.append(event)
            if event == first and not emitter.is_alive() and len(self.gate_seen) == 1:
                emitter.start()
                emitter.join(5)

        self.assertTrue(self.router.attach(gate))
        self.assertFalse(emitter.is_alive())
        self.assertEqual(self.gate_seen, [TextEvent(text="a"), TextEvent(text="b"), TextEvent(text="c"),
                                          TextEvent(text="d")])

    def test_process_exited_sets_exited_and_calls_on_activity(self) -> None:
        self.hold()
        self.router(ProcessExited(returncode=1))
        self.assertTrue(self.router.exited)
        self.assertEqual(self.activity, [True])
        self.assertEqual(self.session_seen, [], "ProcessExited is not a session event: idle, it is dropped")
        self.assertIn("ignoring a later ProcessExited", self.log_text())

    def test_buffer_overflow_drops_the_oldest_and_logs(self) -> None:
        self.hold()
        self.router(TurnBegan(origin="task-notification"))
        total = MAX_BUFFERED_EVENTS + 3
        for index in range(total):
            self.router(TextEvent(text=str(index)))
        self.assertTrue(self.router.attach(self.gate_seen.append))
        self.assertEqual(len(self.gate_seen), MAX_BUFFERED_EVENTS)
        self.assertEqual(self.gate_seen[0], TextEvent(text="3"), "the three oldest events were dropped")
        self.assertEqual(self.gate_seen[-1], TextEvent(text=str(total - 1)))
        self.assertEqual(self.log_text().count("buffer full"), 1, "the overflow is logged once")

    def test_buffer_overflow_never_drops_a_turn_mark(self) -> None:
        self.hold()
        self.router(TurnBegan(origin="task-notification"))
        for index in range(MAX_BUFFERED_EVENTS + 3):
            self.router(TextEvent(text=str(index)))
        self.router(OK)
        self.router(TurnBegan(origin="task-notification"))
        self.router(TextEvent(text="b"))
        self.router(OK)
        self.assertTrue(self.router.attach(self.gate_seen.append))
        self.assertEqual(self.gate_seen[-1], OK, "the first self turn ends at its own result")
        self.assertNotIn(TextEvent(text="b"), self.gate_seen, "the second self turn is not merged into the first")
        self.assertTrue(self.router.self_pending, "the second self turn still waits")
        second_seen: list[object] = []
        self.assertTrue(self.router.attach(second_seen.append))
        self.assertEqual(second_seen, [TextEvent(text="b"), OK])

    def test_buffer_overflow_is_logged_once_per_self_turn(self) -> None:
        self.hold()
        for _ in range(2):
            self.router(TurnBegan(origin="task-notification"))
            for index in range(MAX_BUFFERED_EVENTS + 1):
                self.router(TextEvent(text=str(index)))
            self.assertTrue(self.router.attach(self.gate_seen.append))
            self.router(OK)
        self.assertEqual(self.log_text().count("buffer full"), 2, "each self turn logs its own overflow once")

    def test_a_raising_gate_does_not_stop_the_drain(self) -> None:
        seen: list[object] = []

        def gate(event: object) -> None:
            seen.append(event)
            if len(seen) == 1:
                raise RuntimeError("boom")

        self.router.attach(gate)
        self.router(TextEvent(text="a"))
        self.router(TextEvent(text="b"))
        self.assertEqual(seen, [TextEvent(text="a"), TextEvent(text="b")])
        self.assertIn("a listener raised RuntimeError: boom", self.log_text())


class RaceTests(RouterCase):
    def test_self_turn_that_begins_while_the_gate_end_is_queued_waits_behind_that_end(self) -> None:
        # the drain thread is still delivering the user's text when the user's TurnResult is queued and the CLI starts
        # a self turn: the self turn must wait behind the end, not be ignored, and its events must not reach the gate
        self.hold()
        entered, release = threading.Event(), threading.Event()
        user_seen: list[object] = []

        def slow_gate(event: object) -> None:
            user_seen.append(event)
            if event == TextEvent(text="slow"):
                entered.set()
                release.wait(5)

        self.assertEqual(self.router.attach(slow_gate), Attach.ATTACHED)
        drainer = threading.Thread(target=self.router, args=(TextEvent(text="slow"),))
        drainer.start()
        self.assertTrue(entered.wait(5))
        self.router(OK)  # queued, not delivered: the drainer is busy
        self.router(TurnBegan(origin="task-notification"))
        self_events = [TextEvent(text="b"), OK]
        for event in self_events:
            self.router(event)
        release.set()
        drainer.join(5)
        self.assertEqual(user_seen, [TextEvent(text="slow"), OK], "the user's gate gets its own end and nothing else")
        self.assertTrue(self.router.self_pending, "the self turn began and waits for its gate")
        self_seen: list[object] = []
        self.assertTrue(self.router.attach(self_seen.append))
        self.assertEqual(self_seen, self_events)
        self.assertFalse(self.router.self_pending)

    def test_user_attach_is_refused_while_a_self_turn_waits_and_the_self_turn_runs_first(self) -> None:
        self.hold()
        self.router(TurnBegan(origin="task-notification"))
        self.router(TextEvent(text="bg"))
        self.router(OK)
        self.assertIs(self.router.attach(self.gate_seen.append, user=True), Attach.SELF_FIRST)
        self.assertEqual(self.gate_seen, [], "nothing is attached: the self turn's events are not the user's")
        self.assertTrue(self.router.self_pending)
        self.assertFalse(Attach.SELF_FIRST)
        self.assertTrue(self.router.attach(self.gate_seen.append), "the self turn's own attach takes it")
        self.assertEqual(self.gate_seen, [TextEvent(text="bg"), OK])
        user_seen: list[object] = []
        self.assertIs(self.router.attach(user_seen.append, user=True), Attach.ATTACHED, "after the self turn")
        self.assertEqual(user_seen, [])

    def test_the_gate_ending_flag_is_cleared_by_the_next_attach(self) -> None:
        # a turn's end that was queued and delivered leaves the flag set; the next turn's gate must start with it clear,
        # so a self turn that begins while that gate runs is ignored, as it is for any attached gate
        self.hold()
        self.router.attach(self.gate_seen.append)
        self.router(OK)  # delivered at once: the gate is detached, the end flag stays until the next attach
        user_seen: list[object] = []
        self.assertIs(self.router.attach(user_seen.append), Attach.ATTACHED)
        self.router(TurnBegan(origin="task-notification"))
        self.assertFalse(self.router.self_pending, "a running gate with no pending end: the self turn is ignored")
        self.router(TextEvent(text="a"))
        self.assertEqual(user_seen, [TextEvent(text="a")])

    def test_user_attach_is_refused_while_a_gate_is_attached(self) -> None:
        self.assertIs(self.router.attach(self.gate_seen.append), Attach.ATTACHED)
        self.assertIs(self.router.attach(self.session_seen.append, user=True), Attach.REFUSED)
        self.assertFalse(Attach.REFUSED)
        self.assertTrue(Attach.ATTACHED)


if __name__ == "__main__":
    unittest.main()
