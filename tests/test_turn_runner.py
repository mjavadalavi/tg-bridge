"""TurnRunner (STEP 38) on FakeBackend and FakeClock: port of the scenarios of fakecli/test_run_turn_v2.py (labels
'bypass', 'second turn', 'default mode', 'cap', '⏭ Now'), fakecli/test_resume_fallback.py (the fresh relaunch with the
same id) and fakecli/test_slot_release.py (a turn parked on a card gives its permit back and takes it again).

Also covered: the SessionSpec the turn builds (launcher precedence, allow and deny tools, RULES text, TZ), the shutdown
refusal and the marker kept on a shutdown, a launch failure, the usage on the turns row, the answer collection rules and
the dead-process end. The end-to-end runs with real fake CLIs are in tests/test_turn_runner_e2e.py.
"""

import dataclasses
import os
import unittest
from unittest import mock

from tests.fakes.backend import FAKE_CAPABILITIES, FakeSession, WaitForInterrupt
from tests.support import TempHome
from tests.turn_support import (
    CHAT,
    THREAD,
    TOPIC,
    FeedRecorder,
    ProbingBackend,
    RecordingFeed,
    TurnHarness,
    join_turn,
    run_async,
    wait_until,
)
from tg_bridge.agents.events import (
    ApprovalRequest,
    Decision,
    ProcessExited,
    TextEvent,
    ToolEnd,
    ToolStart,
    TurnResult,
    TurnStatus,
    UsageEvent,
)
from tg_bridge.core.session_router import SessionRouter
from tg_bridge.core.turn_listeners import EndGate
from tg_bridge.core.turn_runner import active_key
from tg_bridge.core.turn_state import TurnOutcome
from tg_bridge.core.turn_text import INTERRUPTED_TEXT, NO_CONVERSATION, RESTART_TEXT
from tg_bridge.logging_ import Logger

OK = TurnResult(TurnStatus.OK, answer="ok:allow", duration_ms=5, num_turns=1, subtype="success")
CAPPED_3S = '⏱ stopped after 3s of work. The session is kept — send "continue" to pick up where it left off.'
STORED = "stored-session-0001"


class RunnerCase(unittest.TestCase):
    """A harness in a TempHome; start() builds the fake backend and the runner for the scripted turns."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self._stop_live_turns)  # runs before the TempHome is removed (cleanups run last-in first-out)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")

    def _stop_live_turns(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)

    def start(self, scripts=(), *, capabilities=FAKE_CAPABILITIES, **config_overrides) -> None:
        self.h.set_config(**config_overrides)
        self.backend = ProbingBackend(scripts, capabilities=capabilities)
        self.backend.probe = lambda: self.h.sessions.get(THREAD).uuid
        self.runner = self.h.runner(self.h.registry_for(self.backend))

    def turn(self, text: str = "hello", topic: str = TOPIC, slot=None):
        return self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=topic, text=text, sender="tester", slot=slot)

    def feed_events(self) -> list:
        """The events the newest feed has received (empty until the turn has created its feed)."""
        return self.h.feeds.feeds[-1].events if self.h.feeds.feeds else []

    def sent(self) -> bool:
        """True once the turn has sent its user text to the newest session."""
        return bool(self.backend.sessions) and any(c[0] == "send_user" for c in self.backend.sessions[-1].calls)

    def turn_rows(self, thread: int = THREAD) -> list[dict]:
        return [dict(row) for row in self.h.db.query("SELECT * FROM turns WHERE thread_id=? ORDER BY id", (thread,))]

    def stored_uuid(self) -> str:
        return self.h.sessions.get(THREAD).uuid


class BypassTurnTests(RunnerCase):
    def test_bypass_first_turn_stores_id_before_spawn(self) -> None:
        # port of run_turn_v2 step 1: the uuid is stored before the process exists and passed as the new id
        """Ported labels:
        Q8: the first turn records its session
        bypass: answer ok:allow
        C: bypass turn finished
        uuid stored before spawn and passed as --session-id
        """
        self.start([[TextEvent(text="hello", final=True), OK]])
        reply = self.turn()
        self.assertEqual((reply.answer, reply.err, reply.outcome), ("ok:allow", None, TurnOutcome.OK))
        spec = self.backend.specs[0]
        self.assertTrue(spec.new_id)
        self.assertIsNone(spec.resume_id)
        self.assertEqual(self.backend.probed, spec.new_id)
        self.assertEqual(self.stored_uuid(), spec.new_id)
        self.assertEqual(self.backend.sessions[0].session_id, spec.new_id)
        self.assertIsNotNone(self.h.history.find(THREAD, spec.new_id))

    def test_a_later_turn_on_the_same_uuid_adds_no_history_row(self) -> None:
        """Ported labels:
        Q8: a later turn on the same uuid adds nothing
        """
        self.start([[TextEvent(text="hello", final=True), OK], [TextEvent(text="again", final=True), OK]])
        self.turn()
        uuid = self.stored_uuid()
        before = self.h.db.query("SELECT COUNT(*) AS n FROM session_history WHERE thread_id=?", (THREAD,))[0]["n"]
        self.turn("second message")
        after = self.h.db.query("SELECT COUNT(*) AS n FROM session_history WHERE thread_id=?", (THREAD,))[0]["n"]
        self.assertEqual((before, after), (1, 1))
        self.assertEqual(self.stored_uuid(), uuid)

    def test_bypass_turn_row_feed_and_marker(self) -> None:
        """Ported labels:
        turns row inserted (status ok)
        last_active updated
        """
        self.start([[OK]])
        self.turn()
        rows = self.turn_rows()
        self.assertEqual([(r["status"], r["subtype"], r["backend"]) for r in rows], [("ok", "success", "fake")])
        self.assertNotEqual(self.h.sessions.get(THREAD).last_active, "")
        feed = self.h.feeds.last
        self.assertTrue(feed.started and feed.closed)
        self.assertTrue(feed.headers and feed.headers[-1].startswith("✅"))
        self.assertIsNone(self.h.kv.get(active_key(THREAD)))

    def test_bypass_plan_has_no_allow_list_and_the_tz_env(self) -> None:
        """Ported labels:
        bypass: no allowedTools
        C: launched in bypassPermissions
        C: no deny_tools -> KillShell only
        """
        self.start([[OK]])
        self.turn()
        spec = self.backend.specs[0]
        self.assertEqual(spec.mode, "bypassPermissions")
        self.assertEqual(spec.allow_rules, ())
        self.assertEqual(spec.deny_tools, ("KillShell",))
        self.assertEqual(spec.env["TZ"], "Asia/Tehran")
        self.assertEqual(spec.launcher.command, "topic-cmd")  # the topic's launcher, no pick

    def test_second_turn_resumes_the_stored_id(self) -> None:
        # port of run_turn_v2 step 2
        """Ported labels:
        second turn: --resume with stored uuid
        second turn row inserted
        """
        self.start([[OK], [TurnResult(TurnStatus.OK, answer="ok:allow", subtype="success")]])
        self.turn()
        first_id = self.stored_uuid()
        self.turn("again")
        second = self.backend.specs[1]
        self.assertEqual((second.resume_id, second.new_id), (first_id, None))
        self.assertEqual(len(self.turn_rows()), 2)


class PlanTests(RunnerCase):
    def test_default_mode_plan_carries_saved_rules_deny_list_and_rules_text(self) -> None:
        """Ported labels:
        C: a single string deny_tools is accepted
        session object carries mode
        """
        self.start([[TurnResult(TurnStatus.OK, answer="ok")]],
                   topics={TOPIC: {"launcher": "topic-cmd", "deny_tools": "WebFetch"}})
        self.h.sessions.set_mode(THREAD, "default")
        self.h.rules.add(THREAD, "Edit")
        self.turn()
        spec = self.backend.specs[0]
        self.assertEqual(spec.mode, "default")
        self.assertEqual(spec.allow_rules, ("Read,Glob,Grep", "Edit"))
        self.assertEqual(spec.deny_tools, ("KillShell", "WebFetch"))
        self.assertIn('from their private Telegram topic "side"', spec.system_rules)
        self.assertIn("tester (the owner)", spec.system_rules)

    def test_launcher_precedence_is_pick_then_topic_then_stored(self) -> None:
        self.start([[OK], [OK], [OK]])
        self.h.sessions.set_launcher_pick(THREAD, "alt")
        self.turn()
        self.h.sessions.set_launcher_pick(THREAD, "")
        self.turn()
        self.turn(topic="OTHER")  # no topic entry: the stored launcher
        self.assertEqual([s.launcher.command for s in self.backend.specs],
                         ["alt-cmd", "topic-cmd", "fake-cmd"])

    def test_blank_launcher_is_refused_before_any_id_is_stored(self) -> None:
        self.start([[OK]])
        self.h.sessions.ensure(9102, "blank", "", self.home.home, "bypassPermissions")
        reply = self.runner.run(thread_id=9102, chat_id=CHAT, topic="blank", text="hi", sender="tester")
        self.assertIsNone(reply.answer)
        self.assertTrue(reply.err.startswith("launcher failed: the launcher is empty"))
        self.assertEqual(reply.outcome, TurnOutcome.ERROR)
        self.assertEqual(self.backend.specs, [])
        self.assertEqual(self.h.sessions.get(9102).uuid, "")
        self.assertEqual(self.turn_rows(9102), [])


class PhotoFallbackTests(RunnerCase):
    def test_photos_a_backend_without_image_input_cannot_take_become_text_notes(self) -> None:
        self.start([[OK]], capabilities=dataclasses.replace(FAKE_CAPABILITIES, images=False))
        self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="", sender="tester",
                        images=("/inbox/p1.jpg", "/inbox/p2.jpg"))
        sent = [call for call in self.backend.sessions[0].calls if call[0] == "send_user"]
        note = "[photo saved for you at /inbox/p1.jpg]\n[photo saved for you at /inbox/p2.jpg]"
        self.assertEqual(sent, [("send_user", note, ())])


class PermissionCardTests(RunnerCase):
    def test_parked_turn_gives_its_permit_back_and_takes_it_again(self) -> None:
        # port of run_turn_v2 step 3 and test_slot_release.py: the card holds the turn, the slot is given back meanwhile
        self.start([[ApprovalRequest(approval_id="r1", kind="command", tool="Bash",
                                     input={"command": "git status"}, summary="Bash",
                                     choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY})),
                     TurnResult(TurnStatus.OK, answer="ok:allow")]])
        self.h.sessions.set_mode(THREAD, "default")
        slot = self.h.slots.new_slot(CHAT, THREAD)
        self.assertTrue(slot.take())
        self.assertEqual(self.h.slots.free(), 1)
        thread, box = run_async(self.runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="write",
                                sender="tester", slot=slot)
        self.assertTrue(wait_until(lambda: len(self.h.asks.pending(THREAD)) == 1))
        self.assertTrue(wait_until(lambda: self.h.slots.free() == 2), "the parked turn kept its permit")
        nonce = self.h.asks.pending(THREAD)[0].nonce
        self.assertTrue(self.h.permissions.answer_ask(nonce, allow=True))
        join_turn(self.h.clock, thread)
        reply = box[0]
        self.assertEqual((reply.answer, reply.err), ("ok:allow", None))
        self.assertTrue(slot.held)
        self.assertEqual(self.h.slots.free(), 1)
        self.assertEqual(self.backend.sessions[0].calls[-1][0], "stop")
        slot.give_back()
        self.assertEqual(self.h.slots.free(), 2)


class CapAndStopTests(RunnerCase):
    def test_cap_stops_the_turn_with_the_notice(self) -> None:
        # port of run_turn_v2 step 4: a turn that never answers stops at max_turn_seconds
        self.start([[WaitForInterrupt()]], max_turn_seconds=3)
        thread, box = run_async(self.runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="long job",
                                sender="tester")
        join_turn(self.h.clock, thread)
        reply = box[0]
        self.assertEqual((reply.answer, reply.err, reply.outcome), (None, CAPPED_3S, TurnOutcome.CAPPED))
        self.assertEqual(self.turn_rows()[-1]["status"], "capped")
        self.assertTrue(self.h.feeds.last.headers[-1].startswith("⏱ stopped after 3s"))
        self.assertIsNone(self.h.live.running(THREAD))

    def test_a_capped_turn_keeps_the_main_text_as_its_answer(self) -> None:
        """A subagent's text is never the answer, and a capped turn still answers with the main text it had.

        Ported labels:
        4: main answer kept (got 'MAIN-ANSWER' followed by the capped notice)
        4: subagent text excluded from the answer
        """
        self.start([[TextEvent(text="side", parent_id="p1"), TextEvent(text="MAIN-ANSWER"), WaitForInterrupt()]],
                   max_turn_seconds=3)
        thread, box = run_async(self.runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="long job",
                                sender="tester")
        join_turn(self.h.clock, thread)
        reply = box[0]
        self.assertEqual((reply.answer, reply.err, reply.outcome),
                         ("MAIN-ANSWER\n\n" + CAPPED_3S, None, TurnOutcome.CAPPED))

    def test_a_turn_waiting_on_its_card_is_not_capped(self) -> None:
        """Ported labels:
        D: not capped, ok:allow
        D: turn still alive after 4s with a pending tap (clock paused)
        """
        self.start([[ApprovalRequest(approval_id="r1", kind="command", tool="Bash",
                                     input={"command": "git status"}, summary="Bash",
                                     choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY})),
                     TurnResult(TurnStatus.OK, answer="ok:allow")]], max_turn_seconds=2)
        self.h.sessions.set_mode(THREAD, "default")
        thread, box = run_async(self.runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="write",
                                sender="tester")
        self.assertTrue(wait_until(lambda: len(self.h.asks.pending(THREAD)) == 1))
        for _ in range(4):
            self.h.clock.advance(1)
        self.assertTrue(thread.is_alive(), "the turn was capped while its card waited for a tap")
        self.assertTrue(self.h.permissions.answer_ask(self.h.asks.pending(THREAD)[0].nonce, allow=True))
        join_turn(self.h.clock, thread)
        self.assertEqual((box[0].answer, box[0].err), ("ok:allow", None))

    def test_stop_tap_without_tools_reports_the_interrupt(self) -> None:
        """Ported labels:
        ⏭: interrupted result {'ans': None, 'err': '⏹ interrupted — session kept'}
        ⏭: _aborted cleared, turn row stopped
        I: turn row recorded as stopped
        I: stop markers cleared
        I: session uuid kept
        """
        # port of run_turn_v2 step 5 (⏭ Now): the stop is requested on purpose and the session is kept
        self.start([[WaitForInterrupt()]])
        thread, box = run_async(self.runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="skip me",
                                sender="tester")
        self.assertTrue(wait_until(lambda: self.sent()))
        running = self.h.live.request_abort(THREAD, stop_tap=False)
        self.assertIsNotNone(running)
        running.session.stop()
        join_turn(self.h.clock, thread)
        self.assertEqual((box[0].err, box[0].outcome), (INTERRUPTED_TEXT, TurnOutcome.STOPPED))
        self.assertFalse(self.h.live.is_aborted(THREAD))
        self.assertEqual(self.turn_rows()[-1]["status"], "stopped")
        self.assertTrue(self.stored_uuid(), "the session id stays stored after a stop")

    def test_stop_tap_names_the_tool_count(self) -> None:
        """Ported labels:
        I: turn ended after the Stop tap
        I: stop text ({'ans': None, 'err': '⏹ stopped after 3 tool calls — session kept; send a message to continue'})
        I: the log shows the tool calls before the tap
        """
        self.start([[ToolStart(tool_id="t1", name="Bash", label="$ ls"), ToolEnd(tool_id="t1", ok=True),
                     WaitForInterrupt()]])
        thread, box = run_async(self.runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="go", sender="tester")
        self.assertTrue(wait_until(lambda: any(isinstance(e, ToolEnd) for e in self.feed_events())))
        self.h.live.request_abort(THREAD, stop_tap=True).session.stop()
        join_turn(self.h.clock, thread)
        self.assertEqual(box[0].err, "⏹ stopped after 1 tool calls — session kept; send a message to continue")


class ResumeFallbackTests(RunnerCase):
    FAILED = [TurnResult(TurnStatus.ERROR, answer=None, error=NO_CONVERSATION, resume_failed=True)]

    def test_missing_session_relaunches_once_with_the_same_id(self) -> None:
        # port of test_resume_fallback.py: the note leads the answer, the launches are resume then session-id
        """Ported labels:
        fallback: the answer carries the visible note, then the normal answer
        fallback: exactly two launches (got 2)
        first launch: --resume <stored uuid>
        second launch: --session-id <same uuid>
        fallback: log line written once
        exactly one turns row, status ok (got ['ok'])
        sessions.uuid unchanged
        session removed from registry after turn
        J7: the fallback reply carries a visible note
        """
        self.start([self.FAILED, [OK]])
        self.h.sessions.set_uuid(THREAD, STORED)
        reply = self.turn()
        self.assertIsNone(reply.err)
        self.assertTrue(reply.answer.startswith(f"⚠️ the earlier conversation {STORED} was not found in"))
        self.assertTrue(reply.answer.endswith("under the same id.\n\nok:allow"))
        first, second = self.backend.specs
        self.assertIn(f"under launcher {first.launcher.command} (each launcher keeps its own transcripts)", reply.answer)
        self.assertEqual((first.resume_id, first.new_id), (STORED, None))
        self.assertEqual((second.resume_id, second.new_id), (None, STORED))
        self.assertEqual(self.stored_uuid(), STORED)
        lines = [line for line in self.h.log_lines() if "not found" in line]
        self.assertEqual(len(lines), 1)
        self.assertIn(f"resume {STORED} not found — restarting as new session with same id", lines[0])
        self.assertEqual([r["status"] for r in self.turn_rows()], ["ok"])
        self.assertIsNone(self.h.live.running(THREAD), "the session is removed from the registry after the turn")

    def test_without_a_preassigned_id_the_relaunch_is_a_new_thread(self) -> None:
        self.start([self.FAILED, [OK]], capabilities=dataclasses.replace(FAKE_CAPABILITIES, preassigned_id=False))
        self.h.sessions.set_uuid(THREAD, STORED)
        reply = self.turn()
        self.assertIn("starts empty in a new thread.", reply.answer)
        self.assertIsNone(self.backend.specs[1].new_id)
        self.assertTrue(self.stored_uuid().startswith("fake-session-"))
        self.assertNotEqual(self.stored_uuid(), STORED)

    def test_a_turn_stopped_by_a_tap_is_not_relaunched(self) -> None:
        self.start([[WaitForInterrupt()], [OK]])
        self.h.sessions.set_uuid(THREAD, STORED)
        thread, box = run_async(self.runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="x", sender="tester")
        self.assertTrue(wait_until(lambda: self.sent()))
        self.h.live.request_abort(THREAD, stop_tap=False).session.stop()
        join_turn(self.h.clock, thread)
        self.assertEqual(len(self.backend.specs), 1)
        self.assertEqual(box[0].outcome, TurnOutcome.STOPPED)


class ClockAdvancingFeed(RecordingFeed):
    """A feed whose close() takes time, as the real one does while it flushes (the sender can take up to 20 s)."""

    def __init__(self, clock, seconds: float) -> None:
        super().__init__()
        self._clock = clock
        self._seconds = seconds

    def close(self) -> None:
        self._clock.advance(self._seconds)
        super().close()


class ClockAdvancingFeeds(FeedRecorder):
    def __init__(self, clock, seconds: float) -> None:
        super().__init__()
        self._clock = clock
        self._seconds = seconds

    def __call__(self, chat_id, thread_id, parked):
        feed = ClockAdvancingFeed(self._clock, self._seconds)
        self.feeds.append(feed)
        return feed


class FinishedTurnNearCapTests(RunnerCase):
    def test_a_turn_that_finished_ok_is_not_reclassified_as_capped_by_its_teardown(self) -> None:
        # the result arrives at 0 s of work; closing the feed then takes 4 s, past max_turn_seconds=3. The teardown
        # time is not the model's work, so the turn stays OK, with no cap notice and no interrupt sent to the CLI
        self.h.feeds = ClockAdvancingFeeds(self.h.clock, seconds=4.0)
        self.start([[OK]], max_turn_seconds=3)
        reply = self.turn()
        self.assertEqual((reply.answer, reply.err, reply.outcome), ("ok:allow", None, TurnOutcome.OK))
        self.assertEqual(self.turn_rows()[-1]["status"], "ok")
        self.assertTrue(self.h.feeds.last.headers[-1].startswith("✅"))
        calls = [call[0] for call in self.backend.sessions[0].calls]
        self.assertIn("end_gracefully", calls)
        self.assertNotIn("interrupt", calls)


class ShutdownTests(RunnerCase):
    def test_shutdown_refuses_the_spawn_and_clears_the_marker(self) -> None:
        self.start([[OK]])
        self.h.live.begin_shutdown()
        reply = self.turn()
        self.assertEqual((reply.answer, reply.err, reply.outcome), (None, None, TurnOutcome.RESTART))
        self.assertNotIn("start", [c[0] for c in self.backend.sessions[0].calls])
        self.assertEqual(self.turn_rows(), [])
        self.assertIsNone(self.h.kv.get(active_key(THREAD)))

    def test_shutdown_mid_turn_keeps_the_marker_and_reports_a_restart(self) -> None:
        """Ported labels:
        1: the turn is running in the CLI before the restart
        1: shutdown signalled the one live session (n=1)
        1: turn row stored as interrupted
        1: active marker kept so recover() tells the topic
        """
        self.start([[WaitForInterrupt()]])
        thread, box = run_async(self.runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="x", sender="tester")
        self.assertTrue(wait_until(lambda: self.sent()), "the turn reached the CLI")
        signalled = list(self.h.live.begin_shutdown())
        for session in signalled:
            session.stop()
        self.assertEqual(len(signalled), 1)
        join_turn(self.h.clock, thread)
        self.assertEqual((box[0].err, box[0].outcome), (RESTART_TEXT, TurnOutcome.RESTART))
        self.assertEqual(self.turn_rows()[-1]["status"], "interrupted")
        self.assertIsNotNone(self.h.kv.get(active_key(THREAD)))


class FailureAndAnswerTests(RunnerCase):
    def test_launch_failure_is_reported_and_recorded(self) -> None:
        self.start([[OK]])
        with mock.patch.object(FakeSession, "start", side_effect=RuntimeError("boom")):
            reply = self.turn()
        self.assertEqual((reply.answer, reply.err, reply.outcome), (None, "launcher failed: boom", TurnOutcome.ERROR))
        self.assertEqual([(r["status"], r["error"]) for r in self.turn_rows()], [("error", "launcher failed: boom")])

    def test_process_exit_without_a_result_is_an_error(self) -> None:
        self.start([[ProcessExited(returncode=3)]])
        reply = self.turn()
        self.assertEqual((reply.answer, reply.err), (None, "process ended without an answer (rc=3)"))
        self.assertEqual(self.turn_rows()[-1]["status"], "error")

    def test_dead_process_ends_the_turn_after_the_dead_delay(self) -> None:
        self.start([[TextEvent(text="partial")]])
        with mock.patch.object(FakeSession, "alive", return_value=False), \
                mock.patch.object(FakeSession, "exit_code", 1):
            thread, box = run_async(self.runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="x",
                                    sender="tester")
            join_turn(self.h.clock, thread)
        # the child's own return code (taken from the stop that reaps it), not None
        self.assertEqual((box[0].answer, box[0].err), (None, "process ended without an answer (rc=1)"))

    def test_usage_is_stored_on_the_turn_row(self) -> None:
        self.start([[UsageEvent(input_tokens=10, output_tokens=4, cost_usd=0.25, cost_known=True),
                     TurnResult(TurnStatus.OK, answer="a", duration_ms=1200, num_turns=2)]])
        self.turn()
        totals = self.h.turns.totals(THREAD)
        self.assertEqual((totals.cost, totals.tokens_in, totals.tokens_out, totals.unknown), (0.25, 10, 4, 0))
        row = self.turn_rows()[-1]
        self.assertEqual((row["duration_ms"], row["num_turns"]), (1200, 2))

    def test_answer_collection_skips_subagent_text_and_falls_back_to_texts(self) -> None:
        self.start([
            [TextEvent(text="side", parent_id="p1"), TextEvent(text="final text"),
             TurnResult(TurnStatus.OK, answer=None)],
            [TextEvent(text="part one"), TurnResult(TurnStatus.OK, answer=None)],
            [TurnResult(TurnStatus.OK, answer=None)],
        ])
        self.assertEqual(self.turn().answer, "final text")
        self.assertEqual(self.turn().answer, "part one")
        self.assertEqual(self.turn().answer, "(no output)")



class GracefulEndTests(RunnerCase):
    def test_a_normal_end_closes_the_input_and_never_interrupts(self) -> None:
        # the live file's normal end: close stdin and let the CLI exit by itself. An interrupt would make the CLI answer
        # again (error_during_execution), and that second result would replace the real usage and the feed's answer.
        self.start([[OK]])
        self.turn()
        calls = self.backend.sessions[0].calls
        self.assertNotIn("interrupt", [c[0] for c in calls])
        self.assertEqual(calls[-2:], [("end_gracefully", 10.0), ("stop", 10)])


class EndGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.seen: list[object] = []
        self.gate = EndGate(self.seen.append, Logger(os.path.join(self.home.home, "bridge.log")))

    def test_events_after_the_first_end_never_reach_the_listeners(self) -> None:
        # a stop answered with a second result: its usage, its result and the exit are all dropped
        answer = TextEvent(text="real answer", final=True)
        for event in (UsageEvent(input_tokens=100, output_tokens=40), answer, OK, UsageEvent(),
                      TurnResult(TurnStatus.ERROR, answer=None, subtype="error_during_execution"),
                      ProcessExited(returncode=0)):
            self.gate(event)
        self.assertEqual(self.seen, [UsageEvent(input_tokens=100, output_tokens=40), answer, OK])
        self.assertEqual(self.gate.dropped, 3)


class SlotRetryTests(RunnerCase):
    def test_a_turn_that_lost_its_permit_takes_one_back_on_a_later_tick(self) -> None:
        # retake() fails when the card is answered (both permits are busy), so the turn runs unslotted; the permit that
        # is freed later goes back to the turn on the next wait tick, not to a queued topic
        self.start([[ApprovalRequest(approval_id="r1", kind="command", tool="Bash", input={"command": "git status"},
                                     summary="Bash", choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY})),
                     WaitForInterrupt()]])
        self.h.sessions.set_mode(THREAD, "default")
        slot = self.h.slots.new_slot(CHAT, THREAD)
        self.assertTrue(slot.take())
        thread, box = run_async(self.runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="write",
                                sender="tester", slot=slot)
        self.assertTrue(wait_until(lambda: len(self.h.asks.pending(THREAD)) == 1))
        self.assertTrue(wait_until(lambda: self.h.slots.free() == 2), "the parked turn gave its permit back")
        busy = [self.h.slots.new_slot(CHAT, 9201), self.h.slots.new_slot(CHAT, 9202)]
        self.assertTrue(all(other.take() for other in busy))
        self.assertEqual(self.h.slots.free(), 0)
        nonce = self.h.asks.pending(THREAD)[0].nonce
        self.assertTrue(self.h.permissions.answer_ask(nonce, allow=True))
        self.assertFalse(slot.held, "every permit was busy when the card was answered: the turn runs unslotted")
        busy[0].give_back()
        self.assertTrue(wait_until(lambda: slot.held), "the next wait tick takes the freed permit back")
        self.assertEqual(self.h.slots.free(), 0)
        self.backend.sessions[0].stop(grace=1)  # the scripted turn waits for an interrupt: end it
        join_turn(self.h.clock, thread)
        slot.give_back()
        busy[1].give_back()


class RouterSinkTests(RunnerCase):
    def test_the_session_sink_is_a_router_that_is_closed_when_the_turn_ends(self) -> None:
        # the session talks to a SessionRouter for its whole life; once the turn is over the router is closed, so an
        # event that arrives later is dropped with the router's own note (not the turn's "turn ended" note)
        self.start([[OK]])
        opened: list[object] = []
        original = self.backend.open_session

        def spy(spec, sink):
            opened.append(sink)
            return original(spec, sink)

        self.backend.open_session = spy
        self.assertEqual(self.turn().answer, "ok:allow")
        self.assertEqual(len(opened), 1)
        self.assertIsInstance(opened[0], SessionRouter)
        opened[0](TextEvent(text="a late text", final=True))
        lines = self.h.log_lines()
        self.assertTrue(any("session router closed: dropping a later TextEvent" in line for line in lines), lines)


if __name__ == "__main__":
    unittest.main()
