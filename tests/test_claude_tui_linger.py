"""ClaudeTuiSession with background work (the tui transport), with the fake TUI (tests/fakes/fake_claude_tui.py): the
transcript tail outlives the turn, so the background work the CLI leaves running is counted (BackgroundWork) and a turn the
CLI starts by itself is reported (TurnBegan, then a new generation with its own TurnResult). Scenarios: a background Bash,
a Monitor, a background Agent and a ScheduleWakeup each leave one piece of work pending, counted from the tool result of the
turn itself. A flag file ends the work, and the CLI's own turn follows: a task notification, or the wakeup's scheduled
turn (a line with isMeta). The lines a typed /exit writes are never a turn, and a deaf TUI that ignores the first /exit after
a wakeup is closed by the signal ladder. A second message keeps the one tail, and no self turn starts while the session
is ending."""

import json
import os
import threading
import time

from tests.fakes.backend import RecordingSink
from tests.test_claude_tui_session import TuiTestCase, read_text, results, wait_until
from tg_bridge.agents.claude.tui_session import ClaudeTuiSession
from tg_bridge.agents.events import BackgroundWork, ProcessExited, TurnBegan, TurnStatus, UsageEvent

WAIT = 15.0
WAKE_LINE = {"type": "user", "isMeta": True, "turnOrigin": "scheduled",
             "message": {"role": "user", "content": "Reply with the single word AWAKE."}}
SCENARIOS = ("bg_bash", "bg_monitor", "bg_agent", "bg_wakeup")
# the three lines a typed /exit writes, as the TUI writes them (no turnOrigin)
EXIT_LINES = (
    {"type": "user", "message": {"role": "user", "content": "<local-command-caveat>caveat</local-command-caveat>"}},
    {"type": "user", "message": {"role": "user", "content": "<command-name>/exit</command-name>"}},
    {"type": "user", "message": {"role": "user", "content": "<local-command-stdout>(no content)</local-command-stdout>"}},
)


def tail_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == "transcript-tail"]


def began(sink: RecordingSink) -> list[TurnBegan]:
    return [e for e in sink.events if isinstance(e, TurnBegan)]


class BackgroundTests(TuiTestCase):
    def flag(self) -> str:
        """The file that ends the background work of the fake TUI. Not created by the test until it releases the work."""
        return os.path.join(self.home.state, "bg.flag")

    def open_background(self, scenario: str, *, extra_env: dict | None = None) -> tuple[RecordingSink, ClaudeTuiSession]:
        env = {"FAKE_TUI_BG_FLAG": self.flag(), **(extra_env or {})}
        return self.open(scenario, extra_env=env)

    def release_work(self) -> None:
        open(self.flag(), "w", encoding="utf-8").close()

    def test_tail_survives_the_turn_and_reports_a_self_turn(self) -> None:
        sink, session = self.open_background("bg_bash")
        session.start()
        session.send_user("run it in the background")
        self.wait_result(sink)
        first = results(sink)[0]
        self.assertEqual((first.status, first.answer), (TurnStatus.OK, "STARTED"))
        self.assertLess(sink.events.index(BackgroundWork(pending=1)), sink.events.index(first))
        self.release_work()
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnBegan), WAIT))
        self.assertTrue(wait_until(lambda: len(results(sink)) == 2))
        self.assertEqual([r.answer for r in results(sink)], ["STARTED", "FINISHED"])
        self.assertEqual(began(sink)[0].origin, "task_notification")
        self.assertLess(sink.events.index(BackgroundWork(pending=0)), sink.events.index(began(sink)[0]))
        self.assertLess(sink.events.index(first), sink.events.index(began(sink)[0]))
        usages = [e for e in sink.events if isinstance(e, UsageEvent)]
        self.assertEqual(len(usages), 2)  # the self turn has its own usage, not the first turn's
        self.assertTrue(session.alive())

    def test_background_work_counts_from_the_tool_result(self) -> None:
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                sink, session = self.open_background(scenario)
                session.start()
                session.send_user("go")
                self.wait_result(sink)
                first = results(sink)[0]
                self.assertIn(BackgroundWork(pending=1), sink.events)
                self.assertLess(sink.events.index(BackgroundWork(pending=1)), sink.events.index(first))
                self.assertTrue(session.end_gracefully(WAIT))

    def test_scheduled_meta_line_starts_a_self_turn(self) -> None:
        sink, session = self.open_background("bg_wakeup")
        session.start()
        session.send_user("wake me")
        self.wait_result(sink)
        self.assertIn(BackgroundWork(pending=1), sink.events)
        self.release_work()  # the wakeup fires: a meta line starts the scheduled turn
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, TurnBegan), WAIT))
        self.assertEqual(began(sink)[0].origin, "scheduled")
        self.assertIn(BackgroundWork(pending=0), sink.events)
        self.assertTrue(wait_until(lambda: len(results(sink)) == 2))
        self.assertEqual([r.answer for r in results(sink)], ["STARTED", "AWAKE"])

    def test_exit_lines_are_not_a_self_turn(self) -> None:
        sink, session = self.open("text")
        session.start()
        session.send_user("hello")
        self.wait_result(sink)
        path = self.transcript_path(session.session_id)
        with open(path, "a", encoding="utf-8") as f:
            for line in EXIT_LINES:  # what a typed /exit writes, while the session is idle
                f.write(json.dumps(line) + "\n")
            # a sentinel after them: a wakeup call's result, which the ledger counts; the lines before it are read by then
            sentinel = {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_s", "content": "ok", "is_error": False}]},
                "toolUseResult": {"scheduledFor": 1}}
            f.write(json.dumps(sentinel) + "\n")
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, BackgroundWork) and e.pending == 1, WAIT))
        self.assertEqual(began(sink), [])
        self.assertEqual(len(results(sink)), 1)
        self.assertTrue(session.end_gracefully(WAIT))
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ProcessExited), WAIT))
        self.assertEqual(began(sink), [])

    def test_deaf_exit_after_wakeup_falls_back_to_signals(self) -> None:
        sink, session = self.open_background("bg_wakeup", extra_env={"FAKE_TUI_EXIT_DEAF": "1"})
        session.start()
        session.send_user("wake me")
        self.wait_result(sink)
        self.assertIn(BackgroundWork(pending=1), sink.events)
        session.stop(grace=1)  # Esc and /exit are typed; the deaf TUI ignores the /exit, so SIGINT ends it
        self.assertFalse(session.alive())
        self.assertNotIn("local-command-stdout", read_text(self.transcript_path(session.session_id)))

    def test_second_message_on_a_live_session_keeps_one_tail(self) -> None:
        known = set(tail_threads())
        sink, session = self.open("text")
        session.start()
        session.send_user("one")
        self.wait_result(sink)
        (tail,) = [t for t in tail_threads() if t not in known]
        session.send_user("two")
        self.assertTrue(wait_until(lambda: len(results(sink)) == 2))
        self.assertEqual([r.answer for r in results(sink)], ["ANSWER", "ANSWER"])
        self.assertTrue(tail.is_alive())  # the second turn follows the same transcript: the same tail thread, not a new one
        self.assertEqual([t for t in tail_threads() if t not in known], [tail])

    def test_no_turn_began_while_ending(self) -> None:
        sink, session = self.open_background("bg_bash")  # the work ends only when /exit releases it
        session.start()
        session.send_user("go")
        self.wait_result(sink)
        self.assertIn(BackgroundWork(pending=1), sink.events)
        self.assertTrue(session.end_gracefully(WAIT))
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ProcessExited), WAIT))
        self.assertIn(BackgroundWork(pending=0), sink.events)  # the end lines were read, while the session was ending
        self.assertEqual(began(sink), [])
        self.assertEqual([r.answer for r in results(sink)], ["STARTED"])

    def test_no_turn_began_while_stopping(self) -> None:
        sink, session = self.open_background("bg_bash")  # the work ends only when the /exit of stop() releases it
        session.start()
        session.send_user("go")
        self.wait_result(sink)
        self.assertIn(BackgroundWork(pending=1), sink.events)
        session.stop(grace=5)  # stop() marks the session as ending before its first key
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ProcessExited), WAIT))
        self.assertIn(BackgroundWork(pending=0), sink.events)  # the end lines were read while the session was stopping
        self.assertEqual(began(sink), [])
        self.assertEqual([r.answer for r in results(sink)], ["STARTED"])

    def test_a_line_from_a_replaced_tail_is_ignored(self) -> None:
        sink, session = self.open("text")
        session.start()
        session.send_user("hello")
        self.wait_result(sink)
        current = session._tail_gen  # the generation of the tail that follows the transcript now
        stale = current - 1  # the generation of a tail that was replaced: its lines must not start a turn
        before = list(sink.events)
        session._on_transcript(stale, WAKE_LINE)
        self.assertEqual(sink.events, before)
        session._on_transcript(current, WAKE_LINE)  # control: a line of the current tail does start the self turn
        self.assertEqual(began(sink)[0].origin, "scheduled")
        session.stop(grace=5)
