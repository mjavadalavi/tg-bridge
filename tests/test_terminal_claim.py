"""Terminal claims (core/terminal_claim.py): the file lock between `tg-bridge view` and the daemon for one topic.

Covers the claim file (pid, started): a live claim is read, a stale claim (dead pid) and a garbled claim are removed
when they are read; take_claim (the pane) is refused while a claim is live or the meta says running or lingering;
begin_turn (the daemon) is refused while a claim is live and otherwise marks the meta running under the same lock;
end_turn only turns running into idle; release_claim removes only the claim of its own pid. The /status turn line
shows the terminal line while a claim is live. Every process id is real: os.getpid() is live, a child that has exited
and been reaped is dead. No claude process is started.
"""

import json
import os
import subprocess
import sys
import unittest

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.events import TurnResult, TurnStatus
from tg_bridge.bot.commands.info import TERMINAL_STATUS_TEXT, _turn_line
from tg_bridge.core import terminal_claim
from tg_bridge.core.view_recorder import META_SUFFIX, STATE_IDLE, STATE_LINGERING, STATE_RUNNING, ViewRecorder
from tg_bridge.logging_ import Logger

THREAD = 5150
OTHER_THREAD = 5151


def dead_pid() -> int:
    """The pid of a child that has exited and been reaped: no process has it now."""
    proc = subprocess.Popen([sys.executable, "-I", "-c", "pass"])
    proc.wait()
    return proc.pid


class ClaimCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.view = os.path.join(self.home.home, "view")
        os.makedirs(self.view, mode=0o700)
        self.clock = FakeClock()

    def claim_file(self, thread_id: int = THREAD) -> str:
        return terminal_claim.claim_path(self.view, thread_id)

    def write_claim(self, pid: int, thread_id: int = THREAD) -> None:
        with open(self.claim_file(thread_id), "w", encoding="utf-8") as handle:
            json.dump({"pid": pid, "started": 1.0}, handle)

    def write_meta(self, state: str, thread_id: int = THREAD, **fields: str) -> None:
        meta = dict(fields, state=state, updated=1.0)
        with open(os.path.join(self.view, f"{thread_id}{META_SUFFIX}"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle)

    def meta(self, thread_id: int = THREAD) -> dict:
        with open(os.path.join(self.view, f"{thread_id}{META_SUFFIX}"), encoding="utf-8") as handle:
            return json.load(handle)


class LiveAndStaleClaims(ClaimCase):
    def test_no_claim_file_means_no_claim(self) -> None:
        self.assertIsNone(terminal_claim.live_claim(self.view, THREAD))

    def test_a_claim_whose_pid_is_alive_is_live_and_kept(self) -> None:
        self.write_claim(os.getpid())
        claim = terminal_claim.live_claim(self.view, THREAD)
        self.assertIsNotNone(claim)
        self.assertEqual(claim.pid, os.getpid())
        self.assertTrue(os.path.exists(self.claim_file()))

    def test_a_stale_claim_with_a_dead_pid_is_ignored_and_removed(self) -> None:
        self.write_claim(dead_pid())
        self.assertIsNone(terminal_claim.live_claim(self.view, THREAD))
        self.assertFalse(os.path.exists(self.claim_file()), "the stale claim was not removed")

    def test_a_garbled_claim_is_removed_and_is_no_claim(self) -> None:
        with open(self.claim_file(), "w", encoding="utf-8") as handle:
            handle.write("{not json")
        self.assertIsNone(terminal_claim.live_claim(self.view, THREAD))
        self.assertFalse(os.path.exists(self.claim_file()))

    def test_a_pid_that_is_not_a_positive_int_is_no_claim(self) -> None:
        with open(self.claim_file(), "w", encoding="utf-8") as handle:
            json.dump({"pid": 0, "started": 1.0}, handle)  # kill(0, 0) would signal the whole group: never used
        self.assertIsNone(terminal_claim.live_claim(self.view, THREAD))
        self.assertFalse(os.path.exists(self.claim_file()))


class PaneSide(ClaimCase):
    def test_take_claim_writes_the_claim_of_the_pane(self) -> None:
        self.assertTrue(terminal_claim.take_claim(self.view, THREAD, os.getpid(), 7.0))
        claim = terminal_claim.live_claim(self.view, THREAD)
        self.assertEqual((claim.pid, claim.started), (os.getpid(), 7.0))

    def test_take_claim_is_refused_while_another_claim_is_live(self) -> None:
        self.write_claim(os.getpid())
        self.assertFalse(terminal_claim.take_claim(self.view, THREAD, os.getpid(), 7.0))

    def test_take_claim_is_refused_while_the_bridge_runs_or_keeps_the_topic(self) -> None:
        for state in (STATE_RUNNING, STATE_LINGERING):
            with self.subTest(state=state):
                self.write_meta(state)
                self.assertFalse(terminal_claim.take_claim(self.view, THREAD, os.getpid(), 7.0))
                self.assertFalse(os.path.exists(self.claim_file()))

    def test_take_claim_is_allowed_when_the_meta_is_idle_or_missing(self) -> None:
        self.write_meta(STATE_IDLE)
        self.assertTrue(terminal_claim.take_claim(self.view, THREAD, os.getpid(), 7.0))
        terminal_claim.release_claim(self.view, THREAD, os.getpid())
        self.assertTrue(terminal_claim.take_claim(self.view, OTHER_THREAD, os.getpid(), 8.0))

    def test_release_claim_removes_only_the_claim_of_its_own_pid(self) -> None:
        self.write_claim(os.getpid())
        self.assertFalse(terminal_claim.release_claim(self.view, THREAD, os.getpid() + 1))
        self.assertTrue(os.path.exists(self.claim_file()))
        self.assertTrue(terminal_claim.release_claim(self.view, THREAD, os.getpid()))
        self.assertFalse(os.path.exists(self.claim_file()))
        self.assertFalse(terminal_claim.release_claim(self.view, THREAD, os.getpid()))


class DaemonSide(ClaimCase):
    def test_begin_turn_is_refused_while_a_claim_is_live_and_marks_nothing(self) -> None:
        self.write_claim(os.getpid())
        self.assertFalse(terminal_claim.begin_turn(self.view, THREAD, self.clock))
        self.assertFalse(os.path.exists(os.path.join(self.view, f"{THREAD}{META_SUFFIX}")))

    def test_begin_turn_marks_running_and_keeps_the_other_meta_fields(self) -> None:
        self.write_meta(STATE_IDLE, topic="Launch notes", transport="tui", launcher="iclaude")
        self.assertTrue(terminal_claim.begin_turn(self.view, THREAD, self.clock))
        meta = self.meta()
        self.assertEqual(meta["state"], STATE_RUNNING)
        self.assertEqual((meta["topic"], meta["transport"], meta["launcher"]), ("Launch notes", "tui", "iclaude"))
        self.assertEqual(meta["updated"], self.clock.now())

    def test_begin_turn_clears_a_stale_claim_and_starts(self) -> None:
        self.write_claim(dead_pid())
        self.assertTrue(terminal_claim.begin_turn(self.view, THREAD, self.clock))
        self.assertEqual(self.meta()["state"], STATE_RUNNING)

    def test_end_turn_turns_running_into_idle(self) -> None:
        terminal_claim.begin_turn(self.view, THREAD, self.clock)
        terminal_claim.end_turn(self.view, THREAD, self.clock)
        self.assertEqual(self.meta()["state"], STATE_IDLE)

    def test_end_turn_leaves_a_lingering_or_idle_topic_alone(self) -> None:
        self.write_meta(STATE_LINGERING)
        terminal_claim.end_turn(self.view, THREAD, self.clock)
        self.assertEqual(self.meta()["state"], STATE_LINGERING)
        self.write_meta(STATE_IDLE)
        terminal_claim.end_turn(self.view, THREAD, self.clock)
        self.assertEqual(self.meta()["state"], STATE_IDLE)

    def test_claims_are_per_topic(self) -> None:
        self.write_claim(os.getpid(), OTHER_THREAD)
        self.assertTrue(terminal_claim.begin_turn(self.view, THREAD, self.clock))


class TurnMarkCase(ClaimCase):
    """The daemon's turn mark: begin_turn sets it, end_turn removes it, and take_claim refuses while it exists. The
    recorder is the real ViewRecorder, whose TurnResult writes 'idle' while the CLI of the turn may still run."""

    def recorder(self) -> ViewRecorder:
        return ViewRecorder(self.view, THREAD, "alpha", "tui", "iclaude", self.clock,
                            Logger(os.path.join(self.home.home, "bridge.log")))


class TurnMarks(TurnMarkCase):
    def test_begin_turn_writes_the_mark_and_end_turn_removes_it(self) -> None:
        self.assertFalse(terminal_claim.turn_marked(self.view, THREAD))
        self.assertTrue(terminal_claim.begin_turn(self.view, THREAD, self.clock))
        self.assertTrue(terminal_claim.turn_marked(self.view, THREAD))
        terminal_claim.end_turn(self.view, THREAD, self.clock)
        self.assertFalse(terminal_claim.turn_marked(self.view, THREAD))

    def test_the_mark_is_per_topic(self) -> None:
        terminal_claim.begin_turn(self.view, THREAD, self.clock)
        self.assertFalse(terminal_claim.turn_marked(self.view, OTHER_THREAD))

    def test_a_turn_result_of_a_real_recorder_does_not_free_the_topic_while_its_process_runs(self) -> None:
        self.assertTrue(terminal_claim.begin_turn(self.view, THREAD, self.clock))
        self.recorder().on_event(TurnResult(TurnStatus.OK, answer="done"))
        self.assertEqual(self.meta()["state"], STATE_IDLE, "the recorder did not write its idle state")
        self.assertFalse(terminal_claim.take_claim(self.view, THREAD, os.getpid(), 7.0),
                         "the pane claimed a topic whose CLI still runs")
        terminal_claim.end_turn(self.view, THREAD, self.clock)  # the process is gone and not kept
        self.assertTrue(terminal_claim.take_claim(self.view, THREAD, os.getpid(), 7.0))

    def test_a_kept_session_refuses_the_claim_through_its_self_turns_until_the_release(self) -> None:
        terminal_claim.begin_turn(self.view, THREAD, self.clock)
        turn = self.recorder()
        turn.on_event(TurnResult(TurnStatus.OK, answer="done"))  # the user turn's result: idle, the process still runs
        turn.set_state(STATE_LINGERING)  # the session is kept
        self.assertFalse(terminal_claim.take_claim(self.view, THREAD, os.getpid(), 7.0))
        self.recorder().on_event(TurnResult(TurnStatus.OK, answer="self turn"))  # a self turn writes idle again
        self.assertEqual(self.meta()["state"], STATE_IDLE)
        self.assertFalse(terminal_claim.take_claim(self.view, THREAD, os.getpid(), 7.0),
                         "the pane claimed a kept session between its self turns")
        terminal_claim.end_turn(self.view, THREAD, self.clock)  # the hold is released
        self.assertTrue(terminal_claim.take_claim(self.view, THREAD, os.getpid(), 7.0))

    def test_clear_turn_marks_removes_the_marks_of_a_stopped_daemon_and_counts_them(self) -> None:
        terminal_claim.begin_turn(self.view, THREAD, self.clock)
        terminal_claim.begin_turn(self.view, OTHER_THREAD, self.clock)
        self.write_meta(STATE_IDLE, thread_id=OTHER_THREAD)
        logger = Logger(os.path.join(self.home.home, "bridge.log"))
        self.assertEqual(terminal_claim.clear_turn_marks(self.view, logger), 2)
        self.assertFalse(terminal_claim.turn_marked(self.view, THREAD))
        self.assertFalse(terminal_claim.turn_marked(self.view, OTHER_THREAD))
        self.assertTrue(os.path.exists(os.path.join(self.view, f"{OTHER_THREAD}{META_SUFFIX}")))

    def test_after_the_startup_sweep_a_stale_mark_no_longer_blocks_the_pane(self) -> None:
        terminal_claim.begin_turn(self.view, THREAD, self.clock)
        self.write_meta(STATE_IDLE)
        terminal_claim.clear_turn_marks(self.view, Logger(os.path.join(self.home.home, "bridge.log")))
        self.assertTrue(terminal_claim.take_claim(self.view, THREAD, os.getpid(), 7.0))


class StatusLine(unittest.TestCase):
    def test_a_live_claim_shows_the_terminal_line_and_otherwise_the_usual_line(self) -> None:
        self.assertEqual(_turn_line(None, None, True), TERMINAL_STATUS_TEXT)
        self.assertEqual(TERMINAL_STATUS_TEXT, "turn: open in the terminal")
        self.assertEqual(_turn_line(None, None), "turn: idle")
        self.assertEqual(_turn_line(None, None, False), "turn: idle")
