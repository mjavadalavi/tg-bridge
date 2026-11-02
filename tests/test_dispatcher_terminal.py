"""Dispatcher and the terminal claim (spec: the Lock section): a live claim of tg-bridge view holds a topic's message.

The DispatcherCase of test_dispatcher.py with a view folder: a live claim (the test's own pid) makes the head message
wait, the topic is told TERMINAL_NOTICE once, and the message runs after the claim file is removed. A stale claim (a
reaped child's pid) is removed and the message runs at once. While a turn runs, its meta says 'running'; after it, 'idle'.
Without a view folder the claim file is not read at all. Polls are POLL seconds; every wait is a bounded poll.
"""

import json
import os
import subprocess
import sys
import threading
import time

from tests.test_dispatcher import CHAT, SENDER, THREAD, TOPIC, DispatcherCase, wait_until
from tests.test_dispatcher_linger import StubLinger
from tg_bridge.core import terminal_claim
from tg_bridge.core.dispatcher import Dispatcher
from tg_bridge.core.linger import HoldEnd
from tg_bridge.core.view_recorder import META_SUFFIX, STATE_IDLE, STATE_RUNNING
from tg_bridge.paths import Paths
from tg_bridge.store.inbox import STATE_DONE, STATE_PENDING

POLL = 0.01  # the real pause of a claim wait, so the tests stay fast
QUIET_SECONDS = 0.1  # a few polls go by while the claim is live


def dead_pid() -> int:
    """The pid of a child that has exited and been reaped: no process has it now."""
    proc = subprocess.Popen([sys.executable, "-I", "-c", "pass"])
    proc.wait()
    return proc.pid


class TerminalDispatcherCase(DispatcherCase):
    """DispatcherCase with a view folder: the terminal claims of the topic are checked before each turn."""

    def setUp(self) -> None:
        super().setUp()
        os.makedirs(self.view, mode=0o700, exist_ok=True)

    @property
    def view(self) -> str:
        return Paths.from_home(self.home).view

    def _make_dispatcher(self, media_wait: float | None = None) -> Dispatcher:
        return Dispatcher(
            turns=self.turns, slots=self.slots, inbox=self.inbox, messenger=self.messenger, answers=self.answers,
            titles=self.titles, client=self.tg, live=self.live, config=self.config, logger=self.logger,
            media_wait_seconds=self.media_wait if media_wait is None else media_wait,
            start_thread=self.thread_log, clock=self.clock, poll_seconds=POLL, view_dir=self.view)

    def alive_runner(self) -> bool:
        return any(thread.is_alive() for thread in self.thread_log.threads)

    def claim_file(self) -> str:
        return terminal_claim.claim_path(self.view, THREAD)

    def write_claim(self, pid: int) -> None:
        with open(self.claim_file(), "w", encoding="utf-8") as handle:
            json.dump({"pid": pid, "started": 1.0}, handle)

    def meta(self) -> dict:
        with open(os.path.join(self.view, f"{THREAD}{META_SUFFIX}"), encoding="utf-8") as handle:
            return json.load(handle)

    def notices(self) -> int:
        return sum(1 for text in self.tg.texts(THREAD) if text == terminal_claim.TERMINAL_NOTICE)


class ClaimHoldsTheTopic(TerminalDispatcherCase):
    def test_a_live_claim_holds_the_message_and_the_topic_is_told_once(self) -> None:
        self.write_claim(os.getpid())
        self.ingest(1)
        self.deliver(1, "first")
        self.assertTrue(wait_until(lambda: self.notices() == 1), "the topic was not told about the terminal")
        time.sleep(QUIET_SECONDS)  # several polls go by while the claim is live
        self.assertEqual(self.turns.texts_run(THREAD), [], "the message ran while the terminal held the topic")
        self.assertEqual(self.notices(), 1, "the notice was sent more than once")
        self.assertEqual(self.state(1), STATE_PENDING, "the row was marked done before the turn started")
        self.assertTrue(self.alive_runner(), "the runner left the topic while the claim was live")

    def test_removing_the_claim_lets_the_queued_message_run(self) -> None:
        self.write_claim(os.getpid())
        self.ingest(1)
        self.deliver(1, "first")
        self.assertTrue(wait_until(lambda: self.notices() == 1))
        os.remove(self.claim_file())
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the message never ran")
        self.assertIn("answer to first", self.tg.texts(THREAD))
        self.assertEqual(self.notices(), 1)
        self.assertEqual(self.state(1), STATE_DONE)

    def test_a_message_queued_behind_the_held_one_runs_after_it_in_order(self) -> None:
        self.write_claim(os.getpid())
        self.ingest(1)
        self.ingest(2)
        self.deliver(1, "first")
        self.deliver(2, "second")
        self.assertTrue(wait_until(lambda: self.notices() == 1))
        os.remove(self.claim_file())
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first", "second"]))
        self.assertEqual(self.notices(), 1)


class StaleClaimIsIgnored(TerminalDispatcherCase):
    def test_a_claim_with_a_dead_pid_is_removed_and_the_message_runs(self) -> None:
        self.write_claim(dead_pid())
        self.ingest(1)
        self.deliver(1, "first")
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the stale claim blocked")
        self.assertFalse(os.path.exists(self.claim_file()), "the stale claim was not removed")
        self.assertEqual(self.notices(), 0)


class TurnMeta(TerminalDispatcherCase):
    def test_the_turn_is_marked_running_before_it_runs_and_idle_after_it(self) -> None:
        gate = self.turns.gates.setdefault("first", threading.Event())
        self.ingest(1)
        self.deliver(1, "first")
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]))
        self.assertEqual(self.meta()["state"], STATE_RUNNING)
        self.assertEqual(self.meta()["updated"], self.clock.now())
        gate.set()
        self.assertTrue(wait_until(lambda: self.meta()["state"] == STATE_IDLE), "the turn end was not recorded")


class KeptTopicKeepsItsMark(TerminalDispatcherCase):
    """A turn that leaves a kept session (the topic is held after it) keeps the turn mark, so the pane stays refused,
    and the tick that ends the hold removes the mark. The hold is a StubLinger: no process runs."""

    def setUp(self) -> None:
        self.linger = StubLinger()
        super().setUp()

    def tearDown(self) -> None:
        with self.linger._lock:
            self.linger.holds = False
        super().tearDown()

    def _make_dispatcher(self, media_wait: float | None = None) -> Dispatcher:
        return Dispatcher(
            turns=self.turns, slots=self.slots, inbox=self.inbox, messenger=self.messenger, answers=self.answers,
            titles=self.titles, client=self.tg, live=self.live, config=self.config, logger=self.logger,
            media_wait_seconds=self.media_wait if media_wait is None else media_wait,
            start_thread=self.thread_log, clock=self.clock, poll_seconds=POLL, view_dir=self.view,
            linger=self.linger, hold_failure_pause=0.0)

    def test_the_mark_stays_through_the_hold_and_goes_when_the_hold_ends(self) -> None:
        self.linger.holds = True  # the turn that runs now is kept
        self.ingest(1)
        self.deliver(1, "first")
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the turn did not run")
        self.assertTrue(wait_until(lambda: self.linger.calls >= 2), "the runner did not tick the hold")
        self.assertTrue(terminal_claim.turn_marked(self.view, THREAD), "the kept session lost its mark")
        self.assertFalse(terminal_claim.take_claim(self.view, THREAD, os.getpid(), 1.0),
                         "the pane claimed a topic whose session is kept")
        self.linger.next_end = HoldEnd.DONE  # the hold ends: its process is released
        self.assertTrue(wait_until(lambda: not terminal_claim.turn_marked(self.view, THREAD)),
                        "the released hold kept its mark")
        self.assertTrue(terminal_claim.take_claim(self.view, THREAD, os.getpid(), 1.0))

    def test_a_hold_that_ends_while_a_begun_turn_waits_for_a_slot_keeps_the_mark_until_the_turn_ends(self) -> None:
        """The mark of a begun turn stays through its slot wait: the hold tick that ends the hold runs inside that wait,
        and it does not clear the mark, since the turn is about to spawn its process. The turn's own end clears it."""
        self.assertTrue(self.slots.try_acquire_permit())  # the pool is full: the held topic's message waits for a slot
        self.assertTrue(self.slots.try_acquire_permit())
        self.addCleanup(self._give_back_permits)
        self.linger.holds = True  # the topic is held; its kept process is the mark's owner
        self.assertTrue(terminal_claim.begin_turn(self.view, THREAD, self.clock))
        self.ingest(1)
        self.deliver(1, "second")
        self.assertTrue(wait_until(lambda: self.linger.calls >= 2), "the slot wait did not tick the hold")
        self.linger.next_end = HoldEnd.DONE  # the tick inside the slot wait ends the hold
        self.assertTrue(wait_until(lambda: not self.linger.holds), "the hold did not end during the slot wait")
        time.sleep(QUIET_SECONDS)  # the tick has returned; a clear made by it would show now
        self.assertTrue(terminal_claim.turn_marked(self.view, THREAD), "the slot wait cleared the begun turn's mark")
        self.assertFalse(terminal_claim.take_claim(self.view, THREAD, os.getpid(), 1.0),
                         "the pane claimed a topic whose message is about to spawn")
        self._give_back_permits()
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["second"]), "the turn never ran")
        self.assertTrue(wait_until(lambda: not self.alive_runner()))
        self.assertFalse(terminal_claim.turn_marked(self.view, THREAD), "the turn's end left its mark behind")

    def _give_back_permits(self) -> None:
        """Gives back the permits this case took; a no-op for those already returned."""
        while self.slots.free() < self.slots.size:
            self.slots.release_permit()

    def test_a_tick_that_releases_the_hold_and_then_raises_clears_the_mark(self) -> None:
        """A release that happens before a tick raises (a stop that fails) leaves no process: the mark goes with it,
        and the runner does not stay on a topic that is no longer held."""
        self.linger.holds = True
        self.ingest(1)
        self.deliver(1, "first")
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the turn did not run")
        self.assertTrue(wait_until(lambda: self.linger.calls >= 1), "the runner did not tick the hold")
        self.assertTrue(terminal_claim.turn_marked(self.view, THREAD), "the kept session lost its mark")
        released = self.linger.wait_once

        def release_then_raise(thread_id: int):
            released(thread_id)  # the release: the record is gone, the process is stopped
            raise RuntimeError("stop failed after the release")

        self.linger.wait_once = release_then_raise
        self.linger.next_end = HoldEnd.DONE
        self.assertTrue(wait_until(lambda: not self.alive_runner()), "the runner stayed on the released topic")
        self.assertFalse(terminal_claim.turn_marked(self.view, THREAD), "the released hold kept its mark")
        self.assertTrue(terminal_claim.take_claim(self.view, THREAD, os.getpid(), 1.0))


class ViewsOff(DispatcherCase):
    """No view folder: a claim file in the view folder is never read, and the message runs at once."""

    def setUp(self) -> None:
        super().setUp()
        view = Paths.from_home(self.home).view
        os.makedirs(view, mode=0o700, exist_ok=True)
        with open(terminal_claim.claim_path(view, THREAD), "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started": 1.0}, handle)
        self.claim = terminal_claim.claim_path(view, THREAD)

    def test_views_off_runs_the_message_and_leaves_the_claim_alone(self) -> None:
        self.ingest(1)
        self.deliver(1, "first")
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the message did not run")
        self.assertNotIn(terminal_claim.TERMINAL_NOTICE, self.tg.texts(THREAD))
        self.assertTrue(os.path.exists(self.claim), "the claim file was touched with views off")
