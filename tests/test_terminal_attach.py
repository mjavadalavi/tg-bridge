"""Dispatcher and an attachable terminal claim (spec T3): a claim with a pane and a session uuid takes the topic's message
when its pane is idle; a busy pane holds the message with one WAITING notice; a claim that goes while it waits gives the
normal path; a claim without a pane keeps the old wait; without a pane probe nothing is attachable.

The TerminalDispatcherCase of test_dispatcher_terminal with the pane probe faked: StubTurns records the attach target of
each run. Polls are POLL seconds; every wait is a bounded poll.
"""

import json
import os
import threading
import time
from collections.abc import Callable

from tests.test_dispatcher import THREAD, wait_until
from tests.test_dispatcher_terminal import QUIET_SECONDS, TerminalDispatcherCase
from tg_bridge.agents.base import AttachSpec
from tg_bridge.core import terminal_claim
from tg_bridge.core.dispatcher import Dispatcher
from tg_bridge.core.terminal_wait import WAITING_TEXT
from tg_bridge.core.turn_kit import TurnReply
from tg_bridge.core.turn_state import TurnOutcome
from tg_bridge.store.inbox import STATE_DONE

UUID = "0a1b2c3d-0000-4000-8000-000000000001"
PANE = "%3"
CONFIG_DIR = "/srv/claude-config"
CWD = "/srv/work"
ATTACH = AttachSpec(socket="", pane=PANE, uuid=UUID, config_dir=CONFIG_DIR, cwd=CWD)
POLL_SECONDS = 0.01
WAIT_SECONDS = 5.0


class PaneProbe:
    """Stands in for the pane probe of the app: busy until idle is set, and it counts its reads."""

    def __init__(self) -> None:
        self.idle = False
        self.calls = 0

    def __call__(self, claim: terminal_claim.Claim) -> bool:
        self.calls += 1
        return self.idle


class AttachableDispatcherCase(TerminalDispatcherCase):
    """The terminal dispatcher case with the pane probe wired (WITH_PROBE False leaves it out, as the default does)."""

    WITH_PROBE = True

    def setUp(self) -> None:
        self.probe = PaneProbe()
        super().setUp()

    def _make_dispatcher(self, media_wait: float | None = None) -> Dispatcher:
        return Dispatcher(
            turns=self.turns, slots=self.slots, inbox=self.inbox, messenger=self.messenger, answers=self.answers,
            titles=self.titles, client=self.tg, live=self.live, config=self.config, logger=self.logger,
            media_wait_seconds=self.media_wait if media_wait is None else media_wait,
            start_thread=self.thread_log, clock=self.clock, poll_seconds=POLL_SECONDS, view_dir=self.view,
            pane_idle=self.probe if self.WITH_PROBE else None)

    def write_attachable(self, pid: int) -> None:
        with open(self.claim_file(), "w", encoding="utf-8") as handle:
            json.dump({"pid": pid, "started": 1.0, "pane": PANE, "uuid": UUID, "config_dir": CONFIG_DIR,
                       "cwd": CWD, "socket": ""}, handle)

    def waiting_notices(self) -> int:
        return sum(1 for text in self.tg.texts(THREAD) if text == WAITING_TEXT)

    def deliver_first(self) -> None:
        self.ingest(1)
        self.deliver(1, "first")


class IdlePaneTakesTheMessage(AttachableDispatcherCase):
    def test_an_idle_pane_runs_the_message_in_its_session_and_marks_nothing(self) -> None:
        self.write_attachable(os.getpid())
        self.probe.idle = True
        self.deliver_first()
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the message did not run")
        self.assertEqual(self.turns.attached("first"), ATTACH, "the turn did not run in the claim's session")
        self.assertEqual(self.notices(), 0, "an idle pane told the topic it waits")
        self.assertEqual(self.waiting_notices(), 0)
        self.assertFalse(terminal_claim.turn_marked(self.view, THREAD), "an attached turn took the turn mark")
        self.assertTrue(os.path.exists(self.claim_file()), "the terminal's claim was removed")
        self.assertTrue(wait_until(lambda: self.state(1) == STATE_DONE))

    def test_a_busy_pane_holds_the_message_and_the_topic_is_told_once(self) -> None:
        self.write_attachable(os.getpid())
        self.deliver_first()
        self.assertTrue(wait_until(lambda: self.waiting_notices() == 1), "the topic was not told it waits")
        time.sleep(QUIET_SECONDS)  # several polls go by while the pane is busy
        self.assertEqual(self.turns.texts_run(THREAD), [], "the message ran while the pane was busy")
        self.assertEqual(self.waiting_notices(), 1, "the waiting notice was sent more than once")
        self.assertEqual(self.notices(), 0, "the old terminal notice was sent for an attachable claim")
        self.probe.idle = True
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the message never ran")
        self.assertEqual(self.turns.attached("first"), ATTACH)
        self.assertEqual(self.waiting_notices(), 1)

    def test_a_claim_that_goes_while_the_message_waits_gives_the_normal_path(self) -> None:
        self.write_attachable(os.getpid())
        self.deliver_first()
        self.assertTrue(wait_until(lambda: self.waiting_notices() == 1))
        os.remove(self.claim_file())
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the message never ran")
        self.assertIsNone(self.turns.attached("first"), "a claim that went still ran in its session")
        self.assertTrue(wait_until(lambda: not terminal_claim.turn_marked(self.view, THREAD)),
                        "the normal turn left its mark behind")
        self.assertGreaterEqual(self.probe.calls, 1, "the pane was never probed")

    def test_a_queued_message_behind_a_waiting_one_runs_after_it_in_its_session(self) -> None:
        self.write_attachable(os.getpid())
        self.ingest(1)
        self.ingest(2)
        self.deliver(1, "first")
        self.deliver(2, "second")
        self.assertTrue(wait_until(lambda: self.waiting_notices() == 1))
        self.probe.idle = True
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first", "second"]))
        self.assertEqual(self.turns.attached("second"), ATTACH)


class TerminalNotReadyRetries(AttachableDispatcherCase):
    """A turn whose terminal was not ready (the owner typed in the tab after the wait) pastes nothing and is refused:
    the message is not consumed and not answered with an error, the topic waits for the terminal, and the next try runs."""

    def refuse_first_run(self, before: Callable[[], None]) -> list:
        """The turns run: the first one calls before() and is refused as not ready; the rest run as the stub does."""
        calls: list = []
        stub_run = self.turns.run

        def run(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                before()
                return TurnReply(None, None, TurnOutcome.ERROR, terminal_busy=True)
            return stub_run(**kwargs)

        self.turns.run = run
        return calls

    def test_typed_in_between_the_message_waits_again_and_then_runs(self) -> None:
        self.write_attachable(os.getpid())
        self.probe.idle = True

        def owner_types() -> None:
            self.probe.idle = False  # the owner typed into the tab between the wait and the paste

        calls = self.refuse_first_run(owner_types)
        self.deliver_first()
        self.assertTrue(wait_until(lambda: self.waiting_notices() == 1), "the topic was not told it waits again")
        self.assertEqual(self.turns.texts_run(THREAD), [], "the refused message ran")
        self.assertEqual(len(calls), 1, "the refused message was run again while the pane was busy")
        self.probe.idle = True
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the message never ran")
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.turns.attached("first"), ATTACH, "the retry did not run in the claim's session")
        self.assertTrue(wait_until(lambda: "answer to first" in self.tg.texts(THREAD)), "no answer was sent")
        self.assertTrue(wait_until(lambda: self.state(1) == STATE_DONE))
        self.assertEqual(self.waiting_notices(), 1, "the topic was told more than once")
        self.assertEqual(self.notices(), 0)

    def test_a_tab_closed_in_between_takes_the_normal_path(self) -> None:
        self.write_attachable(os.getpid())
        self.probe.idle = True

        def tab_closes() -> None:
            os.remove(self.claim_file())  # the view tab is closed: its claim goes with it

        calls = self.refuse_first_run(tab_closes)
        self.deliver_first()
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the message never ran")
        self.assertEqual(len(calls), 2)
        self.assertIsNone(self.turns.attached("first"), "a closed tab's session was used")
        self.assertTrue(wait_until(lambda: "answer to first" in self.tg.texts(THREAD)), "no answer was sent")
        self.assertTrue(wait_until(lambda: not terminal_claim.turn_marked(self.view, THREAD)),
                        "the normal turn left its mark behind")
        self.assertEqual(self.waiting_notices(), 0)


class WaitingForTheTerminal(AttachableDispatcherCase):
    """The dispatcher says whether a topic's head message waits for its terminal (the inbound notice reads it)."""

    def test_a_busy_pane_marks_the_topic_as_waiting_for_its_terminal_until_the_message_runs(self) -> None:
        self.write_attachable(os.getpid())
        self.deliver_first()
        self.assertTrue(wait_until(lambda: self.waiting_notices() == 1))
        self.assertTrue(self.dispatcher.waiting_terminal(THREAD))
        self.probe.idle = True
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]))
        self.assertTrue(wait_until(lambda: not self.dispatcher.waiting_terminal(THREAD)), "still waiting after the run")

    def test_a_topic_without_a_claim_is_not_waiting_for_its_terminal(self) -> None:
        self.deliver_first()
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]))
        self.assertFalse(self.dispatcher.waiting_terminal(THREAD))


class AttachedTurnIsHeldByItsClaim(AttachableDispatcherCase):
    def test_the_attached_turn_checks_the_claim_it_started_with(self) -> None:
        self.write_attachable(os.getpid())
        self.probe.idle = True
        gate = self.turns.gates.setdefault("first", threading.Event())
        self.deliver_first()
        self.assertTrue(self.turns.started("first").wait(WAIT_SECONDS), "the turn did not start")
        still = self.turns.attached("first").still_attached
        self.assertTrue(still(), "the claim is held, but the turn reads it as gone")
        os.remove(self.claim_file())  # the CLI exited: the view pane released its claim
        self.assertFalse(still(), "a released claim still counts as attached")
        gate.set()
        self.assertTrue(wait_until(lambda: self.state(1) == STATE_DONE))


class OldClaimKeepsTheOldWait(AttachableDispatcherCase):
    """A claim without a pane (an old view process) is never attached: the probe is not even asked."""

    def test_a_claim_without_pane_and_uuid_waits_with_the_terminal_notice(self) -> None:
        self.write_claim(os.getpid())
        self.probe.idle = True
        self.deliver_first()
        self.assertTrue(wait_until(lambda: self.notices() == 1), "the old claim did not hold the topic")
        time.sleep(QUIET_SECONDS)
        self.assertEqual(self.turns.texts_run(THREAD), [], "the message ran while the old claim was live")
        self.assertEqual(self.waiting_notices(), 0)
        self.assertEqual(self.probe.calls, 0, "an old claim was probed as a pane")
        os.remove(self.claim_file())
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]))
        self.assertIsNone(self.turns.attached("first"))


class WithoutAProbeNothingIsAttachable(AttachableDispatcherCase):
    WITH_PROBE = False

    def test_an_attachable_claim_keeps_the_old_wait_when_no_probe_is_wired(self) -> None:
        self.write_attachable(os.getpid())
        self.probe.idle = True
        self.deliver_first()
        self.assertTrue(wait_until(lambda: self.notices() == 1), "the claim did not hold the topic")
        self.assertEqual(self.turns.texts_run(THREAD), [], "the message ran in a session with no probe")
        self.assertEqual(self.probe.calls, 0)
        os.remove(self.claim_file())
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]))
        self.assertIsNone(self.turns.attached("first"))
