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
import unittest
from collections.abc import Callable

from tests.test_dispatcher import THREAD, wait_until
from tests.test_dispatcher_terminal import QUIET_SECONDS, TerminalDispatcherCase
from tg_bridge.agents.base import AttachSpec
from tg_bridge.agents.claude import tmux_pane
from tg_bridge.core import terminal_claim
from tg_bridge.core.dispatcher import Dispatcher
from tg_bridge.core.terminal_wait import STALL_TEXT, STALL_TEXTS, WAITING_TEXT
from tg_bridge.core.turn_kit import TurnReply
from tg_bridge.core.turn_state import TurnOutcome
from tg_bridge.store.inbox import STATE_DONE

UUID = "0a1b2c3d-0000-4000-8000-000000000001"
NEW_UUID = "0a1b2c3d-0000-4000-8000-000000000002"
PANE = "%3"
CONFIG_DIR = "/srv/claude-config"
CWD = "/srv/work"
ATTACH = AttachSpec(socket="", pane=PANE, uuid=UUID, config_dir=CONFIG_DIR, cwd=CWD)
POLL_SECONDS = 0.01
WAIT_SECONDS = 5.0
STALL_THRESHOLD_SECONDS = 0.05  # the stall threshold the dispatcher of these tests runs with
STALL_STEP_SECONDS = 1.0  # one clock step: well past the threshold, and under a minute in all


class PaneProbe:
    """Stands in for the pane probe of the app: busy until idle is set, and it counts its reads. reason is what the
    reason probe reports while the pane is not idle."""

    def __init__(self) -> None:
        self.idle = False
        self.calls = 0
        self.reason = "typed"
        self.reason_calls = 0

    def __call__(self, claim: terminal_claim.Claim) -> bool:
        self.calls += 1
        return self.idle

    def reason_of(self, claim: terminal_claim.Claim) -> str:
        self.reason_calls += 1
        return self.reason


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
            pane_idle=self.probe if self.WITH_PROBE else None, pane_reason=self.probe.reason_of,
            stall_seconds=STALL_THRESHOLD_SECONDS)

    def write_attachable(self, pid: int, hooked: bool = False) -> None:
        with open(self.claim_file(), "w", encoding="utf-8") as handle:
            json.dump({"pid": pid, "started": 1.0, "pane": PANE, "uuid": UUID, "config_dir": CONFIG_DIR,
                       "cwd": CWD, "socket": "", "hooked": hooked}, handle)

    def waiting_notices(self) -> int:
        return sum(1 for text in self.tg.texts(THREAD) if text == WAITING_TEXT)

    def stall_notices(self) -> list[str]:
        return [text for text in self.tg.texts(THREAD) if text.startswith(STALL_TEXT.split("{", 1)[0])]

    def stalled(self, count: int) -> Callable[[], bool]:
        """A wait_until predicate: each call moves the clock one step on, and it is true once count stall notices went
        out. The stall starts at the first poll of a wait, so the clock keeps moving until the topic is told."""
        def check() -> bool:
            self.clock.advance(STALL_STEP_SECONDS)
            return len(self.stall_notices()) >= count
        return check

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


class StalledPaneIsExplained(AttachableDispatcherCase):
    """A pane that stays busy or typed in past the stall threshold tells the topic why, once. Nothing is sent to the
    pane."""

    def test_after_the_threshold_the_topic_is_told_the_reason_once(self) -> None:
        self.write_attachable(os.getpid())
        self.probe.reason = "typed"
        self.deliver_first()
        self.assertTrue(wait_until(lambda: self.waiting_notices() == 1), "the topic was not told it waits")
        self.assertTrue(wait_until(self.stalled(1)), "the topic was not told why it waits")
        self.assertEqual(self.stall_notices(), [STALL_TEXT.format(minutes=1, detail=STALL_TEXTS["typed"])])
        self.assertEqual(self.waiting_notices(), 1)
        time.sleep(QUIET_SECONDS)  # more polls go by with the clock moved on
        self.clock.advance(STALL_STEP_SECONDS)
        time.sleep(QUIET_SECONDS)
        self.assertEqual(len(self.stall_notices()), 1, "the stall notice was sent more than once")
        self.assertEqual(self.turns.texts_run(THREAD), [], "the message ran while the pane was typed in")
        self.probe.idle = True
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the message never ran")
        self.assertEqual(self.turns.attached("first"), ATTACH)

    def test_each_reason_has_its_own_text(self) -> None:
        self.write_attachable(os.getpid())
        for number, reason in enumerate(("busy", "dialog", "gone"), start=1):
            self.probe.reason = reason
            self.ingest(number)
            self.deliver(number, reason)
            self.assertTrue(wait_until(self.stalled(number)), f"the {reason} pane was not explained")
            self.assertEqual(self.stall_notices()[-1], STALL_TEXT.format(minutes=1, detail=STALL_TEXTS[reason]))
            self.probe.idle = True
            self.assertTrue(wait_until(lambda: len(self.turns.texts_run(THREAD)) == number), "the message never ran")
            self.probe.idle = False

    def test_a_late_idle_reason_sends_no_stall_notice(self) -> None:
        self.write_attachable(os.getpid())
        self.probe.reason = "idle"  # the pane freed between the idle poll and the reason probe
        self.deliver_first()
        self.assertTrue(wait_until(lambda: self.waiting_notices() == 1), "the topic was not told it waits")
        for _ in range(3):  # the stall is due: the reason is asked for at each poll past the threshold
            self.clock.advance(STALL_STEP_SECONDS)
            time.sleep(QUIET_SECONDS)
        self.assertGreaterEqual(self.probe.reason_calls, 1, "the stall was never due, so the test proves nothing")
        self.assertEqual(self.stall_notices(), [], "an idle pane was explained as stalled")
        self.assertEqual(self.turns.texts_run(THREAD), [], "the message ran while the pane was not idle")
        self.probe.reason = "typed"  # the next poll decides: the stall is still open, so the notice goes out now
        self.assertTrue(wait_until(self.stalled(1)), "the stall was not reported once the pane is not idle")
        self.assertEqual(self.stall_notices(), [STALL_TEXT.format(minutes=1, detail=STALL_TEXTS["typed"])])
        self.probe.idle = True
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the message never ran")

    def test_no_stall_notice_when_the_pane_frees_in_time(self) -> None:
        self.write_attachable(os.getpid())
        self.deliver_first()
        self.assertTrue(wait_until(lambda: self.waiting_notices() == 1), "the topic was not told it waits")
        self.probe.idle = True
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the message never ran")
        self.clock.advance(STALL_STEP_SECONDS)
        time.sleep(QUIET_SECONDS)
        self.assertEqual(self.stall_notices(), [], "a pane that freed in time was reported as stalled")


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


class StallTextsMatchThePaneReasons(unittest.TestCase):
    """The reason strings of terminal_wait are the tmux_pane REASON_* values: every non-idle reason has a stall text,
    and every stall text is for one of those reasons."""

    def test_every_pane_reason_has_a_stall_text(self) -> None:
        reasons = {value for name, value in vars(tmux_pane).items() if name.startswith("REASON_")}
        self.assertIn(tmux_pane.REASON_IDLE, reasons)
        self.assertEqual(set(STALL_TEXTS), reasons - {tmux_pane.REASON_IDLE})


class AttachSpecReportsTheClaim(AttachableDispatcherCase):
    """The attach spec says whether the tab's claim is hooked, and reads the session the claim names right now."""

    def test_the_attach_spec_says_whether_the_tab_is_hooked(self) -> None:
        self.write_attachable(os.getpid(), hooked=True)
        self.probe.idle = True
        self.deliver_first()
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the message never ran")
        self.assertTrue(self.turns.attached("first").hooked, "a hooked claim gave a spec that is not hooked")
        self.assertTrue(wait_until(lambda: self.state(1) == STATE_DONE))

    def test_the_attach_spec_of_an_unhooked_claim_is_not_hooked(self) -> None:
        self.write_attachable(os.getpid())
        self.probe.idle = True
        self.deliver_first()
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the message never ran")
        self.assertFalse(self.turns.attached("first").hooked)
        self.assertTrue(wait_until(lambda: self.state(1) == STATE_DONE))

    def test_the_attach_spec_reads_the_claim_s_current_session(self) -> None:
        self.write_attachable(os.getpid(), hooked=True)
        self.probe.idle = True
        gate = self.turns.gates.setdefault("first", threading.Event())
        self.deliver_first()
        self.assertTrue(self.turns.started("first").wait(WAIT_SECONDS), "the turn did not start")
        claim_uuid = self.turns.attached("first").claim_uuid
        self.assertEqual(claim_uuid(), UUID, "the spec does not read the claim's session")
        self.assertTrue(terminal_claim.update_uuid(self.view, THREAD, os.getpid(), NEW_UUID))
        self.assertEqual(claim_uuid(), NEW_UUID, "the spec does not read the session the claim moved to")
        os.remove(self.claim_file())  # the claim is released: nothing names a session any more
        self.assertEqual(claim_uuid(), "", "a released claim still names a session")
        gate.set()
        self.assertTrue(wait_until(lambda: self.state(1) == STATE_DONE))
