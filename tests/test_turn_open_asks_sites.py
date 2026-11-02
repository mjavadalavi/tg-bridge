"""The three call sites of sync_open_asks (K11): a fresh turn (TurnRunner._drive), a turn of a kept session
(TurnRunner._drive_kept) and a self turn (SelfTurns.run). Each test posts an approval card while no turn is
running, then starts a turn of that binding and checks that the turn waits for the card: its active timer does not
advance and no cap stops it, until the card is answered.

The real PermissionService, WaitRouter, TurnState, LingerService and SelfTurns, on real repositories in a TempHome, with
a FakeClock and FakeBackend. No process starts, and the only sleeps are short pauses that let a wait loop see the clock.
If a call site loses its sync_open_asks line, its test fails.
"""

import dataclasses
import threading
import time
import unittest

from tests.fakes.backend import FAKE_CAPABILITIES, FakeBackend, FakeSession, RecordingSink, WaitForInterrupt, fake_spec
from tests.support import TempHome
from tests.test_self_turn import SelfTurnCase, command_ask
from tests.turn_support import CHAT, THREAD, TOPIC, RecordingFeed, TurnHarness, join_turn, run_async, wait_until
from tg_bridge.agents.events import BackgroundWork, TextEvent, TurnResult, TurnStatus
from tg_bridge.core.permissions import TurnBinding
from tg_bridge.core.turn_kit import TurnPlan
from tg_bridge.core.turn_state import TurnOutcome, TurnPhase, TurnState

OK = TurnResult(TurnStatus.OK, answer="first", duration_ms=5, num_turns=1, subtype="success")
BG = BackgroundWork(pending=1)  # the CLI reports one background task that is still running
MAX_ACTIVE = 2  # seconds: a turn that counts its wait would be capped within the first steps of a test
STEPS = 4  # clock steps of one second each, taken while a test checks that no cap has fired


class PhaseSession(FakeSession):
    """A FakeSession that notes the phase of its turn at the moment the text is sent."""

    def __init__(self, backend, spec, sink, turn: TurnState) -> None:
        super().__init__(backend, spec, sink)
        self.turn = turn
        self.phases: list[TurnPhase] = []

    def send_user(self, text: str, images: tuple[str, ...] = ()) -> None:
        self.phases.append(self.turn.phase)
        super().send_user(text, images)


def advance(clock, steps: int = STEPS) -> None:
    """Moves the FakeClock one second at a time, so a wait loop that is running sees every second pass."""
    for _ in range(steps):
        clock.advance(1)
        time.sleep(0.02)


class FreshTurnSiteTests(unittest.TestCase):
    """TurnRunner._drive: a new session whose binding gets a card while no turn is registered yet."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "default")

    def test_fresh_turn_with_an_open_card_sends_its_text_in_the_waiting_phase(self) -> None:
        backend = FakeBackend()
        launcher = self.h.config.resolve_launcher("", "", "fake-cmd")
        plan = TurnPlan(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", images=(), launcher=launcher,
                        backend=backend, cwd=self.home.home, mode="default", model=None, allow_rules=(),
                        deny_tools=(), system_rules="", stored_uuid="")
        turn = TurnState(THREAD, self.h.clock, max_active_seconds=MAX_ACTIVE, hard_cap_seconds=10_000)
        binding = TurnBinding(thread_id=THREAD, chat_id=CHAT, cwd=self.home.home, rules=backend.rules, mode="default")
        session = PhaseSession(backend, fake_spec(thread_id=THREAD, chat_id=CHAT), RecordingSink(), turn)
        listener = self.h.permissions.bind_turn(binding)
        self.h.permissions.attach(binding, session)
        listener.on_event(command_ask())  # idle: the card is posted and stays open
        self.assertEqual(self.h.permissions.open_asks(binding), 1)
        self.h.waits.register(THREAD, turn)
        done = threading.Event()
        done.set()  # the wait loop has nothing to wait for: the test checks the phase at the send, not the end
        runner = self.h.runner(self.h.registry_for(backend))
        try:
            refused, err = runner._drive(plan, turn, binding, session, RecordingFeed(), done, lambda: False)
        finally:
            self.h.waits.unregister(THREAD, turn)
            session.wait_idle()
        self.assertEqual((refused, err), (False, None))
        self.assertEqual(session.phases, [TurnPhase.WAITING_APPROVAL])


class KeptTurnSiteTests(unittest.TestCase):
    """TurnRunner._drive_kept: a turn that starts on a session kept by linger, while a card of that session is open."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self._close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "default")

    def _close(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)
        self.h.db.close()

    def start(self, scripts) -> None:
        self.h.set_config(linger_seconds=60, linger_max_sessions=3, max_turn_seconds=MAX_ACTIVE)
        capabilities = dataclasses.replace(FAKE_CAPABILITIES, linger=True)
        self.backend = FakeBackend(scripts, capabilities=capabilities)
        self.registry = self.h.registry_for(self.backend)
        self.service = self.h.linger(self.registry)
        self.runner = self.h.runner(self.registry, linger=self.service)

    def calls(self) -> list[str]:
        return [call[0] for call in self.backend.sessions[0].calls]

    def test_kept_turn_with_an_open_card_waits_until_the_card_is_answered(self) -> None:
        # the first turn ends with background work still pending, so its session is kept by linger
        self.start([[TextEvent(text="a"), BG, OK], [WaitForInterrupt()]])
        first = self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        self.assertEqual(first.answer, "first")
        self.assertTrue(self.service.holding(THREAD))
        # the card is posted while the kept session is idle, in default mode (the topic's row is changed for it)
        self.h.db.execute("UPDATE sessions SET mode=? WHERE thread_id=?", ("default", THREAD))
        binding = self.h.permissions._turns[THREAD]
        self.h.permissions.handle_request(binding, command_ask())
        self.assertEqual(len(self.h.asks.pending(THREAD)), 1)
        self.assertTrue(self.backend.sessions[0].wait_idle())

        thread, box = run_async(self.runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="again", sender="tester")
        self.assertTrue(wait_until(lambda: self.calls().count("send_user") == 2))
        advance(self.h.clock)
        self.assertTrue(thread.is_alive(), "the kept turn was capped while its card waited")
        self.assertNotIn("stop", self.calls())

        self.assertTrue(self.h.permissions.answer_ask(self.h.asks.pending(THREAD)[0].nonce, allow=True))
        advance(self.h.clock)  # running again: the active cap is reached and the turn is stopped
        join_turn(self.h.clock, thread)
        self.assertIn("stop", self.calls())
        self.assertEqual(box[0].outcome, TurnOutcome.CAPPED)


class SelfTurnSiteTests(SelfTurnCase):
    """SelfTurns.run: a self turn of a kept session, started while a card of the session is open."""

    def test_self_turn_started_with_an_open_card_waits_until_the_card_is_answered(self) -> None:
        self.h.set_config(max_turn_seconds=MAX_ACTIVE)
        self.keep([WaitForInterrupt()], mode="default")
        self.h.permissions.handle_request(self.binding, command_ask())  # idle: the card is posted before the self turn
        self.assertEqual(len(self.h.asks.pending(THREAD)), 1)
        self.wake()
        box: list[object] = []
        thread = threading.Thread(target=lambda: box.append(self.self_turns().run(self.context())), daemon=True)
        thread.start()
        self.assertTrue(wait_until(lambda: self.h.live.running(THREAD) is not None))
        advance(self.h.clock)
        self.assertTrue(thread.is_alive(), "the self turn was capped while its card waited")
        self.assertNotIn("stop", self.calls())

        self.assertTrue(self.h.permissions.answer_ask(self.h.asks.pending(THREAD)[0].nonce, allow=True))
        advance(self.h.clock)  # running again: the active cap is reached and the self turn is stopped
        join_turn(self.h.clock, thread)
        self.assertIn("stop", self.calls())
        self.assertEqual(box[0].outcome, TurnOutcome.CAPPED)


if __name__ == "__main__":
    unittest.main()
