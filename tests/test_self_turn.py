"""SelfTurns (linger): one self turn of a kept session, on FakeBackend, FakeClock and the real SessionRouter.

The session is opened by hand: its first user turn ends with a Later item, so the fake plays a self turn once the test
sets the Later's gate. The router is held and idle in between, as LingerService holds it. The tests check the 🔔 reply,
the feed and the turns row, the running state, the order of the buffered events, the interrupt of one update, the cap,
the closed router and that the input of the session is never closed.
"""

import threading
import unittest

from tests.fakes.backend import FAKE_CAPABILITIES, FakeBackend, Later, WaitForInterrupt
from tests.fakes.uploads import FakeUploader
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, RecordingFeed, TurnHarness, join_turn, wait_until
from tg_bridge.agents.base import SessionSpec
from tg_bridge.agents.events import ApprovalRequest, Decision, TextEvent, TurnResult, TurnStatus
from tg_bridge.config import LauncherSpec
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.permissions import TurnBinding
from tg_bridge.core.self_turn import SelfTurnContext, SelfTurns
from tg_bridge.core.session_router import SessionRouter
from tg_bridge.core.turn_kit import TurnPlan
from tg_bridge.core.turn_listeners import SessionRecorder
from tg_bridge.core.turn_state import TurnOutcome
from tg_bridge.telegram.answers import AnswerSender

PREFIX = "🔔 background update"
OK_BG = TurnResult(TurnStatus.OK, answer="bg done", duration_ms=5, num_turns=1, subtype="success")


def command_ask() -> ApprovalRequest:
    """A Bash ask that a default-mode session puts on a card, as the turn runner tests do."""
    return ApprovalRequest(approval_id="r1", kind="command", tool="Bash", input={"command": "git status"},
                           summary="Bash", choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY}))


class FailingSender:
    """Stands in for AnswerSender and Messenger: every send raises, as a Telegram outage does."""

    def send_answer(self, *args, **kwargs) -> None:
        raise RuntimeError("telegram unavailable")

    def send_text(self, *args, **kwargs) -> None:
        raise RuntimeError("telegram unavailable")


class WatchFeed(RecordingFeed):
    """A feed that calls probe() after every event it receives."""

    def __init__(self, probe) -> None:
        super().__init__()
        self._probe = probe

    def on_event(self, event: object) -> None:
        super().on_event(event)
        self._probe()


class StartOrderFeed(RecordingFeed):
    """A feed that notes whether start() had been called when its first event arrived."""

    def __init__(self) -> None:
        super().__init__()
        self.started_at_first_event: bool | None = None

    def on_event(self, event: object) -> None:
        if self.started_at_first_event is None:
            self.started_at_first_event = self.started
        super().on_event(event)


class SelfTurnCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self._close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")

    def _close(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)
        if hasattr(self, "router"):
            self.router.close()
        self.h.db.close()

    def keep(self, background, *, mode: str = "bypassPermissions") -> None:
        """Opens a session whose first user turn answers 'first' and ends with a Later of the given background items.
        The router is held and idle when this returns; the Later waits for self.gate."""
        self.gate = threading.Event()
        self.backend = FakeBackend([[TextEvent(text="first"), TurnResult(TurnStatus.OK, answer="first"),
                                     Later(self.gate, tuple(background))]], capabilities=FAKE_CAPABILITIES)
        self.launcher = LauncherSpec(name="fake", backend="fake", command="fake-cmd", argv=("fake-cmd",),
                                     config_dir=self.home.home)
        self.plan = TurnPlan(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="", images=(), launcher=self.launcher,
                             backend=self.backend, cwd=self.home.home, mode=mode, model=None, allow_rules=(),
                             deny_tools=(), system_rules="", stored_uuid="")
        self.router = SessionRouter(self.h.logger)
        spec = SessionSpec(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, launcher=self.launcher, cwd=self.home.home,
                           mode=mode, model=None, resume_id=None, new_id=None, allow_rules=(), deny_tools=(),
                           system_rules="", env={}, errfile_dir=self.h.errfile_dir)
        self.session = self.backend.open_session(spec, self.router)
        self.session.start()
        first: list[object] = []
        self.assertTrue(self.router.attach(first.append))
        self.session.send_user("hello")
        self.assertTrue(wait_until(lambda: any(isinstance(event, TurnResult) for event in first)))
        self.assertTrue(self.session.wait_idle())
        self.router.hold(lambda event: None, lambda: None)
        self.recorder = SessionRecorder(sessions=self.h.sessions, history=self.h.history, kv=self.h.kv,
                                        thread_id=THREAD, text="", cwd=self.home.home, launcher="fake-cmd",
                                        backend="fake", known_uuid="")
        self.binding = TurnBinding(thread_id=THREAD, chat_id=CHAT, cwd=self.home.home, rules=self.backend.rules,
                                   mode=mode)
        self.h.permissions.attach(self.binding, self.session)  # as the first user turn's launch does for a kept session

    def wake(self) -> None:
        """Sets the Later's gate and waits until the router has seen the self turn begin."""
        self.gate.set()
        self.assertTrue(wait_until(lambda: self.router.self_pending))

    def context(self, interrupt: threading.Event | None = None) -> SelfTurnContext:
        return SelfTurnContext(plan=self.plan, session=self.session, router=self.router, binding=self.binding,
                               recorder=self.recorder, interrupt=interrupt or threading.Event())

    def self_turns(self, feeds=None, answers=None, messenger=None) -> SelfTurns:
        return SelfTurns(
            config=self.h.config, clock=self.h.clock, logger=self.h.logger, feeds=feeds or self.h.feeds, views=None,
            permissions=self.h.permissions,
            handoff=HandoffService(self.h.kv, self.h.registry_for(self.backend), self.h.logger), waits=self.h.waits,
            live=self.h.live, turns=self.h.turns,
            answers=answers or AnswerSender(self.h.tg, FakeUploader(), self.h.messenger, self.h.logger, self.h.clock),
            messenger=messenger or self.h.messenger, poll_seconds=0.005)

    def posted(self) -> list[str]:
        return self.h.tg.texts(THREAD)

    def logged(self) -> str:
        with open(self.h.logger.path, encoding="utf-8") as f:
            return f.read()

    def calls(self) -> list[str]:
        return [call[0] for call in self.session.calls]


class SelfTurnReplyTests(SelfTurnCase):
    def test_self_turn_answer_is_posted_with_the_background_prefix(self) -> None:
        self.keep([TextEvent(text="bg"), OK_BG])
        self.wake()
        end = self.self_turns().run(self.context())
        self.assertEqual((end.ran, end.outcome, end.alive), (True, TurnOutcome.OK, True))
        self.assertIn(f"{PREFIX}\n\nbg done", self.posted())

    def test_self_turn_error_is_posted_with_the_prefix(self) -> None:
        failure = TurnResult(TurnStatus.ERROR, answer=None, error="boom", subtype="error_during_execution")
        self.keep([TextEvent(text="bg"), failure])
        self.wake()
        end = self.self_turns().run(self.context())
        self.assertEqual(end.outcome, TurnOutcome.ERROR)
        self.assertTrue(any(text.startswith(f"{PREFIX}\n❌ error_during_execution") for text in self.posted()))

    def test_failed_answer_send_is_logged_and_the_session_is_kept(self) -> None:
        self.keep([TextEvent(text="bg"), OK_BG])
        self.wake()
        end = self.self_turns(answers=FailingSender()).run(self.context())
        self.assertEqual((end.ran, end.outcome, end.alive), (True, TurnOutcome.OK, True))
        self.assertNotIn("stop", self.calls())
        self.assertEqual(self.h.db.query("SELECT status FROM turns WHERE thread_id=?", (THREAD,))[0]["status"], "ok")
        self.assertIn("reply not sent", self.logged())

    def test_failed_error_send_is_logged_and_the_session_is_kept(self) -> None:
        failure = TurnResult(TurnStatus.ERROR, answer=None, error="boom", subtype="error_during_execution")
        self.keep([TextEvent(text="bg"), failure])
        self.wake()
        end = self.self_turns(messenger=FailingSender()).run(self.context())
        self.assertEqual((end.ran, end.outcome, end.alive), (True, TurnOutcome.ERROR, True))
        self.assertNotIn("stop", self.calls())
        self.assertIn("reply not sent", self.logged())

    def test_self_turn_writes_a_turns_row_and_a_feed_summary(self) -> None:
        self.keep([TextEvent(text="bg"), OK_BG])
        self.wake()
        self.self_turns().run(self.context())
        rows = self.h.db.query("SELECT status FROM turns WHERE thread_id=? ORDER BY id", (THREAD,))
        self.assertEqual([row["status"] for row in rows], ["ok"])
        self.assertEqual(len(self.h.feeds.last.headers), 1)

    def test_self_turn_is_a_running_turn_while_it_runs(self) -> None:
        seen: list[bool] = []
        feeds = lambda chat, thread, parked: WatchFeed(lambda: seen.append(self.h.live.running(THREAD) is not None))
        self.keep([TextEvent(text="bg"), OK_BG])
        self.wake()
        self.self_turns(feeds).run(self.context())
        self.assertTrue(seen and all(seen))
        self.assertIsNone(self.h.live.running(THREAD))

    def test_buffered_events_reach_the_self_turn_in_order(self) -> None:
        self.keep([TextEvent(text="bg-1"), TextEvent(text="bg-2"), OK_BG])
        self.wake()
        self.self_turns().run(self.context())
        names = [type(event).__name__ for event in self.h.feeds.last.events]
        self.assertEqual(names, ["TextEvent", "TextEvent", "TurnResult"])
        self.assertEqual([event.text for event in self.h.feeds.last.events[:2]], ["bg-1", "bg-2"])

    def test_buffered_events_reach_a_started_feed(self) -> None:
        made: list[StartOrderFeed] = []

        def feeds(chat, thread, parked):
            made.append(StartOrderFeed())
            return made[-1]

        self.keep([TextEvent(text="bg"), OK_BG])
        self.wake()
        self.self_turns(feeds).run(self.context())
        self.assertIs(made[-1].started_at_first_event, True)

    def test_self_turn_card_uses_the_session_binding_and_is_not_voided(self) -> None:
        self.keep([command_ask(), OK_BG], mode="default")
        self.wake()
        box: list[object] = []
        thread = threading.Thread(target=lambda: box.append(self.self_turns().run(self.context())), daemon=True)
        thread.start()
        self.assertTrue(wait_until(lambda: len(self.h.asks.pending(THREAD)) == 1))
        self.assertTrue(self.h.tg.buttons(THREAD), "the self turn's approval posted no card")
        self.assertTrue(self.h.permissions.answer_ask(self.h.asks.pending(THREAD)[0].nonce, allow=True))
        join_turn(self.h.clock, thread)
        end = box[0]
        self.assertEqual((end.ran, end.outcome), (True, TurnOutcome.OK))
        self.assertIn("answer", self.calls())
        self.assertIn(f"{PREFIX}\n\nbg done", self.posted())
        self.assertFalse(self.binding.closed)

    def test_approval_wait_is_not_counted_against_the_turn_cap(self) -> None:
        self.h.set_config(max_turn_seconds=2)
        self.keep([command_ask(), OK_BG], mode="default")
        self.wake()
        box: list[object] = []
        thread = threading.Thread(target=lambda: box.append(self.self_turns().run(self.context())), daemon=True)
        thread.start()
        self.assertTrue(wait_until(lambda: len(self.h.asks.pending(THREAD)) == 1))
        for _ in range(4):  # one step at a time, so a wait loop that has started sees every second pass
            self.h.clock.advance(1)
            thread.join(0.02)
        self.assertTrue(thread.is_alive(), "the self turn was capped while its card waited for a tap")
        self.assertTrue(self.h.permissions.answer_ask(self.h.asks.pending(THREAD)[0].nonce, allow=True))
        join_turn(self.h.clock, thread)
        self.assertEqual(box[0].outcome, TurnOutcome.OK)


class SelfTurnSessionTests(SelfTurnCase):
    def test_self_turn_never_closes_input(self) -> None:
        self.keep([TextEvent(text="bg"), OK_BG])
        self.wake()
        self.self_turns().run(self.context())
        self.assertNotIn("end_gracefully", self.calls())
        self.assertNotIn("stop", self.calls())
        self.assertTrue(self.session.alive())

    def test_interrupt_request_ends_the_update_and_keeps_the_session(self) -> None:
        self.keep([TextEvent(text="bg"), WaitForInterrupt()])
        self.wake()
        interrupt = threading.Event()
        interrupt.set()
        end = self.self_turns().run(self.context(interrupt))
        self.assertEqual((end.outcome, end.alive), (TurnOutcome.INTERRUPTED, True))
        self.assertIn(f"{PREFIX}\n⏹ interrupted — session kept", self.posted())
        self.assertEqual(self.calls().count("interrupt"), 1)
        self.assertNotIn("stop", self.calls())

    def test_refused_interrupt_stops_the_update_and_releases_the_hold(self) -> None:
        self.keep([TextEvent(text="bg"), WaitForInterrupt()])
        self.wake()
        refusals: list[bool] = []

        def refuse() -> bool:  # the session takes no interrupt, as when the TUI is gone
            refusals.append(True)
            return False

        self.session.interrupt = refuse
        interrupt = threading.Event()
        interrupt.set()
        box: list[object] = []
        thread = threading.Thread(target=lambda: box.append(self.self_turns().run(self.context(interrupt))), daemon=True)
        thread.start()
        join_turn(self.h.clock, thread)
        end = box[0]
        self.assertEqual(len(refusals), 1)
        self.assertEqual((end.outcome, end.alive), (TurnOutcome.STOPPED, False),
                         "a refused interrupt must end the update, not leave it running to the cap")
        self.assertIn("stop", self.calls())
        self.assertTrue(any(text.startswith(PREFIX) and "stopped" in text for text in self.posted()))

    def test_raising_interrupt_stops_the_update_like_a_refusal(self) -> None:
        self.keep([TextEvent(text="bg"), WaitForInterrupt()])
        self.wake()

        def raise_on_interrupt() -> bool:  # the session's control channel fails instead of answering
            raise RuntimeError("control channel gone")

        self.session.interrupt = raise_on_interrupt
        interrupt = threading.Event()
        interrupt.set()
        box: list[object] = []
        thread = threading.Thread(target=lambda: box.append(self.self_turns().run(self.context(interrupt))), daemon=True)
        thread.start()
        join_turn(self.h.clock, thread)
        end = box[0]
        self.assertEqual((end.outcome, end.alive), (TurnOutcome.STOPPED, False),
                         "a raising interrupt must end the update as a refusal, not fail the wait as a launch error")
        self.assertIn("stop", self.calls())
        self.assertIn("interrupt raised", self.logged())

    def test_capped_self_turn_stops_the_session(self) -> None:
        self.h.set_config(max_turn_seconds=3)
        self.keep([TextEvent(text="bg"), WaitForInterrupt()])
        self.wake()
        box: list[object] = []
        thread = threading.Thread(target=lambda: box.append(self.self_turns().run(self.context())), daemon=True)
        thread.start()
        join_turn(self.h.clock, thread)
        end = box[0]
        self.assertEqual((end.outcome, end.alive), (TurnOutcome.CAPPED, False))
        self.assertIn("stop", self.calls())
        self.assertTrue(any(text.startswith(PREFIX) and "The session was closed" in text for text in self.posted()),
                        "a capped self turn closed its session, so its reply must not say the session is kept")

    def test_closed_router_runs_nothing(self) -> None:
        self.keep([TextEvent(text="bg"), OK_BG])
        self.router.close()
        end = self.self_turns().run(self.context())
        self.assertEqual((end.ran, end.outcome), (False, None))
        self.assertEqual(self.posted(), [])
        self.assertEqual(self.h.db.query("SELECT COUNT(*) AS n FROM turns WHERE thread_id=?", (THREAD,))[0]["n"], 0)


if __name__ == "__main__":
    unittest.main()
