"""LingerService (linger): offer, hold, claim and release of a kept session, on FakeBackend, FakeClock and the real
SessionRouter. A stub SelfTurns stands in for the self turn: it attaches a gate the way the real run does, then returns
the end the test scripted. The tests check the refusals of an offer, the end conditions in their order, the quiet
window, the cap counted from the last user turn, the self turns, the claim of a kept session and the locks.
"""

import dataclasses
import json
import threading
import unittest
from collections.abc import Callable
from unittest import mock

from tests.fakes.backend import FAKE_CAPABILITIES, FakeBackend, Later
from tests.fakes.uploads import FakeUploader
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, TurnHarness, wait_until
from tg_bridge.agents.base import SessionSpec
from tg_bridge.agents.events import BackgroundWork, ProcessExited, TextEvent, TurnBegan, TurnResult, TurnStatus
from tg_bridge.config import LauncherError
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core import linger as linger_module
from tg_bridge.core import linger_types
from tg_bridge.core.linger import (
    LINGER_QUIET_SECONDS,
    RELEASE_GRACE_SECONDS,
    HoldEnd,
    LingerOffer,
    LingerService,
    LingerStatus,
    linger_key,
)
from tg_bridge.core.permissions import TurnBinding
from tg_bridge.core.self_turn import SelfTurnContext, SelfTurnEnd, SelfTurns
from tg_bridge.core.session_router import SessionRouter
from tg_bridge.core.topic_key import TopicKey, topic_cwd, topic_key, topic_launcher, topic_mode, topic_model
from tg_bridge.core.turn_kit import TurnPlan, build_kit, wire_gate
from tg_bridge.core.turn_listeners import SessionRecorder
from tg_bridge.core.turn_state import TurnOutcome, TurnState
from tg_bridge.core.turn_text import BACKGROUND_PREFIX
from tg_bridge.core.view_recorder import STATE_IDLE, STATE_LINGERING
from tg_bridge.telegram.answers import AnswerSender

BG = (BackgroundWork(pending=1),)  # a turn that starts background work and leaves it pending
WAIT_SECONDS = 10.0


class StubSelfTurns:
    """Stands in for SelfTurns. Each run attaches a gate to the router (which clears the pending self turn, as the real
    run does), records its context and returns the scripted end. on_run, when set, runs first, on the caller's thread."""

    def __init__(self) -> None:
        self.outcome = TurnOutcome.OK
        self.alive = True
        self.calls: list[SelfTurnContext] = []
        self.events: list[object] = []
        self.on_run: Callable[[SelfTurnContext], None] | None = None

    def run(self, ctx: SelfTurnContext) -> SelfTurnEnd:
        self.calls.append(ctx)
        if self.on_run is not None:
            self.on_run(ctx)
        ctx.router.attach(self.events.append)
        return SelfTurnEnd(ran=True, outcome=self.outcome, alive=self.alive)


class FakeView:
    """A topic view that records its state changes and whether it was finished."""

    def __init__(self) -> None:
        self.states: list[str] = []
        self.finished = False

    def on_event(self, event: object) -> None:
        pass

    def set_state(self, state: str) -> None:
        self.states.append(state)

    def finish(self) -> None:
        self.finished = True


class LingerCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self.h.db.close)  # registered first, so it runs last: the sessions stop before the database closes
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")
        self.h.set_config(linger_seconds=1800, linger_max_sessions=3)
        self.stub = StubSelfTurns()
        self.view = FakeView()
        self.service = self.make_service()

    def make_service(self) -> LingerService:
        return LingerService(
            config=self.h.config, sessions=self.h.sessions, kv=self.h.kv, live=self.h.live,
            permissions=self.h.permissions, self_turns=self.stub, clock=self.h.clock, logger=self.h.logger,
            poll_seconds=0.005)

    def reconfigure(self, **overrides: int | float) -> None:
        """Replaces the config keys and builds the service again, since the service reads the config it was given."""
        self.h.set_config(**{"linger_seconds": 1800, "linger_max_sessions": 3, **overrides})
        self.service = self.make_service()

    def start(self, *, before: tuple[object, ...] = (), after: tuple[object, ...] = (),
              status: TurnStatus = TurnStatus.OK, linger: bool = True) -> None:
        """Runs one user turn on a fresh session: its events, then its result, then a Later that waits for self.gate and
        plays the after items as a self turn. The router is attached to the turn's gate, as TurnRunner does, so the
        result detaches it. The attributes below describe the turn, for offer() and for the tests."""
        self.gate = threading.Event()
        self.addCleanup(self.gate.set)
        caps = dataclasses.replace(FAKE_CAPABILITIES, linger=linger)
        self.backend = FakeBackend([[TextEvent(text="first"), *before, TurnResult(status, answer="first"),
                                     Later(self.gate, tuple(after))]], capabilities=caps)
        row = self.h.sessions.get(THREAD)
        assert row is not None
        launcher = topic_launcher(self.h.config, row, TOPIC)
        self.plan = TurnPlan(
            thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", images=(), launcher=launcher,
            backend=self.backend, cwd=topic_cwd(self.h.config, row, TOPIC), mode=topic_mode(self.h.config, row),
            model=topic_model(row), allow_rules=(), deny_tools=(), system_rules="", stored_uuid=row.uuid)
        self.router = SessionRouter(self.h.logger)
        spec = SessionSpec(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, launcher=launcher, cwd=self.plan.cwd,
                           mode=self.plan.mode, model=self.plan.model, resume_id=None, new_id=None, allow_rules=(),
                           deny_tools=(), system_rules="", env={}, errfile_dir=self.h.errfile_dir)
        self.session = self.backend.open_session(spec, self.router)
        self.addCleanup(self.session.stop, 1)
        self.session.start()
        self.recorder = SessionRecorder(sessions=self.h.sessions, history=self.h.history, kv=self.h.kv,
                                        thread_id=THREAD, text="hello", cwd=self.plan.cwd,
                                        launcher=launcher.command, backend=self.backend.name, known_uuid=row.uuid)
        self.binding = TurnBinding(thread_id=THREAD, chat_id=CHAT, cwd=self.plan.cwd, rules=self.backend.rules,
                                   mode=self.plan.mode)
        handoff = HandoffService(self.h.kv, self.h.registry_for(self.backend), self.h.logger)
        self.kit = build_kit(self.plan, self.recorder, None, clock=self.h.clock, config=self.h.config,
                             feeds=self.h.feeds, permissions=self.h.permissions, handoff=handoff,
                             pending_digest=lambda: "", open_view=lambda: self.view, binding=self.binding)
        self.assertTrue(self.router.attach(wire_gate(self.kit, self.h.logger)))
        self.session.send_user("hello")
        self.assertTrue(self.kit.done.wait(WAIT_SECONDS))
        self.assertTrue(self.session.wait_idle())

    def offer_args(self) -> LingerOffer:
        return LingerOffer(plan=self.plan, session=self.session, router=self.router, binding=self.binding,
                           recorder=self.recorder, kit=self.kit)

    def offer(self) -> bool:
        return self.service.offer(self.offer_args())

    def wake_self_turn(self) -> None:
        """Sets the Later's gate, so the fake plays its self turn, and waits until the router has seen it begin."""
        self.gate.set()
        self.assertTrue(wait_until(lambda: self.router.self_pending))

    def key(self) -> TopicKey:
        """The key the topic resolves to now, as the claiming turn computes it."""
        return topic_key(self.h.config, self.h.sessions.get(THREAD), TOPIC)

    def assertRefused(self) -> None:
        self.assertFalse(self.offer())
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))
        self.assertIsNone(self.h.live.lingering(THREAD))
        self.assertFalse(self.service.holding(THREAD))

    def stop_calls(self) -> list[object]:
        return [call for call in self.session.calls if call[0] == "stop"]

    def feed_self_turns(self, count: int) -> None:
        """Buffers count self turns in the router, each one waiting behind its turn mark, as a busy runner does."""
        for _ in range(count):
            for event in (TurnBegan(origin="task-notification"), TextEvent(text="self"), SELF_RESULT):
                self.router(event)


class OfferTests(LingerCase):
    def test_offer_refusals(self) -> None:
        with self.subTest("the backend cannot keep a session open"):
            self.start(before=BG, linger=False)
            self.assertRefused()
        with self.subTest("linger_seconds is 0"):
            self.reconfigure(linger_seconds=0)
            self.start(before=BG)
            self.assertRefused()
        with self.subTest("linger_max_sessions is 0"):
            self.reconfigure(linger_max_sessions=0)
            self.start(before=BG)
            self.assertRefused()
        self.reconfigure()
        with self.subTest("the turn was stopped"):
            self.start(before=BG)
            self.kit.turn.request_stop()
            self.assertRefused()
        with self.subTest("the turn hit its cap"):
            self.start(before=BG)
            with mock.patch.object(TurnState, "limit_hit", new_callable=mock.PropertyMock, return_value=60.0):
                self.assertRefused()
        with self.subTest("the turn did not end OK"):
            self.start(before=BG, status=TurnStatus.ERROR)
            self.assertRefused()
        with self.subTest("the process is gone"):
            self.start(before=BG)
            self.session.stop(grace=1)
            self.assertRefused()
        with self.subTest("no background work is pending"):
            self.start()
            self.assertRefused()
        with self.subTest("the session cap is full"):
            self.reconfigure(linger_max_sessions=1)
            other = object()
            self.assertTrue(self.h.live.begin_linger(THREAD + 1, other, 1))  # another topic keeps the only slot
            self.start(before=BG)
            self.assertRefused()
            self.h.live.end_linger(THREAD + 1, other)

    def test_refused_while_shutting_down(self) -> None:
        self.start(before=BG)
        self.h.live.begin_shutdown()
        self.assertRefused()
        self.assertTrue(any(f"linger: thread {THREAD} not kept: the bridge is shutting down" in line
                            for line in self.h.log_lines()))

    def test_refused_when_the_turn_was_aborted(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.h.live.begin_turn(THREAD, self.session, self.kit.feed))
        self.assertIsNotNone(self.h.live.request_abort(THREAD, stop_tap=False))
        self.assertRefused()
        self.h.live.end_turn(THREAD, self.session, self.kit.feed)

    def test_a_second_session_cannot_take_a_held_topic(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.start(before=BG)  # a new session with a new router for the same topic
        self.assertFalse(self.offer())
        self.assertTrue(self.service.holding(THREAD))

    def test_offer_keeps_and_marks(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        marker = json.loads(self.h.kv.get(linger_key(THREAD)) or "{}")
        self.assertEqual((marker["chat_id"], marker["name"]), (CHAT, TOPIC))
        self.assertIn("since", marker)
        self.assertIs(self.h.live.lingering(THREAD), self.session)
        self.assertTrue(self.service.holding(THREAD))
        self.assertTrue(self.service.idle_alive(THREAD))
        self.assertEqual(self.view.states, [STATE_LINGERING])
        status = self.service.status(THREAD)
        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual((status.pending, status.seconds_left, status.self_turn), (1, 1800.0, False))

    def test_activity_without_pending_is_kept_then_done_after_quiet(self) -> None:
        self.start(before=(BackgroundWork(pending=0),))
        self.assertTrue(self.offer())
        self.assertIsNone(self.service.wait_once(THREAD).end)  # the first tick sees no pending work: the window opens
        self.h.clock.advance(LINGER_QUIET_SECONDS - 1)
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.h.clock.advance(2)
        self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.DONE)
        self.assertEqual(self.stop_calls(), [("stop", RELEASE_GRACE_SECONDS)])
        self.assertFalse(self.service.holding(THREAD))
        self.assertEqual(self.stub.calls, [])

    def test_a_self_turn_buffered_before_the_offer_keeps_the_session(self) -> None:
        self.start()  # the user turn ends with no background work reported
        # the runner arms a fresh session before its turn: a self turn after the result is buffered
        self.router.arm(lambda event: None)
        self.gate.set()
        self.assertTrue(wait_until(lambda: self.router.self_pending))
        self.assertEqual((self.router.pending, self.router.activity), (0, False))
        self.assertTrue(self.offer(), "a buffered self turn is work to wait for")
        self.assertTrue(self.service.holding(THREAD))

    def test_re_offer_moves_the_deadline(self) -> None:
        self.reconfigure(linger_seconds=60)
        self.start(before=BG)
        self.assertTrue(self.offer())  # deadline t0 + 60
        self.h.clock.advance(50)
        self.assertTrue(self.offer())  # the same session: the deadline is now t0 + 110
        self.h.clock.advance(20)  # t0 + 70: past the first deadline
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.h.clock.advance(40)  # t0 + 110
        self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.CAP)

    def test_cap_counts_from_the_last_user_turn_not_self_turns(self) -> None:
        self.reconfigure(linger_seconds=60)
        self.start(before=BG)
        self.assertTrue(self.offer())  # deadline t0 + 60
        self.h.clock.advance(50)
        self.wake_self_turn()
        self.assertIsNone(self.service.wait_once(THREAD).end)  # the self turn runs; the session is kept
        self.assertEqual(len(self.stub.calls), 1)
        self.h.clock.advance(10)  # t0 + 60
        result = self.service.wait_once(THREAD)
        self.assertEqual((result.end, result.pending), (HoldEnd.CAP, 1))


class HoldTests(LingerCase):
    def test_release_on_request(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.assertTrue(self.service.request_release(THREAD))
        result = self.service.wait_once(THREAD)
        self.assertEqual((result.end, result.pending, result.chat_id), (HoldEnd.STOPPED, 1, CHAT))
        self.assertEqual(self.stop_calls(), [("stop", RELEASE_GRACE_SECONDS)])
        self.assertTrue(self.binding.closed)
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))
        self.assertIsNone(self.h.live.lingering(THREAD))
        self.assertEqual(self.view.states[-1], STATE_IDLE)

    def test_release_on_process_exit(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.router(ProcessExited(returncode=0))  # the reader's event after the exit, as the PTY reader sends it
        self.session.stop(grace=1)
        result = self.service.wait_once(THREAD)
        self.assertEqual((result.end, result.pending), (HoldEnd.EXITED, 1))
        self.assertFalse(self.service.holding(THREAD))

    def test_missing_row_or_launcher_error_is_changed(self) -> None:
        with self.subTest("the session row is gone"):
            self.start(before=BG)
            self.assertTrue(self.offer())
            with mock.patch.object(self.h.sessions, "get", return_value=None):
                self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.CHANGED)
            self.assertFalse(self.service.holding(THREAD))
        with self.subTest("the topic's launcher can no longer be resolved"):
            self.start(before=BG)
            self.assertTrue(self.offer())
            with mock.patch("tg_bridge.core.linger.topic_key", side_effect=LauncherError("gone")):
                self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.CHANGED)
            self.assertFalse(self.service.holding(THREAD))

    def test_stop_that_raises_still_voids_the_cards_and_removes_the_marker(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())

        def failing_stop(grace: float = 5.0) -> int | None:
            raise RuntimeError("stop failed")

        self.session.stop = failing_stop  # type: ignore[method-assign]
        self.assertTrue(self.service.request_release(THREAD))
        with self.assertRaises(RuntimeError):
            self.service.wait_once(THREAD)
        self.assertTrue(self.binding.closed)
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))
        self.assertIsNone(self.h.live.lingering(THREAD))
        self.assertFalse(self.service.holding(THREAD))

    def test_release_on_topic_change(self) -> None:
        cases: list[tuple[str, Callable[[], None]]] = [
            ("mode", lambda: self.h.sessions.set_mode(THREAD, "default")),
            ("model", lambda: self.h.sessions.set_model(THREAD, "other-model")),
            ("cwd", lambda: self.h.sessions.set_cwd(THREAD, self.home.home + "/elsewhere")),
            ("launcher pick", lambda: self.h.sessions.set_launcher_pick(THREAD, "alt")),
        ]
        for what, change in cases:
            with self.subTest(what):
                self.start(before=BG)
                self.assertTrue(self.offer())
                change()
                self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.CHANGED)
                self.assertFalse(self.service.holding(THREAD))
        with self.subTest("uuid cleared by new_session"):
            self.h.sessions.set_uuid(THREAD, "uuid-1")
            self.start(before=BG)
            self.assertTrue(self.offer())
            self.h.sessions.new_session(THREAD)
            self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.CHANGED)

    def test_an_approved_plan_is_never_kept_nor_held_on(self) -> None:
        """Its CLI is still in native plan mode: the offer is refused, and a hold whose plan was approved in a self
        turn ends as CHANGED even though the topic's key did not change."""
        with self.subTest("the offer"):
            self.start(before=BG)
            self.binding.plan_approved = True
            self.assertFalse(self.offer())
        with self.subTest("an existing hold"):
            self.start(before=BG)
            self.assertTrue(self.offer())
            self.binding.plan_approved = True
            self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.CHANGED)
            self.assertFalse(self.service.holding(THREAD))

    def test_request_release_needs_an_idle_session(self) -> None:
        self.assertFalse(self.service.request_release(THREAD))
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.stub.on_run = lambda ctx: self.assertFalse(self.service.request_release(THREAD))
        self.wake_self_turn()
        self.assertIsNone(self.service.wait_once(THREAD).end)

    def test_shutdown_leaves_the_marker_and_does_not_stop(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.h.live.begin_shutdown()
        self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.SHUTDOWN)
        self.assertEqual(self.stop_calls(), [])
        self.assertIsNotNone(self.h.kv.get(linger_key(THREAD)))

    def test_forget_drops_the_record_and_holds_checks_the_session(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.assertTrue(self.service.holds(THREAD, self.session))
        self.assertFalse(self.service.holds(THREAD, object()))  # another session is not the held one
        self.service.forget(THREAD, self.session)
        self.assertFalse(self.service.holding(THREAD))
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))
        self.assertFalse(self.router.attach(lambda event: None))  # the router is closed

    def test_no_lock_held_while_stopping(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        probed: list[bool] = []  # per stop: did a status() call from another thread finish while the stop ran?
        real_stop = self.session.stop

        def stop_and_probe(grace: float = 5.0) -> int | None:
            probe = threading.Thread(target=lambda: self.service.status(THREAD))
            probe.start()
            probe.join(WAIT_SECONDS)  # a lock held by the release would keep the probe waiting here
            probed.append(not probe.is_alive())
            return real_stop(grace)

        self.session.stop = stop_and_probe  # type: ignore[method-assign]
        self.assertTrue(self.service.request_release(THREAD))
        self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.STOPPED)
        self.assertEqual(probed, [True])


class SelfTurnTests(LingerCase):
    def test_wait_once_runs_a_pending_self_turn(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.wake_self_turn()
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.assertEqual(len(self.stub.calls), 1)
        self.assertTrue(self.service.idle_alive(THREAD))
        self.assertEqual(self.view.states[-1], STATE_LINGERING)

    def test_bad_self_turn_releases(self) -> None:
        self.stub.outcome = TurnOutcome.ERROR
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.wake_self_turn()
        self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.SELF_ENDED)
        self.assertFalse(self.service.holding(THREAD))
        self.assertEqual(self.stop_calls(), [("stop", RELEASE_GRACE_SECONDS)])
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))

    def test_interrupted_self_turn_keeps_the_session(self) -> None:
        self.stub.outcome = TurnOutcome.INTERRUPTED
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.wake_self_turn()
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.assertTrue(self.service.holding(THREAD))
        self.assertEqual(self.stop_calls(), [])

    def test_request_interrupt_reaches_the_running_self_turn(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.assertFalse(self.service.request_interrupt(THREAD))  # idle: nothing to interrupt
        results: list[bool] = []
        self.stub.on_run = lambda ctx: results.append(self.service.request_interrupt(THREAD))
        self.wake_self_turn()
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.assertEqual(results, [True])
        self.assertTrue(self.stub.calls[0].interrupt.is_set())
        self.assertFalse(self.service.request_interrupt(THREAD))  # back to idle

    def test_idle_alive_and_status_follow_the_state(self) -> None:
        self.assertIsNone(self.service.status(THREAD))
        self.assertFalse(self.service.idle_alive(THREAD))
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.assertTrue(self.service.idle_alive(THREAD))
        self.wake_self_turn()
        self.assertFalse(self.service.idle_alive(THREAD))
        status = self.service.status(THREAD)
        assert status is not None
        self.assertTrue(status.self_turn)
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.assertTrue(self.service.idle_alive(THREAD))
        status = self.service.status(THREAD)
        assert status is not None
        self.assertFalse(status.self_turn)

    def test_claim_returns_the_record_for_the_same_key(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.assertIsNotNone(self.service.claim(THREAD, self.key()))
        self.assertTrue(self.service.holding(THREAD))
        self.assertFalse(self.service.idle_alive(THREAD))  # a user turn holds it now

    def test_claim_releases_on_a_different_key(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        other = TopicKey(launcher_name="other", launcher_command="other-cmd", cwd=self.plan.cwd,
                         mode=self.plan.mode, model=self.plan.model)
        self.assertIsNone(self.service.claim(THREAD, other))
        self.assertFalse(self.service.holding(THREAD))
        self.assertEqual(self.stop_calls(), [("stop", RELEASE_GRACE_SECONDS)])

    def test_claim_runs_a_pending_self_turn_first(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.wake_self_turn()
        self.assertIsNotNone(self.service.claim(THREAD, self.key()))
        self.assertEqual(len(self.stub.calls), 1)

    def test_claim_runs_every_buffered_self_turn_and_the_user_gate_gets_none_of_them(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self_result = TurnResult(TurnStatus.OK, answer="self", duration_ms=5, num_turns=1, subtype="success")
        first = [TextEvent(text="a"), self_result]
        second = [TextEvent(text="b"), self_result]
        # two self turns start while the runner is busy: both wait in the queue behind their turn marks
        for event in [TurnBegan(origin="task-notification"), *first, TurnBegan(origin="task-notification"), *second]:
            self.router(event)
        self.assertTrue(self.router.self_pending)
        self.assertIsNotNone(self.service.claim(THREAD, self.key()))
        self.assertEqual(len(self.stub.calls), 2, "both buffered self turns ran before the user turn")
        self.assertEqual(self.stub.events, [*first, *second])
        self.assertFalse(self.router.self_pending)
        user_seen: list[object] = []
        self.assertTrue(self.router.attach(user_seen.append))
        self.assertEqual(user_seen, [], "no event of a self turn reaches the user turn's gate")

    def test_claim_and_resume_release_a_hold_whose_plan_was_approved(self) -> None:
        with self.subTest("claim"):
            self.start(before=BG)
            self.assertTrue(self.offer())
            self.binding.plan_approved = True
            self.assertIsNone(self.service.claim(THREAD, self.key()))
            self.assertFalse(self.service.holding(THREAD))
        with self.subTest("resume, after a self turn that approved a plan"):
            self.start(before=BG)
            self.assertTrue(self.offer())
            self.assertIsNotNone(self.service.claim(THREAD, self.key()))
            self.binding.plan_approved = True
            self.assertFalse(self.service.resume(THREAD))
            self.assertFalse(self.service.holding(THREAD))

    def test_claim_whose_self_turn_ends_badly_releases_and_returns_none(self) -> None:
        self.stub.outcome = TurnOutcome.ERROR
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.wake_self_turn()
        self.assertIsNone(self.service.claim(THREAD, self.key()))
        self.assertEqual(len(self.stub.calls), 1)
        self.assertFalse(self.service.holding(THREAD))
        self.assertEqual(self.stop_calls(), [("stop", RELEASE_GRACE_SECONDS)])
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))

    def test_release_request_after_the_end_check_is_honoured_by_claim(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        real_end_reason = self.service._end_reason

        def check_then_request(held: object, key: TopicKey | None, *, dead_counts: bool) -> HoldEnd | None:
            reason = real_end_reason(held, key, dead_counts=dead_counts)  # no end yet; the request lands while IDLE
            self.assertTrue(self.service.request_release(THREAD))
            return reason

        with mock.patch.object(self.service, "_end_reason", side_effect=check_then_request):
            self.assertIsNone(self.service.claim(THREAD, self.key()))
        self.assertFalse(self.service.holding(THREAD))
        self.assertEqual(self.stop_calls(), [("stop", RELEASE_GRACE_SECONDS)])


SELF_RESULT = TurnResult(TurnStatus.OK, answer="self", duration_ms=5, num_turns=1, subtype="success")


class NaturalExitTests(LingerCase):
    """The TUI exits by itself while the session is kept. A complete background update buffered before the exit still
    runs (its reply is posted) and the hold then ends EXITED; an incomplete self turn is never waited for."""

    def exit_process(self) -> None:
        """What the PTY reader does when the TUI exits: ProcessExited reaches the router, then the process is gone."""
        self.router(ProcessExited(returncode=0))
        self.session.stop(grace=1)
        self.stub.alive = False  # the real self turn reads the session's liveness when it ends

    def test_complete_background_update_before_a_natural_exit_runs_then_ends_exited(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.router(TurnBegan(origin="task-notification"))
        self.router(TextEvent(text="update"))
        self.router(SELF_RESULT)
        self.exit_process()
        result = self.service.wait_once(THREAD)
        self.assertEqual((result.end, result.pending), (HoldEnd.EXITED, 1))
        self.assertEqual(len(self.stub.calls), 1)
        self.assertIn(SELF_RESULT, self.stub.events)  # the self turn reached its result: its reply is posted from here
        self.assertFalse(self.service.holding(THREAD))
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))

    def test_an_incomplete_self_turn_at_exit_is_not_waited_for(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.router(TurnBegan(origin="task-notification"))
        self.router(TextEvent(text="half"))
        self.exit_process()
        result = self.service.wait_once(THREAD)
        self.assertEqual((result.end, result.pending), (HoldEnd.EXITED, 1))
        self.assertEqual(self.stub.calls, [])
        self.assertFalse(self.service.holding(THREAD))

    def test_claim_with_an_incomplete_self_turn_at_exit_releases_and_returns_none(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.router(TurnBegan(origin="task-notification"))
        self.exit_process()
        self.assertIsNone(self.service.claim(THREAD, self.key()))
        self.assertEqual(self.stub.calls, [])
        self.assertFalse(self.service.holding(THREAD))

    def test_a_draining_process_is_not_released_before_its_process_exited(self) -> None:
        """The reader reaps the process before it drains the last lines and sends ProcessExited, so a dead process that
        has not sent it yet is still draining: the tick waits, and the hold ends once the event arrives."""
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.session.stop(grace=1)  # reaped; the drain has not reached the router yet
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.assertTrue(self.service.holding(THREAD))
        self.router(ProcessExited(returncode=0))
        self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.EXITED)
        self.assertFalse(self.service.holding(THREAD))

    def test_a_self_turn_buffered_while_the_process_is_gone_runs_before_the_exit_is_read(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.router(TurnBegan(origin="task-notification"))
        self.router(TextEvent(text="half"))
        self.session.stop(grace=1)  # reaped while the drain is still writing: no ProcessExited yet
        self.stub.alive = False
        self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.EXITED)
        self.assertEqual(len(self.stub.calls), 1, "the buffered self turn ran; it was not released before the drain")

    def test_two_complete_updates_before_a_natural_exit_both_run(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.feed_self_turns(2)
        self.exit_process()
        self.assertIsNone(self.service.wait_once(THREAD).end)  # the first update runs; the second one is still buffered
        self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.EXITED)
        self.assertEqual(len(self.stub.calls), 2)
        self.assertEqual(self.stub.events.count(SELF_RESULT), 2)
        self.assertFalse(self.service.holding(THREAD))


class RealSelfTurnExitTests(LingerCase):
    """A complete update buffered before a natural exit, run by the real SelfTurns: its 🔔 reply is posted, then the
    hold ends EXITED."""

    def real_service(self) -> LingerService:
        handoff = HandoffService(self.h.kv, self.h.registry_for(self.backend), self.h.logger)
        self_turns = SelfTurns(
            config=self.h.config, clock=self.h.clock, logger=self.h.logger, feeds=self.h.feeds, views=None,
            permissions=self.h.permissions, handoff=handoff, waits=self.h.waits, live=self.h.live, turns=self.h.turns,
            answers=AnswerSender(self.h.tg, FakeUploader(), self.h.messenger, self.h.logger, self.h.clock),
            messenger=self.h.messenger, poll_seconds=0.005)
        return LingerService(
            config=self.h.config, sessions=self.h.sessions, kv=self.h.kv, live=self.h.live,
            permissions=self.h.permissions, self_turns=self_turns, clock=self.h.clock, logger=self.h.logger,
            poll_seconds=0.005)

    def test_a_complete_update_before_a_natural_exit_posts_its_bell_then_ends_exited(self) -> None:
        self.start(before=BG)
        self.service = self.real_service()
        self.h.permissions.attach(self.binding, self.session)  # as the first user turn's launch does for a kept session
        self.assertTrue(self.offer())
        self.router(TurnBegan(origin="task-notification"))
        self.router(TextEvent(text="update"))
        self.router(SELF_RESULT)
        self.router(ProcessExited(returncode=0))
        self.session.stop(grace=1)
        result = self.service.wait_once(THREAD)
        self.assertEqual((result.end, result.pending), (HoldEnd.EXITED, 1))
        self.assertTrue(any(text.startswith(BACKGROUND_PREFIX) for text in self.h.tg.texts(THREAD)))
        self.assertFalse(self.service.holding(THREAD))


class EndBeforeSelfTurnTests(LingerCase):
    """A release already requested, or a passed cap, ends the hold before a pending self turn runs. Both orders of the
    request and the self turn are covered, on wait_once and on claim, and a stream of self turns cannot pass the cap."""

    def test_wait_once_release_requested_before_the_self_turn_never_runs(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.assertTrue(self.service.request_release(THREAD))  # the session is idle: the request is accepted
        self.wake_self_turn()  # a self turn starts after the answered request
        result = self.service.wait_once(THREAD)
        self.assertEqual((result.end, result.pending), (HoldEnd.STOPPED, 1))
        self.assertEqual(self.stub.calls, [])
        self.assertEqual(self.stop_calls(), [("stop", RELEASE_GRACE_SECONDS)])
        self.assertFalse(self.service.holding(THREAD))
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))

    def test_wait_once_self_turn_pending_then_release_requested_never_runs(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.wake_self_turn()  # the self turn waits behind its mark while the record is still idle
        self.assertTrue(self.service.request_release(THREAD))
        self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.STOPPED)
        self.assertEqual(self.stub.calls, [])
        self.assertFalse(self.service.holding(THREAD))

    def test_claim_release_requested_before_the_self_turn_never_runs(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.assertTrue(self.service.request_release(THREAD))
        self.wake_self_turn()
        self.assertIsNone(self.service.claim(THREAD, self.key()))
        self.assertEqual(self.stub.calls, [])
        self.assertEqual(self.stop_calls(), [("stop", RELEASE_GRACE_SECONDS)])
        self.assertFalse(self.service.holding(THREAD))

    def test_claim_self_turn_pending_then_release_requested_never_runs(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.wake_self_turn()
        self.assertTrue(self.service.request_release(THREAD))
        self.assertIsNone(self.service.claim(THREAD, self.key()))
        self.assertEqual(self.stub.calls, [])
        self.assertFalse(self.service.holding(THREAD))

    def test_wait_once_cap_passed_before_a_pending_self_turn_ends_the_hold(self) -> None:
        self.reconfigure(linger_seconds=60)
        self.start(before=BG)
        self.assertTrue(self.offer())  # deadline t0 + 60
        self.wake_self_turn()
        self.h.clock.advance(60)
        result = self.service.wait_once(THREAD)
        self.assertEqual((result.end, result.pending), (HoldEnd.CAP, 1))
        self.assertEqual(self.stub.calls, [])

    def test_wait_once_a_stream_of_self_turns_cannot_pass_the_cap(self) -> None:
        self.reconfigure(linger_seconds=60)
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.feed_self_turns(2)
        self.stub.on_run = lambda ctx: self.h.clock.advance(60)  # the first self turn takes the whole cap
        self.assertIsNone(self.service.wait_once(THREAD).end)  # the tick's top check ran before the cap passed
        self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.CAP)
        self.assertEqual(len(self.stub.calls), 1, "the second buffered self turn never runs")

    def test_claim_a_stream_of_self_turns_cannot_pass_the_cap(self) -> None:
        self.reconfigure(linger_seconds=60)
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.feed_self_turns(2)
        self.stub.on_run = lambda ctx: self.h.clock.advance(60)
        self.assertIsNone(self.service.claim(THREAD, self.key()))
        self.assertEqual(len(self.stub.calls), 1, "the second buffered self turn never runs")
        self.assertEqual(self.stop_calls(), [("stop", RELEASE_GRACE_SECONDS)])
        self.assertFalse(self.service.holding(THREAD))


class HoldIdentityTests(LingerCase):
    """hold_id numbers the accepted offers of the service; self_seq counts the self turns of the current hold."""

    def status(self) -> LingerStatus:
        status = self.service.status(THREAD)
        assert status is not None
        return status

    def test_every_accepted_offer_gets_a_new_hold_id(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.assertEqual((self.status().hold_id, self.status().self_seq), (1, 0))
        self.assertTrue(self.offer())  # a later user turn on the kept session is an accepted offer too
        self.assertEqual((self.status().hold_id, self.status().self_seq), (2, 0))

    def test_refused_offers_take_no_id(self) -> None:
        self.start()  # no background work: refused
        self.assertRefused()
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.assertEqual(self.status().hold_id, 1)

    def test_a_hold_after_a_release_gets_the_next_id(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.assertTrue(self.service.request_release(THREAD))
        self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.STOPPED)
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.assertEqual(self.status().hold_id, 2)

    def test_a_new_service_counts_its_own_holds(self) -> None:
        other = self.make_service()
        self.start(before=BG)
        self.assertTrue(other.offer(self.offer_args()))
        status = other.status(THREAD)
        assert status is not None
        self.assertEqual(status.hold_id, 1)

    def test_self_seq_counts_the_self_turns_of_the_hold(self) -> None:
        seen: list[LingerStatus | None] = []
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.stub.on_run = lambda ctx: seen.append(self.service.status(THREAD))  # read while the self turn runs
        self.feed_self_turns(2)  # two self turns wait in the router: one runs per tick
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.assertEqual((self.status().self_seq, self.status().self_turn), (1, True))  # the second one is still pending
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.assertEqual((self.status().hold_id, self.status().self_seq), (1, 2))
        running = [status for status in seen if status is not None]
        self.assertEqual([(s.self_seq, s.self_turn) for s in running], [(1, True), (2, True)])

    def test_a_new_offer_restarts_self_seq_under_its_hold_id(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.wake_self_turn()
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.assertEqual(self.status().self_seq, 1)
        self.assertTrue(self.offer())
        self.assertEqual((self.status().hold_id, self.status().self_seq), (2, 0))

    def test_the_moved_names_are_still_importable_from_linger(self) -> None:
        for name in ("HoldEnd", "HoldResult", "HoldState", "LingerStatus", "LingerOffer", "HeldSession",
                     "LingerControl", "LINGER_CAP_TEXT", "LINGER_EXIT_TEXT", "KEEP_OUTCOMES", "POLL_SECONDS"):
            self.assertIs(getattr(linger_module, name), getattr(linger_types, name), name)


class InSelfTurnTests(LingerCase):
    """in_self_turn is True only while the hold's self turn with that number runs (K9: a stop choice acts on that turn
    only). Every other moment of the kept session is a miss: after the self turn, after a newer one started, during a
    user turn and after the hold ended."""

    def test_the_running_self_turn_matches_its_hold_and_number(self) -> None:
        seen: list[tuple[bool, bool, bool]] = []
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.stub.on_run = lambda ctx: seen.append((self.service.in_self_turn(THREAD, 1, 1),
                                                    self.service.in_self_turn(THREAD, 1, 2),
                                                    self.service.in_self_turn(THREAD, 2, 1)))
        self.wake_self_turn()
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.assertEqual(seen, [(True, False, False)])

    def test_after_the_self_turn_ended_it_no_longer_matches(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.wake_self_turn()
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.assertFalse(self.service.in_self_turn(THREAD, 1, 1))

    def test_a_new_self_turn_does_not_match_the_old_number(self) -> None:
        seen: list[tuple[bool, bool]] = []
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.stub.on_run = lambda ctx: seen.append((self.service.in_self_turn(THREAD, 1, 1),
                                                    self.service.in_self_turn(THREAD, 1, 2)))
        self.feed_self_turns(2)
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.assertEqual(seen, [(True, False), (False, True)])

    def test_a_user_turn_on_the_kept_session_matches_nothing(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.assertIsNotNone(self.service.claim(THREAD, self.key()))  # the user turn now runs on the kept session
        self.assertFalse(self.service.in_self_turn(THREAD, 1, 0))
        self.assertFalse(self.service.in_self_turn(THREAD, 1, 1))

    def test_a_released_hold_matches_nothing(self) -> None:
        self.start(before=BG)
        self.assertTrue(self.offer())
        self.wake_self_turn()
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.assertTrue(self.service.request_release(THREAD))
        self.assertEqual(self.service.wait_once(THREAD).end, HoldEnd.STOPPED)
        self.assertFalse(self.service.in_self_turn(THREAD, 1, 1))
