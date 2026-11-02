"""TurnRunner with a LingerService (linger): a user turn offers its session at the end, the next message reuses the
kept process, and every other case stops the session as before. FakeBackend (with linger capability), FakeClock, the
real LingerService and SelfTurns, and the real SessionRouter.

The tests check that a turn with background work keeps its process (no stop, no graceful close, the kv marker), that a
turn without it stops as before, the config and capability refusals, the turns that end badly, the session cap, a
reused process for the next message, a changed key that opens a fresh process, and the quiet window that closes a
reused session once its background work is over.
"""

import dataclasses
import unittest
from unittest import mock

from tests.fakes.backend import FAKE_CAPABILITIES, FakeBackend, WaitForInterrupt, fake_spec
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, TurnHarness, join_turn, run_async, wait_until
from tg_bridge.agents.events import BackgroundWork, TextEvent, TurnBegan, TurnResult, TurnStatus
from tg_bridge.core.linger import LINGER_QUIET_SECONDS, HoldEnd, linger_key
from tg_bridge.core.session_router import Attach, SessionRouter
from tg_bridge.core.turn_kit import GRACEFUL_SECONDS
from tg_bridge.core import self_turn, turn_runner
from tg_bridge.core.turn_runner import BUSY_TEXT, SELF_FIRST_RETRIES, TurnRunner, active_key
from tg_bridge.core.turn_listeners import EndGate
from tg_bridge.core.turn_state import TurnOutcome, TurnPhase, TurnState
from tg_bridge.core.turn_text import BACKGROUND_PREFIX

OK = TurnResult(TurnStatus.OK, answer="first", duration_ms=5, num_turns=1, subtype="success")
BG = BackgroundWork(pending=1)  # the CLI reports one background task that is still running
ENDED = BackgroundWork(pending=0)  # the CLI reports that the background tasks are done
OK_NEXT = TurnResult(TurnStatus.OK, answer="second", duration_ms=5, num_turns=1, subtype="success")
OK_SELF = TurnResult(TurnStatus.OK, answer="background", duration_ms=5, num_turns=1, subtype="success")


class LingerRunnerCase(unittest.TestCase):
    """A harness in a TempHome; start() builds the fake backend, the linger service and the runner for the scripts."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self._stop_live_turns)  # runs before the TempHome is removed (cleanups run last-in first-out)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")

    def _stop_live_turns(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)

    def start(self, scripts=(), *, linger: bool = True, **config_overrides) -> None:
        self.h.set_config(**{"linger_seconds": 60, "linger_max_sessions": 3, **config_overrides})
        capabilities = dataclasses.replace(FAKE_CAPABILITIES, linger=linger)
        self.backend = FakeBackend(scripts, capabilities=capabilities)
        self.registry = self.h.registry_for(self.backend)
        self.service = self.h.linger(self.registry)
        self.runner = self.h.runner(self.registry, linger=self.service)

    def turn(self, text: str = "hello"):
        for session in self.backend.sessions:  # the fake frees its busy flag a moment after its result: wait for it
            self.assertTrue(session.wait_idle())
        return self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text=text, sender="tester")

    def calls(self, index: int = 0) -> list[str]:
        """The calls the session of this index has received so far (none before the session is opened)."""
        if len(self.backend.sessions) <= index:
            return []
        return [call[0] for call in self.backend.sessions[index].calls]

    def assertNotKept(self) -> None:
        self.assertIsNone(self.h.live.lingering(THREAD))
        self.assertIsNone(self.h.kv.get(linger_key(THREAD)))
        self.assertFalse(self.service.holding(THREAD))


class KeepTests(LingerRunnerCase):
    def test_pending_background_keeps_the_session(self) -> None:
        self.start([[TextEvent(text="a"), BG, OK]])
        with mock.patch.object(self.h.permissions, "void_all") as voided:
            reply = self.turn()
        self.assertEqual((reply.answer, reply.err), ("first", None))
        session = self.backend.sessions[0]
        self.assertNotIn("stop", self.calls())
        self.assertNotIn("end_gracefully", self.calls())
        self.assertIs(self.h.live.lingering(THREAD), session)
        self.assertIsNotNone(self.h.kv.get(linger_key(THREAD)))
        self.assertTrue(self.service.holding(THREAD))
        voided.assert_not_called()  # the permission binding lives on with the session

    def test_no_background_stops_as_before(self) -> None:
        self.start([[TextEvent(text="a"), OK]])
        reply = self.turn()
        self.assertEqual((reply.answer, reply.err), ("first", None))
        self.assertEqual(self.calls(), ["start", "send_user", "end_gracefully", "stop"])
        self.assertNotKept()

    def test_pending_work_that_is_not_kept_gets_the_graceful_end_as_before(self) -> None:
        # linger off with background work pending: the turn ends exactly as it did before linger, with the graceful
        # end (input closed, GRACEFUL_SECONDS to exit by itself) and then the stop
        self.start([[TextEvent(text="a"), BG, OK]], linger_seconds=0)
        self.turn()
        self.assertIn(("end_gracefully", GRACEFUL_SECONDS), self.backend.sessions[0].calls)
        self.assertEqual(self.calls(), ["start", "send_user", "end_gracefully", "stop"])
        self.assertNotKept()

    def test_linger_off_by_config(self) -> None:
        for overrides in ({"linger_seconds": 0}, {"linger_max_sessions": 0}):
            with self.subTest(overrides=overrides):
                self.start([[TextEvent(text="a"), BG, OK]], **overrides)
                self.turn()
                self.assertEqual(self.calls(), ["start", "send_user", "end_gracefully", "stop"])
                self.assertNotKept()

    def test_router_is_armed_when_linger_is_on(self) -> None:
        # the control for the test below: with linger on, the turn's router is armed before its gate attaches
        real_arm = SessionRouter.arm
        self.start([[TextEvent(text="a"), BG, OK]])
        with mock.patch.object(SessionRouter, "arm", autospec=True, side_effect=real_arm) as armed:
            self.turn()
        armed.assert_called_once()

    def test_router_is_not_armed_when_linger_is_off_by_config(self) -> None:
        # arm() makes a self turn that starts after the result wait for the offer: with linger off by config it must not
        # be armed at all, so the router keeps its events flowing to the gate as before
        real_arm = SessionRouter.arm
        for overrides in ({"linger_seconds": 0}, {"linger_max_sessions": 0}):
            with self.subTest(overrides=overrides):
                self.start([[TextEvent(text="a"), BG, OK]], **overrides)
                with mock.patch.object(SessionRouter, "arm", autospec=True, side_effect=real_arm) as armed:
                    self.turn()
                armed.assert_not_called()
                self.assertNotKept()

    def test_codex_like_backend_never_lingers(self) -> None:
        self.start([[TextEvent(text="a"), BG, OK]], linger=False)
        self.turn()
        self.assertEqual(self.calls(), ["start", "send_user", "end_gracefully", "stop"])
        self.assertNotKept()

    def test_failed_or_stopped_turn_is_not_kept(self) -> None:
        with self.subTest("error"), mock.patch.object(self.h.permissions, "void_all") as voided:
            self.start([[TextEvent(text="a"), BG, TurnResult(TurnStatus.ERROR, answer=None, error="boom")]])
            self.turn()
            self.assertIn("stop", self.calls())
            self.assertNotKept()
            voided.assert_called_once()
        with self.subTest("stop tap"), mock.patch.object(self.h.permissions, "void_all") as voided:
            self.start([[BG, WaitForInterrupt()]])
            thread, box = run_async(self.runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="long",
                                    sender="tester")
            self.assertTrue(wait_until(lambda: "send_user" in self.calls()))
            running = self.h.live.request_abort(THREAD, stop_tap=True)
            self.assertIsNotNone(running)
            running.session.stop()
            join_turn(self.h.clock, thread)
            self.assertEqual(box[0].outcome, TurnOutcome.STOPPED)
            self.assertNotKept()
            voided.assert_called_once()
        with self.subTest("cap"), mock.patch.object(self.h.permissions, "void_all") as voided:
            self.start([[BG, WaitForInterrupt()]], max_turn_seconds=3)
            thread, box = run_async(self.runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="long",
                                    sender="tester")
            join_turn(self.h.clock, thread)
            self.assertEqual(box[0].outcome, TurnOutcome.CAPPED)
            self.assertIn("stop", self.calls())
            self.assertNotKept()
            voided.assert_called_once()

    def test_max_sessions_reached_gets_the_graceful_end_as_before(self) -> None:
        self.start([[TextEvent(text="a"), BG, OK]], linger_max_sessions=1)
        other = FakeBackend().open_session(fake_spec(thread_id=THREAD + 1), lambda event: None)
        self.assertTrue(self.h.live.begin_linger(THREAD + 1, other, 1))  # another topic keeps the only slot
        self.turn()
        self.assertEqual(self.calls(), ["start", "send_user", "end_gracefully", "stop"])
        self.assertNotKept()
        self.assertIs(self.h.live.lingering(THREAD + 1), other)


class ReuseTests(LingerRunnerCase):
    def test_next_message_reuses_the_process(self) -> None:
        self.start([[TextEvent(text="a"), BG, OK],
                    [TextEvent(text="b"), TurnResult(TurnStatus.OK, answer="second")]])
        self.turn("first message")
        reply = self.turn("second message")
        self.assertEqual((reply.answer, reply.err), ("second", None))
        self.assertEqual(len(self.backend.specs), 1)  # one spawn for both turns
        sends = [call[1] for call in self.backend.sessions[0].calls if call[0] == "send_user"]
        self.assertEqual(sends, ["first message", "second message"])
        self.assertIs(self.h.live.lingering(THREAD), self.backend.sessions[0])

    def test_changed_mode_opens_a_fresh_process(self) -> None:
        self.start([[TextEvent(text="a"), BG, OK], [TextEvent(text="b"), TurnResult(TurnStatus.OK, answer="second")]])
        self.turn()
        self.h.sessions.set_mode(THREAD, "plan")  # the key changes: the kept process is released first
        reply = self.turn("after the change")
        self.assertEqual(reply.answer, "second")
        self.assertIn("stop", self.calls(0))
        self.assertEqual(len(self.backend.specs), 2)
        self.assertEqual(self.backend.specs[1].mode, "plan")
        self.assertEqual(self.calls(1)[:2], ["start", "send_user"])

    def test_reused_turn_that_fails_releases_the_record(self) -> None:
        self.start([[TextEvent(text="a"), BG, OK],
                    [TextEvent(text="b"), TurnResult(TurnStatus.ERROR, answer=None, error="boom")]])
        self.turn()
        reply = self.turn("second message")
        self.assertIsNotNone(reply.err)
        self.assertEqual(len(self.backend.specs), 1)
        self.assertIn("stop", self.calls(0))
        self.assertNotKept()

    def test_reused_turn_whose_background_work_ended_closes_after_the_quiet_window(self) -> None:
        self.start([[TextEvent(text="a"), BG, OK],
                    [TextEvent(text="b"), ENDED, TurnResult(TurnStatus.OK, answer="second")]])
        self.turn()
        self.turn("second message")
        # the router keeps the last count the CLI reported; the work is over, so the hold waits for the quiet window
        self.assertIs(self.h.live.lingering(THREAD), self.backend.sessions[0])
        with mock.patch.object(self.h.permissions, "void_all") as voided:
            self.assertIsNone(self.service.wait_once(THREAD).end)
            self.h.clock.advance(LINGER_QUIET_SECONDS)
            end = self.service.wait_once(THREAD).end
        self.assertEqual(end, HoldEnd.DONE)
        self.assertIn("stop", self.calls(0))
        self.assertNotKept()
        voided.assert_called_once()

    def test_shutdown_that_begins_during_a_held_turn_refuses_it(self) -> None:
        self.start([[TextEvent(text="a"), BG, OK], [TextEvent(text="b"), TurnResult(TurnStatus.OK, answer="second")]])
        self.turn()
        real_start = TurnState.start

        def start_then_shut(turn: TurnState) -> None:  # the shutdown begins after the claim, before the text is sent
            real_start(turn)
            self.h.live.begin_shutdown()

        with mock.patch.object(TurnState, "start", start_then_shut):
            reply = self.turn("second message")
        self.assertEqual((reply.answer, reply.err), (None, None))  # refused: nothing ran
        self.assertEqual(self.calls(0).count("send_user"), 1)  # the second text never reached the process
        self.assertIn("stop", self.calls(0))  # the held process is stopped by the refused turn
        self.assertFalse(self.service.holding(THREAD))
        self.assertIsNone(self.h.live.lingering(THREAD))
        self.assertIsNone(self.h.kv.get(active_key(THREAD)))
        self.assertIsNotNone(self.h.kv.get(linger_key(THREAD)))  # a shutdown keeps the linger marker (spec 1.4)

    def test_closed_router_refuses_the_held_turn(self) -> None:
        self.start([[TextEvent(text="a"), BG, OK], [TextEvent(text="b"), TurnResult(TurnStatus.OK, answer="second")]],
                   max_turn_seconds=3)  # a cap ends the turn if the refused text is never answered (no hang)
        self.turn()
        self.service._get(THREAD).router.close()  # the record is still kept, but its router no longer takes a gate
        thread, box = run_async(self.runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="second message",
                                sender="tester")
        join_turn(self.h.clock, thread)
        reply = box[0]
        self.assertEqual((reply.answer, reply.err), (None, "session closed"))
        self.assertEqual(self.calls(0).count("send_user"), 1)  # the second text never reached the process
        self.assertIn("stop", self.calls(0))
        self.assertNotKept()

    def test_failure_before_the_launch_releases_the_claimed_record(self) -> None:
        self.start([[TextEvent(text="a"), BG, OK]])
        self.turn()
        with mock.patch("tg_bridge.core.turn_runner.build_kit", side_effect=RuntimeError("kit failed")):
            with self.assertRaises(RuntimeError):
                self.turn("second message")
        self.assertIn("stop", self.calls(0))
        self.assertFalse(self.service.holding(THREAD))  # the claimed record is not left in the TURN state
        self.assertNotKept()


class SelfTurnBeforeAttachTests(LingerRunnerCase):
    """A self turn that the CLI starts between the claim of a kept session and the attach of the user's gate (the two
    steps of _drive_kept are not one). The self turn runs first, through the linger service, and the user's message
    then runs alone: its answer is its own, and the self turn's answer is posted with the background prefix."""

    def keep_first_turn(self) -> None:
        self.start([[TextEvent(text="a"), BG, OK], [TextEvent(text="second"), BG, OK_NEXT]])
        self.turn()
        self.assertTrue(self.service.holding(THREAD))

    def test_self_turn_that_begins_before_the_attach_runs_first(self) -> None:
        self.keep_first_turn()
        original = TurnRunner._drive_kept

        def begins_before_attach(runner, plan, kit, router, session, gate, after):
            session._sink(TurnBegan(origin="task-notification"))  # the sink is the session's router
            session._sink(TextEvent(text="background"))
            session._sink(OK_SELF)
            return original(runner, plan, kit, router, session, gate, after)

        live_while_waiting: list[bool] = []
        phases: list[TurnPhase] = []
        real_wait = turn_runner.wait_turn

        def watch_wait(*args, **kwargs):  # the user turn is registered as the running turn while it waits
            live_while_waiting.append(self.h.live.running(THREAD) is not None)
            # the user turn must still be the thread's wait observer after the self turn: a waiting change reaches it
            self.h.waits.waiting_changed(THREAD, True)
            phases.append(args[0].phase)
            self.h.waits.waiting_changed(THREAD, False)
            return real_wait(*args, **kwargs)

        with mock.patch.object(TurnRunner, "_drive_kept", begins_before_attach), \
                mock.patch.object(turn_runner, "wait_turn", side_effect=watch_wait):
            reply = self.turn("next")
        self.assertEqual(phases, [TurnPhase.WAITING_APPROVAL], "the self turn took the user turn's wait observer")
        self.assertEqual(live_while_waiting[-1:], [True], "the self turn's end did not remove the user turn's entry")
        self.assertEqual((reply.answer, reply.err), ("second", None), "the user's gate got only its own end")
        self.assertEqual(self.calls(0), ["start", "send_user", "send_user"], "the user's text is sent once, at the end")
        self.assertNotIn("stop", self.calls(0))
        self.assertTrue(any(text.startswith(BACKGROUND_PREFIX) for text in self.h.tg.texts(THREAD)),
                        "the self turn's answer is posted with the background prefix")
        self.assertTrue(self.service.holding(THREAD), "the session is kept after the user's turn again")

    def test_self_turn_that_runs_first_does_not_count_against_the_user_turn_caps(self) -> None:
        # a 31 s self turn runs in front of a user turn capped at 30 s: the user turn starts after that self turn, so
        # it begins its wait with no active time and keeps its full cap
        self.start([[TextEvent(text="a"), BG, OK], [TextEvent(text="second"), BG, OK_NEXT]], max_turn_seconds=30)
        self.turn()
        self.assertTrue(self.service.holding(THREAD))
        original = TurnRunner._drive_kept

        def self_turn_first(runner, plan, kit, router, session, gate, after):
            session._sink(TurnBegan(origin="task-notification"))  # the sink is the session's router
            session._sink(TextEvent(text="background"))
            session._sink(OK_SELF)
            return original(runner, plan, kit, router, session, gate, after)

        real_self_wait = self_turn.wait_turn

        def long_self_wait(*args, **kwargs):  # the self turn takes 31 s of the clock
            self.h.clock.advance(31)
            return real_self_wait(*args, **kwargs)

        active_at_wait: list[float] = []
        real_wait = turn_runner.wait_turn

        def watch_wait(*args, **kwargs):  # the user turn's active time when its wait begins
            active_at_wait.append(args[0].active_seconds())
            return real_wait(*args, **kwargs)

        with mock.patch.object(TurnRunner, "_drive_kept", self_turn_first), \
                mock.patch.object(self_turn, "wait_turn", side_effect=long_self_wait), \
                mock.patch.object(turn_runner, "wait_turn", side_effect=watch_wait):
            reply = self.turn("next")
        self.assertEqual(active_at_wait, [0.0], "the self turn's 31 s were counted against the user turn")
        self.assertEqual((reply.answer, reply.err), ("second", None), "the user turn was stopped by its cap")
        self.assertTrue(any(text.startswith(BACKGROUND_PREFIX) for text in self.h.tg.texts(THREAD)),
                        "the self turn ran and posted its answer")
        self.assertEqual(self.calls(0), ["start", "send_user", "send_user"], "the user's text is sent once, at the end")

    def test_self_turn_that_begins_while_the_user_end_is_queued_is_kept_for_the_offer(self) -> None:
        # the user's text is being delivered when its end is queued and the CLI starts a self turn: the self turn waits
        # behind the end (it is not ignored), the offer keeps the session for it, and the next wait runs it
        self.start([[TextEvent(text="a"), BG, OK], [TextEvent(text="second")]])
        self.turn()
        self.assertTrue(self.service.holding(THREAD))
        session = self.backend.sessions[0]
        real_call = EndGate.__call__
        injected: list[bool] = []

        def ends_then_self_turn(gate, event):
            if event == TextEvent(text="second") and not injected:
                injected.append(True)
                session._sink(OK_NEXT)  # the user's end, queued while the user's gate is still delivering this event
                session._sink(TurnBegan(origin="task-notification"))
                session._sink(TextEvent(text="background"))
                session._sink(OK_SELF)
            return real_call(gate, event)

        with mock.patch.object(EndGate, "__call__", ends_then_self_turn):
            reply = self.turn("next")
        self.assertEqual((reply.answer, reply.err), ("second", None))
        self.assertTrue(self.service.holding(THREAD), "the self turn that began meanwhile keeps the session")
        self.assertIsNone(self.service.wait_once(THREAD).end)
        self.assertNotIn("stop", self.calls(0))
        self.assertTrue(any(text.startswith(BACKGROUND_PREFIX) for text in self.h.tg.texts(THREAD)),
                        "the self turn's answer is posted with the background prefix")
        self.assertTrue(self.service.holding(THREAD))

    def test_attach_that_keeps_finding_a_self_turn_is_bounded(self) -> None:
        self.keep_first_turn()
        def self_first(router, gate, user=False):
            return Attach.SELF_FIRST if user else Attach.REFUSED

        with mock.patch.object(SessionRouter, "attach", autospec=True, side_effect=self_first) as attach:
            reply = self.turn("next")
        self.assertEqual(attach.call_count, SELF_FIRST_RETRIES + 1, "one attach per try, then no more")
        self.assertIsNone(reply.answer)
        self.assertIn(BUSY_TEXT, reply.err)
        self.assertEqual(self.calls(0), ["start", "send_user", "stop"], "the user's text is never sent")
        self.assertNotKept()
