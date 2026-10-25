"""core/turn_state.py (STEP 36): caps, waiting excluded from active time, illegal transitions, slot give_back/retake,
the classification table and the dead-process rule.

Port of the turn-state scenarios of fakecli/test_run_turn_v2.py (cap notice, bypass and stop taps), test_slot_release.py
(a parked turn gives its permit back and takes it again) and test_review_fixes.py (a process that exits without a result
while a grandchild holds stdout open is reported after seconds, not at the cap). Everything runs on FakeClock: no sleeps,
no subprocess, no Telegram.
"""

import threading
import unittest

from tests.fakes.clock import FakeClock
from tg_bridge.agents.events import TurnResult, TurnStatus
from tg_bridge.core.turn_state import (
    DEAD_PROCESS_SECONDS,
    TRANSITIONS,
    ActiveTimer,
    IllegalTransition,
    TurnFacts,
    TurnOutcome,
    TurnPhase,
    TurnState,
    classify,
)

THREAD = 4411
OTHER_THREAD = 4412


class SlotSpy:
    """Records give_back/retake calls. retake_result decides what retake() answers (False = every permit busy)."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.retake_result = True

    def give_back(self) -> None:
        self.calls.append("give_back")

    def retake(self) -> bool:
        self.calls.append("retake")
        return self.retake_result

    def count(self, name: str) -> int:
        return self.calls.count(name)


def result(status: TurnStatus) -> TurnResult:
    return TurnResult(status=status, answer="done" if status is TurnStatus.OK else None)


def make_state(clock: FakeClock, slot: SlotSpy | None = None, **caps) -> TurnState:
    return TurnState(THREAD, clock, slot=slot, **caps)


def drive_to(state: TurnState, phase: TurnPhase) -> None:
    """Walk a fresh TurnState to phase using only public methods."""
    if phase is TurnPhase.STARTING:
        return
    state.start()
    if phase is TurnPhase.RUNNING:
        state.mark_running()
    elif phase is TurnPhase.WAITING_APPROVAL:
        state.waiting_changed(THREAD, True)
    elif phase is TurnPhase.STOPPING:
        state.begin_stopping()
    elif phase is TurnPhase.FINISHED:
        state.finish(None)


class ActiveTimerTests(unittest.TestCase):
    def test_paused_stretches_are_not_counted(self) -> None:
        """A: running 10 s, paused 500 s, running 5 s -> 15 s of work."""
        clock = FakeClock()
        timer = ActiveTimer(clock)
        timer.start()
        clock.advance(10)
        timer.pause()
        clock.advance(500)
        timer.resume()
        clock.advance(5)
        self.assertAlmostEqual(timer.seconds(), 15.0)
        self.assertTrue(timer.running)

    def test_pause_and_resume_are_idempotent_and_safe_before_start(self) -> None:
        """B: pause() before start() and a second pause()/resume() change nothing."""
        clock = FakeClock()
        timer = ActiveTimer(clock)
        timer.pause()
        timer.resume()
        self.assertEqual(timer.seconds(), 0.0)
        timer.start()
        clock.advance(3)
        timer.resume()  # already running: no double count
        timer.pause()
        timer.pause()
        clock.advance(100)
        self.assertAlmostEqual(timer.seconds(), 3.0)

    def test_second_start_raises(self) -> None:
        timer = ActiveTimer(FakeClock())
        timer.start()
        with self.assertRaises(RuntimeError):
            timer.start()


class CapTests(unittest.TestCase):
    def test_active_cap_fires_at_max_active_seconds(self) -> None:
        """C: max_turn_seconds 30 -> not hit at 29 s of work, hit at 30 s, and the hit is sticky."""
        clock = FakeClock()
        state = make_state(clock, max_active_seconds=30, hard_cap_seconds=10_000)
        state.start()
        clock.advance(29)
        self.assertIsNone(state.check_caps())
        clock.advance(1)
        self.assertEqual(state.check_caps(), 30.0)
        clock.advance(500)
        self.assertEqual(state.check_caps(), 30.0)  # sticky
        self.assertEqual(state.limit_hit, 30.0)

    def test_waiting_time_does_not_count_towards_the_active_cap(self) -> None:
        """D: 20 s of work, 500 s parked on a card, 9 s more of work -> still under a 30 s active cap."""
        clock = FakeClock()
        state = make_state(clock, max_active_seconds=30, hard_cap_seconds=10_000)
        state.start()
        clock.advance(20)
        state.waiting_changed(THREAD, True)
        clock.advance(500)
        self.assertIsNone(state.check_caps())
        state.waiting_changed(THREAD, False)
        clock.advance(9)
        self.assertIsNone(state.check_caps())
        clock.advance(1)
        self.assertEqual(state.check_caps(), 30.0)

    def test_hard_cap_counts_wall_time_including_waiting(self) -> None:
        """E: hard_cap_seconds 120 -> a turn parked for 120 s (with 10 s of work) hits the hard cap, not the active cap."""
        clock = FakeClock()
        state = make_state(clock, max_active_seconds=1000, hard_cap_seconds=120)
        state.start()
        clock.advance(10)
        state.waiting_changed(THREAD, True)
        clock.advance(110)
        self.assertEqual(state.check_caps(), 120.0)

    def test_active_cap_is_reported_before_hard_cap_when_both_are_reached(self) -> None:
        """F: the live file checks the active cap first, so a turn past both caps reports the active one."""
        clock = FakeClock()
        state = make_state(clock, max_active_seconds=30, hard_cap_seconds=40)
        state.start()
        clock.advance(40)
        self.assertEqual(state.check_caps(), 30.0)

    def test_caps_are_not_checked_before_start(self) -> None:
        clock = FakeClock()
        state = make_state(clock, max_active_seconds=1, hard_cap_seconds=1)
        clock.advance(50)
        self.assertIsNone(state.check_caps())

    def test_non_positive_caps_are_refused(self) -> None:
        clock = FakeClock()
        with self.assertRaises(ValueError):
            make_state(clock, max_active_seconds=0)
        with self.assertRaises(ValueError):
            make_state(clock, hard_cap_seconds=-1)
        with self.assertRaises(ValueError):
            make_state(clock, dead_after_seconds=-0.5)


class TransitionTests(unittest.TestCase):
    def test_transition_table_matches_the_design(self) -> None:
        """G: the table has the five phases; FINISHED is terminal; every other phase may stop."""
        self.assertEqual(set(TRANSITIONS), set(TurnPhase))
        self.assertEqual(TRANSITIONS[TurnPhase.FINISHED], frozenset())
        for phase in (TurnPhase.STARTING, TurnPhase.RUNNING, TurnPhase.WAITING_APPROVAL):
            self.assertIn(TurnPhase.STOPPING, TRANSITIONS[phase])
            self.assertIn(TurnPhase.FINISHED, TRANSITIONS[phase])
        self.assertEqual(TRANSITIONS[TurnPhase.STOPPING], frozenset({TurnPhase.FINISHED}))

    def test_every_pair_of_phases_is_allowed_only_when_the_table_says_so(self) -> None:
        """H: each (from, to) pair either moves or raises IllegalTransition, exactly as TRANSITIONS lists it."""
        for source in TurnPhase:
            for target in TurnPhase:
                with self.subTest(source=source.value, target=target.value):
                    state = make_state(FakeClock())
                    drive_to(state, source)
                    self.assertEqual(state.phase, source)
                    if target in TRANSITIONS[source]:
                        state._move(target)
                        self.assertEqual(state.phase, target)
                    else:
                        with self.assertRaises(IllegalTransition):
                            state._move(target)
                        self.assertEqual(state.phase, source)

    def test_finishing_twice_raises(self) -> None:
        """I: a turn has one outcome; a second finish() is an illegal transition."""
        state = make_state(FakeClock())
        state.start()
        state.finish(result(TurnStatus.OK))
        with self.assertRaises(IllegalTransition):
            state.finish(result(TurnStatus.OK))

    def test_starting_twice_raises(self) -> None:
        state = make_state(FakeClock())
        state.start()
        with self.assertRaises(RuntimeError):
            state.start()

    def test_mark_running_does_not_undo_a_card_that_arrived_first(self) -> None:
        """J: a card can arrive before send_user() returns; mark_running() then leaves WAITING_APPROVAL alone."""
        state = make_state(FakeClock())
        state.start()
        state.waiting_changed(THREAD, True)
        state.mark_running()
        self.assertEqual(state.phase, TurnPhase.WAITING_APPROVAL)

    def test_waiting_changes_after_stopping_are_ignored(self) -> None:
        """K: once the turn is stopping, a card that shows up late does not move the phase."""
        state = make_state(FakeClock())
        drive_to(state, TurnPhase.RUNNING)
        state.begin_stopping()
        self.assertFalse(state.waiting_changed(THREAD, True))
        self.assertEqual(state.phase, TurnPhase.STOPPING)

    def test_waiting_changes_for_another_thread_are_ignored(self) -> None:
        slot = SlotSpy()
        state = make_state(FakeClock(), slot=slot)
        state.start()
        self.assertFalse(state.waiting_changed(OTHER_THREAD, True))
        self.assertEqual(state.phase, TurnPhase.STARTING)
        self.assertEqual(slot.calls, [])

    def test_repeated_waiting_true_does_not_raise(self) -> None:
        state = make_state(FakeClock())
        state.start()
        self.assertTrue(state.waiting_changed(THREAD, True))
        self.assertFalse(state.waiting_changed(THREAD, True))
        self.assertTrue(state.waiting_changed(THREAD, False))
        self.assertFalse(state.waiting_changed(THREAD, False))


class SlotTests(unittest.TestCase):
    def test_slot_is_given_back_on_entering_waiting_and_retaken_on_leaving(self) -> None:
        """L: one give_back when the first card opens and one retake when the last one closes."""
        slot = SlotSpy()
        state = make_state(FakeClock(), slot=slot)
        state.start()
        state.mark_running()
        state.waiting_changed(THREAD, True)
        self.assertEqual(slot.calls, ["give_back"])
        state.waiting_changed(THREAD, False)
        self.assertEqual(slot.calls, ["give_back", "retake"])
        self.assertFalse(state.unslotted)

    def test_a_second_card_does_not_give_the_slot_back_twice(self) -> None:
        """M: waiting is reported once per turn (PermissionService counts the cards), so the slot moves once."""
        slot = SlotSpy()
        state = make_state(FakeClock(), slot=slot)
        state.start()
        state.waiting_changed(THREAD, True)
        state.waiting_changed(THREAD, True)
        state.waiting_changed(THREAD, False)
        self.assertEqual(slot.count("give_back"), 1)
        self.assertEqual(slot.count("retake"), 1)

    def test_a_busy_pool_leaves_the_turn_unslotted_until_a_retake_succeeds(self) -> None:
        """N: retake() False (every permit busy) -> the turn runs unslotted and says so; the next retake clears it."""
        slot = SlotSpy()
        state = make_state(FakeClock(), slot=slot)
        state.start()
        state.waiting_changed(THREAD, True)
        slot.retake_result = False
        state.waiting_changed(THREAD, False)
        self.assertTrue(state.unslotted)
        state.waiting_changed(THREAD, True)
        slot.retake_result = True
        state.waiting_changed(THREAD, False)
        self.assertFalse(state.unslotted)

    def test_no_slot_means_no_calls_and_no_error(self) -> None:
        state = make_state(FakeClock(), slot=None)
        state.start()
        self.assertTrue(state.waiting_changed(THREAD, True))
        self.assertTrue(state.waiting_changed(THREAD, False))

    def test_concurrent_waiting_changes_keep_the_slot_balanced(self) -> None:
        """O: four threads toggle the waiting flag 100 times each. The give-backs and retakes always pair up, so the
        final give-back count minus the retake count is 1 exactly when the turn ends WAITING_APPROVAL."""
        slot = SlotSpy()
        state = make_state(FakeClock(), slot=slot)
        state.start()
        state.mark_running()

        def toggle() -> None:
            for _ in range(100):
                state.waiting_changed(THREAD, True)
                state.waiting_changed(THREAD, False)

        workers = [threading.Thread(target=toggle) for _ in range(4)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        held = slot.count("give_back") - slot.count("retake")
        self.assertEqual(held, 1 if state.phase is TurnPhase.WAITING_APPROVAL else 0)
        self.assertGreater(slot.count("give_back"), 0)


class StopTests(unittest.TestCase):
    def test_stop_tap_during_a_run_classifies_as_stopped(self) -> None:
        """P: ⏹ tap -> STOPPING, then the process ends without an answer -> STOPPED (not ERROR)."""
        state = make_state(FakeClock())
        state.start()
        state.mark_running()
        self.assertTrue(state.request_stop())
        self.assertEqual(state.phase, TurnPhase.STOPPING)
        self.assertEqual(state.finish(None), TurnOutcome.STOPPED)

    def test_stop_after_a_cap_still_counts_as_a_stop(self) -> None:
        """Q: the live file checks the stop tap before the cap, so a tap that lands while a cap stop runs wins."""
        clock = FakeClock()
        state = make_state(clock, max_active_seconds=5)
        state.start()
        clock.advance(5)
        state.check_caps()
        state.begin_stopping()
        state.request_stop()
        self.assertEqual(state.finish(None), TurnOutcome.STOPPED)

    def test_stop_after_finish_is_refused_without_error(self) -> None:
        state = make_state(FakeClock())
        state.start()
        state.finish(result(TurnStatus.OK))
        self.assertFalse(state.request_stop())
        self.assertFalse(state.begin_stopping())
        self.assertEqual(state.outcome, TurnOutcome.OK)

    def test_a_result_freezes_the_active_timer_and_stops_new_caps(self) -> None:
        """Q: the time after the TurnResult (closing input, stopping the process, flushing the feed) is not work, so it
        neither counts as active time nor records a cap that would reclassify a finished turn."""
        clock = FakeClock()
        state = make_state(clock, max_active_seconds=5, hard_cap_seconds=8)
        state.start()
        clock.advance(2)
        state.mark_result_received()
        clock.advance(10)
        self.assertEqual(state.active_seconds(), 2.0)
        self.assertIsNone(state.check_caps())
        self.assertIsNone(state.limit_hit)
        self.assertEqual(state.finish(result(TurnStatus.OK)), TurnOutcome.OK)

    def test_a_cap_stop_without_a_tap_classifies_as_capped(self) -> None:
        clock = FakeClock()
        state = make_state(clock, max_active_seconds=5)
        state.start()
        clock.advance(5)
        self.assertEqual(state.check_caps(), 5.0)
        self.assertTrue(state.begin_stopping())
        self.assertFalse(state.stop_requested)
        self.assertEqual(state.finish(None), TurnOutcome.CAPPED)

    def test_finish_freezes_the_active_timer(self) -> None:
        clock = FakeClock()
        state = make_state(clock)
        state.start()
        clock.advance(7)
        state.finish(result(TurnStatus.OK))
        clock.advance(100)
        self.assertAlmostEqual(state.active_seconds(), 7.0)


class DeadProcessTests(unittest.TestCase):
    def test_a_process_gone_for_three_seconds_is_dead(self) -> None:
        """R: the wrapper exited, a child keeps stdout open: dead only after DEAD_PROCESS_SECONDS (3 s) without a result."""
        clock = FakeClock()
        state = make_state(clock)
        state.start()
        self.assertEqual(DEAD_PROCESS_SECONDS, 3.0)
        self.assertFalse(state.observe_alive(False))
        clock.advance(2)
        self.assertFalse(state.observe_alive(False))
        clock.advance(1)
        self.assertTrue(state.observe_alive(False))

    def test_a_live_process_resets_the_dead_timer(self) -> None:
        clock = FakeClock()
        state = make_state(clock)
        state.start()
        state.observe_alive(False)
        clock.advance(2)
        self.assertFalse(state.observe_alive(True))
        state.observe_alive(False)
        clock.advance(2)
        self.assertFalse(state.observe_alive(False))

    def test_the_dead_timer_does_not_count_the_active_timer(self) -> None:
        """A parked turn whose process is gone is still reported dead: waiting does not hide a dead wrapper."""
        clock = FakeClock()
        state = make_state(clock)
        state.start()
        state.waiting_changed(THREAD, True)
        state.observe_alive(False)
        clock.advance(3)
        self.assertTrue(state.observe_alive(False))


class ClassifyTests(unittest.TestCase):
    def test_classification_table(self) -> None:
        """S: the outcome for each combination of facts, in the live file's order of checks."""
        ok, err, intr = result(TurnStatus.OK), result(TurnStatus.ERROR), result(TurnStatus.INTERRUPTED)
        cases = [
            (TurnFacts(result=ok), TurnOutcome.OK),
            (TurnFacts(result=err), TurnOutcome.ERROR),
            (TurnFacts(result=intr), TurnOutcome.INTERRUPTED),
            (TurnFacts(result=None), TurnOutcome.ERROR),  # process ended without an answer
            (TurnFacts(launch_failed=True, result=ok), TurnOutcome.ERROR),
            (TurnFacts(stop_requested=True), TurnOutcome.STOPPED),
            (TurnFacts(stop_requested=True, result=ok), TurnOutcome.STOPPED),
            (TurnFacts(stop_requested=True, shutting_down=True), TurnOutcome.STOPPED),
            (TurnFacts(shutting_down=True, result=None), TurnOutcome.RESTART),
            (TurnFacts(shutting_down=True, result=err), TurnOutcome.RESTART),
            (TurnFacts(shutting_down=True, result=intr), TurnOutcome.RESTART),
            (TurnFacts(shutting_down=True, result=ok), TurnOutcome.OK),
            (TurnFacts(limit_hit=True, result=ok), TurnOutcome.CAPPED),
            (TurnFacts(limit_hit=True, result=err), TurnOutcome.CAPPED),
            (TurnFacts(limit_hit=True, result=None), TurnOutcome.CAPPED),
            (TurnFacts(limit_hit=True, shutting_down=True, result=None), TurnOutcome.RESTART),
            (TurnFacts(limit_hit=True, stop_requested=True), TurnOutcome.STOPPED),
        ]
        for facts, expected in cases:
            with self.subTest(facts=facts):
                self.assertIs(classify(facts), expected)

    def test_stored_status_keeps_the_live_words(self) -> None:
        """T: turns.status is written with the same words as the live file; a restart is stored as 'interrupted'."""
        self.assertEqual(TurnOutcome.RESTART.stored, "interrupted")
        self.assertEqual(TurnOutcome.CAPPED.stored, "capped")
        self.assertEqual(TurnOutcome.STOPPED.stored, "stopped")
        self.assertEqual(TurnOutcome.OK.stored, "ok")
        self.assertEqual(TurnOutcome.ERROR.stored, "error")

    def test_finish_takes_the_outcome_from_the_state(self) -> None:
        clock = FakeClock()
        state = make_state(clock)
        state.start()
        state.mark_running()
        outcome = state.finish(result(TurnStatus.OK))
        self.assertIs(outcome, TurnOutcome.OK)
        self.assertIs(state.outcome, TurnOutcome.OK)
        self.assertIs(state.phase, TurnPhase.FINISHED)

    def test_shutdown_during_a_card_is_a_restart(self) -> None:
        """U: the bridge shuts down while a card is open -> RESTART (the runner re-marks the inbox row pending)."""
        slot = SlotSpy()
        state = make_state(FakeClock(), slot=slot)
        state.start()
        state.waiting_changed(THREAD, True)
        self.assertIs(state.finish(None, shutting_down=True), TurnOutcome.RESTART)
        self.assertEqual(slot.calls, ["give_back"])  # no retake: the turn is over


if __name__ == "__main__":
    unittest.main()
