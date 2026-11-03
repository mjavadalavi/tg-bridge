"""TerminalGuard (cards H1, K-C, L1): a /new or /resume that would move a topic off the session its terminal tab
has open waits for the owner's answer. close_now applies it once the tab is free, by its own poll or by the
dispatcher's before_turn hook, whichever comes first, and reports it. settle waits while the turn mark exists;
before_turn does not. A tab that cannot be closed or does not close in time keeps the card answerable. A claim is a
real file naming this process, so it is live; the close and the thread are fakes.
"""

import json
import os
import threading
import unittest.mock

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.bot import terminal_guard as tg
from tg_bridge.bot.keep_guard import KEEP_NOT_APPLIED_TEXT, ApplyResult
from tg_bridge.core import terminal_claim
from tg_bridge.telegram.keyboards import PICKER_PREFIX

THREAD = 5150
OTHER_THREAD = 5151
WHAT = "session"
CHILD = 4242
REFUSED_TEXT = "the session is busy elsewhere."
NEVER_TOKEN = "0" * (2 * tg.TERMINAL_TOKEN_BYTES)
CLOSED_DONE = tg.TERMINAL_CLOSED_TEXT.format(text=tg.TERMINAL_DONE_TEXT)
LATER_DONE = tg.TERMINAL_LATER_DONE_TEXT.format(text=tg.TERMINAL_DONE_TEXT)


class Applied:
    """An apply callable that counts its runs and returns the result it was built with."""

    def __init__(self, result: ApplyResult = ApplyResult(True)) -> None:
        self.result = result
        self.runs = 0

    def __call__(self) -> ApplyResult:
        self.runs += 1
        return self.result


class HookClock(FakeClock):
    """A FakeClock that calls on_sleep after every sleep: a test can change the world while a loop waits."""

    def __init__(self) -> None:
        super().__init__()
        self.on_sleep = lambda: None

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        self.on_sleep()


class FakeLive:
    """Stands in for LiveRegistry: a topic runs a turn while busy is set."""

    def __init__(self) -> None:
        self.busy = False

    def running(self, thread_id: int):
        return object() if self.busy else None


class FakeThread:
    def __init__(self, *, target, args, daemon) -> None:
        self.target, self.args, self.daemon, self.started = target, args, daemon, False

    def start(self) -> None:
        self.started = True


class ThreadRecorder:
    """Stands in for threading.Thread: records each thread the guard asks for and never runs it."""

    def __init__(self) -> None:
        self.made: list[FakeThread] = []

    def __call__(self, *, target, args=(), daemon=False) -> FakeThread:
        thread = FakeThread(target=target, args=args, daemon=daemon)
        self.made.append(thread)
        return thread


class TerminalGuardCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.view = os.path.join(self.home.home, "view")
        os.makedirs(self.view, mode=0o700)
        self.clock = HookClock()
        self.live = FakeLive()
        self.threads = ThreadRecorder()
        self.guard = self.build()

    def build(self, **overrides) -> tg.TerminalGuard:
        options = {"view_dir": self.view, "live": self.live, "clock": self.clock, "close": self.close_tab,
                   "start_thread": self.threads}
        options.update(overrides)
        return tg.TerminalGuard(**options)

    def close_tab(self, claim: terminal_claim.Claim) -> bool:
        self.release_claim()
        return True

    def close_tab_and_turn_starts(self, claim: terminal_claim.Claim) -> bool:
        """The fake close, then the dispatcher takes the turn mark."""
        self.close_tab(claim)
        self.mark_turn()
        return True

    def write_claim(self, thread_id: int = THREAD, child: int = CHILD) -> None:
        with open(terminal_claim.claim_path(self.view, thread_id), "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started": 1.0, "child": child}, handle)

    def write_ended(self) -> None:
        terminal_claim.write_ended(self.view, THREAD, "a" * 8, "", "/cwd", 2.0)

    def release_claim(self) -> None:
        os.remove(terminal_claim.claim_path(self.view, THREAD))

    def mark_turn(self) -> None:
        open(terminal_claim.turn_path(self.view, THREAD), "wb").close()

    def ended_exists(self) -> bool:
        return os.path.exists(terminal_claim.ended_path(self.view, THREAD))

    def claim_is_live(self) -> bool:
        return terminal_claim.live_claim(self.view, THREAD) is not None

    def hold_card(self, apply: Applied, *, thread: int = THREAD) -> str:
        self.write_claim(thread)
        return self.token_of(self.guard.hold(thread, WHAT, apply))

    def token_of(self, picker) -> str:
        return picker.rows[0][0][1].args[2]

    def run_watcher(self) -> None:
        watcher = self.threads.made[-1]
        watcher.target(*watcher.args)


class CardRows(TerminalGuardCase):
    def test_nothing_is_held_without_a_view_dir_a_claim_or_a_readable_claim_dir(self) -> None:
        applied = Applied()
        self.assertIsNone(self.build(view_dir="").hold(THREAD, WHAT, applied))
        self.assertIsNone(self.guard.hold(THREAD, WHAT, applied))
        self.write_claim(OTHER_THREAD)
        self.assertIsNone(self.guard.hold(THREAD, WHAT, applied))
        plain = os.path.join(self.home.home, "plain-file")
        open(plain, "w").close()
        self.assertIsNone(self.build(view_dir=plain).hold(THREAD, WHAT, applied))
        self.assertEqual(applied.runs, 0)

    def test_a_live_claim_returns_the_card_with_three_rows_and_runs_nothing(self) -> None:
        self.write_claim()
        applied = Applied()
        picker = self.guard.hold(THREAD, WHAT, applied)
        self.assertEqual(applied.runs, 0)
        self.assertEqual(picker.text, tg.TERMINAL_CARD_TEXT.format(what=WHAT))
        (close_row,), (later_row,), (cancel_row,) = picker.rows
        self.assertEqual([close_row[0], later_row[0], cancel_row[0]],
                         [tg.TERMINAL_CLOSE_BUTTON, tg.TERMINAL_LATER_BUTTON, tg.TERMINAL_NO_BUTTON])
        token = self.token_of(picker)
        self.assertEqual(close_row[1].prefix, PICKER_PREFIX)
        self.assertEqual(close_row[1].args, (str(THREAD), tg.TERMINAL_CLOSE_KIND, token))
        self.assertEqual(later_row[1].args, (str(THREAD), tg.TERMINAL_LATER_KIND, token))
        self.assertEqual(cancel_row[1].args, (str(THREAD), tg.TERMINAL_NO_KIND, token))
        self.assertEqual(len(token), 2 * tg.TERMINAL_TOKEN_BYTES)


class Cancel(TerminalGuardCase):
    def test_cancel_drops_the_change_and_a_second_cancel_or_an_unknown_token_is_expired(self) -> None:
        applied = Applied()
        token = self.hold_card(applied)
        self.assertEqual(self.guard.cancel(OTHER_THREAD, token), tg.TERMINAL_EXPIRED_TEXT)  # another topic's tap
        self.assertEqual(self.guard.cancel(THREAD, token), tg.TERMINAL_KEPT_TEXT)
        self.assertEqual(self.guard.cancel(THREAD, token), tg.TERMINAL_EXPIRED_TEXT)
        self.assertEqual((self.guard.cancel(THREAD, NEVER_TOKEN), applied.runs), (tg.TERMINAL_EXPIRED_TEXT, 0))

    def test_cancel_takes_a_deferred_change_out_and_its_watcher_stops(self) -> None:
        applied = Applied()
        token = self.hold_card(applied)
        self.guard.defer(THREAD, token, lambda text: None)
        self.assertEqual(self.guard.cancel(THREAD, token), tg.TERMINAL_KEPT_TEXT)
        self.release_claim()
        self.run_watcher()
        self.assertEqual((applied.runs, self.clock.sleeps), (0, []))


class CloseNow(TerminalGuardCase):
    def test_close_ends_the_tab_and_its_own_poll_applies_once_and_reports(self) -> None:
        self.write_ended()
        applied = Applied(ApplyResult(True, "launched the new one."))
        reports: list[str] = []
        token = self.hold_card(applied)
        self.assertIsNone(self.guard.close_now(THREAD, token, reports.append))
        self.assertEqual((applied.runs, reports), (1, [tg.TERMINAL_CLOSED_TEXT.format(text="launched the new one.")]))
        self.assertFalse(self.ended_exists())

    def test_no_child_applies_nothing_and_the_entry_stays_for_a_later_defer(self) -> None:
        self.write_ended()
        applied = Applied()
        token = self.hold_card(applied)
        self.write_claim(child=0)
        self.assertEqual(self.guard.close_now(THREAD, token, lambda text: None), tg.TERMINAL_NO_CHILD_TEXT)
        self.assertEqual(applied.runs, 0)
        self.assertTrue(self.claim_is_live() and self.ended_exists())
        self.release_claim()
        self.assertEqual(self.guard.defer(THREAD, token, lambda text: None), tg.TERMINAL_DEFERRED_TEXT)

    def test_a_close_that_fails_or_an_unreadable_claim_applies_nothing(self) -> None:
        applied = Applied()
        self.write_claim()
        guard = self.build(close=lambda claim: False)
        token = self.token_of(guard.hold(THREAD, WHAT, applied))
        self.assertEqual(guard.close_now(THREAD, token, lambda text: None), tg.TERMINAL_NO_CHILD_TEXT)
        with unittest.mock.patch.object(terminal_claim, "live_claim", side_effect=OSError("unreadable")):
            self.assertEqual(guard.close_now(THREAD, token, lambda text: None), tg.TERMINAL_UNREADABLE_TEXT)
            guard.defer(THREAD, token, lambda text: None)
            self.assertEqual((guard.settle(THREAD), guard.before_turn(THREAD)), (None, None))
        self.assertEqual((applied.runs, guard.cancel(THREAD, token)), (0, tg.TERMINAL_KEPT_TEXT))

    def test_a_tab_that_does_not_close_in_time_times_out_and_a_later_defer_still_works(self) -> None:
        self.write_ended()
        applied = Applied()
        reports: list[str] = []
        self.guard = self.build(close=lambda claim: True)  # the CLI is signalled but the tab stays
        token = self.hold_card(applied)
        self.assertEqual(self.guard.close_now(THREAD, token, reports.append),
                         tg.TERMINAL_CLOSE_TIMEOUT_TEXT.format(seconds=tg.TERMINAL_CLOSE_WAIT_SECONDS))
        self.assertEqual(len(self.clock.sleeps), int(tg.TERMINAL_CLOSE_WAIT_SECONDS / tg.TERMINAL_POLL_SECONDS))
        self.assertEqual((applied.runs, self.ended_exists()), (0, True))
        self.assertEqual(self.guard.defer(THREAD, token, reports.append), tg.TERMINAL_DEFERRED_TEXT)
        self.release_claim()
        self.run_watcher()
        self.assertEqual((applied.runs, reports), (1, [LATER_DONE]))

    def test_a_claim_that_is_already_gone_applies_at_once_and_returns_the_text(self) -> None:
        applied = Applied()
        token = self.hold_card(applied)
        self.release_claim()
        self.write_ended()
        reports: list[str] = []
        self.assertEqual(self.guard.close_now(THREAD, token, reports.append), CLOSED_DONE)
        self.assertEqual((applied.runs, reports, self.ended_exists()), (1, [], False))

    def test_a_refused_apply_reports_its_refusal_unwrapped(self) -> None:
        reports: list[str] = []
        token = self.hold_card(Applied(ApplyResult(False, REFUSED_TEXT)))
        self.assertIsNone(self.guard.close_now(THREAD, token, reports.append))
        token = self.hold_card(Applied(ApplyResult(False)))
        self.assertIsNone(self.guard.close_now(THREAD, token, reports.append))
        self.assertEqual(reports, [REFUSED_TEXT, KEEP_NOT_APPLIED_TEXT])

    def test_a_second_close_on_the_same_card_is_expired_and_a_deferred_card_cannot_be_closed(self) -> None:
        applied = Applied()
        token = self.hold_card(applied)
        self.assertIsNone(self.guard.close_now(THREAD, token, lambda text: None))
        self.assertEqual(self.guard.close_now(THREAD, token, lambda text: None), tg.TERMINAL_EXPIRED_TEXT)
        self.assertEqual(self.guard.cancel(THREAD, token), tg.TERMINAL_EXPIRED_TEXT)
        token = self.hold_card(Applied())
        self.guard.defer(THREAD, token, lambda text: None)
        self.assertEqual(self.guard.close_now(THREAD, token, lambda text: None), tg.TERMINAL_EXPIRED_TEXT)
        self.assertEqual(applied.runs, 1)

    def test_the_dispatcher_hook_applies_first_and_the_ended_record_is_gone_before_it_returns(self) -> None:
        self.write_ended()
        applied = Applied()
        reports: list[str] = []
        seen: list[tuple[bool, int]] = []
        self.guard = self.build(close=self.close_tab_and_turn_starts)
        token = self.hold_card(applied)

        def dispatcher_turn() -> None:  # inside close_now's wait: the tab is free and the dispatcher holds the mark
            self.guard.before_turn(THREAD)
            seen.append((self.ended_exists(), applied.runs))

        self.clock.on_sleep = dispatcher_turn
        self.assertIsNone(self.guard.close_now(THREAD, token, reports.append))
        self.assertEqual(seen, [(False, 1)])
        self.assertEqual((applied.runs, reports, len(self.clock.sleeps)), (1, [CLOSED_DONE], 1))

    def test_cancel_during_the_wait_means_nothing_applies_afterwards(self) -> None:
        applied = Applied()
        reports: list[str] = []
        self.guard = self.build(close=self.close_tab_and_turn_starts)
        token = self.hold_card(applied)
        self.clock.on_sleep = lambda: self.assertEqual(self.guard.cancel(THREAD, token), tg.TERMINAL_KEPT_TEXT)
        self.assertEqual(self.guard.close_now(THREAD, token, reports.append), tg.TERMINAL_KEPT_TEXT)
        os.remove(terminal_claim.turn_path(self.view, THREAD))
        self.guard.before_turn(THREAD)
        self.assertIsNone(self.guard.settle(THREAD))
        self.assertEqual((applied.runs, reports), (0, []))


class Defer(TerminalGuardCase):
    def test_defer_starts_a_daemon_watcher_and_applies_nothing_yet(self) -> None:
        applied = Applied()
        token = self.hold_card(applied)
        self.assertEqual((self.guard.defer(THREAD, token, lambda text: None), applied.runs),
                         (tg.TERMINAL_DEFERRED_TEXT, 0))
        (watcher,) = self.threads.made
        self.assertEqual((watcher.daemon, watcher.started, watcher.args), (True, True, (THREAD, token)))
        self.assertTrue(self.claim_is_live())

    def test_defer_of_an_unknown_or_another_topics_card_is_expired(self) -> None:
        token = self.hold_card(Applied())
        self.assertEqual(self.guard.defer(THREAD, NEVER_TOKEN, lambda text: None), tg.TERMINAL_EXPIRED_TEXT)
        self.assertEqual(self.guard.defer(OTHER_THREAD, token, lambda text: None), tg.TERMINAL_EXPIRED_TEXT)
        self.assertEqual(self.threads.made, [])

    def test_settle_and_before_turn_while_the_tab_is_live_do_nothing(self) -> None:
        applied = Applied()
        token = self.hold_card(applied)
        self.guard.defer(THREAD, token, lambda text: None)
        self.assertEqual((self.guard.settle(THREAD), self.guard.settle(THREAD)), (None, None))
        self.guard.before_turn(THREAD)
        self.assertEqual(applied.runs, 0)

    def test_settle_after_the_tab_closed_applies_once_reports_and_consumes_the_ended_record(self) -> None:
        self.write_ended()
        self.assertIsNone(self.guard.settle(THREAD))  # nothing deferred yet: the record stays
        self.assertTrue(self.ended_exists())
        applied = Applied()
        reports: list[str] = []
        token = self.hold_card(applied)
        self.guard.defer(THREAD, token, reports.append)
        self.release_claim()
        self.assertEqual(self.guard.settle(THREAD), LATER_DONE)
        self.assertEqual((applied.runs, reports, self.ended_exists()), (1, [LATER_DONE], False))
        self.assertIsNone(self.guard.settle(THREAD))
        self.assertEqual(applied.runs, 1)

    def test_settle_waits_while_a_turn_runs_after_the_tab_closed(self) -> None:
        applied = Applied()
        token = self.hold_card(applied)
        self.guard.defer(THREAD, token, lambda text: None)
        self.release_claim()
        self.live.busy = True
        self.assertIsNone(self.guard.settle(THREAD))
        self.live.busy = False
        self.assertIsNotNone(self.guard.settle(THREAD))
        self.assertEqual(applied.runs, 1)

    def test_settle_waits_for_the_turn_mark_but_the_dispatcher_hook_does_not(self) -> None:
        applied = Applied()
        reports: list[str] = []
        token = self.hold_card(applied)
        self.guard.defer(THREAD, token, reports.append)
        self.release_claim()
        self.mark_turn()
        self.assertIsNone(self.guard.settle(THREAD))
        self.assertEqual(applied.runs, 0)
        self.guard.before_turn(THREAD)
        self.assertEqual((applied.runs, reports), (1, [LATER_DONE]))
        self.assertIsNone(self.guard.settle(THREAD))

    def test_a_refusal_is_reported_and_a_failing_report_does_not_lose_the_result(self) -> None:
        def broken_report(text: str) -> None:
            raise RuntimeError("telegram is down")

        reports: list[str] = []
        token = self.hold_card(Applied(ApplyResult(False, REFUSED_TEXT)))
        self.guard.defer(THREAD, token, reports.append)
        self.release_claim()
        self.assertEqual(self.guard.settle(THREAD), REFUSED_TEXT)
        self.assertEqual(reports, [REFUSED_TEXT])
        applied = Applied()
        token = self.hold_card(applied)
        self.guard.defer(THREAD, token, broken_report)
        self.release_claim()
        self.assertEqual((self.guard.settle(THREAD), applied.runs), (LATER_DONE, 1))

    def test_the_watcher_settles_when_the_tab_closes_and_stops_after(self) -> None:
        applied = Applied()
        reports: list[str] = []
        token = self.hold_card(applied)
        self.guard.defer(THREAD, token, reports.append)
        calls: list[int] = []

        def close_on_second_sleep() -> None:
            calls.append(1)
            if len(calls) == 2:
                self.release_claim()

        self.clock.on_sleep = close_on_second_sleep
        self.run_watcher()
        self.assertEqual((applied.runs, reports, len(self.clock.sleeps)), (1, [LATER_DONE], 2))

    def test_the_watcher_sleeps_before_its_first_settle_even_when_the_tab_is_already_closed(self) -> None:
        applied = Applied()
        token = self.hold_card(applied)
        self.guard.defer(THREAD, token, lambda text: None)
        self.release_claim()  # the tab closed before the watcher started
        seen_at_sleep: list[tuple[int, int]] = []
        self.clock.on_sleep = lambda: seen_at_sleep.append((len(self.clock.sleeps), applied.runs))
        self.run_watcher()
        self.assertEqual(seen_at_sleep[0], (1, 0))  # the first recorded action is a sleep; nothing applied yet
        self.assertEqual(applied.runs, 1)

    def test_a_poll_and_the_dispatcher_hook_racing_apply_the_change_exactly_once(self) -> None:
        applied = Applied()
        reports: list[str] = []
        token = self.hold_card(applied)
        self.guard.defer(THREAD, token, reports.append)
        self.release_claim()
        barrier = threading.Barrier(2)
        calls = [lambda: self.guard.settle(THREAD), lambda: self.guard.before_turn(THREAD)]
        workers = [threading.Thread(target=lambda call=call: (barrier.wait(), call())) for call in calls]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=10)
        self.assertEqual((applied.runs, reports), (1, [LATER_DONE]))


class ReplacedCards(TerminalGuardCase):
    def test_a_new_hold_replaces_the_earlier_card(self) -> None:
        first = Applied()
        second = Applied()
        old_token = self.hold_card(first)
        new_token = self.token_of(self.guard.hold(THREAD, WHAT, second))
        self.assertNotEqual(old_token, new_token)
        self.assertEqual(self.guard.close_now(THREAD, old_token, lambda text: None), tg.TERMINAL_EXPIRED_TEXT)
        self.assertIsNone(self.guard.close_now(THREAD, new_token, lambda text: None))
        self.assertEqual((first.runs, second.runs), (0, 1))

    def test_a_new_hold_replaces_a_deferred_change(self) -> None:
        first = Applied()
        old_token = self.hold_card(first)
        self.guard.defer(THREAD, old_token, lambda text: None)
        new_token = self.token_of(self.guard.hold(THREAD, WHAT, Applied()))
        self.assertEqual(self.guard.defer(THREAD, old_token, lambda text: None), tg.TERMINAL_EXPIRED_TEXT)
        self.assertEqual(self.guard.cancel(THREAD, old_token), tg.TERMINAL_EXPIRED_TEXT)
        self.release_claim()
        self.assertIsNone(self.guard.settle(THREAD))
        self.run_watcher()  # the replaced card's watcher stops at once
        self.assertEqual((first.runs, self.clock.sleeps), (0, []))
        self.assertEqual(self.guard.cancel(THREAD, new_token), tg.TERMINAL_KEPT_TEXT)

    def test_a_hold_of_another_topic_leaves_this_card_answerable(self) -> None:
        applied = Applied()
        token = self.hold_card(applied)
        self.write_claim(OTHER_THREAD)
        self.guard.hold(OTHER_THREAD, WHAT, Applied())
        self.assertIsNone(self.guard.close_now(THREAD, token, lambda text: None))
        self.assertEqual(applied.runs, 1)


if __name__ == "__main__":
    unittest.main()
