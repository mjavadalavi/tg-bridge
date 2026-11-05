"""Dispatcher hold loop (linger): the topic's runner stays on a held session, ticks the LingerControl, posts the cap and
exit notices, hands a quiet message over and gives up after repeated failed ticks (C8c).

A StubLinger stands in for LingerService. Its holding() is True while the test holds the topic, wait_once() returns
None ticks until the test scripts an end (next_end), and a fail plan makes some ticks raise. Everything else is the
DispatcherCase of test_dispatcher.py: a StubTurns, the SQLite inbox, the real TopicQueues and Messenger. Every wait is a
bounded poll, and tearDown releases the hold, so a broken hold fails in seconds instead of hanging.
"""

import threading
import time
import unittest
from unittest import mock

from tests.test_dispatcher import CHAT, SENDER, THREAD, TOPIC, DispatcherCase, wait_until
from tg_bridge.core.dispatcher import Dispatcher
from tg_bridge.core.linger import LINGER_CAP_TEXT, LINGER_EXIT_TEXT, HoldEnd, HoldResult
from tg_bridge.core.topic_queues import Admission

NOTICE_PREFIXES = ("⌛", "⚠️")  # the first character of the linger notices, to find them among the answers
CAP_MINUTES = 30  # the default linger_seconds (1800) in minutes


class StubLinger:
    """Stands in for LingerService. holds: the topic has a held session. idle: the held session is idle, so a message is
    handed over quietly. next_end: the end the next tick reports (the tick then clears holds, as a release does); None
    makes the tick wait a moment and report nothing. fail_plan: one bool per tick, True makes that tick raise."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.holds = False
        self.idle = False
        self.pending = 0
        self.next_end: HoldEnd | None = None
        self.fail_plan: list[bool] = []
        self.calls = 0
        self.wakes: list[int] = []
        self.releases: list[int] = []

    def holding(self, thread_id: int) -> bool:
        with self._lock:
            return self.holds

    def idle_alive(self, thread_id: int) -> bool:
        with self._lock:
            return self.holds and self.idle

    def wake(self, thread_id: int) -> None:
        with self._lock:
            self.wakes.append(thread_id)

    def request_release(self, thread_id: int) -> bool:
        with self._lock:
            self.releases.append(thread_id)
            self.next_end = HoldEnd.STOPPED  # what the real service does on its next tick
            return True

    def wait_once(self, thread_id: int) -> HoldResult:
        with self._lock:
            self.calls += 1
            if self.fail_plan and self.fail_plan.pop(0):
                raise RuntimeError("scripted tick failure")
            end = self.next_end
            if end is not None:
                self.holds = False
                self.next_end = None
            pending = self.pending
        if end is None:
            time.sleep(0.005)
            return HoldResult(None, pending, CHAT)
        return HoldResult(end, pending, CHAT)


class LingerDispatcherCase(DispatcherCase):
    """DispatcherCase with a StubLinger injected and no pause after a failed tick."""

    def setUp(self) -> None:
        self.linger = StubLinger()
        super().setUp()

    def tearDown(self) -> None:
        with self.linger._lock:
            self.linger.holds = False  # no runner is left holding a topic when the case ends
        super().tearDown()

    def _make_dispatcher(self, media_wait: float | None = None) -> Dispatcher:
        return Dispatcher(
            turns=self.turns, slots=self.slots, inbox=self.inbox, messenger=self.messenger, answers=self.answers,
            titles=self.titles, client=self.tg, live=self.live, config=self.config, logger=self.logger,
            media_wait_seconds=self.media_wait if media_wait is None else media_wait,
            start_thread=self.thread_log, linger=self.linger, hold_failure_pause=0.0, clock=self.clock)

    def alive(self) -> bool:
        return any(thread.is_alive() for thread in self.thread_log.threads)

    def start_held_topic(self) -> None:
        """A first turn runs and ends; the runner then holds the topic and ticks."""
        self.linger.holds = True
        self.ingest(1)
        self.deliver(1, "first")
        self.assertTrue(wait_until(lambda: self.linger.calls >= 1), "the runner never ticked")

    def notices(self, thread_id: int = THREAD) -> list[str]:
        return [text for text in self.tg.texts(thread_id) if text.startswith(NOTICE_PREFIXES)]


class HoldLoop(LingerDispatcherCase):
    def test_runner_stays_while_held_and_ticks(self) -> None:
        self.start_held_topic()
        self.assertTrue(wait_until(lambda: self.linger.calls >= 3), "the hold did not tick again")
        self.assertTrue(self.alive(), "the runner left the topic while its session was held")
        self.linger.next_end = HoldEnd.DONE
        self.assertTrue(wait_until(lambda: not self.alive()), "the runner did not end after the release")
        self.assertEqual(self.turns.texts_run(THREAD), ["first"])

    def test_message_to_an_idle_kept_session_has_no_queued_notice_and_wakes(self) -> None:
        self.linger.idle = True
        self.start_held_topic()
        self.ingest(2)
        delivery = self.deliver(2, "second")
        self.assertIs(delivery.admission, Admission.HANDED)
        self.assertFalse(delivery.notify, "a quiet message must not announce its queue place")
        self.assertEqual(self.linger.wakes, [THREAD], "the held runner was not woken")
        self.linger.next_end = HoldEnd.DONE
        self.assertTrue(wait_until(lambda: not self.alive()))
        self.assertEqual(self.turns.texts_run(THREAD), ["first", "second"])
        self.assertIn("answer to second", self.tg.texts(THREAD))

    def test_message_during_a_self_turn_is_queued_with_its_notice(self) -> None:
        self.linger.idle = False
        self.start_held_topic()
        self.ingest(2)
        delivery = self.deliver(2, "second")
        self.assertIs(delivery.admission, Admission.QUEUED)
        self.assertTrue(delivery.notify, "a message during a self turn keeps the queued notice")
        self.assertEqual(self.linger.wakes, [])
        self.linger.next_end = HoldEnd.DONE
        self.assertTrue(wait_until(lambda: not self.alive()))
        self.assertEqual(self.turns.texts_run(THREAD), ["first", "second"])

    def test_media_placeholder_behind_an_idle_kept_session_has_no_notice(self) -> None:
        self.linger.idle = True
        self.start_held_topic()
        reserved = self.dispatcher.reserve(THREAD, TOPIC, CHAT, SENDER, 2)
        self.assertIs(reserved.admission, Admission.RESERVED)
        self.assertFalse(reserved.notify)
        filled = self.dispatcher.deliver(THREAD, TOPIC, CHAT, SENDER, 2, "voice text", reserved=reserved.nid)
        self.assertIs(filled.admission, Admission.FILLED)
        self.assertFalse(filled.notify, "a placeholder filled behind an idle kept session is not announced")
        self.linger.next_end = HoldEnd.DONE
        self.assertTrue(wait_until(lambda: not self.alive()))
        self.assertEqual(self.turns.texts_run(THREAD), ["first", "voice text"])

    def test_runner_returns_after_release_and_the_next_message_starts_a_new_runner(self) -> None:
        self.start_held_topic()
        self.linger.next_end = HoldEnd.STOPPED
        self.assertTrue(wait_until(lambda: not self.alive()))
        self.assertEqual(len(self.thread_log.threads), 1)
        self.ingest(2)
        delivery = self.deliver(2, "second")
        self.assertIs(delivery.admission, Admission.STARTED, "the topic was not idle after the release")
        self.assertEqual(len(self.thread_log.threads), 2)
        self.assertTrue(wait_until(lambda: not self.alive()))
        self.assertEqual(self.turns.texts_run(THREAD), ["first", "second"])

    def test_shutdown_during_hold_ends_the_runner(self) -> None:
        self.start_held_topic()
        self.live.begin_shutdown()
        self.assertTrue(wait_until(lambda: not self.alive()), "the hold did not end on shutdown")
        self.assertEqual(self.turns.texts_run(THREAD), ["first"])
        self.assertEqual(self.linger.releases, [], "a shutdown must not ask the session to close")


class HoldNotices(LingerDispatcherCase):
    def end_after_first_turn(self, end: HoldEnd, pending: int) -> None:
        self.linger.pending = pending
        self.linger.next_end = end
        self.linger.holds = True
        self.ingest(1)
        self.deliver(1, "first")
        self.assertTrue(wait_until(lambda: not self.alive()), f"the runner did not end on {end}")

    def test_cap_with_pending_sends_the_cap_notice(self) -> None:
        self.end_after_first_turn(HoldEnd.CAP, pending=2)
        self.assertEqual(self.notices(), [LINGER_CAP_TEXT.format(minutes=CAP_MINUTES)])

    def test_exit_with_pending_sends_the_exit_notice(self) -> None:
        self.end_after_first_turn(HoldEnd.EXITED, pending=1)
        self.assertEqual(self.notices(), [LINGER_EXIT_TEXT.format(pending=1)])

    def test_a_lost_notice_is_logged_and_is_not_a_failed_tick(self) -> None:
        real_send = self.messenger.send_text

        def fail_notices(chat_id: int | str, thread_id: int | str, text: str) -> None:
            if text.startswith(NOTICE_PREFIXES):
                raise RuntimeError("scripted send failure")
            real_send(chat_id, thread_id, text)

        with mock.patch.object(self.messenger, "send_text", side_effect=fail_notices), \
                mock.patch.object(self.logger, "log") as logged:
            self.end_after_first_turn(HoldEnd.CAP, pending=2)
        lines = [call.args[0] for call in logged.call_args_list]
        self.assertFalse(any("hold tick failed" in line for line in lines), "a lost notice counted as a failed tick")
        self.assertTrue(any("notice not sent" in line for line in lines), "the lost notice was not logged")
        self.assertEqual(self.linger.releases, [], "a lost notice must not ask the session to close")
        self.assertEqual(self.notices(), [])
        self.assertIn("answer to first", self.tg.texts(THREAD))

    def test_done_sends_nothing(self) -> None:
        self.end_after_first_turn(HoldEnd.DONE, pending=0)
        self.assertEqual(self.notices(), [])
        self.assertIn("answer to first", self.tg.texts(THREAD))

    def test_cap_or_exit_without_pending_sends_nothing(self) -> None:
        for index, end in enumerate((HoldEnd.CAP, HoldEnd.EXITED, HoldEnd.STOPPED, HoldEnd.CHANGED, HoldEnd.SHUTDOWN)):
            with self.subTest(end=end.value):
                thread = THREAD + 1 + index  # one topic per subtest: each one's notices are its own
                self.linger.pending = 0
                self.linger.holds = True
                self.linger.next_end = end
                self.ingest(200 + index, thread_id=thread)
                self.deliver(200 + index, "first", thread_id=thread)
                self.assertTrue(wait_until(lambda: not self.alive()))
                self.assertEqual(self.notices(thread), [])


class HoldFailures(LingerDispatcherCase):
    def test_failed_ticks_are_logged_and_the_runner_goes_on(self) -> None:
        self.linger.fail_plan = [True, True]
        self.linger.holds = True
        self.ingest(1)
        self.deliver(1, "first")
        self.assertTrue(wait_until(lambda: self.linger.calls >= 3), "the runner stopped after failed ticks")
        self.assertTrue(self.alive(), "two failed ticks ended the runner")
        self.linger.next_end = HoldEnd.DONE
        self.assertTrue(wait_until(lambda: not self.alive()))
        self.assertEqual(self.linger.releases, [])

    def test_a_good_tick_resets_the_failure_count(self) -> None:
        self.linger.fail_plan = [True, True, False, True, True, False]
        self.linger.holds = True
        self.ingest(1)
        self.deliver(1, "first")
        self.assertTrue(wait_until(lambda: self.linger.calls >= 6), "the runner stopped after failed ticks")
        self.assertEqual(self.linger.releases, [], "two failures, a good tick, two failures: no release")
        self.linger.next_end = HoldEnd.DONE
        self.assertTrue(wait_until(lambda: not self.alive()))

    def test_three_failed_ticks_in_a_row_request_a_release(self) -> None:
        self.linger.fail_plan = [True, True, True]
        self.linger.holds = True
        self.ingest(1)
        self.deliver(1, "first")
        self.assertTrue(wait_until(lambda: self.linger.releases == [THREAD]), "no release was requested")
        self.assertTrue(wait_until(lambda: not self.alive()), "the requested release did not end the hold")
        self.assertEqual(self.linger.releases, [THREAD])


if __name__ == "__main__":
    unittest.main()
