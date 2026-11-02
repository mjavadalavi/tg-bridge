"""LiveRegistry: the registries of the live file (_sessions, _feeds, _aborted, _stop_taps, _shutting_down) behind one lock.

Covers the stop paths of the live file (/stop, ⏭ Now, the feed's Stop button), the spawn guard of _run_turn_body and
the shutdown snapshot of shutdown_sessions. Sessions and feeds are plain sentinel objects: the registry only compares
them by identity. The aborted-read test holds the registry's lock on purpose to show that readers wait for it.
"""

import threading
import time
import unittest

from tg_bridge.core.live import AbortFlags, LiveRegistry, RunningTurn

THREAD_A, THREAD_B = 9301, 9302
WAIT = 5.0


class RegistrationTests(unittest.TestCase):
    def test_begin_turn_makes_the_turn_visible(self) -> None:
        reg = LiveRegistry()
        session, feed = object(), object()
        self.assertTrue(reg.begin_turn(THREAD_A, session, feed))
        running = reg.running(THREAD_A)
        self.assertIsInstance(running, RunningTurn)
        self.assertIs(running.session, session)
        self.assertIs(running.feed, feed)
        self.assertIsNone(reg.running(THREAD_B))

    def test_end_turn_removes_the_turn(self) -> None:
        reg = LiveRegistry()
        session, feed = object(), object()
        reg.begin_turn(THREAD_A, session, feed)
        reg.end_turn(THREAD_A, session, feed)
        self.assertIsNone(reg.running(THREAD_A))

    def test_a_late_end_turn_never_drops_a_newer_turn(self) -> None:
        """The old finally blocks deleted by identity: a stale cleanup must leave the next turn of the topic in place."""
        reg = LiveRegistry()
        old_session, old_feed = object(), object()
        new_session, new_feed = object(), object()
        reg.begin_turn(THREAD_A, old_session, old_feed)
        reg.end_turn(THREAD_A, old_session, old_feed)
        reg.begin_turn(THREAD_A, new_session, new_feed)
        reg.end_turn(THREAD_A, old_session, old_feed)  # stale: must not touch the new turn
        running = reg.running(THREAD_A)
        self.assertIsNotNone(running)
        self.assertIs(running.session, new_session)
        self.assertIs(running.feed, new_feed)

    def test_topics_are_independent(self) -> None:
        reg = LiveRegistry()
        reg.begin_turn(THREAD_A, object(), object())
        self.assertIsNone(reg.running(THREAD_B))
        self.assertIsNone(reg.request_abort(THREAD_B, stop_tap=True))
        self.assertFalse(reg.is_aborted(THREAD_B))


class AbortTests(unittest.TestCase):
    def test_request_abort_marks_a_running_turn_and_returns_it(self) -> None:
        """/stop: the turn is marked aborted and handed back so the caller stops the session on its own thread."""
        reg = LiveRegistry()
        session, feed = object(), object()
        reg.begin_turn(THREAD_A, session, feed)
        running = reg.request_abort(THREAD_A, stop_tap=False)
        self.assertIsNotNone(running)
        self.assertIs(running.session, session)
        self.assertTrue(reg.is_aborted(THREAD_A))
        self.assertEqual(reg.consume_abort(THREAD_A), AbortFlags(aborted=True, stop_tap=False))

    def test_feed_stop_button_sets_both_flags(self) -> None:
        reg = LiveRegistry()
        reg.begin_turn(THREAD_A, object(), object())
        self.assertIsNotNone(reg.request_abort(THREAD_A, stop_tap=True))
        self.assertEqual(reg.consume_abort(THREAD_A), AbortFlags(aborted=True, stop_tap=True))

    def test_request_abort_on_an_idle_topic_marks_nothing(self) -> None:
        reg = LiveRegistry()
        self.assertIsNone(reg.request_abort(THREAD_A, stop_tap=True))
        self.assertFalse(reg.is_aborted(THREAD_A))
        self.assertEqual(reg.consume_abort(THREAD_A), AbortFlags(aborted=False, stop_tap=False))

    def test_consume_abort_clears_the_flags_for_the_next_turn(self) -> None:
        reg = LiveRegistry()
        session, feed = object(), object()
        reg.begin_turn(THREAD_A, session, feed)
        reg.request_abort(THREAD_A, stop_tap=True)
        reg.consume_abort(THREAD_A)
        self.assertEqual(reg.consume_abort(THREAD_A), AbortFlags(aborted=False, stop_tap=False))
        self.assertFalse(reg.is_aborted(THREAD_A))

    def test_is_aborted_reads_under_the_registry_lock(self) -> None:
        """The CODE-MAP hazard fix: an aborted read waits for the registry lock like every other read."""
        reg = LiveRegistry()
        done = threading.Event()
        result: list[bool] = []

        def reader() -> None:
            result.append(reg.is_aborted(THREAD_A))
            done.set()

        with reg._lock:  # the registry's own lock, held here to prove the reader waits for it
            th = threading.Thread(target=reader, daemon=True)
            th.start()
            self.assertFalse(done.wait(0.2), "is_aborted returned while the registry lock was held")
        self.assertTrue(done.wait(WAIT))
        th.join(WAIT)
        self.assertEqual(result, [False])


class ShutdownTests(unittest.TestCase):
    def test_begin_turn_is_refused_after_shutdown_begins(self) -> None:
        """The spawn guard of _run_turn_body: nothing is registered once shutdown has taken its snapshot."""
        reg = LiveRegistry()
        self.assertFalse(reg.is_shutting_down())
        reg.begin_shutdown()
        self.assertTrue(reg.is_shutting_down())
        self.assertFalse(reg.begin_turn(THREAD_A, object(), object()))
        self.assertIsNone(reg.running(THREAD_A))

    def test_begin_shutdown_returns_the_sessions_live_at_that_moment(self) -> None:
        reg = LiveRegistry()
        first, second = object(), object()
        reg.begin_turn(THREAD_A, first, object())
        reg.begin_turn(THREAD_B, second, object())
        snapshot = reg.begin_shutdown()
        self.assertEqual(len(snapshot), 2)
        self.assertTrue(any(s is first for s in snapshot))
        self.assertTrue(any(s is second for s in snapshot))

    def test_turns_that_end_after_shutdown_still_unregister(self) -> None:
        """A turn cut short by a shutdown is still removed by its own cleanup."""
        reg = LiveRegistry()
        session, feed = object(), object()
        reg.begin_turn(THREAD_A, session, feed)
        reg.begin_shutdown()
        reg.end_turn(THREAD_A, session, feed)
        self.assertIsNone(reg.running(THREAD_A))

    def test_concurrent_turns_leave_the_registry_empty(self) -> None:
        reg = LiveRegistry()
        errors: list[BaseException] = []

        def topic(thread_id: int) -> None:
            try:
                for _ in range(200):
                    session, feed = object(), object()
                    if reg.begin_turn(thread_id, session, feed):
                        reg.request_abort(thread_id, stop_tap=False)
                        reg.consume_abort(thread_id)
                        reg.end_turn(thread_id, session, feed)
            except BaseException as exc:  # surfaced after join
                errors.append(exc)

        threads = [threading.Thread(target=topic, args=(THREAD_A + i,), daemon=True) for i in range(6)]
        t0 = time.monotonic()
        for th in threads:
            th.start()
        for th in threads:
            th.join(WAIT * 4)
            self.assertFalse(th.is_alive())
        self.assertLess(time.monotonic() - t0, WAIT * 4)
        self.assertEqual(errors, [])
        for i in range(6):
            self.assertIsNone(reg.running(THREAD_A + i))


class LingerMapTests(unittest.TestCase):
    """The kept sessions of topics between their turns: counted against a limit, part of the shutdown snapshot."""

    def test_begin_and_end_linger(self) -> None:
        reg = LiveRegistry()
        session = object()
        self.assertIsNone(reg.lingering(THREAD_A))
        self.assertTrue(reg.begin_linger(THREAD_A, session, limit=3))
        self.assertIs(reg.lingering(THREAD_A), session)
        self.assertEqual(reg.lingering_count(), 1)
        reg.end_linger(THREAD_A, session)
        self.assertIsNone(reg.lingering(THREAD_A))
        self.assertEqual(reg.lingering_count(), 0)

    def test_limit_counts_other_threads_only(self) -> None:
        reg = LiveRegistry()
        self.assertFalse(reg.begin_linger(THREAD_A, object(), limit=0))
        self.assertTrue(reg.begin_linger(THREAD_A, object(), limit=1))
        self.assertFalse(reg.begin_linger(THREAD_B, object(), limit=1))  # one other topic already holds the slot
        again = object()
        self.assertTrue(reg.begin_linger(THREAD_A, again, limit=1))  # the same topic replaces its own entry
        self.assertIs(reg.lingering(THREAD_A), again)
        self.assertEqual(reg.lingering_count(), 1)

    def test_refused_after_shutdown(self) -> None:
        reg = LiveRegistry()
        reg.begin_shutdown()
        self.assertFalse(reg.begin_linger(THREAD_A, object(), limit=3))
        self.assertIsNone(reg.lingering(THREAD_A))

    def test_shutdown_snapshot_includes_lingering_once(self) -> None:
        reg = LiveRegistry()
        running, kept, both = object(), object(), object()
        reg.begin_turn(THREAD_A, running, object())
        reg.begin_turn(THREAD_B, both, object())
        reg.begin_linger(THREAD_B, both, limit=3)
        reg.begin_linger(9303, kept, limit=3)
        snapshot = reg.begin_shutdown()
        self.assertEqual(len(snapshot), 3)
        for expected in (running, both, kept):
            self.assertEqual(sum(1 for s in snapshot if s is expected), 1)

    def test_end_linger_ignores_another_session(self) -> None:
        reg = LiveRegistry()
        held, other = object(), object()
        reg.begin_linger(THREAD_A, held, limit=3)
        reg.end_linger(THREAD_A, other)
        self.assertIs(reg.lingering(THREAD_A), held)


if __name__ == "__main__":
    unittest.main()
