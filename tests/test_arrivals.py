"""Arrivals: the wait before a turn is taken (burst batching). A FakeClock moves only when a test says so, and its
sleep does not block, so each case runs at once. The cap case feeds a steady stream from inside the sleep, and walks
the wait the way the dispatcher's settle does (remaining, then a sleep of at most one second)."""

import unittest
from collections.abc import Callable

from tests.fakes.clock import FakeClock
from tg_bridge.core.arrivals import BURST_CAP_SECONDS, BURST_GAP_SECONDS, Arrivals
from tg_bridge.core.topic_queues import Admission, Delivery

THREAD = 7101
OTHER = 7102


class StreamClock(FakeClock):
    """A FakeClock that calls on_sleep after each sleep, as a message stream arriving while the runner waits."""

    def __init__(self) -> None:
        super().__init__()
        self.on_sleep: Callable[[], None] | None = None

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        if self.on_sleep is not None:
            self.on_sleep()


class ScriptedClock:
    """A clock whose monotonic() returns the given readings one after another, so a test can make time go backward."""

    def __init__(self, readings: list[float]) -> None:
        self._readings = list(readings)
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self._readings.pop(0)

    def now(self) -> float:
        return 0.0

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class ArrivalsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = StreamClock()
        self.arrivals = Arrivals(self.clock)

    def test_a_topic_without_messages_is_not_delayed(self) -> None:
        self.assertEqual(self.arrivals.remaining(THREAD, self.clock.monotonic()), 0.0)

    def test_a_fresh_message_waits_one_gap(self) -> None:
        self.arrivals.mark(THREAD)
        self.assertEqual(self.arrivals.remaining(THREAD, self.clock.monotonic()), BURST_GAP_SECONDS)

    def test_a_message_quiet_for_the_gap_is_not_delayed(self) -> None:
        self.arrivals.mark(THREAD)
        self.clock.advance(BURST_GAP_SECONDS + 0.5)
        self.assertEqual(self.arrivals.remaining(THREAD, self.clock.monotonic()), 0.0)

    def test_only_the_remaining_part_of_the_gap_is_waited(self) -> None:
        self.arrivals.mark(THREAD)
        self.clock.advance(1.25)
        self.assertEqual(self.arrivals.remaining(THREAD, self.clock.monotonic()), BURST_GAP_SECONDS - 1.25)

    def test_remaining_does_not_sleep(self) -> None:
        self.arrivals.mark(THREAD)
        self.arrivals.remaining(THREAD, self.clock.monotonic())
        self.assertEqual(self.clock.sleeps, [])

    def test_another_topic_does_not_delay_this_one(self) -> None:
        self.arrivals.mark(OTHER)
        self.assertEqual(self.arrivals.remaining(THREAD, self.clock.monotonic()), 0.0)

    def test_mark_reports_a_follower_within_the_gap(self) -> None:
        self.assertFalse(self.arrivals.mark(THREAD), "the first message of a topic is not a follower")
        self.clock.advance(BURST_GAP_SECONDS - 0.5)
        self.assertTrue(self.arrivals.mark(THREAD))

    def test_mark_after_the_gap_is_not_a_follower(self) -> None:
        self.arrivals.mark(THREAD)
        self.clock.advance(BURST_GAP_SECONDS)  # exactly the gap is already a new arrival
        self.assertFalse(self.arrivals.mark(THREAD))

    def test_mark_of_another_topic_is_not_a_follower(self) -> None:
        self.arrivals.mark(OTHER)
        self.assertFalse(self.arrivals.mark(THREAD))

    def test_a_refusal_follows_only_a_refusal_within_the_gap(self) -> None:
        """Only a refusal right after another refusal is a follower: the first of a burst of overflow gets the notice,
        and an accepted message in between clears the state."""
        full, queued = Delivery(Admission.FULL), Delivery(Admission.QUEUED)
        self.assertFalse(self.arrivals.admitted(THREAD, full, burst=True).burst, "no refusal came before it")
        self.assertTrue(self.arrivals.admitted(THREAD, full, burst=True).burst, "it follows a refusal in the gap")
        self.assertFalse(self.arrivals.admitted(THREAD, queued, burst=True).burst)
        self.assertFalse(self.arrivals.admitted(THREAD, full, burst=True).burst, "the accepted message cleared it")
        self.assertFalse(self.arrivals.admitted(THREAD, full, burst=False).burst, "after the gap: a new burst")

    def test_a_reversed_clock_call_order_never_moves_the_topic_time_backward(self) -> None:
        """Two marks whose clock reads come out of order: the topic keeps the later time, so the next mark still
        counts from it (10.0 is kept, not 5.0), and 11.5 is a follower of the message read at 10.0."""
        clock = ScriptedClock([10.0, 5.0, 11.5])
        arrivals = Arrivals(clock)
        self.assertFalse(arrivals.mark(THREAD))
        self.assertTrue(arrivals.mark(THREAD), "read at 5.0 is within the gap of 10.0")
        self.assertTrue(arrivals.mark(THREAD), "11.5 is 1.5 s after 10.0, not 6.5 s after 5.0")

    def test_a_steady_stream_is_capped(self) -> None:
        """Every sleep is followed by a new message, so the topic never goes quiet: the wait ends at the cap, measured
        from when it started, not from the last message."""
        self.arrivals.mark(THREAD)
        self.clock.on_sleep = lambda: self.arrivals.mark(THREAD)
        started = self.clock.monotonic()

        left = self.arrivals.remaining(THREAD, started)
        while left > 0:
            self.clock.sleep(min(left, 1.0))
            left = self.arrivals.remaining(THREAD, started)

        self.assertAlmostEqual(self.clock.monotonic() - started, BURST_CAP_SECONDS)
        self.assertGreater(len(self.clock.sleeps), 1)


if __name__ == "__main__":
    unittest.main()
