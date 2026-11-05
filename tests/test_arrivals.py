"""Arrivals: the settle before a turn is taken (burst batching). A FakeClock moves only when a test says so, and its
sleep does not block, so each case runs at once. The cap case feeds a steady stream from inside the sleep."""

import unittest
from collections.abc import Callable

from tests.fakes.clock import FakeClock
from tg_bridge.core.arrivals import BURST_CAP_SECONDS, BURST_GAP_SECONDS, Arrivals

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


class ArrivalsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = StreamClock()
        self.arrivals = Arrivals(self.clock)

    def test_a_topic_without_messages_is_not_delayed(self) -> None:
        self.arrivals.settle(THREAD)
        self.assertEqual(self.clock.sleeps, [])

    def test_a_fresh_message_waits_one_gap(self) -> None:
        self.arrivals.mark(THREAD)
        self.arrivals.settle(THREAD)
        self.assertEqual(self.clock.sleeps, [BURST_GAP_SECONDS])

    def test_a_message_quiet_for_the_gap_is_not_delayed(self) -> None:
        self.arrivals.mark(THREAD)
        self.clock.advance(BURST_GAP_SECONDS + 0.5)
        self.arrivals.settle(THREAD)
        self.assertEqual(self.clock.sleeps, [])

    def test_only_the_remaining_part_of_the_gap_is_slept(self) -> None:
        self.arrivals.mark(THREAD)
        self.clock.advance(1.25)
        self.arrivals.settle(THREAD)
        self.assertEqual(self.clock.sleeps, [BURST_GAP_SECONDS - 1.25])

    def test_another_topic_does_not_delay_this_one(self) -> None:
        self.arrivals.mark(OTHER)
        self.arrivals.settle(THREAD)
        self.assertEqual(self.clock.sleeps, [])

    def test_a_steady_stream_is_capped(self) -> None:
        """Every sleep is followed by a new message, so the topic never goes quiet: the settle ends at the cap."""
        self.arrivals.mark(THREAD)
        self.clock.on_sleep = lambda: self.arrivals.mark(THREAD)

        self.arrivals.settle(THREAD)

        self.assertAlmostEqual(sum(self.clock.sleeps), BURST_CAP_SECONDS)
        self.assertTrue(all(0 < step <= BURST_GAP_SECONDS for step in self.clock.sleeps))
        self.assertGreater(len(self.clock.sleeps), 1)


if __name__ == "__main__":
    unittest.main()
