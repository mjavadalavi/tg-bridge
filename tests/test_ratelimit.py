"""RateGate: port of the gate scenarios of fakecli/test_rate_park.py (A, B, E) on a FakeClock.

Scenario C (idle time in the log header) and D (poll thread never stalls) belong to the feed and the poller and are
ported with those modules. Nothing here sleeps for real: every wait goes through FakeClock.sleep.
"""

import threading
import unittest

from tests.fakes.clock import FakeClock
from tg_bridge.telegram.ratelimit import (
    DEFAULT_FLOOD_WAIT,
    FEED_SENDS_PER_MINUTE,
    GROUP_SENDS_PER_MINUTE,
    RETRY_AFTER_MAX,
    RateGate,
)


def _gate(clock: FakeClock, **overrides: int) -> RateGate:
    return RateGate(clock, **overrides)


class SendBudgetTests(unittest.TestCase):
    """B) the shared send gate: 20 sends a minute, the 21st waits for the oldest one to age out."""

    def test_group_budget_constant_is_twenty(self) -> None:
        self.assertEqual(GROUP_SENDS_PER_MINUTE, 20)
        self.assertEqual(FEED_SENDS_PER_MINUTE, 10)

    def test_twenty_sends_in_a_minute_need_no_wait(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        for _ in range(GROUP_SENDS_PER_MINUTE):
            gate.slot()
        self.assertEqual(clock.sleeps, [])

    def test_twenty_first_send_waits_for_the_oldest_to_age_out(self) -> None:
        """Ported labels:
            B: 21st send in a minute waits for a slot ([59.99998474121094])"""
        clock = FakeClock()
        gate = _gate(clock)
        for _ in range(GROUP_SENDS_PER_MINUTE):
            gate.slot()
        gate.slot()
        self.assertEqual(clock.sleeps, [60.0])

    def test_wait_is_only_as_long_as_the_oldest_send_needs(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        for _ in range(GROUP_SENDS_PER_MINUTE):
            gate.slot()
        clock.advance(45)
        gate.slot()
        self.assertEqual(clock.sleeps, [15.0])

    def test_sends_older_than_a_minute_are_forgotten(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        for _ in range(GROUP_SENDS_PER_MINUTE):
            gate.slot()
        clock.advance(60)
        for _ in range(GROUP_SENDS_PER_MINUTE):
            gate.slot()
        self.assertEqual(clock.sleeps, [])

    def test_budget_refills_as_the_window_slides(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        for _ in range(GROUP_SENDS_PER_MINUTE):
            gate.slot()
        gate.slot()  # waits 60 s for the first stamp to age out
        for _ in range(GROUP_SENDS_PER_MINUTE - 1):
            gate.slot()  # the window has room again: these 19 sends need no wait
        self.assertEqual(clock.sleeps, [60.0])

    def test_custom_group_budget(self) -> None:
        clock = FakeClock()
        gate = _gate(clock, group_per_min=4)
        for _ in range(4):
            gate.slot()
        self.assertEqual(clock.sleeps, [])
        gate.slot()
        self.assertEqual(clock.sleeps, [60.0])


class FeedShareTests(unittest.TestCase):
    """E) the live feed keeps to its share: answers, cards and replies still find room in the budget."""

    def test_feed_may_use_ten_sends_without_waiting(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        for _ in range(FEED_SENDS_PER_MINUTE):
            gate.slot(feed=True)
        self.assertEqual(clock.sleeps, [])

    def test_feed_waits_once_its_share_is_used(self) -> None:
        """Ported labels:
            E: a feed send waits once its share is used (1.50 s)"""
        clock = FakeClock()
        gate = _gate(clock)
        for _ in range(FEED_SENDS_PER_MINUTE):
            gate.slot(feed=True)
        gate.slot(feed=True)
        self.assertEqual(clock.sleeps, [60.0])

    def test_reply_is_not_held_back_while_the_feed_share_is_used(self) -> None:
        """Ported labels:
            E: a reply gets a slot while the feed share is used (0.00 s)"""
        clock = FakeClock()
        gate = _gate(clock)
        for _ in range(FEED_SENDS_PER_MINUTE):
            gate.slot(feed=True)
        gate.slot()
        self.assertEqual(clock.sleeps, [])

    def test_feed_and_replies_share_one_window(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        for _ in range(FEED_SENDS_PER_MINUTE):
            gate.slot(feed=True)
        for _ in range(GROUP_SENDS_PER_MINUTE - FEED_SENDS_PER_MINUTE):
            gate.slot()
        self.assertEqual(clock.sleeps, [])
        gate.slot()
        self.assertEqual(clock.sleeps, [60.0])

    def test_feed_waits_when_group_budget_is_full_even_below_its_share(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        for _ in range(GROUP_SENDS_PER_MINUTE):
            gate.slot()
        gate.slot(feed=True)
        self.assertEqual(clock.sleeps, [60.0])

    def test_custom_feed_budget(self) -> None:
        clock = FakeClock()
        gate = _gate(clock, group_per_min=6, feed_per_min=2)
        gate.slot(feed=True)
        gate.slot(feed=True)
        self.assertEqual(clock.sleeps, [])
        gate.slot(feed=True)
        self.assertEqual(clock.sleeps, [60.0])


class FloodWaitTests(unittest.TestCase):
    """A) a 429 waits the server's retry_after before the next send; the gate is closed for everyone."""

    def test_flood_wait_returns_the_retry_after_seconds(self) -> None:
        gate = _gate(FakeClock())
        self.assertEqual(gate.flood_wait(30), 30)

    def test_next_send_sleeps_out_the_retry_after(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        gate.flood_wait(30)
        gate.slot()
        self.assertEqual(clock.sleeps, [30.0])
        gate.slot()
        self.assertEqual(clock.sleeps, [30.0])

    def test_flood_closes_the_gate_for_feed_sends_too(self) -> None:
        """Ported labels:
            B: 429 closes the shared gate for 30 s

        Ported labels:
        M: the flood gate holds a feed slot while a 429 is in force
        """
        clock = FakeClock()
        gate = _gate(clock)
        gate.flood_wait(12)
        gate.slot(feed=True)
        self.assertEqual(clock.sleeps, [12.0])

    def test_flood_closes_the_gate_for_replies_too(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        gate.flood_wait(12)
        gate.slot()
        self.assertEqual(clock.sleeps, [12.0])

    def test_flood_retry_after_is_capped(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        self.assertEqual(gate.flood_wait(10_000), RETRY_AFTER_MAX)
        gate.slot()
        self.assertEqual(clock.sleeps, [float(RETRY_AFTER_MAX)])

    def test_custom_retry_after_cap(self) -> None:
        self.assertEqual(_gate(FakeClock(), retry_after_max=45).flood_wait(120), 45)

    def test_missing_or_zero_retry_after_counts_as_thirty_seconds(self) -> None:
        gate = _gate(FakeClock())
        self.assertEqual(gate.flood_wait(None), DEFAULT_FLOOD_WAIT)
        self.assertEqual(gate.flood_wait(0), DEFAULT_FLOOD_WAIT)
        self.assertEqual(DEFAULT_FLOOD_WAIT, 30)

    def test_negative_retry_after_is_at_least_one_second(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        self.assertEqual(gate.flood_wait(-5), 1)
        gate.slot()
        self.assertEqual(clock.sleeps, [1.0])

    def test_a_shorter_later_flood_never_shortens_the_gate(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        gate.flood_wait(30)
        gate.flood_wait(5)
        gate.slot()
        self.assertEqual(clock.sleeps, [30.0])

    def test_a_longer_later_flood_extends_the_gate(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        gate.flood_wait(5)
        clock.advance(2)
        gate.flood_wait(30)
        gate.slot()
        self.assertEqual(clock.sleeps, [30.0])

    def test_flood_during_a_full_budget_waits_for_both(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        start = clock.now()
        for _ in range(GROUP_SENDS_PER_MINUTE):
            gate.slot()
        gate.flood_wait(10)
        gate.slot()
        # first the 10 s flood gate, then the rest of the minute the oldest send needs
        self.assertEqual(clock.sleeps, [10.0, 50.0])
        self.assertEqual(clock.now() - start, 60.0)

    def test_flood_wait_does_not_consume_budget(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        gate.flood_wait(5)
        clock.advance(5)
        for _ in range(GROUP_SENDS_PER_MINUTE):
            gate.slot()
        self.assertEqual(clock.sleeps, [])


class ValidationTests(unittest.TestCase):
    def test_budgets_must_be_positive_integers(self) -> None:
        clock = FakeClock()
        for kwargs in ({"group_per_min": 0}, {"feed_per_min": -1}, {"retry_after_max": 0}, {"group_per_min": 2.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RateGate(clock, **kwargs)  # type: ignore[arg-type]


class ConcurrencyTests(unittest.TestCase):
    def test_parallel_senders_never_exceed_the_window(self) -> None:
        clock = FakeClock()
        gate = _gate(clock)
        start = clock.now()
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                for _ in range(4):
                    gate.slot()
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]  # 24 sends in total
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertTrue(all(not t.is_alive() for t in threads))
        # 24 sends with a budget of 20 a minute: the 21st can only go out once the first one is a minute old
        self.assertGreaterEqual(clock.now() - start, 60.0)


if __name__ == "__main__":
    unittest.main()
