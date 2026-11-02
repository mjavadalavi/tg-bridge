"""SlotPool and Slot: port of the slot scenarios of fakecli/test_slot_release.py (labels kept in docstrings).

Scenario 1 (parked topics free their slot) and scenario 2 (every slot taken: the queued turn says so once, then runs
when a permit frees up) run here against real threads, a FakeTelegram and a real Messenger. Turns are simulated by
Slot objects; the dispatcher that drives them is ported with core/dispatcher.py.
"""

import os
import random
import threading
import time
import unittest

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.core.slots import SLOT_COUNT, Slot, SlotPool
from tg_bridge.logging_ import Logger
from tg_bridge.telegram.messenger import Messenger

CHAT = -1001
TOPIC_A, TOPIC_B, TOPIC_C = 9301, 9302, 9303
POLL = 0.01  # queued take() checks for shutdown every 10 ms in tests
WAIT = 5.0
NOTICE_FRAGMENT = "waiting for a free slot"


class Harness:
    """A SlotPool over a Messenger on a FakeTelegram. shutdown is an Event the test sets when it wants a stop."""

    def __init__(self, home: TempHome, size: int = SLOT_COUNT) -> None:
        self.tg = FakeTelegram(clock=FakeClock())
        self.messenger = Messenger(self.tg, Logger(os.path.join(home.state, "bridge.log")), FakeClock())
        self.shutdown = threading.Event()
        self.pool = SlotPool(size, self.messenger, self.shutdown.is_set, poll_seconds=POLL)

    def slot(self, thread_id: int) -> Slot:
        return self.pool.new_slot(CHAT, thread_id)

    def notices(self, thread_id: int) -> list[str]:
        return [t for t in self.tg.texts(thread_id) if NOTICE_FRAGMENT in t]


def _in_thread(fn) -> tuple[threading.Thread, list[object]]:
    """Run fn on its own thread; the result lands in the returned list when it finishes."""
    box: list[object] = []
    th = threading.Thread(target=lambda: box.append(fn()), daemon=True)
    th.start()
    return th, box


def _wait_until(pred, timeout: float = WAIT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return False


class PoolBasicsTests(unittest.TestCase):
    def test_size_below_one_is_refused(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            with self.assertRaises(ValueError):
                SlotPool(0, h.messenger, h.shutdown.is_set)

    def test_free_permits_are_taken_without_a_notice(self) -> None:
        """A free permit is taken at once and nobody is told anything."""
        with TempHome() as home:
            h = Harness(home)
            first, second = h.slot(TOPIC_A), h.slot(TOPIC_B)
            self.assertTrue(first.take())
            self.assertTrue(second.take())
            self.assertEqual(h.pool.free(), 0)
            self.assertEqual(h.tg.texts(TOPIC_A), [])
            self.assertEqual(h.tg.texts(TOPIC_B), [])

    def test_take_is_idempotent_for_one_slot(self) -> None:
        """A second take() on a slot that already holds its permit does not take a second permit."""
        with TempHome() as home:
            h = Harness(home)
            slot = h.slot(TOPIC_A)
            self.assertTrue(slot.take())
            self.assertTrue(slot.take())
            self.assertEqual(h.pool.free(), SLOT_COUNT - 1)
            self.assertTrue(slot.held)

    def test_slot_count(self) -> None:
        self.assertEqual(SLOT_COUNT, 2)


class QueuedTurnTests(unittest.TestCase):
    def test_busy_bridge_says_so_once_then_runs_when_a_slot_frees(self) -> None:
        """2: every slot taken: the queued turn says so once, runs only after a permit frees, permits balance.

        Ported labels:
            slot freed: queued turn ran
            permits balanced after the queued turn
        """
        with TempHome() as home:
            h = Harness(home)
            holders = [h.slot(TOPIC_A), h.slot(TOPIC_B)]
            for holder in holders:
                self.assertTrue(holder.take())

            waiter = h.slot(TOPIC_C)
            th, result = _in_thread(waiter.take)
            self.assertTrue(_wait_until(lambda: len(h.notices(TOPIC_C)) == 1),
                            "the queued topic was never told it is waiting")
            time.sleep(POLL * 20)  # several poll cycles: the notice must not repeat
            self.assertEqual(len(h.notices(TOPIC_C)), 1)
            self.assertTrue(th.is_alive(), "the queued turn ran before a permit was free")
            self.assertEqual(result, [])
            self.assertEqual(h.notices(TOPIC_A), [])
            self.assertEqual(h.notices(TOPIC_B), [])

            holders[0].give_back()
            th.join(WAIT)
            self.assertFalse(th.is_alive())
            self.assertEqual(result, [True])
            self.assertTrue(waiter.held)
            self.assertEqual(h.pool.free(), 0)

            holders[1].give_back()
            waiter.give_back()
            self.assertEqual(h.pool.free(), SLOT_COUNT)

    def test_notice_names_the_pool_size(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            holders = [h.slot(TOPIC_A), h.slot(TOPIC_B)]
            for holder in holders:
                holder.take()
            th, _ = _in_thread(h.slot(TOPIC_C).take)
            self.assertTrue(_wait_until(lambda: len(h.notices(TOPIC_C)) == 1))
            self.assertIn(f"{SLOT_COUNT} topics are busy", h.notices(TOPIC_C)[0])
            holders[0].give_back()
            th.join(WAIT)

    def test_shutdown_stops_a_queued_wait_and_leaks_no_permit(self) -> None:
        """Shutdown while queued: take() returns False, no permit is taken, the holders' permits are untouched."""
        with TempHome() as home:
            h = Harness(home)
            holders = [h.slot(TOPIC_A), h.slot(TOPIC_B)]
            for holder in holders:
                holder.take()
            waiter = h.slot(TOPIC_C)
            th, result = _in_thread(waiter.take)
            self.assertTrue(_wait_until(lambda: len(h.notices(TOPIC_C)) == 1))
            h.shutdown.set()
            th.join(WAIT)
            self.assertFalse(th.is_alive())
            self.assertEqual(result, [False])
            self.assertFalse(waiter.held)
            self.assertEqual(h.pool.free(), 0)
            for holder in holders:
                holder.give_back()
            self.assertEqual(h.pool.free(), SLOT_COUNT)

    def test_shutdown_does_not_block_a_free_permit(self) -> None:
        """Parity with the live acquire(timeout=1) loop: a permit that is free is taken even during shutdown."""
        with TempHome() as home:
            h = Harness(home)
            h.shutdown.set()
            slot = h.slot(TOPIC_A)
            self.assertTrue(slot.take())
            self.assertTrue(slot.held)


class ParkedTurnTests(unittest.TestCase):
    def test_parked_topics_free_their_slot(self) -> None:
        """1: two turns parked on a card give their permits back; a third topic runs at once and is not told to wait."""
        with TempHome() as home:
            h = Harness(home)
            parked_a, parked_b = h.slot(TOPIC_A), h.slot(TOPIC_B)
            parked_a.take()
            parked_b.take()
            parked_a.give_back()
            parked_b.give_back()
            self.assertEqual(h.pool.free(), SLOT_COUNT)

            other = h.slot(TOPIC_C)
            t0 = time.monotonic()
            th, result = _in_thread(other.take)
            th.join(WAIT)
            self.assertLess(time.monotonic() - t0, 2.0)
            self.assertEqual(result, [True])
            self.assertEqual(h.notices(TOPIC_C), [])
            other.give_back()

            self.assertTrue(parked_a.retake())
            self.assertTrue(parked_b.retake())
            self.assertEqual(h.pool.free(), 0)
            parked_a.give_back()
            parked_b.give_back()
            self.assertEqual(h.pool.free(), SLOT_COUNT)

    def test_give_back_releases_exactly_once(self) -> None:
        """Two give_back() calls release one permit only: no over-release, the pool never grows past its size."""
        with TempHome() as home:
            h = Harness(home)
            slot = h.slot(TOPIC_A)
            slot.take()
            slot.give_back()
            slot.give_back()
            self.assertEqual(h.pool.free(), SLOT_COUNT)
            self.assertFalse(slot.held)
            self.assertTrue(h.pool.try_acquire_permit())
            self.assertTrue(h.pool.try_acquire_permit())
            self.assertFalse(h.pool.try_acquire_permit())
            h.pool.release_permit()
            h.pool.release_permit()

    def test_retake_is_non_blocking_and_reports_busy_pool(self) -> None:
        """retake() on a free pool takes the permit; with every permit busy it returns False and holds nothing."""
        with TempHome() as home:
            h = Harness(home)
            parked = h.slot(TOPIC_A)
            parked.take()
            parked.give_back()
            busy = [h.slot(TOPIC_B), h.slot(TOPIC_C)]
            for holder in busy:
                holder.take()

            self.assertFalse(parked.retake())
            self.assertFalse(parked.held)
            self.assertEqual(h.pool.free(), 0)

            busy[0].give_back()
            self.assertTrue(parked.retake())
            self.assertTrue(parked.held)
            self.assertEqual(h.pool.free(), 0)

    def test_retake_while_holding_takes_no_second_permit(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            slot = h.slot(TOPIC_A)
            slot.take()
            self.assertTrue(slot.retake())
            self.assertEqual(h.pool.free(), SLOT_COUNT - 1)


class TimedTakeTests(unittest.TestCase):
    """take_for: one timed slice of the wait. It neither announces the queue nor checks shutdown; the caller does."""

    def test_take_for_gives_up_after_its_slice_when_every_permit_is_busy(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            holders = [h.slot(TOPIC_B), h.slot(TOPIC_C)]
            for holder in holders:
                holder.take()
            waiter = h.slot(TOPIC_A)
            self.assertFalse(waiter.take_for(POLL))
            self.assertFalse(waiter.held)
            self.assertEqual(h.pool.free(), 0)
            self.assertEqual(h.notices(TOPIC_A), [], "a timed slice must not announce the queue")

    def test_take_for_gets_a_permit_released_during_its_slice(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            holders = [h.slot(TOPIC_B), h.slot(TOPIC_C)]
            for holder in holders:
                holder.take()
            waiter = h.slot(TOPIC_A)
            releaser = threading.Timer(POLL / 2, holders[0].give_back)
            releaser.start()
            self.assertTrue(waiter.take_for(WAIT))
            releaser.join()
            self.assertTrue(waiter.held)
            self.assertEqual(h.pool.free(), 0)

    def test_take_for_while_holding_takes_no_second_permit(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            slot = h.slot(TOPIC_A)
            slot.take()
            self.assertTrue(slot.take_for(POLL))
            self.assertEqual(h.pool.free(), SLOT_COUNT - 1)


class BalanceTests(unittest.TestCase):
    def test_permits_balanced_after_many_cycles(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            slots = [h.slot(TOPIC_A), h.slot(TOPIC_B)]
            for _ in range(50):
                for slot in slots:
                    self.assertTrue(slot.take())
                for slot in slots:
                    slot.give_back()
            self.assertEqual(h.pool.free(), SLOT_COUNT)

    def test_concurrent_turns_never_hold_more_than_the_pool_size(self) -> None:
        """Eight topics take and give back permits at random; the number held at once never exceeds SLOT_COUNT."""
        with TempHome() as home:
            h = Harness(home)
            lock = threading.Lock()
            holding, peak = [0], [0]
            errors: list[BaseException] = []

            def topic(thread_id: int) -> None:
                rng = random.Random(thread_id)
                slot = h.slot(thread_id)
                try:
                    for _ in range(20):
                        self.assertTrue(slot.take())
                        with lock:
                            holding[0] += 1
                            peak[0] = max(peak[0], holding[0])
                        time.sleep(rng.random() * 0.002)
                        with lock:
                            holding[0] -= 1
                        slot.give_back()
                except BaseException as exc:  # surfaced after join: a thread must not die silently
                    errors.append(exc)

            threads = [threading.Thread(target=topic, args=(TOPIC_A + i,), daemon=True) for i in range(8)]
            for th in threads:
                th.start()
            for th in threads:
                th.join(WAIT * 4)
                self.assertFalse(th.is_alive())
            self.assertEqual(errors, [])
            self.assertLessEqual(peak[0], SLOT_COUNT)
            self.assertEqual(h.pool.free(), SLOT_COUNT)


if __name__ == "__main__":
    unittest.main()
