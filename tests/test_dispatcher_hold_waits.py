"""Dispatcher hold during the two waits a head message can block in (K10): the media placeholder wait and the slot wait.

While a topic's kept session is held, its runner keeps ticking the hold during both waits, so self turns, the cap, the
exit and /stop releases run as they do between turns. A hold that ends during a wait leaves the wait to finish as a
plain wait. Time is a FakeClock: every hold tick advances it by TICK_SECONDS, so the 300-second media wait runs in
milliseconds. Everything else is the LingerDispatcherCase of test_dispatcher_linger.py, with the clock and a short poll
slice injected.
"""

import unittest

from tests.test_dispatcher import CHAT, SENDER, THREAD, TOPIC, WAIT, wait_until
from tests.test_dispatcher_linger import CAP_MINUTES, LingerDispatcherCase
from tg_bridge.core.dispatcher import MEDIA_WAIT_SECONDS, Dispatcher
from tg_bridge.core.linger import LINGER_CAP_TEXT, HoldEnd
from tg_bridge.core.topic_queues import Admission
from tg_bridge.store.inbox import STATE_PENDING

TICK_SECONDS = 10.0  # fake time that passes on each hold tick
POLL = 0.01  # the real slice of a media wait and the real pool poll of a slot wait, so the tests stay fast
MEDIA_TIMEOUT_TEXT = "still being processed"
SLOT_WAIT_TEXT = "waiting for a free slot"


class HoldWaitCase(LingerDispatcherCase):
    """LingerDispatcherCase whose hold ticks move the FakeClock, and whose dispatcher reads that clock."""

    def setUp(self) -> None:
        super().setUp()  # builds the dispatcher with self.linger (a StubLinger) and self.clock
        real_tick = self.linger.wait_once

        def ticking(thread_id: int):
            self.clock.advance(TICK_SECONDS)
            return real_tick(thread_id)

        self.linger.wait_once = ticking  # every hold tick moves fake time forward

    def _make_dispatcher(self, media_wait: float | None = None) -> Dispatcher:
        return Dispatcher(
            turns=self.turns, slots=self.slots, inbox=self.inbox, messenger=self.messenger, answers=self.answers,
            titles=self.titles, client=self.tg, live=self.live, config=self.config, logger=self.logger,
            media_wait_seconds=self.media_wait if media_wait is None else media_wait,
            start_thread=self.thread_log, linger=self.linger, hold_failure_pause=0.0, clock=self.clock,
            poll_seconds=POLL)

    def texts_with(self, fragment: str) -> list[str]:
        return [text for text in self.tg.texts(THREAD) if fragment in text]

    def cap_text(self) -> str:
        return LINGER_CAP_TEXT.format(minutes=CAP_MINUTES)


class MediaWaitWhileHeld(HoldWaitCase):
    def test_media_wait_ticks_the_hold_and_times_out_on_the_fake_clock(self) -> None:
        """A placeholder that never fills keeps the held runner ticking for the whole media wait (fake time), then the
        topic is told and the runner stays on the held topic."""
        self.linger.holds = True
        media = self.ingest(4100)
        started_at = self.clock.monotonic()
        self.dispatcher.reserve(THREAD, TOPIC, CHAT, SENDER, media)
        self.assertTrue(wait_until(lambda: self.texts_with(MEDIA_TIMEOUT_TEXT)), "the media wait never timed out")
        self.assertGreaterEqual(self.clock.monotonic() - started_at, MEDIA_WAIT_SECONDS)
        self.assertGreaterEqual(self.linger.calls, int(MEDIA_WAIT_SECONDS / TICK_SECONDS) - 1,
                                "the hold was not ticked during the media wait")
        self.assertEqual(self.dispatcher.queue_snapshot(THREAD), ())
        self.assertEqual(self.state(media), STATE_PENDING)
        self.assertEqual(self.turns.texts_run(THREAD), [])
        self.assertTrue(self.alive(), "the held runner left the topic after the media timeout")
        self.linger.next_end = HoldEnd.DONE
        self.assertTrue(wait_until(lambda: not self.alive()))

    def test_a_cap_during_the_media_wait_ends_the_hold_and_a_fill_after_it_still_runs(self) -> None:
        """The cap is checked during the media wait: its notice is posted then. The hold is gone, so the wait goes on
        as a plain wait, and the placeholder filled after that still runs as the message."""
        self.linger.holds = True
        self.linger.pending = 2
        voice = self.ingest(4200)
        reserved = self.dispatcher.reserve(THREAD, TOPIC, CHAT, SENDER, voice)
        self.assertIs(reserved.admission, Admission.STARTED)
        self.assertTrue(wait_until(lambda: self.linger.calls >= 3), "the media wait did not tick the hold")
        self.linger.next_end = HoldEnd.CAP
        self.assertTrue(wait_until(lambda: self.texts_with(self.cap_text())), "the cap was not checked in the wait")
        self.assertEqual(self.turns.texts_run(THREAD), [], "the placeholder ran before it was filled")
        self.assertTrue(self.alive(), "the runner left the topic before the placeholder was filled")
        self.dispatcher.deliver(THREAD, TOPIC, CHAT, SENDER, voice, "voice text", reserved=reserved.nid)
        self.assertTrue(wait_until(lambda: not self.alive()))
        self.assertEqual(self.turns.texts_run(THREAD), ["voice text"])
        self.assertEqual(self.texts_with(MEDIA_TIMEOUT_TEXT), [], "a filled placeholder timed out")


class SlotWaitWhileHeld(HoldWaitCase):
    def setUp(self) -> None:
        super().setUp()
        self.assertTrue(self.slots.try_acquire_permit())  # both permits are taken by other topics
        self.assertTrue(self.slots.try_acquire_permit())

    def tearDown(self) -> None:
        for _ in range(2):  # the permits of the other topics come back (a no-op if a test already gave them back)
            if self.slots.free() < self.slots.size:
                self.slots.release_permit()
        super().tearDown()

    def test_a_queued_turn_keeps_ticking_the_hold_while_it_waits_for_a_slot(self) -> None:
        """The held topic's message waits for a permit. The hold is ticked during the wait, so a cap is posted while
        no permit is free; the wait then goes on, and the turn runs once a permit comes back."""
        self.linger.holds = True
        self.linger.pending = 2
        self.ingest(1)
        self.deliver(1, "first")
        self.assertTrue(wait_until(lambda: self.linger.calls >= 3), "the slot wait did not tick the hold")
        self.assertEqual(self.turns.texts_run(THREAD), [])
        self.linger.next_end = HoldEnd.CAP
        self.assertTrue(wait_until(lambda: self.texts_with(self.cap_text())), "the cap was not checked in the wait")
        self.assertEqual(self.turns.texts_run(THREAD), [], "the turn ran without a free permit")
        self.assertTrue(self.alive(), "the runner left the topic while the message waited for a slot")
        self.slots.release_permit()
        self.slots.release_permit()
        self.assertTrue(wait_until(lambda: self.turns.texts_run(THREAD) == ["first"]), "the turn never ran")
        self.assertTrue(wait_until(lambda: not self.alive()))
        self.assertIn("answer to first", self.tg.texts(THREAD))

    def test_the_queued_notice_is_sent_once_even_when_the_hold_ends_during_the_wait(self) -> None:
        """A hold that ends while the message waits posts the queued notice once, not again on the plain wait."""
        self.linger.holds = True
        self.ingest(1)
        self.deliver(1, "first")
        self.assertTrue(wait_until(lambda: self.linger.calls >= 2))
        self.linger.next_end = HoldEnd.DONE
        self.assertTrue(wait_until(lambda: not self.linger.holds), "the hold did not end during the wait")
        self.assertTrue(wait_until(lambda: self.texts_with(SLOT_WAIT_TEXT)), "the queued notice was not sent")
        self.assertEqual(len(self.texts_with(SLOT_WAIT_TEXT)), 1)
        self.assertEqual(self.turns.texts_run(THREAD), [])
        self.slots.release_permit()
        self.slots.release_permit()
        self.assertTrue(wait_until(lambda: not self.alive()))
        self.assertEqual(self.turns.texts_run(THREAD), ["first"])
        self.assertEqual(len(self.texts_with(SLOT_WAIT_TEXT)), 1, "the queued notice was sent again")

    def test_shutdown_during_a_held_slot_wait_replays_the_message(self) -> None:
        """A shutdown that comes during the slot wait refuses the message: its row stays pending for the next start."""
        self.linger.holds = True
        self.ingest(1)
        delivery = self.deliver(1, "first")
        self.assertIs(delivery.admission, Admission.STARTED)
        self.assertTrue(wait_until(lambda: self.linger.calls >= 2))
        self.live.begin_shutdown()
        self.assertTrue(wait_until(lambda: not self.alive()), "the slot wait ignored the shutdown")
        self.assertEqual(self.turns.texts_run(THREAD), [])
        self.assertEqual(self.state(1), STATE_PENDING)


if __name__ == "__main__":
    unittest.main()
