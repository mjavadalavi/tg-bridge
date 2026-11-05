"""TopicQueues quiet delivery and keep-busy head (card C8a of the linger design).

A quiet message is one that a kept, idle session takes straight away: when the topic is busy it is HANDED (no queued
notice, no buttons, no new runner), and a quiet reserve is filled without a notice. A kept session keeps its topic
busy when the queue runs empty (take_head keep_busy), so a message that arrives meanwhile is handed over, not started.
The queue holds data only, so these tests run without threads.
"""

import unittest

from tg_bridge.core.topic_queues import Admission, TopicQueues

THREAD = 7001
CHAT = -100
SENDER = "42"


class QuietDeliveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.queues = TopicQueues()

    def test_quiet_delivery_to_a_busy_topic_is_handed_without_notice(self) -> None:
        first = self.queues.deliver(THREAD, CHAT, SENDER, 1, "first")
        self.assertIs(first.admission, Admission.STARTED)

        quiet = self.queues.deliver(THREAD, CHAT, SENDER, 2, "while kept", quiet=True)
        self.assertIs(quiet.admission, Admission.HANDED)
        self.assertEqual(quiet.position, 2)
        self.assertIsNotNone(quiet.nid)
        self.assertFalse(quiet.notify)
        self.assertEqual([item.text for item in self.queues.snapshot(THREAD)], ["first", "while kept"])

        plain = self.queues.deliver(THREAD, CHAT, SENDER, 3, "plain")
        self.assertIs(plain.admission, Admission.QUEUED)
        self.assertTrue(plain.notify)

    def test_busy_topic_queues_twenty_messages_in_order(self) -> None:
        self.assertIs(self.queues.deliver(THREAD, CHAT, SENDER, 1, "running").admission, Admission.STARTED)
        results = [self.queues.deliver(THREAD, CHAT, SENDER, 10 + i, f"bug {i}") for i in range(20)]
        self.assertEqual([r.admission for r in results], [Admission.QUEUED] * 20)
        self.assertEqual([r.position for r in results], list(range(2, 22)))  # position 1 is the running turn
        self.assertEqual([item.text for item in self.queues.snapshot(THREAD)[1:]], [f"bug {i}" for i in range(20)])

    def test_quiet_is_ignored_for_an_idle_topic(self) -> None:
        idle = self.queues.deliver(THREAD, CHAT, SENDER, 1, "hello", quiet=True)
        self.assertIs(idle.admission, Admission.STARTED)
        self.assertEqual(idle.position, 1)
        self.assertTrue(self.queues.busy(THREAD))
        head = self.queues.take_head(THREAD)
        self.assertEqual(head.text, "hello")

    def test_quiet_reserve_fills_without_notice(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 1, "running")

        quiet_place = self.queues.reserve(THREAD, CHAT, SENDER, 2, quiet=True)
        self.assertIs(quiet_place.admission, Admission.RESERVED)
        filled = self.queues.deliver(THREAD, CHAT, SENDER, 2, "voice text", reserved=quiet_place.nid)
        self.assertIs(filled.admission, Admission.FILLED)
        self.assertFalse(filled.notify)

        loud_place = self.queues.reserve(THREAD, CHAT, SENDER, 3)
        self.assertIs(loud_place.admission, Admission.RESERVED)
        loud = self.queues.deliver(THREAD, CHAT, SENDER, 3, "photo text", reserved=loud_place.nid)
        self.assertIs(loud.admission, Admission.FILLED)
        self.assertTrue(loud.notify)

    def test_full_queue_still_refuses_quiet(self) -> None:
        queues = TopicQueues(max_items=2)
        self.assertIs(queues.deliver(THREAD, CHAT, SENDER, 1, "running").admission, Admission.STARTED)
        self.assertIs(queues.deliver(THREAD, CHAT, SENDER, 2, "queued").admission, Admission.QUEUED)

        self.assertIs(queues.deliver(THREAD, CHAT, SENDER, 3, "quiet", quiet=True).admission, Admission.FULL)
        self.assertIs(queues.reserve(THREAD, CHAT, SENDER, 4, quiet=True).admission, Admission.FULL)
        self.assertEqual(len(queues.snapshot(THREAD)), 2)

    def test_keep_busy_head_keeps_the_topic_busy(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 1, "only")
        self.assertEqual(self.queues.take_head(THREAD).text, "only")

        empty = self.queues.take_head(THREAD, keep_busy=True)
        self.assertEqual(empty.kind.value, "empty")
        self.assertTrue(self.queues.busy(THREAD))

        handed = self.queues.deliver(THREAD, CHAT, SENDER, 2, "next turn", quiet=True)
        self.assertIs(handed.admission, Admission.HANDED)

    def test_empty_head_without_keep_busy_ends_the_busy_state(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 1, "only")
        self.queues.take_head(THREAD)

        self.assertEqual(self.queues.take_head(THREAD).kind.value, "empty")
        self.assertFalse(self.queues.busy(THREAD))
        self.assertIs(self.queues.deliver(THREAD, CHAT, SENDER, 2, "again", quiet=True).admission, Admission.STARTED)


if __name__ == "__main__":
    unittest.main()
