"""TopicQueues quiet delivery and keep-busy head (card C8a of the linger design).

A quiet message is one that a kept, idle session takes straight away: when the topic is busy it is HANDED (no queued
notice, no buttons, no new runner), and a quiet reserve is filled without a notice. A kept session keeps its topic
busy when the queue runs empty (take_head keep_busy), so a message that arrives meanwhile is handed over, not started.
The queue holds data only, so these tests run without threads.
"""

import unittest

from tg_bridge.core.topic_queues import (
    BURST_FOOTER, MERGE_MAX, Admission, Placeholder, TopicQueues, _HeadKind, burst_text,
)

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


class BatchTakeTests(unittest.TestCase):
    """take_head on a busy topic merges the head and the ready messages behind it into one batch (READY)."""

    def setUp(self) -> None:
        self.queues = TopicQueues()
        self.assertIs(self.queues.deliver(THREAD, CHAT, SENDER, 1, "running").admission, Admission.STARTED)
        self.assertEqual(self.queues.take_head(THREAD).text, "running")

    def test_three_burst_entries_are_one_head_with_parts_footer_and_images(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 11, "bug A", images=("/tmp/a.png",))
        self.queues.deliver(THREAD, CHAT, SENDER, 12, "bug B", burst=True)
        self.queues.deliver(THREAD, CHAT, SENDER, 13, "bug C", images=("/tmp/c1.png", "/tmp/c2.png"), burst=True)

        head = self.queues.take_head(THREAD)

        self.assertEqual(head.kind.value, "ready")
        expected = ("bug A\n\n"
                    "--- message 2/3, sent together with the one above ---\nbug B\n\n"
                    "--- message 3/3, sent together with the one above ---\nbug C\n\n"
                    f"{BURST_FOOTER.format(count=3)}")
        self.assertEqual(head.text, expected)
        self.assertEqual(head.images, ("/tmp/a.png", "/tmp/c1.png", "/tmp/c2.png"))
        self.assertEqual([e.update_id for e in head.entries], [11, 12, 13])
        self.assertEqual(head.entry.update_id, 11)
        self.assertEqual(self.queues.snapshot(THREAD), ())
        self.assertEqual(self.queues.take_head(THREAD, keep_busy=True).kind.value, "empty")

    def test_the_merged_text_starts_with_the_first_text_and_ends_with_the_footer(self) -> None:
        self.assertEqual(burst_text(["one", "two", "three"]).split("\n\n")[0], "one")
        self.assertTrue(burst_text(["one", "two", "three"]).endswith(BURST_FOOTER.format(count=3)))
        self.assertEqual(burst_text(["solo"]), "solo")

    def test_first_text_is_the_first_entrys_own_text(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 41, "topic title source", burst=False)
        self.queues.deliver(THREAD, CHAT, SENDER, 42, "second", burst=True)

        head = self.queues.take_head(THREAD)

        self.assertEqual(head.first_text, "topic title source")
        self.assertNotEqual(head.first_text, head.text)

    def test_single_entry_text_and_images_are_unchanged(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 21, "just one", images=("/tmp/one.png",))

        head = self.queues.take_head(THREAD)

        self.assertEqual(head.text, "just one")
        self.assertEqual(head.first_text, "just one")
        self.assertEqual(head.images, ("/tmp/one.png",))
        self.assertEqual(len(head.entries), 1)
        self.assertIs(head.entry, head.entries[0])

    def test_only_burst_followers_merge_into_the_head(self) -> None:
        """A message sent minutes later (burst=False) is not merged, even though it is ready and queued behind."""
        self.queues.deliver(THREAD, CHAT, SENDER, 51, "first")
        self.queues.deliver(THREAD, CHAT, SENDER, 52, "minutes later")
        self.queues.deliver(THREAD, CHAT, SENDER, 53, "part of that burst", burst=True)

        first = self.queues.take_head(THREAD)
        self.assertEqual(first.text, "first")
        self.assertEqual(len(first.entries), 1)

        second = self.queues.take_head(THREAD)
        self.assertEqual(second.text, burst_text(["minutes later", "part of that burst"]))

    def test_a_slash_command_head_runs_alone(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 61, "/compact", burst=True)
        self.queues.deliver(THREAD, CHAT, SENDER, 62, "later", burst=True)

        command = self.queues.take_head(THREAD)
        self.assertEqual(command.text, "/compact")
        self.assertEqual(len(command.entries), 1)

        self.assertEqual(self.queues.take_head(THREAD).text, "later")

    def test_a_slash_command_follower_ends_the_batch(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 71, "one")
        self.queues.deliver(THREAD, CHAT, SENDER, 72, "  /x", burst=True)
        self.queues.deliver(THREAD, CHAT, SENDER, 73, "two", burst=True)

        self.assertEqual(self.queues.take_head(THREAD).text, "one")
        self.assertEqual(self.queues.take_head(THREAD).text, "  /x")
        self.assertEqual(self.queues.take_head(THREAD).text, "two")

    def test_a_bash_mode_bang_runs_alone_as_head_and_as_follower(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 81, "!ls")
        self.queues.deliver(THREAD, CHAT, SENDER, 82, "after the ls", burst=True)
        self.assertEqual(self.queues.take_head(THREAD).text, "!ls")
        self.assertEqual(self.queues.take_head(THREAD).text, "after the ls")

        self.queues.deliver(THREAD, CHAT, SENDER, 83, "before the ls")
        self.queues.deliver(THREAD, CHAT, SENDER, 84, "!ls", burst=True)
        self.assertEqual(self.queues.take_head(THREAD).text, "before the ls")
        self.assertEqual(self.queues.take_head(THREAD).text, "!ls")

    def test_an_unfilled_burst_placeholder_behind_the_head_is_waited_for_and_nothing_is_popped(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 91, "a")
        voice = self.queues.reserve(THREAD, CHAT, SENDER, 92, burst=True)

        head = self.queues.take_head(THREAD)

        self.assertIs(head.kind, _HeadKind.WAIT)
        self.assertEqual(head.entry.nid, voice.nid)
        self.assertIsInstance(head.entry.body, Placeholder)
        self.assertEqual([item.text for item in self.queues.snapshot(THREAD)], ["a", None])

        self.queues.deliver(THREAD, CHAT, SENDER, 92, "voice text", voice.nid)
        merged = self.queues.take_head(THREAD)
        self.assertEqual(merged.text, burst_text(["a", "voice text"]))
        self.assertEqual([e.update_id for e in merged.entries], [91, 92])

    def test_a_non_burst_placeholder_behind_the_head_does_not_block(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 101, "a")
        self.queues.reserve(THREAD, CHAT, SENDER, 102)  # sent minutes later: not a follower

        head = self.queues.take_head(THREAD)

        self.assertIs(head.kind, _HeadKind.READY)
        self.assertEqual(head.text, "a")
        self.assertEqual([item.text for item in self.queues.snapshot(THREAD)], [None])

    def test_batch_stops_at_an_unfilled_placeholder_which_stays_the_head(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 31, "a")
        self.queues.deliver(THREAD, CHAT, SENDER, 32, "b", burst=True)
        voice = self.queues.reserve(THREAD, CHAT, SENDER, 33, burst=True)
        self.queues.deliver(THREAD, CHAT, SENDER, 34, "c", burst=True)

        waiting = self.queues.take_head(THREAD)
        self.assertEqual(waiting.kind.value, "wait")
        self.assertEqual(waiting.entry.nid, voice.nid)
        self.assertEqual(waiting.entries, ())
        self.assertEqual([item.text for item in self.queues.snapshot(THREAD)], ["a", "b", None, "c"])

        self.queues.deliver(THREAD, CHAT, SENDER, 33, "voice text", voice.nid)
        filled = self.queues.take_head(THREAD)
        self.assertEqual(filled.text, burst_text(["a", "b", "voice text", "c"]))
        self.assertEqual([e.update_id for e in filled.entries], [31, 32, 33, 34])
        self.assertIsInstance(filled.entries[2].body, Placeholder)
        self.assertTrue(filled.entries[2].body.ev.is_set())

    def test_a_filled_placeholder_keeps_its_burst_flag(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 111, "a")
        voice = self.queues.reserve(THREAD, CHAT, SENDER, 112, burst=True)
        filled = self.queues.deliver(THREAD, CHAT, SENDER, 112, "voice", voice.nid)
        self.assertIs(filled.admission, Admission.FILLED)
        self.assertTrue(filled.burst)

    def test_merge_is_bounded_by_merge_max(self) -> None:
        total = MERGE_MAX + 5
        for i in range(total):
            self.assertIs(self.queues.deliver(THREAD, CHAT, SENDER, 100 + i, f"m{i}", burst=True).admission,
                          Admission.QUEUED)

        first = self.queues.take_head(THREAD)
        second = self.queues.take_head(THREAD)

        self.assertEqual(len(first.entries), MERGE_MAX)
        self.assertEqual(len(second.entries), 5)
        self.assertEqual(first.text, burst_text([f"m{i}" for i in range(MERGE_MAX)]))
        self.assertEqual(second.text, burst_text([f"m{i}" for i in range(MERGE_MAX, total)]))


class GroupTests(unittest.TestCase):
    """A burst is a group identity: moves, drops, releases and refusals never merge two groups."""

    def setUp(self) -> None:
        self.queues = TopicQueues()
        self.queues.deliver(THREAD, CHAT, SENDER, 1, "running")
        self.queues.take_head(THREAD)

    def texts(self) -> list[str | None]:
        return [item.text for item in self.queues.snapshot(THREAD)]

    def test_now_on_a_burst_member_moves_the_whole_group_in_order(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 11, "a")
        self.queues.deliver(THREAD, CHAT, SENDER, 12, "b", burst=True)
        self.queues.deliver(THREAD, CHAT, SENDER, 13, "c")
        d = self.queues.deliver(THREAD, CHAT, SENDER, 14, "d", burst=True)

        self.assertTrue(self.queues.to_front(THREAD, d.nid))

        self.assertEqual(self.texts(), ["c", "d", "a", "b"])
        self.assertEqual(self.queues.take_head(THREAD).text, burst_text(["c", "d"]))
        self.assertEqual(self.queues.take_head(THREAD).text, burst_text(["a", "b"]))

    def test_now_on_a_lone_message_moves_only_it_and_no_group_is_split(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 21, "a")
        self.queues.deliver(THREAD, CHAT, SENDER, 22, "b", burst=True)
        lone = self.queues.deliver(THREAD, CHAT, SENDER, 23, "lone")

        self.assertTrue(self.queues.to_front(THREAD, lone.nid))

        self.assertEqual(self.queues.take_head(THREAD).text, "lone")
        self.assertEqual(self.queues.take_head(THREAD).text, burst_text(["a", "b"]))

    def test_dropping_a_group_lead_does_not_merge_its_follower_into_the_group_before(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 31, "a")
        lead = self.queues.deliver(THREAD, CHAT, SENDER, 32, "b")
        self.queues.deliver(THREAD, CHAT, SENDER, 33, "c", burst=True)

        self.assertIsNotNone(self.queues.drop(THREAD, lead.nid))

        self.assertEqual(self.queues.take_head(THREAD).text, "a")
        self.assertEqual(self.queues.take_head(THREAD).text, "c")

    def test_a_released_placeholder_does_not_merge_two_groups(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 41, "a")
        voice = self.queues.reserve(THREAD, CHAT, SENDER, 42)
        self.queues.deliver(THREAD, CHAT, SENDER, 43, "after the voice", burst=True)

        self.assertTrue(self.queues.unreserve(THREAD, voice.nid))

        self.assertEqual(self.queues.take_head(THREAD).text, "a")
        self.assertEqual(self.queues.take_head(THREAD).text, "after the voice")

    def test_a_refusal_ends_the_chain(self) -> None:
        queues = TopicQueues(max_items=2)
        queues.deliver(THREAD, CHAT, SENDER, 1, "running")
        queues.take_head(THREAD)
        queues.deliver(THREAD, CHAT, SENDER, 51, "a")
        b = queues.deliver(THREAD, CHAT, SENDER, 52, "b", burst=True)
        self.assertIs(queues.deliver(THREAD, CHAT, SENDER, 53, "refused", burst=True).admission, Admission.FULL)
        queues.drop(THREAD, b.nid)

        queues.deliver(THREAD, CHAT, SENDER, 54, "after the refusal", burst=True)

        self.assertEqual(queues.take_head(THREAD).text, "a")
        self.assertEqual(queues.take_head(THREAD).text, "after the refusal")

    def test_a_wait_on_a_member_behind_the_head_names_the_head_as_lead(self) -> None:
        a = self.queues.deliver(THREAD, CHAT, SENDER, 61, "a")
        voice = self.queues.reserve(THREAD, CHAT, SENDER, 62, burst=True)

        head = self.queues.take_head(THREAD)

        self.assertIs(head.kind, _HeadKind.WAIT)
        self.assertEqual((head.entry.nid, head.lead), (voice.nid, a.nid))
        self.assertEqual(self.queues.head_nid(THREAD), a.nid)

    def test_a_wait_on_the_head_itself_names_it_as_lead(self) -> None:
        voice = self.queues.reserve(THREAD, CHAT, SENDER, 71)

        head = self.queues.take_head(THREAD)

        self.assertEqual((head.entry.nid, head.lead), (voice.nid, voice.nid))

    def test_a_follower_after_a_refusal_is_announced_as_a_new_message(self) -> None:
        queues = TopicQueues(max_items=2)
        queues.deliver(THREAD, CHAT, SENDER, 1, "running")
        queues.take_head(THREAD)
        queues.deliver(THREAD, CHAT, SENDER, 81, "a")
        b = queues.deliver(THREAD, CHAT, SENDER, 82, "b", burst=True)
        self.assertTrue(b.burst)
        queues.deliver(THREAD, CHAT, SENDER, 83, "refused", burst=True)
        queues.drop(THREAD, b.nid)

        after = queues.deliver(THREAD, CHAT, SENDER, 84, "c", burst=True)

        self.assertFalse(after.burst, "a new group: its queued notice and buttons are sent")

    def test_a_follower_of_a_burst_whose_turn_was_taken_starts_its_own_group(self) -> None:
        self.queues.deliver(THREAD, CHAT, SENDER, 91, "a")
        self.queues.deliver(THREAD, CHAT, SENDER, 92, "b", burst=True)
        self.queues.take_head(THREAD)

        late = self.queues.deliver(THREAD, CHAT, SENDER, 93, "late part", burst=True)
        voice = self.queues.reserve(THREAD, CHAT, SENDER, 94, burst=True)

        self.assertFalse(late.burst, "nothing of its burst is queued: it is announced")
        self.assertTrue(voice.burst, "it joins the late part")
        self.queues.deliver(THREAD, CHAT, SENDER, 94, "voice", voice.nid)
        self.assertEqual(self.queues.take_head(THREAD).text, burst_text(["late part", "voice"]))

    def test_head_nid_of_an_empty_queue_is_none(self) -> None:
        self.assertIsNone(self.queues.head_nid(THREAD))


if __name__ == "__main__":
    unittest.main()
