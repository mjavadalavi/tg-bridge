"""inbound, a late update for a topic being removed (card F4): the removal forgets the topic, so its row is None when the
update arrives. The guard in continue_message must drop the update (inbox row done, no row created, no runner) before any
session is ensured, and a topic whose removal was refused (mark cleared) is created as usual.
"""

from tg_bridge.bot.inbound import Disposition
from tests.inbound_support import CHAT, THREAD, TOPIC, InboundCase


class RemovingTests(InboundCase):
    """A late update for a topic being removed must not re-create its row (the removal forgets it, so the row is None)."""

    UPDATE_ID = 7701

    def test_a_late_update_for_a_removed_topic_creates_no_row_and_its_inbox_row_is_done(self) -> None:
        self.name_topic(THREAD, TOPIC)
        self.assertTrue(self.inbox.ingest(self.UPDATE_ID, {"update_id": self.UPDATE_ID}, thread_id=THREAD,
                                          chat_id=CHAT, sender="tester"))
        self.removals.begin(THREAD)
        self.assertIs(self.handle(self.message("hi"), update_id=self.UPDATE_ID), Disposition.DONE)
        self.assertIsNone(self.row())
        self.assertEqual(self.starts.started, [])
        self.assertEqual(self.sent(), [])
        self.assertEqual(self.inbox.get(self.UPDATE_ID).state, "done")
        self.assertTrue(self.logged(f"late message for a topic being removed: dropped (thread {THREAD})"))

    def test_after_the_removal_is_forgotten_and_its_mark_is_gone_the_topic_is_created_again(self) -> None:
        self.name_topic(THREAD, TOPIC)
        self.removals.begin(THREAD)
        self.removals.cancel(THREAD)  # a refused removal clears the mark
        self.assertIs(self.handle(self.message("hi")), Disposition.DEFERRED)
        self.assertEqual(self.row().name, TOPIC)
