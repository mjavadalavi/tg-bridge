"""InboxService (STEP 51): port of the ingest, process, replay and spool scenarios of fakecli/test_inbox.py (labels kept
in the test names and docstrings).

Every test runs in a TempHome with FakeTelegram and FakeClock, so no network, no real clock and no real state directory
are touched. Old rows are made by ingesting at a known clock time and then advancing the clock.
"""

import json
import os
import unittest
from typing import Any

from tests.inbox_support import CHAT, GROUP_CHAT, THREAD, InboxHarness, callback_update, message_update
from tests.support import TempHome
from tg_bridge.bot.inbound import Disposition
from tg_bridge.clock import utc_stamp
from tg_bridge.daemon.inbox import CALLBACK_REPLAY_SECONDS
from tg_bridge.store.kv import KvRepo

DAY = 86400
GROUP_ID = str(GROUP_CHAT["id"])


class SpyKv(KvRepo):
    """Records every write, with the inbox state of the update the value names (for the offset)."""

    def __init__(self, db: Any) -> None:
        super().__init__(db)
        self.writes: list[tuple[str, str, str | None]] = []
        self.probe: Any = None  # set by the test: value -> inbox state of that update

    def set(self, key: str, value: str) -> None:
        state = self.probe(value) if self.probe is not None and key == "offset" else None
        self.writes.append((key, value, state))
        super().set(key, value)


class InboxCase(unittest.TestCase):
    """Opens a harness under a TempHome for each test."""

    kv_class: type[KvRepo] = KvRepo

    def setUp(self) -> None:
        self._home = TempHome()
        self._home.__enter__()
        self.h = InboxHarness(self._home, self.kv_class)

    def tearDown(self) -> None:
        self.h.close()
        self._home.__exit__(None, None, None)

    def _raw_row(self, update_id: int, payload: str, state: str = "pending", created: str | None = None) -> None:
        """Inserts a row the way an old file would have written it (no validation of the payload)."""
        stamp = created or self._stamp()
        self.h.db.execute(
            "INSERT INTO inbox_queue(update_id, thread_id, chat_id, sender, payload, created, state) "
            "VALUES(?,?,?,?,?,?,?)",
            (update_id, THREAD, CHAT, "42", payload, stamp, state),
        )

    def _stamp(self) -> str:
        return utc_stamp(self.h.clock.now())


class IngestTests(InboxCase):
    def test_ingest_stores_pending_and_reports_redelivery(self) -> None:
        """ingest_update: new update reports True, a re-delivery reports False; the row is pending and the offset moved.

        Ported labels:
            ingest alone does not process the update
        """
        update = message_update(880001, 1)
        self.assertTrue(self.h.service.ingest_update(update))
        self.assertEqual(self.h.state(880001), "pending")
        self.assertEqual(self.h.kv.get("offset"), "880001")
        self.assertFalse(self.h.service.ingest_update(update))
        self.assertEqual(self.h.messages.seen, [])  # ingest alone does not process the update

    def test_ingest_stores_thread_chat_and_sender(self) -> None:
        """The inbox row keeps the topic, the chat and the sender id, as the live INSERT does."""
        self.h.service.ingest_update(message_update(880002, 2, thread=777))
        record = self.h.inbox.get(880002)
        assert record is not None
        self.assertEqual((record.thread_id, record.chat_id, record.sender), (777, CHAT, "42"))

    def test_callback_stored_as_pending(self) -> None:
        """A callback tap is stored like a message: pending until processed."""
        self.assertTrue(self.h.service.ingest_update(callback_update(880003)))
        self.assertEqual(self.h.state(880003), "pending")


class OffsetOrderTests(InboxCase):
    kv_class = SpyKv

    def test_offset_written_only_after_row_is_pending(self) -> None:
        """ingest_update: the offset is written only once the row is stored (the state is seen at write time)."""
        self.h.kv.probe = lambda value: self.h.state(int(value))
        self.h.service.ingest_update(message_update(880001, 1))
        self.assertEqual(self.h.kv.writes, [("offset", "880001", "pending")])


class ProcessTests(InboxCase):
    def test_message_processed_once_and_marked_done(self) -> None:
        """process_update ran the message handler once with the update id; the row is done afterwards.

        Ported labels:
            message row state done after processing
        """
        update = message_update(880001, 1)
        self.h.service.ingest_update(update)
        self.h.service.process_update(update)
        self.assertEqual(self.h.messages.seen, [(1, 880001)])
        self.assertEqual(self.h.state(880001), "done")

    def test_deferred_message_stays_pending(self) -> None:
        """A message the handler deferred (queued or in a media worker) keeps its row pending for the replay."""
        update = message_update(880001, 1)
        self.h.messages.outcomes[1] = Disposition.DEFERRED
        self.h.service.ingest_update(update)
        self.h.service.process_update(update)
        self.assertEqual(self.h.state(880001), "pending")

    def test_handler_exception_is_logged_and_row_done(self) -> None:
        """A message handler that raises is logged as 'handle:' and the row is still finished (live behaviour)."""
        update = message_update(880001, 1)
        self.h.messages.outcomes[1] = RuntimeError("boom")
        self.h.service.ingest_update(update)
        self.h.service.process_update(update)
        self.assertIn("handle: boom", self.h.log_text())
        self.assertEqual(self.h.state(880001), "done")

    def test_callback_processed_and_marked_done(self) -> None:
        """callback goes through the same path: dispatched once, row done."""
        update = callback_update(880002)
        self.h.service.ingest_update(update)
        self.h.service.process_update(update)
        self.assertEqual(self.h.callbacks.queries, ["cb880002"])
        self.assertEqual(self.h.state(880002), "done")

    def test_callback_exception_is_logged_and_row_done(self) -> None:
        """A failing callback handler is logged as 'callback:' and the row is finished."""
        update = callback_update(880002)
        self.h.callbacks.fail = RuntimeError("tap failed")
        self.h.service.ingest_update(update)
        self.h.service.process_update(update)
        self.assertIn("callback: tap failed", self.h.log_text())
        self.assertEqual(self.h.state(880002), "done")

    def test_group_learned_from_forum_message(self) -> None:
        """A forum group's message stores its id as group_id and its title under chat:<id>."""
        update = message_update(880001, 1)
        update["message"]["chat"]["title"] = "My AIs"
        self.h.service.process_update(update)
        self.assertEqual(self.h.kv.get("group_id"), GROUP_ID)
        self.assertEqual(self.h.kv.get(f"chat:{CHAT}"), "My AIs")

    def test_private_chat_learns_title_but_not_group(self) -> None:
        """A private chat is remembered by its type, and it does not become the group."""
        update = message_update(880001, 1, thread=None, chat={"id": 42, "type": "private"})
        self.h.service.process_update(update)
        self.assertIsNone(self.h.kv.get("group_id"))
        self.assertEqual(self.h.kv.get("chat:42"), "private")


class ReplayTests(InboxCase):
    def test_replay_processes_pending_in_update_id_order(self) -> None:
        """replay processes every pending row, lowest update_id first, and marks each done.

        Ported labels:
            replay processes pending row (got [('msg', 10)])
        """
        for update_id in (880030, 880010, 880020):
            self.h.service.ingest_update(message_update(update_id, update_id % 1000))
        self.h.service.replay_inbox()
        self.assertEqual(self.h.messages.seen, [(10, 880010), (20, 880020), (30, 880030)])
        self.assertTrue(all(self.h.state(u) == "done" for u in (880010, 880020, 880030)))

    def test_replay_leaves_deferred_row_pending(self) -> None:
        """A replayed message that is deferred again stays pending for the next start."""
        self.h.service.ingest_update(message_update(880010, 10))
        self.h.messages.outcomes[10] = Disposition.DEFERRED
        self.h.service.replay_inbox()
        self.assertEqual(self.h.state(880010), "pending")

    def test_stale_callback_20_min_marked_done_not_processed(self) -> None:
        """A callback older than 10 minutes is marked done without running; a fresh one (2 min) is replayed.

        Ported labels:
            fresh callback (2 min) replayed
        """
        self.h.service.ingest_update(callback_update(880020))
        self.h.clock.advance(18 * 60)
        self.h.service.ingest_update(callback_update(880021))
        self.h.clock.advance(2 * 60 + 1)  # 880020 is 20 min old; 880021 is 2 min old
        self.h.service.replay_inbox()
        self.assertEqual(self.h.state(880020), "done")
        self.assertNotIn("cb880020", self.h.callbacks.queries)
        self.assertIn("cb880021", self.h.callbacks.queries)
        self.assertEqual(self.h.state(880021), "done")
        self.assertIn("replay: update 880020 skipped", self.h.log_text())

    def test_callback_exactly_at_replay_limit_still_runs(self) -> None:
        """The limit is strict: a callback exactly CALLBACK_REPLAY_SECONDS old is still replayed."""
        self.h.service.ingest_update(callback_update(880040))
        self.h.clock.advance(CALLBACK_REPLAY_SECONDS)
        self.h.service.replay_inbox()
        self.assertEqual(self.h.callbacks.queries, ["cb880040"])

    def test_done_row_older_than_7_days_deleted_at_start(self) -> None:
        """Finished rows older than seven days are deleted at start; a recent finished row is kept."""
        self.h.service.ingest_update(message_update(880030, 30))
        self.h.inbox.done(880030)
        self.h.clock.advance(8 * DAY)
        self.h.service.ingest_update(message_update(880031, 31))
        self.h.inbox.done(880031)
        self.h.service.replay_inbox()
        self.assertIsNone(self.h.inbox.get(880030))
        self.assertEqual(self.h.state(880031), "done")

    def test_unreadable_payload_marked_done_not_processed(self) -> None:
        """A payload that is not JSON, or not a JSON object, is finished without running."""
        self._raw_row(880050, "not json")
        self._raw_row(880051, "[1, 2]")
        self.h.service.replay_inbox()
        self.assertEqual(self.h.messages.seen, [])
        self.assertEqual(self.h.state(880050), "done")
        self.assertEqual(self.h.state(880051), "done")

    def test_unreadable_timestamp_counts_as_stale_callback(self) -> None:
        """A callback whose stored stamp cannot be read is treated as stale (it is not run)."""
        self._raw_row(880060, json.dumps(callback_update(880060)), created="garbage")
        self.h.service.replay_inbox()
        self.assertEqual(self.h.callbacks.queries, [])
        self.assertEqual(self.h.state(880060), "done")


class SpoolTests(InboxCase):
    def setUp(self) -> None:
        super().setUp()
        self.h.sessions.ensure(THREAD, "side", "claude", "")
        self.h.kv.set("group_id", GROUP_ID)
        self.spool = self.h.paths.spool

    def _drop(self, name: str, body: str) -> str:
        path = os.path.join(self.spool, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(body)
        return path

    def test_spool_file_relayed_to_topic_then_removed(self) -> None:
        """drain_spool sends {topic, text} to the topic's thread in the group, then deletes the file."""
        path = self._drop("a.json", json.dumps({"topic": "side", "text": "hello from the watcher"}))
        self.h.service.drain_spool()
        self.assertEqual(self.h.tg.texts(THREAD), ["hello from the watcher"])
        self.assertFalse(os.path.exists(path))

    def test_spool_unknown_topic_left_in_place(self) -> None:
        """A file for a topic that has no session row stays in the folder and is logged."""
        path = self._drop("b.json", json.dumps({"topic": "NOPE", "text": "x"}))
        self.h.service.drain_spool()
        self.assertTrue(os.path.exists(path))
        self.assertIn("unknown topic 'NOPE'", self.h.log_text())
        self.assertEqual(self.h.tg.texts(THREAD), [])

    def test_spool_waits_until_group_is_learned(self) -> None:
        """Without a learned group the file stays in place, as the live drain does."""
        self.h.kv.delete("group_id")
        path = self._drop("c.json", json.dumps({"topic": "side", "text": "later"}))
        self.h.service.drain_spool()
        self.assertTrue(os.path.exists(path))
        self.assertEqual(self.h.tg.texts(THREAD), [])

    def test_bad_spool_file_logged_and_other_files_ignored(self) -> None:
        """A broken JSON file is logged and left; a file without the .json suffix is not touched at all."""
        bad = self._drop("d.json", "{not json")
        notes = self._drop("notes.txt", json.dumps({"topic": "side", "text": "ignored"}))
        self.h.service.drain_spool()
        self.assertTrue(os.path.exists(bad))
        self.assertTrue(os.path.exists(notes))
        self.assertIn("spool d.json:", self.h.log_text())
        self.assertEqual(self.h.tg.texts(THREAD), [])

    def test_missing_spool_folder_is_not_an_error(self) -> None:
        """No spool folder means nothing to drain."""
        os.rmdir(self.spool)
        self.h.service.drain_spool()
        self.assertEqual(self.h.tg.calls, ())


if __name__ == "__main__":
    unittest.main()
