"""TopicRemover (card D1): forgetting a topic completely, with an optional Telegram deletion.

The App is built over a TempHome with a FakeTelegram and a FakeClock, so the real repositories, the real LiveRegistry
and the real Dispatcher run. A FakeSession stands in for a live agent session: its stop() ends the turn the way the
turn runner does (end_turn), so the removal's wait finishes at once. Nothing touches the real state directory or the
real Telegram API.
"""

import os
import unittest

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.app import App
from tg_bridge.config import Config
from tg_bridge.core.cwd import PENDING_KEY
from tg_bridge.core.handoff import handoff_key
from tg_bridge.core.linger import linger_key
from tg_bridge.core.topic_queues import TopicQueues
from tg_bridge.core.topic_remover import remove_view_files
from tg_bridge.core.turn_listeners import slash_key
from tg_bridge.core.turn_runner import active_key
from tg_bridge.paths import Paths
from tg_bridge.store.asks import AskRepo
from tg_bridge.store.forget import THREAD_KV_PREFIXES, TopicForget
from tg_bridge.store.inbox import InboxRepo
from tg_bridge.store.rules import RuleRepo
from tg_bridge.store.sessions import HistoryRepo, SessionRepo
from tg_bridge.store.turns import TurnRepo

CHAT = -1009901
GONE_THREAD = 7
OTHER_THREAD = 8
ALLOWED_UID = 42
GROUP_KEY = "group_id"


class FakeSession:
    """Stands in for a live agent session. stop() ends the turn it runs in, as the turn runner does on a stop."""

    def __init__(self, live, thread_id: int, feed: object) -> None:
        self.live = live
        self.thread_id = thread_id
        self.feed = feed
        self.stops = 0

    def stop(self, grace: float = 5.0) -> None:
        self.stops += 1
        self.live.end_turn(self.thread_id, self, self.feed)


def _config(home: str) -> Config:
    return Config({"allow_user_ids": [ALLOWED_UID]}, home=home, env={})


class RemoverCase(unittest.TestCase):
    """An App over a TempHome, with two topics that each have rows in every thread table and kv key."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.clock = FakeClock()
        self.telegram = FakeTelegram(self.clock)
        self.app = App.build(Paths.from_home(self.home.home), _config(self.home.home), telegram=self.telegram,
                             clock=self.clock, env={})
        self.addCleanup(self.app.close)
        self.app.kv.set(GROUP_KEY, str(CHAT))
        self.view = self.app.paths.view
        db = self.app.db
        self.sessions_repo = SessionRepo(db, self.clock)
        self.history = HistoryRepo(db, self.clock)
        self.rules = RuleRepo(db, self.clock)
        self.asks = AskRepo(db, self.clock)
        self.inbox = InboxRepo(db, self.clock)
        self.turns = TurnRepo(db)
        for thread in (GONE_THREAD, OTHER_THREAD):
            self._fill(thread)

    def _fill(self, thread: int) -> None:
        self.sessions_repo.ensure(thread, f"topic{thread}", "claude-cmd", self.home.home)
        self.history.note_session(thread, f"uuid-{thread}", "hello", self.home.home, "claude-cmd", "claude")
        self.rules.add(thread, "Edit")
        self.asks.insert(nonce=f"nonce-{thread}", thread_id=thread, chat_id=CHAT, request_id="r", tool_name="Bash",
                         input={"command": "ls"})
        self.inbox.ingest(1000 + thread, {"text": "queued"}, thread_id=thread, chat_id=CHAT, sender="u")
        self.turns.insert(thread, started="2026-01-01T00:00:00Z", ended="2026-01-01T00:00:01Z", status="ok",
                          cost=0.5, in_tok=1, out_tok=1)
        for key in (f"thread:{thread}", slash_key(thread), linger_key(thread), handoff_key(thread),
                    active_key(thread), PENDING_KEY.format(thread=thread), f"pane:topic{thread}"):
            self.app.kv.set(key, "x")

    def _count(self, sql: str, args: tuple = ()) -> int:
        return int(self.app.db.query(sql, args)[0][0])

    def _kv_keys(self) -> set[str]:
        return {row["k"] for row in self.app.db.query("SELECT k FROM kv")}


class ForgetTopicTests(RemoverCase):
    """TopicForget.forget_topic: every thread table and kv key of one topic goes; turns and other topics stay."""

    def _forget(self, thread: int) -> None:
        TopicForget(db=self.app.db, sessions=self.sessions_repo, history=self.history, rules=self.rules,
                    asks=self.asks, inbox=self.inbox, kv=self.app.kv).forget_topic(thread)

    def test_every_thread_table_loses_only_the_forgotten_topic(self) -> None:
        self._forget(GONE_THREAD)
        for table in ("sessions", "session_history", "perm_rules", "asks", "inbox_queue"):
            self.assertEqual(self._count(f"SELECT COUNT(*) FROM {table} WHERE thread_id=?", (GONE_THREAD,)), 0,
                             table)
            self.assertEqual(self._count(f"SELECT COUNT(*) FROM {table} WHERE thread_id=?", (OTHER_THREAD,)), 1,
                             table)

    def test_turns_are_kept_for_the_forgotten_topic(self) -> None:
        self._forget(GONE_THREAD)
        self.assertEqual(self._count("SELECT COUNT(*) FROM turns WHERE thread_id=?", (GONE_THREAD,)), 1)

    def test_thread_kv_keys_and_pane_key_go_and_other_topics_keys_stay(self) -> None:
        self._forget(GONE_THREAD)
        keys = self._kv_keys()
        for prefix in THREAD_KV_PREFIXES:
            self.assertNotIn(f"{prefix}{GONE_THREAD}", keys, prefix)
        self.assertNotIn(f"pane:topic{GONE_THREAD}", keys)
        self.assertIn(f"thread:{OTHER_THREAD}", keys)
        self.assertIn(f"pane:topic{OTHER_THREAD}", keys)
        self.assertIn(GROUP_KEY, keys)


class RemoveTests(RemoverCase):
    """TopicRemover.remove: the session, the Telegram topic, the forget and the view files."""

    def test_a_live_session_is_stopped_before_the_topic_is_forgotten(self) -> None:
        feed = object()
        session = FakeSession(self.app.live, GONE_THREAD, feed)
        self.assertTrue(self.app.live.begin_turn(GONE_THREAD, session, feed))
        text = self.app.remover.remove(GONE_THREAD, delete_in_telegram=False)
        self.assertEqual(session.stops, 1, "the running session was stopped once")
        self.assertIsNone(self.app.live.running(GONE_THREAD))
        self.assertEqual(text, f"topic {GONE_THREAD} removed (telegram: kept)")
        self.assertEqual(self._count("SELECT COUNT(*) FROM sessions WHERE thread_id=?", (GONE_THREAD,)), 0)

    def test_telegram_topic_invalid_counts_as_already_gone_and_is_forgotten(self) -> None:
        for description in ("Bad Request: TOPIC_ID_INVALID", "Bad Request: message thread not found"):
            with self.subTest(description=description):
                self.telegram.fail_next("deleteForumTopic", 400, description=description)
                text = self.app.remover.remove(GONE_THREAD, delete_in_telegram=True)
                self.assertEqual(text, f"topic {GONE_THREAD} removed (telegram: already gone)")
                self.assertEqual(self._count("SELECT COUNT(*) FROM sessions WHERE thread_id=?", (GONE_THREAD,)), 0)
                self._fill(GONE_THREAD)

    def test_other_telegram_error_refuses_the_removal_and_forgets_nothing(self) -> None:
        self.telegram.fail_next("deleteForumTopic", 403, description="Forbidden: not enough rights")
        text = self.app.remover.remove(GONE_THREAD, delete_in_telegram=True)
        self.assertTrue(text.startswith(f"topic {GONE_THREAD} not removed"), text)
        self.assertEqual(self._count("SELECT COUNT(*) FROM sessions WHERE thread_id=?", (GONE_THREAD,)), 1)
        self.assertIn(f"thread:{GONE_THREAD}", self._kv_keys())

    def test_delete_in_telegram_calls_delete_forum_topic_in_the_group(self) -> None:
        text = self.app.remover.remove(GONE_THREAD, delete_in_telegram=True)
        calls = [c for c in self.telegram.calls if c.method == "deleteForumTopic"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].params, {"chat_id": CHAT, "message_thread_id": GONE_THREAD})
        self.assertEqual(text, f"topic {GONE_THREAD} removed (telegram: deleted)")

    def test_without_delete_in_telegram_no_api_call_is_made(self) -> None:
        self.app.remover.remove(GONE_THREAD, delete_in_telegram=False)
        self.assertEqual([c for c in self.telegram.calls if c.method == "deleteForumTopic"], [])

    def test_view_files_of_the_topic_go_and_its_lock_stays(self) -> None:
        names = [f"{GONE_THREAD}.json", f"{GONE_THREAD}.feed", f"{GONE_THREAD}.term", f"{GONE_THREAD}.turn",
                 f"{GONE_THREAD}.lock", f"{OTHER_THREAD}.json", f"{GONE_THREAD}0.json"]
        for name in names:
            with open(os.path.join(self.view, name), "w", encoding="utf-8") as handle:
                handle.write("{}")
        self.app.remover.remove(GONE_THREAD, delete_in_telegram=False)
        remaining = set(os.listdir(self.view))
        self.assertEqual(remaining, {f"{GONE_THREAD}.lock", f"{OTHER_THREAD}.json", f"{GONE_THREAD}0.json"})

    def test_remove_view_files_with_views_off_removes_nothing(self) -> None:
        self.assertEqual(remove_view_files("", GONE_THREAD), 0)


class QueueClearTests(unittest.TestCase):
    """TopicQueues.clear drops every queued message of one topic and leaves the others alone."""

    def test_clear_drops_only_the_topic_queue(self) -> None:
        queues = TopicQueues()
        queues.reserve(GONE_THREAD, CHAT, "u", 1)
        queues.deliver(GONE_THREAD, CHAT, "u", 2, "second")
        queues.deliver(OTHER_THREAD, CHAT, "u", 3, "other")
        self.assertEqual(queues.clear(GONE_THREAD), 2)
        self.assertEqual(queues.snapshot(GONE_THREAD), ())
        self.assertEqual([item.text for item in queues.snapshot(OTHER_THREAD)], ["other"])


if __name__ == "__main__":
    unittest.main()


class ViewFilesTests(unittest.TestCase):
    """remove_view_files (the topic's files in the view folder): the pane's .session record goes with the topic, like the
    claim, while the .lock file and the delete request files stay."""

    def test_the_session_record_is_removed_with_the_topic_view_files(self) -> None:
        home = self.enterContext(TempHome()).home
        view = os.path.join(home, "view")
        os.makedirs(view)
        names = ["5150.session", "5150.json", "5150.term", "5150.turn", "5150.lock", "5150.delete"]
        for name in names:
            open(os.path.join(view, name), "w", encoding="utf-8").close()
        open(os.path.join(view, "5151.session"), "w", encoding="utf-8").close()  # another topic: untouched
        self.assertEqual(remove_view_files(view, 5150), 4)
        self.assertEqual(sorted(os.listdir(view)), ["5150.delete", "5150.lock", "5151.session"])

