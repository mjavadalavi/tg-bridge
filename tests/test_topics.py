"""topics: port of the topic resolution of the live _continue_message and _session_name (C18), plus the General guard
of handle_message. Scenario labels follow the live order: session row, learned name, config topic, spawn alias,
default_topic, unknown.

Real Database, KvRepo and SessionRepo under a TempHome; Config is built from a plain dict. The reply texts are compared
with the literal strings of tg-bridge.py.
"""

import os
import unittest

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.config import Config
from tg_bridge.core.topics import (
    GENERAL_TEXT,
    NO_TOPIC_TEXT,
    TopicKind,
    TopicResolver,
    guard_text,
)
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.db import Database
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.sessions import SessionRepo

CLAUDE_CMD = "claude --alpha"
CONFIG = {
    "topics": {
        "side": {"launcher": "altclaude", "cwd": "/work/alt"},
        "pane_topic": {"type": "pane", "pane": "%7", "label": "seat"},
        "empty": {},
    },
    "spawn": {"budget": "budget-claude"},
    "default_topic": {"launcher": "alpha", "cwd": "/work/default"},
}
NO_DEFAULT = {"topics": CONFIG["topics"], "spawn": CONFIG["spawn"]}


class TopicCase(unittest.TestCase):
    """A migrated Database under a TempHome, with a resolver built on the given config dict."""

    def setUp(self) -> None:
        self._home = TempHome()
        self.home = self._home.__enter__().home
        self.clock = FakeClock()
        self.db = Database(Paths.from_home(self.home))
        self.kv = KvRepo(self.db)
        self.sessions = SessionRepo(self.db, self.clock)
        self.log_path = os.path.join(self._home.state, "bridge.log")
        self.logger = Logger(self.log_path)

    def tearDown(self) -> None:
        self.db.close()
        self._home.__exit__(None, None, None)

    def resolver(self, data: dict | None = None) -> TopicResolver:
        config = Config(CONFIG if data is None else data, home=self.home, env={})
        return TopicResolver(config=config, kv=self.kv, sessions=self.sessions, logger=self.logger)

    def log_text(self) -> str:
        if not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, encoding="utf-8") as handle:
            return handle.read()


class ResolveOrderTests(TopicCase):
    def test_existing_session_row_wins_over_everything(self) -> None:
        self.sessions.ensure(11, "side", CLAUDE_CMD, "/work/row")
        self.kv.set("thread:11", "budget.other")
        target = self.resolver().resolve(11)
        self.assertEqual((target.kind, target.name), (TopicKind.SESSION, "side"))
        self.assertEqual(target.config, {})

    def test_learned_kv_name_selects_a_configured_topic(self) -> None:
        self.kv.set("thread:12", "side")
        target = self.resolver().resolve(12)
        self.assertEqual(target.kind, TopicKind.CONFIGURED)
        self.assertEqual(target.name, "side")
        self.assertEqual(target.config, {"launcher": "altclaude", "cwd": "/work/alt"})

    def test_reply_header_name_is_used_and_learned_into_kv(self) -> None:
        reply = {"forum_topic_created": {"name": "side"}}
        target = self.resolver().resolve(13, reply_to=reply)
        self.assertEqual((target.kind, target.name), (TopicKind.CONFIGURED, "side"))
        self.assertEqual(self.kv.get("thread:13"), "side")

    def test_kv_name_is_preferred_to_the_reply_header(self) -> None:
        self.kv.set("thread:14", "side")
        reply = {"forum_topic_created": {"name": "empty"}}
        target = self.resolver().resolve(14, reply_to=reply)
        self.assertEqual(target.name, "side")
        self.assertEqual(self.kv.get("thread:14"), "side")

    def test_learn_writes_the_kv_name(self) -> None:
        self.resolver().learn(15, "other")
        self.assertEqual(self.kv.get("thread:15"), "other")

    def test_an_empty_config_entry_counts_as_missing(self) -> None:
        self.kv.set("thread:16", "empty")
        target = self.resolver().resolve(16)
        self.assertEqual(target.kind, TopicKind.DEFAULT)
        self.assertEqual(target.config, {"launcher": "alpha", "cwd": "/work/default"})


class SpawnAndDefaultTests(TopicCase):
    def test_spawn_alias_with_task_gives_the_command_and_stripped_task(self) -> None:
        self.kv.set("thread:21", "budget. mytask ")
        target = self.resolver().resolve(21)
        self.assertEqual(target.kind, TopicKind.SPAWN)
        self.assertEqual(target.name, "budget. mytask ")
        self.assertEqual(target.spawn_command, "budget-claude")
        self.assertEqual(target.spawn_name, "mytask")
        self.assertIn("topic resolved: thread 21 -> budget. mytask ", self.log_text())

    def test_spawn_alias_with_a_blank_task_is_not_a_spawn(self) -> None:
        self.kv.set("thread:22", "budget.   ")
        target = self.resolver(NO_DEFAULT).resolve(22)
        self.assertEqual(target.kind, TopicKind.UNKNOWN)

    def test_unknown_alias_is_not_a_spawn(self) -> None:
        self.kv.set("thread:23", "nope.task")
        self.assertEqual(self.resolver(NO_DEFAULT).resolve(23).kind, TopicKind.UNKNOWN)

    def test_default_topic_makes_a_fresh_topic_its_own_session(self) -> None:
        self.kv.set("thread:24", "a brand new topic")
        target = self.resolver().resolve(24)
        self.assertEqual(target.kind, TopicKind.DEFAULT)
        self.assertEqual(target.name, "a brand new topic")
        self.assertEqual(target.config["cwd"], "/work/default")

    def test_default_topic_applies_to_a_topic_with_no_name_at_all(self) -> None:
        target = self.resolver().resolve(25)
        self.assertEqual((target.kind, target.name), (TopicKind.DEFAULT, ""))

    def test_no_default_topic_leaves_an_unnamed_topic_unknown(self) -> None:
        target = self.resolver(NO_DEFAULT).resolve(26)
        self.assertEqual((target.kind, target.name), (TopicKind.UNKNOWN, ""))
        self.assertIn("unresolved topic: thread 26", self.log_text())


class UnknownTextTests(TopicCase):
    def test_unknown_text_lists_config_and_spawn_names_sorted_once(self) -> None:
        resolver = self.resolver(NO_DEFAULT)
        target = resolver.resolve(31)
        self.assertEqual(
            resolver.unknown_text(target),
            'Unknown topic "31" (thread 31).\n'
            "Known: budget, empty, pane_topic, side\n"
            "To spawn a new live seat, name the topic like work.mytask — a session starts in a new tmux window "
            "and your first message becomes its first prompt.",
        )

    def test_unknown_text_quotes_the_learned_name_when_there_is_one(self) -> None:
        resolver = self.resolver(NO_DEFAULT)
        self.kv.set("thread:32", "mystery")
        text = resolver.unknown_text(resolver.resolve(32))
        self.assertTrue(text.startswith('Unknown topic "mystery" (thread 32).\n'))


class QueuedSessionNameTests(TopicCase):
    def test_a_plain_session_topic_is_queued_under_its_row_name(self) -> None:
        self.sessions.ensure(41, "side", CLAUDE_CMD, "/work")
        self.assertEqual(self.resolver().queued_session_name(41), "side")

    def test_pane_topics_are_not_queued(self) -> None:
        self.sessions.ensure(42, "pane_topic", CLAUDE_CMD, "/work")
        self.assertIsNone(self.resolver().queued_session_name(42))

    def test_a_kv_pane_binding_makes_a_topic_a_pane_topic(self) -> None:
        self.sessions.ensure(43, "seat-x", CLAUDE_CMD, "/work")
        self.kv.set("pane:seat-x", "%9")
        self.assertIsNone(self.resolver().queued_session_name(43))

    def test_no_row_means_nothing_to_queue(self) -> None:
        self.assertIsNone(self.resolver().queued_session_name(44))


class GuardTextTests(unittest.TestCase):
    def test_general_topic_of_a_group_gets_the_board_text(self) -> None:
        self.assertEqual(guard_text("supergroup", 1), GENERAL_TEXT)
        self.assertEqual(GENERAL_TEXT, "This is the general board — open a topic and message me there.")

    def test_a_group_message_without_a_thread_gets_the_board_text(self) -> None:
        self.assertEqual(guard_text("supergroup", None), GENERAL_TEXT)

    def test_a_private_chat_without_a_thread_gets_the_add_me_text(self) -> None:
        self.assertEqual(guard_text("private", None), NO_TOPIC_TEXT)
        self.assertEqual(NO_TOPIC_TEXT, 'Add me to the "My AIs" forum group and talk in a topic.')

    def test_a_private_chat_in_thread_one_continues(self) -> None:
        self.assertIsNone(guard_text("private", 1))

    def test_a_real_topic_continues(self) -> None:
        self.assertIsNone(guard_text("supergroup", 77))


if __name__ == "__main__":
    unittest.main()
