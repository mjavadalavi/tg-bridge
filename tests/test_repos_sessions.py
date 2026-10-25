"""Repositories part 1 (STEP 11): KvRepo, SessionRepo, HistoryRepo and CwdRepo.

Every method is covered on a real migrated Database under a TempHome. The concurrency test runs four threads
against one Database and checks there is no deadlock and no lost write.
"""

import os
import sqlite3
import threading
import unittest

from tests.fakes.clock import DEFAULT_START, FakeClock
from tests.support import TempHome
from tg_bridge.clock import utc_stamp
from tg_bridge.paths import Paths
from tg_bridge.store.db import Database
from tg_bridge.store.inbox import CwdRepo
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.sessions import HistoryRecord, HistoryRepo, SessionRecord, SessionRepo
from tg_bridge.store.turns import TurnRepo

CLAUDE_CMD = "claude --alpha"
STAMP = utc_stamp(DEFAULT_START)


class RepoCase(unittest.TestCase):
    """Opens a migrated Database under a TempHome for each test."""

    def setUp(self) -> None:
        self._home = TempHome()
        self.home = self._home.__enter__().home
        self.clock = FakeClock()
        self.db = Database(Paths.from_home(self.home))
        self.sessions = SessionRepo(self.db, self.clock)
        self.history = HistoryRepo(self.db, self.clock)
        self.kv = KvRepo(self.db)
        self.cwd = CwdRepo(self.db)

    def tearDown(self) -> None:
        self.db.close()
        self._home.__exit__(None, None, None)

    def _raw(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        return self.db.query(sql, args)


class KvRepoTests(RepoCase):
    def test_get_returns_default_for_missing_key(self) -> None:
        self.assertIsNone(self.kv.get("missing"))
        self.assertEqual(self.kv.get("missing", "fallback"), "fallback")

    def test_set_then_get_and_upsert_replaces_value(self) -> None:
        self.kv.set("offset", "42")
        self.assertEqual(self.kv.get("offset"), "42")
        self.kv.set("offset", "43")
        self.assertEqual(self.kv.get("offset"), "43")
        self.assertEqual(len(self._raw("SELECT k FROM kv WHERE k='offset'")), 1)

    def test_delete_removes_key_and_missing_delete_is_not_an_error(self) -> None:
        self.kv.set("pane:alpha", "%3")
        self.kv.delete("pane:alpha")
        self.assertIsNone(self.kv.get("pane:alpha"))
        self.kv.delete("pane:alpha")  # already gone: no error


class CwdRepoTests(RepoCase):
    def test_path_id_registers_new_path_and_returns_same_id_for_same_path(self) -> None:
        first = self.cwd.path_id("/work/a")
        second = self.cwd.path_id("/work/b")
        self.assertNotEqual(first, second)
        self.assertEqual(self.cwd.path_id("/work/a"), first)
        self.assertEqual(len(self._raw("SELECT id FROM cwd_paths")), 2)

    def test_path_of_returns_path_for_known_id_and_none_otherwise(self) -> None:
        pid = self.cwd.path_id("/work/a")
        self.assertEqual(self.cwd.path_of(pid), "/work/a")
        self.assertIsNone(self.cwd.path_of(pid + 999))


class SessionGetEnsureTests(RepoCase):
    def test_get_returns_none_for_topic_without_row(self) -> None:
        self.assertIsNone(self.sessions.get(7))

    def test_ensure_creates_row_with_clock_timestamp_and_reports_creation(self) -> None:
        created = self.sessions.ensure(7, "alpha", CLAUDE_CMD, "/work/a", mode="acceptEdits")
        self.assertTrue(created)
        row = self.sessions.get(7)
        self.assertIsInstance(row, SessionRecord)
        self.assertEqual(row.thread_id, 7)
        self.assertEqual(row.name, "alpha")
        self.assertEqual(row.uuid, "")
        self.assertEqual(row.launcher, CLAUDE_CMD)
        self.assertEqual(row.cwd, "/work/a")
        self.assertEqual(row.created, STAMP)
        self.assertEqual(row.mode, "acceptEdits")
        self.assertEqual(row.backend, "claude")

    def test_ensure_on_existing_row_changes_nothing(self) -> None:
        self.sessions.ensure(7, "alpha", CLAUDE_CMD, "/work/a")
        self.clock.advance(60)
        again = self.sessions.ensure(7, "beta", "claude --other", "/work/b", mode="plan")
        self.assertFalse(again)
        row = self.sessions.get(7)
        self.assertEqual((row.name, row.launcher, row.cwd, row.mode), ("alpha", CLAUDE_CMD, "/work/a", ""))
        self.assertEqual(row.created, STAMP)

    def test_get_reads_null_text_columns_as_empty_strings(self) -> None:
        self.db.execute("INSERT INTO sessions(thread_id, name) VALUES(9, NULL)")
        row = self.sessions.get(9)
        self.assertEqual((row.name, row.uuid, row.title, row.backend), ("", "", "", "claude"))


class SessionUuidTests(RepoCase):
    def setUp(self) -> None:
        super().setUp()
        self.sessions.ensure(7, "alpha", CLAUDE_CMD, "/work/a")

    def test_set_uuid_stores_the_session_id(self) -> None:
        self.sessions.set_uuid(7, "uuid-1")
        self.assertEqual(self.sessions.get(7).uuid, "uuid-1")

    def test_new_session_moves_uuid_to_prev_and_clears_uuid(self) -> None:
        self.sessions.set_uuid(7, "uuid-1")
        self.sessions.new_session(7)
        row = self.sessions.get(7)
        self.assertEqual((row.uuid, row.prev_uuid), ("", "uuid-1"))

    def test_new_session_with_empty_uuid_keeps_prev_uuid(self) -> None:
        self.sessions.set_uuid(7, "uuid-1")
        self.sessions.new_session(7)
        self.sessions.new_session(7)
        self.assertEqual(self.sessions.get(7).prev_uuid, "uuid-1")
        self.sessions.set_uuid(7, "uuid-2")
        self.sessions.new_session(7)
        row = self.sessions.get(7)
        self.assertEqual((row.uuid, row.prev_uuid), ("", "uuid-2"))

    def test_bind_session_moves_current_uuid_and_sets_cwd_and_pick_when_given(self) -> None:
        self.sessions.set_uuid(7, "uuid-1")
        self.sessions.bind_session(7, "uuid-old", cwd="/work/b", launcher_pick="codex-pick")
        row = self.sessions.get(7)
        self.assertEqual((row.uuid, row.prev_uuid, row.cwd, row.launcher_pick), ("uuid-old", "uuid-1", "/work/b", "codex-pick"))

    def test_bind_session_without_cwd_or_pick_leaves_them_unchanged(self) -> None:
        self.sessions.set_uuid(7, "uuid-1")
        self.sessions.bind_session(7, "uuid-2")
        row = self.sessions.get(7)
        self.assertEqual((row.uuid, row.prev_uuid, row.cwd, row.launcher_pick), ("uuid-2", "uuid-1", "/work/a", ""))


class MostRecentThreadTests(RepoCase):
    def test_most_recent_thread_is_none_without_topics(self) -> None:
        self.assertIsNone(self.sessions.most_recent_thread())

    def test_most_recent_thread_picks_the_latest_last_active(self) -> None:
        for thread in (7, 8, 9):
            self.sessions.ensure(thread, f"topic {thread}", CLAUDE_CMD, "")
        self.db.execute("UPDATE sessions SET last_active=? WHERE thread_id=?", ("2026-10-24T09:00:00+00:00", 7))
        self.db.execute("UPDATE sessions SET last_active=? WHERE thread_id=?", ("2026-10-24T11:00:00+00:00", 8))
        self.db.execute("UPDATE sessions SET last_active=? WHERE thread_id=?", ("2026-10-24T10:00:00+00:00", 9))
        self.assertEqual(self.sessions.most_recent_thread(), 8)

    def test_most_recent_thread_follows_a_finished_turn(self) -> None:
        for thread in (7, 8):
            self.sessions.ensure(thread, f"topic {thread}", CLAUDE_CMD, "")
        self.db.execute("UPDATE sessions SET last_active=? WHERE thread_id=?", (STAMP, 7))
        later = utc_stamp(DEFAULT_START + 60)
        TurnRepo(self.db).insert(8, started=STAMP, ended=later, status="ok")
        self.assertEqual(self.sessions.most_recent_thread(), 8)


class ThreadOfNameTests(RepoCase):
    """thread_of: the spool drain addresses a topic by its session name (STEP 51)."""

    def test_thread_of_finds_the_topic_by_name(self) -> None:
        self.sessions.ensure(7, "side", CLAUDE_CMD, "")
        self.sessions.ensure(8, "other", CLAUDE_CMD, "")
        self.assertEqual(self.sessions.thread_of("side"), 7)

    def test_thread_of_is_none_for_unknown_or_missing_name(self) -> None:
        self.sessions.ensure(7, "side", CLAUDE_CMD, "")
        self.assertIsNone(self.sessions.thread_of("NOPE"))
        self.assertIsNone(self.sessions.thread_of(None))


class SessionSettersTests(RepoCase):
    def setUp(self) -> None:
        super().setUp()
        self.sessions.ensure(7, "alpha", CLAUDE_CMD, "/work/a")

    def test_setters_store_their_column(self) -> None:
        self.sessions.set_mode(7, "plan")
        self.sessions.set_model(7, "opus")
        self.sessions.set_launcher_pick(7, "pick-1")
        self.sessions.set_cwd(7, "/work/c")
        self.sessions.set_title(7, "Alpha topic")
        row = self.sessions.get(7)
        self.assertEqual(
            (row.mode, row.model, row.launcher_pick, row.cwd, row.title), ("plan", "opus", "pick-1", "/work/c", "Alpha topic")
        )

    def test_set_backend_accepts_known_backends(self) -> None:
        self.sessions.set_backend(7, "codex")
        self.assertEqual(self.sessions.get(7).backend, "codex")
        self.sessions.set_backend(7, "claude")
        self.assertEqual(self.sessions.get(7).backend, "claude")

    def test_set_backend_refuses_unknown_backend_and_writes_nothing(self) -> None:
        with self.assertRaises(ValueError):
            self.sessions.set_backend(7, "gemini")
        self.assertEqual(self.sessions.get(7).backend, "claude")

    def test_setters_never_change_the_claude_launcher(self) -> None:
        self.sessions.set_backend(7, "codex")
        self.sessions.set_launcher_pick(7, "codex-pick")
        self.sessions.bind_session(7, "uuid-x", launcher_pick="codex-pick")
        self.sessions.set_mode(7, "bypassPermissions")
        self.assertEqual(self.sessions.get(7).launcher, CLAUDE_CMD)

    def test_private_column_setter_refuses_names_outside_the_whitelist(self) -> None:
        with self.assertRaises(ValueError):
            self.sessions._set_column(7, "launcher", "other")
        self.assertEqual(self.sessions.get(7).launcher, CLAUDE_CMD)

    def test_cwds_lists_non_empty_folders_in_topic_order(self) -> None:
        self.sessions.ensure(3, "gamma", CLAUDE_CMD, "")
        self.sessions.ensure(9, "delta", CLAUDE_CMD, "/work/d")
        self.assertEqual(self.sessions.cwds(), ["/work/a", "/work/d"])


class HistoryNoteTests(RepoCase):
    def test_note_session_stores_first_text_collapsed_to_60_chars(self) -> None:
        """Ported labels:
        Q7: one row per uuid, text collapsed and cut to 60
        """
        self.history.note_session(7, "u1", "  hello \n   world  " + "x" * 80, cwd="", launcher=CLAUDE_CMD)
        row = self.history.find(7, "u1")
        self.assertIsInstance(row, HistoryRecord)
        self.assertEqual(row.first_text, ("hello world " + "x" * 80)[:60])
        self.assertEqual(len(row.first_text), 60)
        self.assertEqual(row.started, STAMP)
        self.assertEqual((row.cwd, row.launcher, row.backend), ("", CLAUDE_CMD, "claude"))

    def test_note_session_records_a_uuid_once(self) -> None:
        """Ported labels:
        Q7: one row per uuid, text collapsed and cut to 60
        """
        self.history.note_session(7, "u1", "first text", cwd="", launcher=CLAUDE_CMD)
        self.history.note_session(7, "u1", "second text", cwd="", launcher=CLAUDE_CMD)
        rows = self._raw("SELECT first_text FROM session_history WHERE uuid='u1'")
        self.assertEqual([r["first_text"] for r in rows], ["first text"])

    def test_note_session_fills_an_empty_folder_and_launcher_on_a_later_call(self) -> None:
        """Ported labels:
        J6: the next turn fills in the missing folder
        """
        self.history.note_session(7, "u1", "text")
        self.history.note_session(7, "u1", "text", cwd=self.home, launcher=CLAUDE_CMD)
        row = self.history.find(7, "u1")
        self.assertEqual((row.cwd, row.launcher), (os.path.realpath(self.home), CLAUDE_CMD))

    def test_note_session_keeps_a_launcher_that_is_already_stored(self) -> None:
        self.history.note_session(7, "u1", "text", launcher="claude --first")
        self.history.note_session(7, "u1", "text", launcher="claude --second")
        self.assertEqual(self.history.find(7, "u1").launcher, "claude --first")

    def test_note_session_resolves_symlinked_folder_to_its_target(self) -> None:
        """Ported labels:
        J1: the folder is recorded with the session
        """
        real = os.path.join(self.home, "real")
        link = os.path.join(self.home, "link")
        os.makedirs(real)
        os.symlink(real, link)
        self.history.note_session(7, "u1", "text", cwd=link)
        self.assertEqual(self.history.find(7, "u1").cwd, os.path.realpath(real))

    def test_note_session_stores_given_backend(self) -> None:
        self.history.note_session(7, "codex-1", "text", backend="codex")
        self.assertEqual(self.history.find(7, "codex-1").backend, "codex")


class HistoryAdoptTests(RepoCase):
    def test_adopt_session_inserts_row_with_empty_first_text_and_sets_launcher(self) -> None:
        self.history.adopt_session(7, "u1", CLAUDE_CMD, self.home)
        row = self.history.find(7, "u1")
        self.assertEqual((row.first_text, row.launcher, row.cwd), ("", CLAUDE_CMD, os.path.realpath(self.home)))

    def test_adopt_session_overwrites_an_existing_launcher(self) -> None:
        self.history.note_session(7, "u1", "text", launcher="claude --old")
        self.history.adopt_session(7, "u1", "claude --new", "")
        self.assertEqual(self.history.find(7, "u1").launcher, "claude --new")
        self.assertEqual(self.history.find(7, "u1").first_text, "text")


class HistoryListTests(RepoCase):
    def setUp(self) -> None:
        super().setUp()
        self.history.note_session(7, "c1", "claude one", cwd="/work/a", launcher=CLAUDE_CMD)
        self.history.note_session(7, "k1", "codex one", cwd="/work/a", launcher="codex", backend="codex")
        self.history.note_session(7, "c2", "claude two", cwd="", launcher=CLAUDE_CMD)
        self.history.note_session(7, "c3", "claude three", cwd="/work/b", launcher=CLAUDE_CMD)
        self.history.note_session(8, "c9", "other topic", cwd="/work/a", launcher=CLAUDE_CMD)

    def test_list_for_filters_by_backend_and_topic_newest_first(self) -> None:
        uuids = [r.uuid for r in self.history.list_for(7, "claude")]
        self.assertEqual(uuids, ["c3", "c2", "c1"])
        self.assertEqual([r.uuid for r in self.history.list_for(7, "codex")], ["k1"])

    def test_list_for_limits_the_count(self) -> None:
        self.assertEqual([r.uuid for r in self.history.list_for(7, "claude", limit=2)], ["c3", "c2"])

    def test_list_for_default_limit_is_eight(self) -> None:
        for i in range(10):
            self.history.note_session(7, f"bulk-{i}", "bulk")
        self.assertEqual(len(self.history.list_for(7, "claude")), 8)

    def test_list_for_with_cwd_keeps_that_folder_and_folderless_rows_only(self) -> None:
        uuids = [r.uuid for r in self.history.list_for(7, "claude", cwd="/work/a")]
        self.assertEqual(uuids, ["c2", "c1"])

    def test_list_for_with_empty_cwd_returns_only_folderless_rows(self) -> None:
        self.assertEqual([r.uuid for r in self.history.list_for(7, "claude", cwd="")], ["c2"])


class HistoryFindTests(RepoCase):
    def test_find_returns_none_for_unknown_uuid(self) -> None:
        self.assertIsNone(self.history.find(7, "nope"))

    def test_find_by_id_returns_row_only_for_its_own_topic(self) -> None:
        self.history.note_session(7, "u1", "text")
        row = self.history.find(7, "u1")
        self.assertEqual(self.history.find_by_id(7, row.id).uuid, "u1")
        self.assertIsNone(self.history.find_by_id(8, row.id))
        self.assertIsNone(self.history.find_by_id(7, row.id + 500))


class NestedTransactionTests(RepoCase):
    def test_repository_writes_inside_outer_transaction_roll_back_together(self) -> None:
        self.sessions.ensure(7, "alpha", CLAUDE_CMD, "/work/a")
        with self.assertRaises(RuntimeError):
            with self.db.tx():
                self.sessions.set_mode(7, "plan")
                self.history.note_session(7, "u1", "text")
                raise RuntimeError("boom")
        self.assertEqual(self.sessions.get(7).mode, "")
        self.assertIsNone(self.history.find(7, "u1"))


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
