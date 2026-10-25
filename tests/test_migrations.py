"""Database and migrations (STEP 10): fresh DB, additive v2, idempotence."""

import os
import sqlite3
import stat
import unittest
from unittest import mock

from tests.support import TempHome
from tg_bridge.paths import Paths
from tg_bridge.store import migrations
from tg_bridge.store.db import Database
from tg_bridge.store.migrations import Migration, SchemaTooNew

BACKEND_TABLES = ("sessions", "session_history", "turns")


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def _schema(conn: sqlite3.Connection) -> list[tuple]:
    return [tuple(r) for r in conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY name")]




class FreshDatabaseTests(unittest.TestCase):
    def test_fresh_database_reaches_latest_version(self) -> None:
        with TempHome() as home:
            db = Database(Paths.from_home(home.home))
            try:
                self.assertEqual(db.schema_version, 2)
                self.assertEqual(migrations.LATEST_VERSION, 2)
                for table in BACKEND_TABLES:
                    self.assertIn("backend", _columns_of(db, table))
                self.assertNotIn("backend", _columns_of(db, "asks"))
                self.assertNotIn("backend", _columns_of(db, "perm_rules"))
            finally:
                db.close()

    def test_backend_defaults_to_claude_on_new_rows(self) -> None:
        with TempHome() as home:
            db = Database(Paths.from_home(home.home))
            try:
                db.execute("INSERT INTO sessions(thread_id, name) VALUES(1, 'alpha')")
                db.execute("INSERT INTO session_history(thread_id, uuid) VALUES(1, 'u1')")
                db.execute("INSERT INTO turns(thread_id, status) VALUES(1, 'ok')")
                for table in BACKEND_TABLES:
                    self.assertEqual(db.query(f"SELECT backend FROM {table}")[0]["backend"], "claude", table)
            finally:
                db.close()

    def test_state_spool_inbox_are_private_and_db_file_is_0600(self) -> None:
        with TempHome() as home:
            paths = Paths.from_home(home.home)
            db = Database(paths)
            try:
                for directory in (paths.state, paths.spool, paths.inbox):
                    self.assertEqual(stat.S_IMODE(os.stat(directory).st_mode), 0o700, directory)
                self.assertEqual(stat.S_IMODE(os.stat(paths.db).st_mode), 0o600)
            finally:
                db.close()


def _columns_of(db: Database, table: str) -> set[str]:
    return {r["name"] for r in db.query(f"PRAGMA table_info({table})")}


class HistorySeedTests(unittest.TestCase):
    CREATED = "2026-10-01T00:00:00+00:00"

    def test_history_is_seeded_once_from_the_uuids_of_sessions(self) -> None:
        """Ported labels:
        Q9: backfill from uuid and prev_uuid, once
        """
        with TempHome() as home:
            paths = Paths.from_home(home.home)
            Database(paths).close()
            conn = _connect(paths.db)
            try:
                conn.execute("DELETE FROM session_history")
                conn.execute("INSERT INTO sessions(thread_id, name, uuid, prev_uuid, created) VALUES(9405, 'side', "
                             "'cur-uuid', 'old-uuid', ?)", (self.CREATED,))
                conn.execute("PRAGMA user_version = 0")
                migrations.migrate(conn, steps=migrations.MIGRATIONS[:1])
                rows = [tuple(r) for r in conn.execute("SELECT thread_id, uuid, started FROM session_history ORDER BY uuid")]
                self.assertEqual(rows, [(9405, "cur-uuid", self.CREATED), (9405, "old-uuid", self.CREATED)])
                migrations.migrate(conn)
                conn.execute("INSERT INTO sessions(thread_id, name, uuid, created) VALUES(9406, 'other', 'new-uuid', ?)",
                             (self.CREATED,))
                migrations.migrate(conn)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM session_history").fetchone()[0], 2)
            finally:
                conn.close()




class MigrationStepTests(unittest.TestCase):
    def test_running_migrations_twice_is_a_no_op(self) -> None:
        with TempHome() as home:
            paths = Paths.from_home(home.home)
            Database(paths).close()
            conn = _connect(paths.db)
            try:
                schema_before = _schema(conn)
                statements: list[str] = []
                conn.set_trace_callback(statements.append)
                self.assertEqual(migrations.migrate(conn), 2)
                conn.set_trace_callback(None)
                self.assertEqual(_schema(conn), schema_before)
                writes = [s for s in statements if s.lstrip().upper().startswith(
                    ("BEGIN", "CREATE", "ALTER", "INSERT", "UPDATE", "DELETE", "PRAGMA USER_VERSION ="))]
                self.assertEqual(writes, [])
            finally:
                conn.close()

    def test_v2_applies_to_a_database_whose_backend_columns_already_exist(self) -> None:
        with TempHome() as home:
            paths = Paths.from_home(home.home)
            Database(paths).close()
            conn = _connect(paths.db)
            try:
                conn.execute("PRAGMA user_version = 1")  # the columns exist but the version was not bumped
                self.assertEqual(migrations.migrate(conn), 2)
            finally:
                conn.close()

    def test_database_newer_than_this_code_is_refused_and_untouched(self) -> None:
        with TempHome() as home:
            paths = Paths.from_home(home.home)
            Database(paths).close()
            conn = _connect(paths.db)
            try:
                conn.execute("PRAGMA user_version = 99")
                with self.assertRaises(SchemaTooNew):
                    migrations.migrate(conn)
                self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 99)
            finally:
                conn.close()

    def test_failed_step_rolls_back_itself_and_keeps_earlier_steps(self) -> None:
        def make_table_a(conn: sqlite3.Connection) -> None:
            conn.execute("CREATE TABLE a(x)")

        def make_table_b_then_fail(conn: sqlite3.Connection) -> None:
            conn.execute("CREATE TABLE b(x)")
            raise RuntimeError("boom")

        steps = (Migration(1, "a", make_table_a), Migration(2, "b", make_table_b_then_fail))
        with TempHome() as home:
            conn = _connect(Paths.from_home(home.home).db)
            try:
                with self.assertRaises(RuntimeError):
                    migrations.migrate(conn, steps)
                self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 1)
                names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertIn("a", names)
                self.assertNotIn("b", names)
                self.assertFalse(conn.in_transaction)
            finally:
                conn.close()


class TransactionTests(unittest.TestCase):
    def test_tx_commits_on_success_and_rolls_back_on_error(self) -> None:
        with TempHome() as home:
            db = Database(Paths.from_home(home.home))
            try:
                with db.tx():
                    db.execute("INSERT INTO kv(k, v) VALUES('k1', 'v1')")
                with self.assertRaises(ValueError):
                    with db.tx():
                        db.execute("INSERT INTO kv(k, v) VALUES('k2', 'v2')")
                        raise ValueError("stop")
                keys = {r["k"] for r in db.query("SELECT k FROM kv")}
                self.assertIn("k1", keys)
                self.assertNotIn("k2", keys)
            finally:
                db.close()

    def test_nested_tx_joins_the_outer_transaction(self) -> None:
        with TempHome() as home:
            db = Database(Paths.from_home(home.home))
            try:
                with self.assertRaises(RuntimeError):
                    with db.tx():
                        with db.tx():
                            db.execute("INSERT INTO kv(k, v) VALUES('inner', 'x')")
                        db.execute("INSERT INTO kv(k, v) VALUES('outer', 'y')")
                        raise RuntimeError("outer fails")
                self.assertEqual(db.query("SELECT k FROM kv WHERE k IN ('inner', 'outer')"), [])
            finally:
                db.close()

    def test_failed_commit_rolls_back_and_the_next_tx_commits(self) -> None:
        """COMMIT blocked by a reader's SHARED lock raises; the connection must leave the transaction."""
        with TempHome() as home, mock.patch("tg_bridge.store.db.BUSY_TIMEOUT_SECONDS", 0.1):
            paths = Paths.from_home(home.home)
            db = Database(paths)
            reader = sqlite3.connect(paths.db, isolation_level=None)
            try:
                reader.execute("BEGIN")
                reader.execute("SELECT k FROM kv").fetchall()  # holds SHARED until the reader's COMMIT
                with self.assertRaises(sqlite3.OperationalError):
                    with db.tx():
                        db.execute("INSERT INTO kv(k, v) VALUES('a', '1')")
                self.assertFalse(db._conn.in_transaction)
                reader.execute("COMMIT")

                with db.tx():
                    db.execute("INSERT INTO kv(k, v) VALUES('b', '2')")
                self.assertFalse(db._conn.in_transaction)
                keys = {r["k"] for r in db.query("SELECT k FROM kv")}
                self.assertIn("b", keys)
                self.assertNotIn("a", keys)
            finally:
                reader.close()
                db.close()


if __name__ == "__main__":
    unittest.main()
