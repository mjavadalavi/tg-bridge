"""Ordered, forward-only schema migrations keyed by PRAGMA user_version.

Version 1 is the live file's schema code (its db() table script and _migrate) ported verbatim. Every statement is
idempotent, so a database the live file created, or one already migrated, is accepted as it is. Version 2 only adds
the `backend` column to sessions, session_history and turns (asks and perm_rules stay as they are: approval ids use
asks.request_id, and rules share one grammar for both backends). Only additive changes are allowed, so the live file
keeps working on a migrated database (rollback safety, DESIGN section 8 item 13).

Each migration runs in its own BEGIN IMMEDIATE transaction together with its user_version bump, so a failure leaves
the database at the previous version. The apply functions must not commit; they need a connection whose row_factory
is sqlite3.Row (Database sets it).
"""

import datetime
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass

LEGACY_LAUNCHER_COLUMNS = (
    ("mode", "TEXT DEFAULT ''"),
    ("model", "TEXT DEFAULT ''"),
    ("prev_uuid", "TEXT DEFAULT ''"),
    ("last_active", "TEXT DEFAULT ''"),
    ("launcher_pick", "TEXT DEFAULT ''"),
)

BACKEND_TABLES = ("sessions", "session_history", "turns")
BACKEND_DEFAULT = "claude"

BASE_TABLES_SCRIPT = """
CREATE TABLE IF NOT EXISTS sessions(
  thread_id INTEGER PRIMARY KEY, name TEXT, uuid TEXT, launcher TEXT,
  cwd TEXT, created TEXT, title TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS cwd_paths(id INTEGER PRIMARY KEY, path TEXT UNIQUE);
"""

LEGACY_TABLES_SCRIPT = """
CREATE TABLE IF NOT EXISTS perm_rules(id INTEGER PRIMARY KEY, thread_id INTEGER, rule TEXT, created TEXT, hits INTEGER DEFAULT 0, UNIQUE(thread_id, rule));
CREATE TABLE IF NOT EXISTS asks(nonce TEXT PRIMARY KEY, thread_id INTEGER, chat_id INTEGER, request_id TEXT, tool_name TEXT, input TEXT, suggestions TEXT, message_id INTEGER, created TEXT, answer TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY, thread_id INTEGER, started TEXT, ended TEXT, status TEXT, subtype TEXT, cost REAL, cost_known INTEGER DEFAULT 1, in_tok INTEGER, out_tok INTEGER, num_turns INTEGER, tools INTEGER, duration_ms INTEGER, error TEXT);
CREATE TABLE IF NOT EXISTS inbox_queue(id INTEGER PRIMARY KEY, update_id INTEGER UNIQUE, thread_id INTEGER, chat_id INTEGER, sender TEXT, payload TEXT, created TEXT, state TEXT DEFAULT 'pending');
CREATE TABLE IF NOT EXISTS session_history(id INTEGER PRIMARY KEY, thread_id INTEGER, uuid TEXT, started TEXT, first_text TEXT DEFAULT '', UNIQUE(thread_id, uuid));
"""


class SchemaTooNew(RuntimeError):
    """The database was migrated by a newer version of the code; this code will not touch it."""


@dataclass(frozen=True)
class Migration:
    """One schema step: the version it brings the database to, a label, and its apply function."""

    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


def run_script(conn: sqlite3.Connection, script: str) -> None:
    """Executes each statement of a script inside the current transaction.

    sqlite3's executescript() commits any open transaction first, which would break the one-transaction-per-migration
    rule, so the script is split on ';' (none of the scripts here has a ';' inside a string literal).
    """
    pending = ""
    for piece in script.split(";"):
        pending += piece + ";"
        if sqlite3.complete_statement(pending):
            statement = pending.strip().rstrip(";").strip()
            if statement:
                conn.execute(statement)
            pending = ""


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Column names of a table (PRAGMA table_info, column 1 is the name)."""
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def user_version(conn: sqlite3.Connection) -> int:
    """The schema version stored in the database header."""
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _v1_base(conn: sqlite3.Connection) -> None:
    """Live file db() table script, then its _migrate(c) body, ported verbatim (commit calls dropped)."""
    run_script(conn, BASE_TABLES_SCRIPT)
    have = columns(conn, "sessions")
    for col, ddl in LEGACY_LAUNCHER_COLUMNS:
        if col not in have:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {ddl}")
    run_script(conn, LEGACY_TABLES_SCRIPT)
    if "cwd" not in columns(conn, "session_history"):
        conn.execute("ALTER TABLE session_history ADD COLUMN cwd TEXT DEFAULT ''")
    if "launcher" not in columns(conn, "session_history"):
        conn.execute("ALTER TABLE session_history ADD COLUMN launcher TEXT DEFAULT ''")
    if not conn.execute("SELECT 1 FROM session_history LIMIT 1").fetchone():  # seed once from the uuids already known
        conn.execute(
            "INSERT OR IGNORE INTO session_history(thread_id, uuid, started) "
            "SELECT thread_id, uuid, created FROM sessions WHERE uuid!=''"
        )
        conn.execute(
            "INSERT OR IGNORE INTO session_history(thread_id, uuid, started) "
            "SELECT thread_id, prev_uuid, created FROM sessions WHERE prev_uuid!=''"
        )
    _move_legacy_kv(conn)


def _move_legacy_kv(conn: sqlite3.Connection) -> None:
    """The one-time kv -> table move (guarded by kv key 'migrated:v2') of the live file's _migrate."""
    if conn.execute("SELECT 1 FROM kv WHERE k='migrated:v2'").fetchone():
        return
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows = [(r["k"], r["v"]) for r in conn.execute("SELECT k, v FROM kv")]
    for k, v in rows:
        try:
            if k.startswith("auto:"):
                conn.execute(
                    "UPDATE sessions SET mode='bypassPermissions' WHERE thread_id=?", (int(k[len("auto:"):]),)
                )
            elif k.startswith("grant:") and v == "always":
                for rule in ("Edit", "Write"):
                    conn.execute(
                        "INSERT OR IGNORE INTO perm_rules(thread_id, rule, created) VALUES(?,?,?)",
                        (int(k[len("grant:"):]), rule, now),
                    )
        except ValueError:
            continue  # malformed legacy key: skip it, it is deleted below anyway
    for k, _ in rows:
        if k.startswith(("ask_answer:", "grant:", "auto:")):
            conn.execute("DELETE FROM kv WHERE k=?", (k,))
    conn.execute("INSERT INTO kv(k,v) VALUES('migrated:v2','1') ON CONFLICT(k) DO UPDATE SET v=excluded.v")


def _v2_backend(conn: sqlite3.Connection) -> None:
    """Adds sessions.backend, session_history.backend and turns.backend (existing rows read as Claude)."""
    for table in BACKEND_TABLES:
        if "backend" not in columns(conn, table):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN backend TEXT DEFAULT '{BACKEND_DEFAULT}'")


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "base schema (live file _migrate)", _v1_base),
    Migration(2, "backend columns", _v2_backend),
)

LATEST_VERSION = MIGRATIONS[-1].version


def migrate(conn: sqlite3.Connection, steps: Sequence[Migration] = MIGRATIONS) -> int:
    """Brings the database up to the latest version and returns it. Running it again is a no-op.

    Raises SchemaTooNew when the database is already newer than the last step, and leaves it untouched.
    """
    current = user_version(conn)
    newest = max((step.version for step in steps), default=0)
    if current > newest:
        raise SchemaTooNew(f"database schema v{current} is newer than this code (v{newest})")
    for step in steps:
        if user_version(conn) >= step.version:
            continue
        conn.execute("BEGIN IMMEDIATE")
        try:
            if user_version(conn) < step.version:  # another process may have migrated while we waited for the lock
                step.apply(conn)
                conn.execute(f"PRAGMA user_version = {int(step.version)}")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    return user_version(conn)
