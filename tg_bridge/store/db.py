"""Database: the one sqlite connection of the bridge, shared by every repository.

Locking: `Database.lock` is an RLock held by every query and by tx(). Repositories take it themselves, and callers
never hold it across a Telegram call or a backend call. tx() is re-entrant: a nested tx() joins the outer
transaction, so an exception in the outer block rolls back the inner writes too.

Opening the database creates the state, spool and inbox folders (0700) and migrates the schema (see migrations.py);
the db file itself is set to 0600 after the migration, as the live file does.
"""

import os
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from tg_bridge.paths import Paths
from tg_bridge.store import migrations

PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
BUSY_TIMEOUT_SECONDS = 15


def _chmod(path: str, mode: int) -> None:
    """Best effort chmod: a file system that refuses it keeps its modes, as in the live file."""
    try:
        os.chmod(path, mode)
    except OSError:
        pass


class Database:
    """One sqlite connection (autocommit, check_same_thread off) guarded by `lock`, migrated on construction."""

    def __init__(self, paths: Paths) -> None:
        self.lock = threading.RLock()
        for directory in (paths.state, paths.spool, paths.inbox):
            os.makedirs(directory, exist_ok=True)
            _chmod(directory, PRIVATE_DIR_MODE)
        self._conn = sqlite3.connect(
            paths.db, timeout=BUSY_TIMEOUT_SECONDS, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        try:
            with self.lock:
                migrations.migrate(self._conn)
        except BaseException:
            self._conn.close()
            raise
        _chmod(paths.db, PRIVATE_FILE_MODE)

    @property
    def schema_version(self) -> int:
        """The migrated schema version (PRAGMA user_version)."""
        with self.lock:
            return migrations.user_version(self._conn)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """All rows of a read statement, fetched under the lock."""
        with self.lock:
            return self._conn.execute(sql, params).fetchall()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """One write statement outside an explicit transaction (autocommit). Returns the affected row count."""
        with self.lock:
            return self._conn.execute(sql, params).rowcount

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """A write transaction: BEGIN IMMEDIATE, COMMIT on success, ROLLBACK on any exception. Re-entrant."""
        with self.lock:
            if self._conn.in_transaction:
                yield self._conn
                return
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            try:
                self._conn.execute("COMMIT")
            except BaseException:
                # A failed COMMIT (e.g. 'database is locked' after the busy timeout) leaves the transaction open;
                # roll it back so the connection is not stuck in_transaction and holding its write lock.
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        """Closes the connection. Further use raises sqlite3.ProgrammingError."""
        with self.lock:
            self._conn.close()
