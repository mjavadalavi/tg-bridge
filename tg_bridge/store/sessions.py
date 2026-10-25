"""SessionRepo (sessions table) and HistoryRepo (session_history table).

SessionRepo keeps one row per topic: the Claude or Codex session id in use (`uuid`), the id it replaced (`prev_uuid`),
the folder, the mode and model, the launcher pick and the backend. `launcher` is written only once, by ensure(), and it
is always a Claude launcher command: a Codex topic keeps its Claude command there so that a rollback to the old file
still finds a valid command (DESIGN section 8, item 13).

HistoryRepo keeps every session a topic has run, newest last, for the /resume list. Each row remembers the folder and
the launcher it ran under, because each launcher keeps its own transcripts.

Every method goes through Database, which takes Database.lock; callers never hold a lock of their own across these
calls. Writes that touch more than one statement run in one transaction.
"""

import os
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, fields

from tg_bridge.clock import Clock, utc_stamp
from tg_bridge.store.db import Database

KNOWN_BACKENDS = ("claude", "codex")
FIRST_TEXT_CHARS = 60
HISTORY_LIST_LIMIT = 8
_SESSION_COLUMNS = frozenset({"mode", "model", "launcher_pick", "cwd", "title", "backend"})


def _folder(cwd: str) -> str:
    """The folder as the live file stores it: user-expanded and resolved, or '' when unknown."""
    return os.path.realpath(os.path.expanduser(cwd)) if cwd else ""


def _text(row: sqlite3.Row | Mapping[str, object], key: str) -> str:
    """A text column as a str; a NULL reads as ''."""
    value = row[key]
    return "" if value is None else str(value)


@dataclass(frozen=True)
class SessionRecord:
    """One row of the sessions table (one topic)."""

    thread_id: int
    name: str
    uuid: str
    launcher: str
    cwd: str
    created: str
    title: str
    mode: str
    model: str
    prev_uuid: str
    last_active: str
    launcher_pick: str
    backend: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "SessionRecord":
        values = {f.name: _text(row, f.name) for f in fields(cls) if f.name != "thread_id"}
        return cls(thread_id=int(row["thread_id"]), **values)


@dataclass(frozen=True)
class HistoryRecord:
    """One row of the session_history table (one session of one topic)."""

    id: int
    thread_id: int
    uuid: str
    started: str
    first_text: str
    cwd: str
    launcher: str
    backend: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "HistoryRecord":
        return cls(
            id=int(row["id"]),
            thread_id=int(row["thread_id"]),
            uuid=_text(row, "uuid"),
            started=_text(row, "started"),
            first_text=_text(row, "first_text"),
            cwd=_text(row, "cwd"),
            launcher=_text(row, "launcher"),
            backend=_text(row, "backend") or "claude",
        )


def _check_backend(backend: str) -> str:
    if backend not in KNOWN_BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {KNOWN_BACKENDS}")
    return backend


class SessionRepo:
    """The sessions table: one row per topic, created by ensure() and updated column by column."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def get(self, thread_id: int) -> SessionRecord | None:
        """The topic's row, or None when the topic has no session row yet."""
        rows = self._db.query("SELECT * FROM sessions WHERE thread_id=?", (thread_id,))
        return SessionRecord.from_row(rows[0]) if rows else None

    def ensure(self, thread_id: int, name: str, claude_launcher: str, cwd: str, mode: str = "") -> bool:
        """Creates the topic's row if it is missing (INSERT OR IGNORE: an existing row is never changed).

        `claude_launcher` is the only writer of sessions.launcher and must be a Claude launcher command. Returns True
        when a row was created.
        """
        created = self._db.execute(
            "INSERT OR IGNORE INTO sessions(thread_id, name, uuid, launcher, cwd, created, mode) "
            "VALUES(?,?,'',?,?,?,?)",
            (thread_id, name, claude_launcher, cwd, utc_stamp(self._clock.now()), mode),
        )
        return created == 1

    def set_uuid(self, thread_id: int, uuid: str) -> None:
        """Stores the session id the topic runs on now."""
        self._db.execute("UPDATE sessions SET uuid=? WHERE thread_id=?", (uuid, thread_id))

    def new_session(self, thread_id: int) -> None:
        """Forgets the current session id (/new): it moves to prev_uuid, and uuid becomes ''.

        With an empty uuid already, prev_uuid is left as it was.
        """
        self._db.execute(
            "UPDATE sessions SET prev_uuid=CASE WHEN uuid!='' THEN uuid ELSE prev_uuid END, uuid='' WHERE thread_id=?",
            (thread_id,),
        )

    def bind_session(
        self, thread_id: int, uuid: str, cwd: str | None = None, launcher_pick: str | None = None
    ) -> None:
        """Binds the topic to an existing session id (/resume). The id in use moves to prev_uuid in the same statement.

        cwd and launcher_pick change only when given (not None).
        """
        sets = ["prev_uuid=CASE WHEN uuid!='' THEN uuid ELSE prev_uuid END", "uuid=?"]
        args: list[object] = [uuid]
        if cwd is not None:
            sets.append("cwd=?")
            args.append(cwd)
        if launcher_pick is not None:
            sets.append("launcher_pick=?")
            args.append(launcher_pick)
        with self._db.tx() as conn:
            conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE thread_id=?", (*args, thread_id))

    def set_mode(self, thread_id: int, mode: str) -> None:
        self._set_column(thread_id, "mode", mode)

    def set_model(self, thread_id: int, model: str) -> None:
        self._set_column(thread_id, "model", model)

    def set_launcher_pick(self, thread_id: int, pick: str) -> None:
        """The /launcher choice. It never changes sessions.launcher, which stays a Claude command."""
        self._set_column(thread_id, "launcher_pick", pick)

    def set_cwd(self, thread_id: int, cwd: str) -> None:
        self._set_column(thread_id, "cwd", cwd)

    def set_title(self, thread_id: int, title: str) -> None:
        self._set_column(thread_id, "title", title)

    def set_backend(self, thread_id: int, backend: str) -> None:
        """Which backend the topic runs on now. Raises ValueError for an unknown backend."""
        self._set_column(thread_id, "backend", _check_backend(backend))

    def most_recent_thread(self) -> int | None:
        """The topic with the latest last_active (the outage notice goes there), or None when there are no topics.

        last_active is written by TurnRepo.insert when a turn finishes, not by this repo.
        """
        rows = self._db.query("SELECT thread_id FROM sessions ORDER BY last_active DESC LIMIT 1")
        return int(rows[0]["thread_id"]) if rows else None

    def thread_of(self, name: str | None) -> int | None:
        """The topic whose session row has this name (the spool drain addresses topics by name), or None."""
        rows = self._db.query("SELECT thread_id FROM sessions WHERE name=? ORDER BY thread_id LIMIT 1", (name,))
        return int(rows[0]["thread_id"]) if rows else None

    def cwds(self) -> list[str]:
        """The non-empty folders of all topics, in topic order (duplicates kept; callers de-duplicate)."""
        return [_text(row, "cwd") for row in self._db.query("SELECT cwd FROM sessions WHERE cwd!='' ORDER BY thread_id")]

    def _set_column(self, thread_id: int, column: str, value: str) -> None:
        if column not in _SESSION_COLUMNS:  # the column name is never user input, but it is spliced into SQL
            raise ValueError(f"not a settable sessions column: {column}")
        self._db.execute(f"UPDATE sessions SET {column}=? WHERE thread_id=?", (value, thread_id))


class HistoryRepo:
    """The session_history table: every session a topic has run, for the /resume list."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def note_session(self, thread_id: int, uuid: str, text: str, cwd: str = "", launcher: str = "",
                     backend: str = "claude") -> None:
        """Records a session the first time a turn runs on it (the port of the live _note_session).

        The first 60 characters of the user text (whitespace collapsed) are its label. A uuid is recorded once: a second
        call changes only an empty folder or an empty launcher, so a row recorded before those were stored takes them
        from the next turn.
        """
        first = " ".join(str(text or "").split())[:FIRST_TEXT_CHARS]
        folder = _folder(cwd)
        launcher = launcher or ""
        with self._db.tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO session_history(thread_id, uuid, started, first_text, cwd, launcher, backend) "
                "VALUES(?,?,?,?,?,?,?)",
                (thread_id, uuid, utc_stamp(self._clock.now()), first, folder, launcher, backend),
            )
            conn.execute(
                "UPDATE session_history SET cwd=? WHERE thread_id=? AND uuid=? AND cwd=''", (folder, thread_id, uuid)
            )
            conn.execute(
                "UPDATE session_history SET launcher=? WHERE thread_id=? AND uuid=? AND launcher=''",
                (launcher, thread_id, uuid),
            )

    def adopt_session(self, thread_id: int, uuid: str, launcher: str, cwd: str, backend: str = "claude") -> None:
        """Records that the session now runs under `launcher` (it stayed in its folder, or was copied into it).

        The port of the live _adopt_session: insert-or-ignore, then the launcher is set even when a row already had one.
        """
        with self._db.tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO session_history(thread_id, uuid, started, first_text, cwd, launcher, backend) "
                "VALUES(?,?,?,'',?,?,?)",
                (thread_id, uuid, utc_stamp(self._clock.now()), _folder(cwd), launcher, backend),
            )
            conn.execute(
                "UPDATE session_history SET launcher=? WHERE thread_id=? AND uuid=?", (launcher, thread_id, uuid)
            )

    def list_for(
        self, thread_id: int, backend: str, cwd: str | None = None, limit: int = HISTORY_LIST_LIMIT
    ) -> list[HistoryRecord]:
        """The newest `limit` sessions of the topic that ran on `backend`, newest first.

        With `cwd` given, only the sessions recorded in that folder (or with no folder recorded) are returned. None (the
        default) returns sessions from every folder.
        """
        sql = "SELECT * FROM session_history WHERE thread_id=? AND backend=?"
        args: list[object] = [thread_id, backend]
        if cwd is not None:
            sql += " AND (cwd=? OR cwd='')"
            args.append(_folder(cwd))
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        return [HistoryRecord.from_row(row) for row in self._db.query(sql, args)]

    def find(self, thread_id: int, uuid: str) -> HistoryRecord | None:
        """The history row of one session of the topic, or None."""
        rows = self._db.query("SELECT * FROM session_history WHERE thread_id=? AND uuid=?", (thread_id, uuid))
        return HistoryRecord.from_row(rows[0]) if rows else None

    def find_by_id(self, thread_id: int, history_id: int) -> HistoryRecord | None:
        """The history row with this id, if it belongs to the topic (a /resume button carries the id)."""
        rows = self._db.query("SELECT * FROM session_history WHERE id=? AND thread_id=?", (history_id, thread_id))
        return HistoryRecord.from_row(rows[0]) if rows else None
