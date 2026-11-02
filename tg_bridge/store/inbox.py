"""CwdRepo (cwd_paths) and InboxRepo (inbox_queue) of the bridge.

CwdRepo: the stable ids of the folders offered by the /cwd buttons. A button carries the id, so a tap applies the
folder its label showed even after the topics' folders changed.

InboxRepo: every Telegram update is stored here before it is acknowledged (ingest), so a crash can re-deliver an update
but never drop one. A row is 'pending' until its work is done (done) or until a deferred message is handled again
(mark_pending). prune() drops finished rows older than the retention period. The update's payload is stored as JSON
text; the InboxService decides what to do with it.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from tg_bridge.clock import Clock, utc_stamp
from tg_bridge.store.db import Database

INBOX_RETENTION = timedelta(days=7)
STATE_PENDING = "pending"
STATE_DONE = "done"


class CwdRepo:
    """Folder path <-> stable id, backed by the cwd_paths table."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def path_id(self, path: str) -> int:
        """The id of a folder path, registering it first when it is new (INSERT OR IGNORE, then the id)."""
        with self._db.tx() as conn:
            conn.execute("INSERT OR IGNORE INTO cwd_paths(path) VALUES(?)", (path,))
            return int(conn.execute("SELECT id FROM cwd_paths WHERE path=?", (path,)).fetchone()["id"])

    def path_of(self, path_id: int) -> str | None:
        """The folder registered under the id, or None when there is no such id."""
        rows = self._db.query("SELECT path FROM cwd_paths WHERE id=?", (path_id,))
        return rows[0]["path"] if rows else None


@dataclass(frozen=True)
class InboxRecord:
    """One row of the inbox_queue table. `payload` is the update as JSON text, `created` an ISO UTC stamp."""

    update_id: int
    thread_id: int | None
    chat_id: int | None
    sender: str
    payload: str
    created: str
    state: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "InboxRecord":
        thread_id, chat_id = row["thread_id"], row["chat_id"]
        return cls(
            update_id=int(row["update_id"]),
            thread_id=None if thread_id is None else int(thread_id),
            chat_id=None if chat_id is None else int(chat_id),
            sender=str(row["sender"] or ""),
            payload=str(row["payload"] or ""),
            created=str(row["created"] or ""),
            state=str(row["state"] or STATE_PENDING),
        )


class InboxRepo:
    """The inbox_queue table: ingest, mark done or pending, list the pending ones, prune old finished rows."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def ingest(
        self,
        update_id: int,
        payload: Mapping[str, Any],
        *,
        thread_id: int | None = None,
        chat_id: int | None = None,
        sender: str = "",
    ) -> bool:
        """Stores a new update as pending. Returns False when this update_id was already stored (a re-delivery)."""
        return self._db.execute(
            "INSERT OR IGNORE INTO inbox_queue(update_id, thread_id, chat_id, sender, payload, created, state) "
            "VALUES(?,?,?,?,?,?,?)",
            (update_id, thread_id, chat_id, sender, json.dumps(dict(payload), ensure_ascii=False),
             utc_stamp(self._clock.now()), STATE_PENDING),
        ) == 1

    def get(self, update_id: int) -> InboxRecord | None:
        """The stored row of one update, or None."""
        rows = self._db.query("SELECT * FROM inbox_queue WHERE update_id=?", (update_id,))
        return InboxRecord.from_row(rows[0]) if rows else None

    def done(self, update_id: int) -> None:
        """Marks one update as finished. An unknown update_id is ignored."""
        self._db.execute("UPDATE inbox_queue SET state=? WHERE update_id=?", (STATE_DONE, update_id))

    def mark_pending(self, update_id: int) -> None:
        """Puts a row back to pending, so the next start replays it (a deferred message)."""
        self._db.execute("UPDATE inbox_queue SET state=? WHERE update_id=?", (STATE_PENDING, update_id))

    def pending(self) -> list[InboxRecord]:
        """Every pending update, in update_id order (the replay order)."""
        rows = self._db.query("SELECT * FROM inbox_queue WHERE state=? ORDER BY update_id", (STATE_PENDING,))
        return [InboxRecord.from_row(row) for row in rows]

    def prune(self) -> int:
        """Deletes finished rows created more than INBOX_RETENTION ago. Returns how many were deleted."""
        cutoff = utc_stamp(self._clock.now() - INBOX_RETENTION.total_seconds())
        return self._db.execute(
            "DELETE FROM inbox_queue WHERE state=? AND created < ?", (STATE_DONE, cutoff)
        )

    def delete_thread(self, thread_id: int) -> int:
        """Removes every queued or finished update of the topic (forget_topic), so nothing of it replays."""
        return self._db.execute("DELETE FROM inbox_queue WHERE thread_id=?", (thread_id,))
