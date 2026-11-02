"""AskRepo (asks table): one row per permission question put to the user (an ask), keyed by its nonce.

An ask is pending while its answer is ''. answer() is claim-first: it returns True only for the call that changed a
pending ask, so a second tap on the same card (or a replayed callback) finds nothing to change. The permission service
relies on this to answer the session exactly once (DESIGN section 2, answer idempotence). void_all() closes asks whose
turn ended or whose bridge restarted.

JSON columns (input, suggestions) are written from Python values and read back as the stored text.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tg_bridge.clock import Clock, utc_stamp
from tg_bridge.store.db import Database

PENDING = ""
VOID = "void"


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _pending_filter(thread_id: int | None) -> tuple[str, tuple[Any, ...]]:
    """The WHERE clause that selects pending asks, optionally of one topic."""
    if thread_id is None:
        return "answer=?", (PENDING,)
    return "answer=? AND thread_id=?", (PENDING, thread_id)


@dataclass(frozen=True)
class AskRecord:
    """One row of the asks table. `input` and `suggestions` are the JSON text as stored."""

    nonce: str
    thread_id: int
    chat_id: int
    request_id: str
    tool_name: str
    input: str
    suggestions: str
    message_id: int | None
    created: str
    answer: str

    @property
    def pending(self) -> bool:
        return self.answer == PENDING

    def tool_input(self) -> dict[str, Any]:
        """The tool input as a dict; text that is not a JSON object reads as {} (as the live recovery does)."""
        try:
            value = json.loads(self.input or "{}")
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "AskRecord":
        message_id = row["message_id"]
        return cls(
            nonce=str(row["nonce"]),
            thread_id=int(row["thread_id"]),
            chat_id=int(row["chat_id"] or 0),
            request_id=str(row["request_id"] or ""),
            tool_name=str(row["tool_name"] or ""),
            input=str(row["input"] or ""),
            suggestions=str(row["suggestions"] or ""),
            message_id=None if message_id is None else int(message_id),
            created=str(row["created"] or ""),
            answer=str(row["answer"] or ""),
        )


class AskRepo:
    """The asks table: insert when a card is about to be posted, answer or void once it is closed."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def insert(
        self,
        *,
        nonce: str,
        thread_id: int,
        chat_id: int,
        request_id: str,
        tool_name: str,
        input: Mapping[str, Any],
        suggestions: Sequence[Any] = (),
    ) -> None:
        """Registers a pending ask. Its card has no message id yet (set_message_id after the post)."""
        self._db.execute(
            "INSERT INTO asks(nonce, thread_id, chat_id, request_id, tool_name, input, suggestions, message_id, "
            "created, answer) VALUES(?,?,?,?,?,?,?,NULL,?,?)",
            (nonce, thread_id, chat_id, request_id, tool_name, _dump(dict(input)), _dump(list(suggestions)),
             utc_stamp(self._clock.now()), PENDING),
        )

    def set_message_id(self, nonce: str, message_id: int | None) -> None:
        """Stores the Telegram message id of the ask's card. An unknown nonce is ignored."""
        self._db.execute("UPDATE asks SET message_id=? WHERE nonce=?", (message_id, nonce))

    def answer(self, nonce: str, verdict: str) -> bool:
        """Closes a pending ask with its verdict (allow, deny, ...). True only for the call that closed it.

        A second call on the same nonce, or a call for an unknown nonce, returns False and changes nothing.
        """
        if not verdict or verdict == PENDING:
            raise ValueError("an answer needs a non-empty verdict")
        return self._db.execute(
            "UPDATE asks SET answer=? WHERE nonce=? AND answer=?", (verdict, nonce, PENDING)
        ) == 1

    def by_nonce(self, nonce: str) -> AskRecord | None:
        """The ask with this nonce, pending or closed, or None."""
        rows = self._db.query("SELECT * FROM asks WHERE nonce=?", (nonce,))
        return AskRecord.from_row(rows[0]) if rows else None

    def pending(self, thread_id: int | None = None) -> list[AskRecord]:
        """The pending asks, oldest first: all topics, or only those of `thread_id`."""
        where, args = _pending_filter(thread_id)
        rows = self._db.query(f"SELECT * FROM asks WHERE {where} ORDER BY created, rowid", args)
        return [AskRecord.from_row(row) for row in rows]

    def void_all(self, thread_id: int | None = None) -> list[str]:
        """Closes every pending ask (all topics, or only `thread_id`) as void. Returns the nonces it closed."""
        where, args = _pending_filter(thread_id)
        with self._db.tx() as conn:
            nonces = [str(row["nonce"]) for row in conn.execute(f"SELECT nonce FROM asks WHERE {where} "
                                                                  "ORDER BY created, rowid", args).fetchall()]
            for nonce in nonces:
                conn.execute("UPDATE asks SET answer=? WHERE nonce=? AND answer=?", (VOID, nonce, PENDING))
        return nonces

    def delete_thread(self, thread_id: int) -> int:
        """Removes every ask row of the topic, pending ones included (forget_topic). Returns how many were removed."""
        return self._db.execute("DELETE FROM asks WHERE thread_id=?", (thread_id,))
