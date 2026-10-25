"""TurnRepo (turns table): one row per finished turn, and the /cost aggregates over those rows.

Every method is one statement (or one read) under Database.lock, except insert, which also stamps the topic's
sessions.last_active in the same transaction (the live file does the same after each finished turn). A turn's
cost is NULL when the total is not known; cost_known is 0 then, and the /cost line marks it with '*'. As in the live
file, SUM() skips NULLs, so a summary whose turns all have an unknown cost reports cost None ('unknown').
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tg_bridge.store.db import Database

DEFAULT_BACKEND = "claude"
_TOTALS_COLUMNS = (
    "COUNT(*) AS n, SUM(cost) AS cost, SUM(in_tok) AS tin, SUM(out_tok) AS tout, "
    "SUM(CASE WHEN cost_known=0 THEN 1 ELSE 0 END) AS unknown"
)


@dataclass(frozen=True)
class CostTotals:
    """The /cost numbers of one group of turns (one backend, or all of them)."""

    turns: int
    cost: float | None   # None when no turn of the group has a known cost
    tokens_in: int
    tokens_out: int
    unknown: int         # turns whose cost is not known

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "CostTotals":
        cost = row["cost"]
        return cls(
            turns=int(row["n"]),
            cost=None if cost is None else float(cost),
            tokens_in=int(row["tin"] or 0),
            tokens_out=int(row["tout"] or 0),
            unknown=int(row["unknown"] or 0),
        )


class TurnRepo:
    """The turns table: insert one row per finished turn, and read the /cost totals of a topic."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def insert(
        self,
        thread_id: int,
        *,
        started: str,
        ended: str,
        status: str,
        subtype: str = "",
        cost: float | None = None,
        cost_known: bool = True,
        in_tok: int | None = None,
        out_tok: int | None = None,
        num_turns: int | None = None,
        duration_ms: int | None = None,
        error: str | None = None,
        backend: str = DEFAULT_BACKEND,
    ) -> int:
        """Stores one finished turn, stamps the topic's last_active with `ended`, and returns the row id.

        Timestamps are ISO strings, as the live file writes them. A topic with no sessions row is not an error: the
        UPDATE then changes nothing.
        """
        with self._db.tx() as conn:
            cursor = conn.execute(
                "INSERT INTO turns(thread_id, started, ended, status, subtype, cost, cost_known, in_tok, out_tok, "
                "num_turns, duration_ms, error, backend) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (thread_id, started, ended, status, subtype, cost, 1 if cost_known else 0, in_tok, out_tok,
                 num_turns, duration_ms, error, backend),
            )
            conn.execute("UPDATE sessions SET last_active=? WHERE thread_id=?", (ended, thread_id))
            return int(cursor.lastrowid or 0)

    def totals(self, thread_id: int) -> CostTotals:
        """The /cost numbers of the topic over all backends (zeros and cost None for a topic with no turns)."""
        rows = self._db.query(f"SELECT {_TOTALS_COLUMNS} FROM turns WHERE thread_id=?", (thread_id,))
        return CostTotals.from_row(rows[0])

    def summary_by_backend(self, thread_id: int) -> dict[str, CostTotals]:
        """The /cost numbers of the topic, one entry per backend that has turns, keyed by backend name."""
        rows = self._db.query(
            f"SELECT COALESCE(backend, '') AS backend, {_TOTALS_COLUMNS} FROM turns WHERE thread_id=? "
            "GROUP BY COALESCE(backend, '') ORDER BY backend",
            (thread_id,),
        )
        return {str(row["backend"]): CostTotals.from_row(row) for row in rows}
