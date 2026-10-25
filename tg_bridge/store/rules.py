"""RuleRepo (perm_rules table): the always-allow rules of each topic, e.g. `Edit` or `Bash(git status *)`.

The rule text is only stored here; matching is done by agents/rules.py RuleGrammar for both backends. A topic's rules
are unique (UNIQUE(thread_id, rule)): adding a rule that is already there changes nothing. hits counts how often a
rule answered an ask, and /rules shows it. Every method is one statement under Database.lock.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tg_bridge.clock import Clock, utc_stamp
from tg_bridge.store.db import Database


@dataclass(frozen=True)
class RuleRecord:
    """One always-allow rule of a topic. `id` is the number the /rules list and /forget use."""

    id: int
    thread_id: int
    rule: str
    hits: int
    created: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "RuleRecord":
        return cls(
            id=int(row["id"]),
            thread_id=int(row["thread_id"]),
            rule=str(row["rule"]),
            hits=int(row["hits"] or 0),
            created=str(row["created"] or ""),
        )


class RuleRepo:
    """The perm_rules table: add, list, count a hit, and delete one or all rules of a topic."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def add(self, thread_id: int, rule: str) -> bool:
        """Saves a rule for the topic. Returns True when it was new, False when the topic already had it."""
        return self._db.execute(
            "INSERT OR IGNORE INTO perm_rules(thread_id, rule, created) VALUES(?,?,?)",
            (thread_id, rule, utc_stamp(self._clock.now())),
        ) == 1

    def for_thread(self, thread_id: int) -> list[RuleRecord]:
        """The topic's rules in the order they were added (the /rules numbering)."""
        rows = self._db.query("SELECT * FROM perm_rules WHERE thread_id=? ORDER BY id", (thread_id,))
        return [RuleRecord.from_row(row) for row in rows]

    def texts(self, thread_id: int) -> list[str]:
        """Only the rule texts of the topic, in the order they were added (what the matcher needs)."""
        return [record.rule for record in self.for_thread(thread_id)]

    def hit(self, thread_id: int, rule: str) -> None:
        """Counts one use of a rule of the topic. An unknown rule is ignored."""
        self._db.execute("UPDATE perm_rules SET hits=hits+1 WHERE thread_id=? AND rule=?", (thread_id, rule))

    def delete(self, thread_id: int, rule_id: int) -> bool:
        """Removes one rule by its number, only if it belongs to the topic. True when a rule was removed."""
        return self._db.execute("DELETE FROM perm_rules WHERE thread_id=? AND id=?", (thread_id, rule_id)) == 1

    def delete_all(self, thread_id: int) -> int:
        """Removes every rule of the topic and returns how many were removed."""
        return self._db.execute("DELETE FROM perm_rules WHERE thread_id=?", (thread_id,))
