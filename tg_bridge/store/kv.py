"""KvRepo: the small key/value table (kv) of the bridge.

Each call is one autocommitted statement under Database.lock, so callers never hold a lock of their own. Values are
text, as in the live file (kv_get, kv_set, kv_del).
"""

from tg_bridge.store.db import Database


class KvRepo:
    """Read, write and delete of kv rows."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, key: str, default: str | None = None) -> str | None:
        """The stored value, or `default` when the key is absent."""
        rows = self._db.query("SELECT v FROM kv WHERE k=?", (key,))
        return rows[0]["v"] if rows else default

    def set(self, key: str, value: str) -> None:
        """Stores the value under the key, replacing any earlier one (upsert)."""
        self._db.execute("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (key, value))

    def delete(self, key: str) -> None:
        """Removes the key; a missing key is not an error."""
        self._db.execute("DELETE FROM kv WHERE k=?", (key,))

    def items_with_prefix(self, prefix: str) -> list[tuple[str, str]]:
        """Every (key, value) whose key starts with prefix, ordered by key (the active:<thread> markers at start-up)."""
        rows = self._db.query("SELECT k, v FROM kv WHERE substr(k, 1, ?) = ? ORDER BY k", (len(prefix), prefix))
        return [(row["k"], row["v"]) for row in rows]
