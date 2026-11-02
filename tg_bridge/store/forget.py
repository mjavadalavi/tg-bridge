"""TopicForget: removes one topic's rows and kv keys in one transaction (the store part of a topic removal).

Cleared: the sessions row, every session_history row, the perm_rules rows (RuleRepo.delete_all), the asks rows, the
inbox_queue rows of the topic, and the kv keys tied to its thread id (THREAD_KV_PREFIXES) plus the pane:<name> key
of its topic name. Kept on purpose: turns (the cost history) and cwd_paths (keyed by folder, not by topic), and every
other topic's rows. forget_topic runs under Database.tx(), so either all of it is removed or none of it is.
"""

from tg_bridge.store.asks import AskRepo
from tg_bridge.store.db import Database
from tg_bridge.store.inbox import InboxRepo
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.rules import RuleRepo
from tg_bridge.store.sessions import HistoryRepo, SessionRepo

# kv keys written as '<prefix><thread id>' by core: thread (topic name), slash (CLI commands), linger (kept session),
# handoff (pending digest), active (turn in progress), cwd_pending (folder tapped before the row existed)
THREAD_KV_PREFIXES = ("thread:", "slash:", "linger:", "handoff:", "active:", "cwd_pending:")
PANE_KV_PREFIX = "pane:"  # keyed by the topic name, not by the thread id


class TopicForget:
    """Forgets one topic: every table and kv key that belongs to its thread id."""

    def __init__(self, *, db: Database, sessions: SessionRepo, history: HistoryRepo, rules: RuleRepo, asks: AskRepo,
                 inbox: InboxRepo, kv: KvRepo) -> None:
        self._db = db
        self._sessions = sessions
        self._history = history
        self._rules = rules
        self._asks = asks
        self._inbox = inbox
        self._kv = kv

    def forget_topic(self, thread_id: int) -> None:
        """Removes everything the topic left behind, in one transaction. A failure rolls all of it back."""
        with self._db.tx():
            record = self._sessions.get(thread_id)
            if record is not None and record.name:
                self._kv.delete(f"{PANE_KV_PREFIX}{record.name}")
            for prefix in THREAD_KV_PREFIXES:
                self._kv.delete(f"{prefix}{thread_id}")
            self._sessions.delete_thread(thread_id)
            self._history.delete_thread(thread_id)
            self._rules.delete_all(thread_id)
            self._asks.delete_thread(thread_id)
            self._inbox.delete_thread(thread_id)
