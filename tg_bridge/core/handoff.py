"""HandoffService: digest (newest kept, ~12k cap) stored in kv handoff:<thread>.

When a topic switches between Claude and Codex the new agent starts with no memory of the conversation. This service
turns the old session's entries (AgentBackend.handoff_entries, read from a transcript or rollout) into one digest of at
most DIGEST_CAP characters and keeps it in kv under handoff:<thread> until the next fresh session takes it with take().

build_digest keeps the NEWEST entries that fit. When the newest single entry is too long on its own it is cut to the
room left, and when anything older was dropped the digest says so with OMITTED. The digest holds chat text, so it lives
only in kv and is never logged; the log line only states its length. take() is one-shot: the row is deleted in the
same step that reads it, under the service lock, so two callers can never both receive one digest.
"""

import threading
from collections.abc import Iterable
from types import MappingProxyType

from tg_bridge.agents.events import HandoffEntry
from tg_bridge.agents.registry import BackendRegistry
from tg_bridge.config import LauncherSpec
from tg_bridge.logging_ import Logger
from tg_bridge.store.kv import KvRepo

DIGEST_CAP = 12000
KEY_PREFIX = "handoff:"
HEADER = "Context from the previous agent in this topic: "
OMITTED = "[earlier conversation omitted]"
ELLIPSIS = "…"
LABELS = MappingProxyType({"user": "User", "assistant": "Agent", "tool": "Tool"})


def handoff_key(thread: int) -> str:
    """The kv key that holds a topic's pending digest."""
    return f"{KEY_PREFIX}{thread}"


def build_digest(entries: Iterable[HandoffEntry], cap: int = DIGEST_CAP) -> str:
    """The digest text for entries given oldest first, at most `cap` characters; '' when nothing is left to carry.

    Entries with an unknown role or empty text are skipped. The result is HEADER followed by the kept lines, oldest
    first, one per line, with OMITTED on the line before them when older entries were dropped.
    """
    lines = [line for line in (_line(entry) for entry in entries) if line]
    if not lines:
        return ""
    full = HEADER + "\n".join(lines)
    if len(full) <= cap:
        return full
    budget = cap - len(HEADER) - len(OMITTED) - len("\n")
    kept = _newest_that_fit(lines, budget)
    if not kept:
        return ""
    return HEADER + OMITTED + "\n" + "\n".join(kept)


def _line(entry: HandoffEntry) -> str:
    """'User: ...', 'Agent: ...' or 'Tool: ...' for one entry; '' for an entry that is skipped."""
    label = LABELS.get(entry.role)
    text = entry.text.strip()
    if label is None or not text:
        return ""
    return f"{label}: {text}"


def _newest_that_fit(lines: list[str], budget: int) -> list[str]:
    """The longest suffix of lines (oldest first) whose joined length fits budget.

    Only the newest line may be cut: when it is too long for the room left it is shortened to that room. Nothing older
    is cut, so once a line does not fit the scan stops.
    """
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        separator = len("\n") if kept else 0
        room = budget - used - separator
        if len(line) <= room:
            kept.append(line)
            used += separator + len(line)
            continue
        if not kept and room > 0:
            kept.append(_cut(line, room))
        break
    kept.reverse()
    return kept


def _cut(line: str, room: int) -> str:
    """The line shortened to at most `room` characters, ending in an ellipsis when there is space for one."""
    if len(line) <= room:
        return line
    if room > len(ELLIPSIS):
        return line[: room - len(ELLIPSIS)] + ELLIPSIS
    return line[:room]


class HandoffService:
    """Builds, stores, and takes the handoff digest of a topic.

    Lock: `_lock` makes take() read-then-delete atomic. It is never held across a backend call; prepare() reads the old
    session's entries before taking it.
    """

    def __init__(self, kv: KvRepo, registry: BackendRegistry, logger: Logger) -> None:
        self._kv = kv
        self._registry = registry
        self._logger = logger
        self._lock = threading.Lock()

    def prepare(self, thread: int, from_backend: str, from_spec: LauncherSpec,
                session_id: str | None, cwd: str) -> bool:
        """Reads the old session of a topic, stores its digest in kv and returns True; False when nothing is carried.

        from_backend is the registry name of the old session's backend. An empty result also deletes any digest that an
        earlier handoff left for this topic, so a stale one can never reach a later session. With no session id there is
        nothing to read, so a pending digest is kept as it is and True is returned when one exists.
        """
        if not session_id:
            # nothing to read: a digest left by an earlier switch still describes the conversation (a quick
            # Claude -> Codex -> Claude toggle before any message), so it stays pending for the next fresh session
            return self.peek(thread) is not None
        entries = self._registry.get(from_backend).handoff_entries(session_id, cwd, from_spec)
        digest = build_digest(entries)
        key = handoff_key(thread)
        if not digest:
            self._kv.delete(key)
            return False
        with self._lock:
            self._kv.set(key, digest)
        self._logger.log(f"handoff: thread {thread} digest of {len(digest)} chars prepared from {from_backend}")
        return True

    def take(self, thread: int) -> str | None:
        """The pending digest of a topic, removed from kv in the same step. None when there is none."""
        key = handoff_key(thread)
        with self._lock:
            digest = self._kv.get(key)
            self._kv.delete(key)
        return digest or None

    def peek(self, thread: int) -> str | None:
        """The pending digest of a topic, left in kv. The turn runner reads it for a fresh session and calls discard()
        only once that session has started, so a failed start keeps the digest for the next try."""
        return self._kv.get(handoff_key(thread)) or None

    def discard(self, thread: int) -> None:
        """Removes the pending digest of a topic from kv. Idempotent."""
        with self._lock:
            self._kv.delete(handoff_key(thread))
