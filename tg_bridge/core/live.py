"""LiveRegistry: sessions, feeds, aborted, stop_taps, lingering, shutting_down (one lock).

Port of the live registries of tg-bridge.py (old _popen_lk and the sets and maps it guards): thread id -> the live
agent session, thread id -> the turn's progress feed, the threads whose turn was stopped on purpose, the threads
stopped by a feed Stop button, and the shutdown flag. The lingering map holds the sessions kept open after a turn for
background work (one per topic, counted against the session cap); it is part of the shutdown snapshot.

Lock rule: every read and write of these maps happens under self._lock. The lock is never held across a call into
another service (a session stop, a Telegram send): callers copy what they need and act after the lock is released.
A session and its feed are always added and removed together, so "session live" and "feed live" are the same fact.
"""

import threading
from dataclasses import dataclass
from typing import Any

from tg_bridge.agents.base import AgentSession


@dataclass(frozen=True)
class RunningTurn:
    """The live session of one topic and the progress feed of its turn. feed is typed Any: core must not import feed."""

    session: AgentSession
    feed: Any


@dataclass(frozen=True)
class AbortFlags:
    """Why a turn was stopped on purpose: aborted is set by /stop, ⏭ Now and the feed's Stop button; stop_tap by the button."""

    aborted: bool
    stop_tap: bool


class LiveRegistry:
    """The process-wide table of running turns. One instance is created by the App and shared by its services."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[int, AgentSession] = {}
        self._feeds: dict[int, Any] = {}
        self._aborted: set[int] = set()
        self._stop_taps: set[int] = set()
        self._lingering: dict[int, AgentSession] = {}
        self._shutting_down = threading.Event()

    def begin_turn(self, thread_id: int, session: AgentSession, feed: Any) -> bool:
        """Register a turn that is about to run. False (nothing registered) once shutdown has begun: the spawn is refused."""
        with self._lock:
            if self._shutting_down.is_set():
                return False
            self._sessions[thread_id] = session
            self._feeds[thread_id] = feed
            return True

    def end_turn(self, thread_id: int, session: AgentSession, feed: Any) -> None:
        """Remove a finished turn. Each entry is removed only when it still holds this exact object, so a late
        cleanup never drops the entry of a newer turn in the same topic."""
        with self._lock:
            if self._sessions.get(thread_id) is session:
                del self._sessions[thread_id]
            if self._feeds.get(thread_id) is feed:
                del self._feeds[thread_id]

    def running(self, thread_id: int) -> RunningTurn | None:
        """The live turn of a topic, or None when nothing is running there."""
        with self._lock:
            session = self._sessions.get(thread_id)
            feed = self._feeds.get(thread_id)
            if session is None or feed is None:
                return None
            return RunningTurn(session, feed)

    def request_abort(self, thread_id: int, stop_tap: bool) -> RunningTurn | None:
        """Mark the running turn of a topic as stopped on purpose and return it, so the caller can stop the session
        on its own thread. Returns None and marks nothing when no turn is running. stop_tap also marks the feed button."""
        with self._lock:
            session = self._sessions.get(thread_id)
            feed = self._feeds.get(thread_id)
            if session is None or feed is None:
                return None
            self._aborted.add(thread_id)
            if stop_tap:
                self._stop_taps.add(thread_id)
            return RunningTurn(session, feed)

    def is_aborted(self, thread_id: int) -> bool:
        """Whether the topic's turn was stopped on purpose. Read under the lock, like every other registry read."""
        with self._lock:
            return thread_id in self._aborted

    def consume_abort(self, thread_id: int) -> AbortFlags:
        """Return the stop flags of a finished turn and clear them, so the next turn starts unmarked."""
        with self._lock:
            flags = AbortFlags(thread_id in self._aborted, thread_id in self._stop_taps)
            self._aborted.discard(thread_id)
            self._stop_taps.discard(thread_id)
            return flags

    def begin_linger(self, thread_id: int, session: AgentSession, limit: int) -> bool:
        """Keep a session open for a topic after its turn. False (nothing stored) once shutdown has begun, or when
        limit other topics already keep a session. A topic that lingers again replaces its own entry."""
        with self._lock:
            if self._shutting_down.is_set():
                return False
            others = sum(1 for held in self._lingering if held != thread_id)
            if others >= limit:
                return False
            self._lingering[thread_id] = session
            return True

    def end_linger(self, thread_id: int, session: AgentSession) -> None:
        """Forget a kept session. Removed only when the entry still holds this exact session."""
        with self._lock:
            if self._lingering.get(thread_id) is session:
                del self._lingering[thread_id]

    def lingering(self, thread_id: int) -> AgentSession | None:
        """The session kept open for a topic, or None."""
        with self._lock:
            return self._lingering.get(thread_id)

    def lingering_count(self) -> int:
        with self._lock:
            return len(self._lingering)

    def is_shutting_down(self) -> bool:
        return self._shutting_down.is_set()

    def begin_shutdown(self) -> list[AgentSession]:
        """Set the shutdown flag and return the sessions live at this moment, running and lingering ones, each once
        (by identity). Under the same lock as begin_turn, so no session can start after the snapshot is taken."""
        with self._lock:
            self._shutting_down.set()
            by_identity: dict[int, AgentSession] = {}
            for session in list(self._sessions.values()) + list(self._lingering.values()):
                by_identity.setdefault(id(session), session)
            return list(by_identity.values())
