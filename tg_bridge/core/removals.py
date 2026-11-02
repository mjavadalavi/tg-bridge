"""Removals: the mark of a topic being removed (card F2), shared by TopicRemover and the Dispatcher.

A marked topic is being removed: the dispatcher drops its messages (their inbox rows are marked done) and starts no turn
for it. TopicRemover sets the mark first, before it clears the queue and stops the session, so a message delivered from
then on cannot start a turn. After a removal that was forgotten, the mark stays GRACE_SECONDS more, so a late update
cannot re-create the topic's rows. A refused removal clears the mark at once (cancel).

Lock map: one lock guards the table. No caller holds it while it calls another service; the clock is read under it,
and a clock never calls back.
Threads: the remover's caller (a callback, the probe or the view), the inbound handler and the dispatcher's runners.
"""

import threading

from tg_bridge.clock import Clock

GRACE_SECONDS = 60.0  # how long a forgotten topic stays marked, so a late update cannot re-create its row


class Removals:
    """The marked topics. A removal marks a topic with None (until it is forgotten or refused); a forgotten topic
    keeps a deadline on the clock's monotonic time."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._marks: dict[int | str, float | None] = {}

    def begin(self, thread_id: int | str) -> bool:
        """Marks the topic as being removed. False (nothing changed) when a removal of it is already running. A topic
        forgotten a moment ago may be removed again: its mark simply runs again."""
        with self._lock:
            if thread_id in self._marks and self._marks[thread_id] is None:
                return False
            self._marks[thread_id] = None
            return True

    def active(self, thread_id: int | str) -> bool:
        """True while the topic is marked: its messages are dropped and no turn starts for it."""
        with self._lock:
            return self._active_locked(thread_id)

    def finish(self, thread_id: int | str) -> None:
        """The topic was forgotten: the mark stays for GRACE_SECONDS more."""
        with self._lock:
            self._marks[thread_id] = self._clock.monotonic() + GRACE_SECONDS

    def cancel(self, thread_id: int | str) -> None:
        """Clears the mark at once (a refused removal)."""
        with self._lock:
            self._marks.pop(thread_id, None)

    def _active_locked(self, thread_id: int | str) -> bool:
        if thread_id not in self._marks:
            return False
        deadline = self._marks[thread_id]
        if deadline is None or self._clock.monotonic() < deadline:
            return True
        del self._marks[thread_id]  # the grace is over: the mark goes
        return False
