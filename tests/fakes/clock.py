"""FakeClock: a Clock whose time moves only when a test says so.

advance(s) moves time forward. sleep(s) does not block: it records the request in `sleeps` and moves time forward
by s, so a loop that sleeps until a deadline finishes at once. now() and monotonic() return the same value,
because a test never needs the two to differ.
"""

import threading

DEFAULT_START = 1_700_000_000.0


class FakeClock:
    """A thread-safe, manually advanced Clock."""

    def __init__(self, start: float = DEFAULT_START) -> None:
        self._now = float(start)
        self._lock = threading.Lock()
        self.sleeps: list[float] = []

    def now(self) -> float:
        with self._lock:
            return self._now

    def monotonic(self) -> float:
        with self._lock:
            return self._now

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("sleep() needs a non-negative number of seconds")
        with self._lock:
            self.sleeps.append(float(seconds))
            self._now += float(seconds)

    def advance(self, seconds: float) -> None:
        """Move time forward by seconds without recording a sleep."""
        if seconds < 0:
            raise ValueError("advance() needs a non-negative number of seconds")
        with self._lock:
            self._now += float(seconds)
