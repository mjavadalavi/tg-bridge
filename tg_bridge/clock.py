"""Clock protocol (now/monotonic/sleep) + SystemClock; FakeClock lives in tests/fakes/clock.py.

Every service that waits or times something (RateGate, TurnState caps, nudges, outage timers) takes a Clock
in its constructor instead of calling time.* directly. Production passes SystemClock; tests pass a FakeClock so
caps and flood waits run in milliseconds.
"""

import datetime
import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """What a service may ask of time: wall seconds, monotonic seconds, and a blocking sleep."""

    def now(self) -> float:
        """Wall-clock time in seconds since the epoch (time.time)."""
        ...

    def monotonic(self) -> float:
        """Monotonic seconds for measuring intervals (time.monotonic)."""
        ...

    def sleep(self, seconds: float) -> None:
        """Block for the given number of seconds."""
        ...


class SystemClock:
    """The real clock. Stateless, so one instance can be shared by every service."""

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def utc_stamp(seconds: float) -> str:
    """An ISO-8601 UTC timestamp with whole seconds, the format the live file writes into sessions and session_history."""
    return datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc).isoformat(timespec="seconds")
