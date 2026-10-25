"""RateGate: per-minute budgets (group 20, feed 10), flood-wait gate (C04).

Port of _send_slot and _flood_wait of the live tg-bridge.py (C04). Every sendMessage of the group shares one
sliding 60 s window: at most group_per_min sends in any minute, across every topic. The live feed may use only
feed_per_min of them, so answers, cards and replies always find room. A HTTP 429 closes the gate for everyone for
the server's retry_after (capped at retry_after_max).

The gate owns its lock and its timestamps (no module-level state) and waits through the injected Clock, so tests
run the whole budget in milliseconds. Parsing the 429 body stays in the client; the gate only receives the seconds.
"""

import threading
from collections import deque

from tg_bridge.clock import Clock

GROUP_SENDS_PER_MINUTE = 20  # Telegram's per-group limit is about 20 messages a minute
FEED_SENDS_PER_MINUTE = 10  # the live feed's share; the other half stays free for answers, cards and replies
RETRY_AFTER_MAX = 300  # a 429 never closes the gate for longer than this many seconds
DEFAULT_FLOOD_WAIT = 30  # used when a 429 carries no usable retry_after
MIN_FLOOD_WAIT = 1
WINDOW_SECONDS = 60.0
MIN_POLL_SECONDS = 0.05  # the shortest pause between two checks of a full budget


class RateGate:
    """Shared send budget and flood-wait gate for one bot. Thread-safe; blocks in slot() until there is room."""

    def __init__(
        self,
        clock: Clock,
        group_per_min: int = GROUP_SENDS_PER_MINUTE,
        feed_per_min: int = FEED_SENDS_PER_MINUTE,
        retry_after_max: int = RETRY_AFTER_MAX,
    ) -> None:
        for name, value in (
            ("group_per_min", group_per_min),
            ("feed_per_min", feed_per_min),
            ("retry_after_max", retry_after_max),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._clock = clock
        self._group_cap = group_per_min
        self._feed_cap = feed_per_min
        self._retry_after_max = retry_after_max
        self._lock = threading.Lock()
        self._stamps: deque[float] = deque()  # clock.monotonic() stamps of the sends in the last minute
        self._flood_until = 0.0  # a 429 keeps the gate closed until this clock.monotonic() time

    def slot(self, feed: bool = False) -> None:
        """Block until the send budget has room, then take one slot. feed=True uses the feed's smaller cap."""
        cap = self._feed_cap if feed else self._group_cap
        while True:
            with self._lock:
                now = self._clock.monotonic()
                while self._stamps and now - self._stamps[0] >= WINDOW_SECONDS:
                    self._stamps.popleft()
                wait = self._flood_until - now
                if wait <= 0 and len(self._stamps) < cap:
                    self._stamps.append(now)
                    return
                if wait <= 0:
                    wait = self._stamps[0] + WINDOW_SECONDS - now
            self._clock.sleep(max(wait, MIN_POLL_SECONDS))

    def flood_wait(self, retry_after: float | int | None) -> int:
        """Close the gate for everyone for retry_after seconds (a 429). Returns the seconds the gate was closed for.

        A missing or zero retry_after counts as DEFAULT_FLOOD_WAIT; the value is clamped to
        [MIN_FLOOD_WAIT, retry_after_max]. A later, shorter 429 never shortens an earlier one.
        """
        secs = int(retry_after) if retry_after else DEFAULT_FLOOD_WAIT
        secs = min(max(secs, MIN_FLOOD_WAIT), self._retry_after_max)
        with self._lock:
            self._flood_until = max(self._flood_until, self._clock.monotonic() + secs)
        return secs
