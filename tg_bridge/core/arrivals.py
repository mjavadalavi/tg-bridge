"""Arrivals: the settle a runner makes before it takes the next turn, so a burst of messages runs as one turn.

Telegram delivers a forwarded batch of messages milliseconds apart. Before the runner of a topic takes the head of its
queue, it waits while messages are still arriving: it sleeps until BURST_GAP_SECONDS have passed since the latest
message handed to the topic, but for no more than BURST_CAP_SECONDS in all, so a steady stream still starts a turn.
Then TopicQueues.take_head hands over the whole burst as one turn.

Arrivals keeps the monotonic time of the latest message of each topic and sleeps through the injected clock. The
inbound handler calls mark, the topic runners call settle; the lock guards the dict only, and no sleep runs under it.
"""

import threading

from tg_bridge.clock import Clock

BURST_GAP_SECONDS = 2.0  # a quiet gap this long ends a burst; a forwarded batch arrives within ~1 s
BURST_CAP_SECONDS = 15.0  # a steady stream still starts a turn within this


class Arrivals:
    """The time of the latest message handed to each topic, and the settle that waits for a quiet gap."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._lock = threading.Lock()  # guards _last
        self._last: dict[int | str, float] = {}

    def mark(self, thread_id: int | str) -> None:
        """Records that a message was just handed to the topic."""
        now = self._clock.monotonic()
        with self._lock:
            self._last[thread_id] = now

    def settle(self, thread_id: int | str) -> None:
        """Sleeps while the topic's latest message is less than BURST_GAP_SECONDS old. The total sleep of one call is
        at most BURST_CAP_SECONDS. A topic with no message, or one that has been quiet for the gap, is not delayed."""
        waited = 0.0
        while waited < BURST_CAP_SECONDS:
            with self._lock:
                last = self._last.get(thread_id)
            if last is None:
                return
            quiet = BURST_GAP_SECONDS - (self._clock.monotonic() - last)
            if quiet <= 0:
                return
            step = min(quiet, BURST_CAP_SECONDS - waited)
            self._clock.sleep(step)
            waited += step
