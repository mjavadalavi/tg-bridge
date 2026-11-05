"""Arrivals: the settle a runner makes before it takes the next turn, so a burst of messages runs as one turn.

Telegram delivers a forwarded batch of messages milliseconds apart. Before the runner of a topic takes the head of its
queue, it waits while messages are still arriving: it sleeps until BURST_GAP_SECONDS have passed since the latest
message handed to the topic, but for no more than BURST_CAP_SECONDS in all, so a steady stream still starts a turn.
Then TopicQueues.take_head hands over the whole burst as one turn.

Arrivals keeps the monotonic time of the latest message of each topic and never sleeps. The Dispatcher calls mark for
each message it is handed (mark also says whether the message is a burst follower), and its topic runner asks remaining
for the time still to wait, sleeping itself in slices, so a removal, a shutdown or a held session is never blocked by a
settle. The lock guards the dict and the clock read in mark; no sleep runs under it.
"""

import threading
from dataclasses import replace

from tg_bridge.clock import Clock
from tg_bridge.core.topic_queues import Admission, Delivery

BURST_GAP_SECONDS = 2.0  # a quiet gap this long ends a burst; a forwarded batch arrives within ~1 s
BURST_CAP_SECONDS = 15.0  # a steady stream still starts a turn within this


class Arrivals:
    """The time of the latest message handed to each topic, the topics whose latest message was refused, and the wait
    that ends a burst: the rest of a quiet gap, capped."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._lock = threading.Lock()  # guards _last
        self._last: dict[int | str, float] = {}
        self._refused: set[int | str] = set()  # topics whose latest message was refused (queue full)

    def mark(self, thread_id: int | str) -> bool:
        """Records that a message was just handed to the topic. True when the topic's previous message was less than
        BURST_GAP_SECONDS ago: this one is a burst follower, and it is announced by nobody. The time is read under the
        lock, so the calls are recorded in the order of the lock; the stored time never moves backward."""
        with self._lock:
            now = self._clock.monotonic()
            last = self._last.get(thread_id)
            self._last[thread_id] = now if last is None else max(last, now)
        return last is not None and now - last < BURST_GAP_SECONDS

    def remaining(self, thread_id: int | str, started: float) -> float:
        """Seconds the topic's runner still waits before it takes the head: the rest of the gap since the topic's latest
        message, but no further than BURST_CAP_SECONDS after started (the monotonic time the settle began). 0.0 means
        take now: no message yet, the gap has passed, or the cap is reached. It does not sleep."""
        with self._lock:
            last = self._last.get(thread_id)
        if last is None:
            return 0.0
        now = self._clock.monotonic()
        return max(0.0, min(BURST_GAP_SECONDS - (now - last), BURST_CAP_SECONDS - (now - started)))

    def admitted(self, thread_id: int | str, delivery: Delivery, burst: bool) -> Delivery:
        """The delivery of a message handed to the topic's queue. A refusal (queue full) carries burst when it follows a
        refused message within the gap, so a burst of overflow gets one queue-full notice. Every new message updates the
        topic's refusal state, under the lock; a filled placeholder is an earlier message, so it leaves the state alone."""
        if delivery.admission is Admission.FILLED:
            return delivery
        refused = delivery.admission is Admission.FULL
        with self._lock:
            follows = refused and burst and thread_id in self._refused
            if refused:
                self._refused.add(thread_id)
            else:
                self._refused.discard(thread_id)
        return replace(delivery, burst=follows) if refused else delivery
