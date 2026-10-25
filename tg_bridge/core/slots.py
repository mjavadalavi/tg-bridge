"""SlotPool (size permits, default 2) + Slot take / give_back / retake (C15).

Port of the live file's _sem (Semaphore(_SLOTS)) and class _Slot. A turn holds one permit while it runs. A turn that
is parked on a permission card gives its permit back (give_back) so the other topics keep running, and takes it again
(retake) when the card is answered. take() waits for a permit and tells the topic once that it is queued.

Thread rule: a Slot belongs to one turn thread. The pool's counters are guarded by its lock; the Slot's own held flag
is only touched by its owner. Waiting in take() gives up when the shutdown callable turns true.
"""

import threading
from collections.abc import Callable

from tg_bridge.telegram.messenger import Messenger

SLOT_COUNT = 2  # turns that may run at once
WAIT_POLL_SECONDS = 1.0  # how often a queued take() checks for shutdown (the live file used acquire(timeout=1))


class SlotPool:
    """The permits shared by every topic. One instance per App."""

    def __init__(
        self,
        size: int,
        messenger: Messenger,
        shutting_down: Callable[[], bool],
        poll_seconds: float = WAIT_POLL_SECONDS,
    ) -> None:
        if size < 1:
            raise ValueError("a SlotPool needs at least one permit")
        self.size = size
        self._messenger = messenger
        self._shutting_down = shutting_down
        self._poll_seconds = poll_seconds
        self._semaphore = threading.Semaphore(size)
        self._lock = threading.Lock()  # guards _in_use
        self._in_use = 0

    @property
    def poll_seconds(self) -> float:
        """How long a queued take() waits between shutdown checks."""
        return self._poll_seconds

    def shutting_down(self) -> bool:
        return self._shutting_down()

    def announce_queued(self, chat_id: int | str, thread_id: int | str) -> None:
        """Tell a topic, once, that its turn waits for a permit."""
        self._messenger.send_text(
            chat_id,
            thread_id,
            f"⏳ waiting for a free slot — {self.size} topics are busy right now; "
            "this message runs as soon as one frees up.",
        )

    def new_slot(self, chat_id: int | str, thread_id: int | str) -> "Slot":
        """A slot for one turn of one topic. It holds no permit until take()."""
        return Slot(self, chat_id, thread_id)

    def free(self) -> int:
        """Permits not held by any slot right now."""
        with self._lock:
            return self.size - self._in_use

    def acquire_permit(self, timeout: float | None) -> bool:
        """Take one permit, waiting up to timeout seconds (None: wait for as long as it takes)."""
        return self._record(self._semaphore.acquire(timeout=timeout))

    def try_acquire_permit(self) -> bool:
        """Take one permit only when one is free right now. Never waits."""
        return self._record(self._semaphore.acquire(blocking=False))

    def _record(self, acquired: bool) -> bool:
        if acquired:
            with self._lock:
                self._in_use += 1
        return acquired

    def release_permit(self) -> None:
        """Give one permit back. The counter drops before the semaphore is released, so it never exceeds size."""
        with self._lock:
            self._in_use -= 1
        self._semaphore.release()


class Slot:
    """One turn's hold on a permit. Idempotent take and give_back: a second take() does not take a second permit."""

    def __init__(self, pool: SlotPool, chat_id: int | str, thread_id: int | str) -> None:
        self._pool = pool
        self.chat_id = chat_id
        self.thread_id = thread_id
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    def take(self) -> bool:
        """Wait for a permit. The topic is told once that it is queued. False: shutdown came first, no permit taken."""
        if self._held:
            return True
        notified = False
        while not self._pool.acquire_permit(timeout=self._pool.poll_seconds):
            if self._pool.shutting_down():
                return False
            if not notified:
                notified = True
                self._pool.announce_queued(self.chat_id, self.thread_id)
        self._held = True
        return True

    def give_back(self) -> None:
        """Release the permit while the turn waits for a person (a permission card). No-op when not held."""
        if self._held:
            self._held = False
            self._pool.release_permit()

    def retake(self) -> bool:
        """Take the permit back without waiting. False when every permit is busy (the turn then runs unslotted)."""
        if self._held:
            return True
        if self._pool.try_acquire_permit():
            self._held = True
            return True
        return False
