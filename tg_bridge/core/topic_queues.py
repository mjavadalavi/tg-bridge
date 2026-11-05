"""The per-topic message queue: the data behind the Dispatcher.

It holds the placeholders, admissions and deliveries of a topic's messages, and the queue of each topic with its busy
flag. Port of the live file's _queues, _queue_lk, QUEUE_MAX and _reserve_slot's queue part (tg-bridge.py:1983-2276).
A topic runs one turn at a time. Messages that arrive while a turn runs wait in the topic's queue, in order, at most
QUEUE_MAX of them. A media message reserves its place at once as a Placeholder; its download and transcription fill it
later. The runner, in dispatcher.py, waits for an unfilled placeholder when it reaches the head of the queue.
Groups: every queued message carries a group id. A burst follower (it arrived within the burst gap of the topic's
previous message, see arrivals.py) takes the group of that message; any other message starts a group of its own. A
refused message (queue full) ends the chain, so the next follower starts a new group. A turn takes a batch: the head and
the later messages of its group, at most MERGE_MAX, merged into one text, so a forwarded batch runs as one turn. The
group is an identity, not a position: ⏭ Now / ⏩ Next move a whole group, and a Drop or a released placeholder never
lets two groups merge. A message that is a command (its first non-space character is "/" or "!") always runs alone.

TopicQueues holds the queue data only. It starts no thread and sends nothing. Its lock guards the queues and the set of
busy topics, and no caller of it ever holds that lock while calling another service.

Threads: the inbound handler, the callback handlers and the runner threads of the Dispatcher all call TopicQueues. Each
call takes the lock once for a short section, copies or returns a Delivery, and never waits for a turn or a download.
Only the Dispatcher starts runner threads; this module starts none.
"""

import enum
import os
import threading
from dataclasses import dataclass

QUEUE_MAX = 100  # messages a topic may hold behind its running turn; a safety bound only, high enough for a batch of forwarded messages
MERGE_MAX = 50  # most queued messages one turn takes in; a larger batch runs as several turns, in order
BURST_PART = "--- message {index}/{count}, sent together with the one above ---\n{text}"  # index 2 and up; the first text leads
BURST_FOOTER = "[{count} messages sent together — read them all, then answer them as one task]"
ID_BYTES = 5


class Placeholder:
    """A queue place reserved for a media message. Its worker fills text, images and sets ev; busy: a turn was running
    when the place was reserved (the topic is then told where the message stands once it is filled)."""

    __slots__ = ("ev", "text", "images", "busy")

    def __init__(self, busy: bool) -> None:
        self.ev = threading.Event()
        self.text: str | None = None
        self.images: tuple[str, ...] = ()
        self.busy = busy


class Admission(enum.Enum):
    """What happened to a message handed to the queue."""

    QUEUED = "queued"  # a text behind a running turn: the topic is told its position
    STARTED = "started"  # the topic was idle: its runner was started with this message
    HANDED = "handed"  # a quiet message behind a running turn of a kept idle session: no notice, no new runner
    RESERVED = "reserved"  # a media placeholder behind a running turn: no notice until it is filled
    FILLED = "filled"  # a placeholder was filled in place (its place in the order is kept)
    FULL = "full"  # the queue already holds QUEUE_MAX messages: nothing was queued
    DROPPED = "dropped"  # the topic is being removed: the message was dropped (its inbox row is done by the dispatcher)


@dataclass(frozen=True)
class Delivery:
    """The outcome of one queue operation. nid is the id the Now/Next/Drop buttons carry; position is 1-based."""

    admission: Admission
    nid: str | None = None
    position: int = 0
    busy: bool = False  # FILLED: whether a turn was running when the placeholder was reserved
    burst: bool = False  # the message followed another within the burst gap: it is announced by nobody, it joins a turn

    @property
    def notify(self) -> bool:
        """Whether the topic should be told where its message stands (the queued notice with its buttons)."""
        return self.admission is Admission.QUEUED or (self.admission is Admission.FILLED and self.busy)


@dataclass(frozen=True)
class QueuedItem:
    """A queued message as a copy: text is None while a media placeholder is not filled yet."""

    nid: str
    text: str | None
    chat_id: int | str
    sender: str
    update_id: int | None


@dataclass(frozen=True)
class _Entry:
    nid: str
    body: str | Placeholder
    chat_id: int | str
    sender: str
    update_id: int | None
    images: tuple[str, ...] = ()  # local image paths of a text entry; a placeholder keeps its own (Placeholder.images)
    group: str = ""  # the burst this message belongs to: its own nid, or the group of the message it followed

    @property
    def follower(self) -> bool:
        """The message joined the group of an earlier one (it arrived within the burst gap)."""
        return bool(self.group) and self.group != self.nid


class _HeadKind(enum.Enum):
    EMPTY = "empty"  # nothing left: the topic is no longer busy
    WAIT = "wait"  # the head is a placeholder that is not filled yet
    READY = "ready"  # the head was taken off the queue and carries its text


@dataclass(frozen=True)
class _Head:
    kind: _HeadKind
    entry: _Entry | None = None  # the first popped entry: chat id and sender of the turn
    text: str | None = None  # the merged text of the batch
    images: tuple[str, ...] = ()  # the images of every popped entry, in order
    entries: tuple[_Entry, ...] = ()  # every popped entry, in order (a READY head has at least one)
    first_text: str | None = None  # the first entry's own text, not the merged one (the topic title is taken from it)
    lead: str | None = None  # WAIT: the nid of the queue head when the wait began (the awaited entry may be behind it)


def _new_id() -> str:
    return os.urandom(ID_BYTES).hex()


def _ready(entry: _Entry) -> bool:
    """A text entry, or a placeholder whose media has been filled in."""
    return not isinstance(entry.body, Placeholder) or entry.body.text is not None


def _text_of(entry: _Entry) -> str:
    return entry.body if isinstance(entry.body, str) else entry.body.text or ""


def _images_of(entry: _Entry) -> tuple[str, ...]:
    return entry.images if isinstance(entry.body, str) else entry.body.images


def _stands_alone(text: str) -> bool:
    """A command (Claude slash command or bash mode, "/" or "!" as the first non-space character): it runs alone."""
    return text.lstrip().startswith(("/", "!"))


def burst_text(texts: list[str]) -> str:
    """The text of a turn. One message is returned as it is. Several start with the first message's own text, so the
    text leads with what the sender wrote; each later one follows as a numbered part, and a footer counts them last."""
    if len(texts) == 1:
        return texts[0]
    count = len(texts)
    parts = [texts[0]]
    parts += [BURST_PART.format(index=i, count=count, text=text) for i, text in enumerate(texts[1:], 2)]
    parts.append(BURST_FOOTER.format(count=count))
    return "\n\n".join(parts)


def _snapshot(entry: _Entry) -> QueuedItem:
    text = entry.body if isinstance(entry.body, str) else entry.body.text
    return QueuedItem(entry.nid, text, entry.chat_id, entry.sender, entry.update_id)


class TopicQueues:
    """The message queue of every topic. Thread-safe; holds data only, so it is tested without threads."""

    def __init__(self, max_items: int = QUEUE_MAX) -> None:
        self._max = max_items
        self._lock = threading.Lock()  # guards _queues, _busy and _groups
        self._queues: dict[int | str, list[_Entry]] = {}
        self._busy: set[int | str] = set()  # topics whose runner is running (or about to run)
        self._groups: dict[int | str, str] = {}  # the group of each topic's latest queued message; gone after a refusal

    def busy(self, thread_id: int | str) -> bool:
        with self._lock:
            return thread_id in self._busy

    def reserve(self, thread_id: int | str, chat_id: int | str, sender: str, update_id: int | None,
                quiet: bool = False, burst: bool = False) -> Delivery:
        """Put a placeholder at the back of the topic's queue, so a media message keeps its place while its download
        and transcription run. An idle topic is marked busy here, and the caller starts its runner (STARTED). A quiet
        place behind a running turn is not announced when it is filled, so its placeholder is not busy. A burst follower
        keeps the flag, so its notice is skipped when it is filled too."""
        nid = _new_id()
        with self._lock:
            queue = self._queues.setdefault(thread_id, [])
            if len(queue) >= self._max:
                return self._refuse(thread_id)
            busy = thread_id in self._busy
            queue.append(_Entry(nid, Placeholder(busy and not quiet), chat_id, sender, update_id,
                                group=self._group(thread_id, nid, burst)))
            if busy:
                return Delivery(Admission.RESERVED, nid, len(queue), burst=burst)
            self._busy.add(thread_id)
            return Delivery(Admission.STARTED, nid, len(queue), burst=burst)

    def deliver(self, thread_id: int | str, chat_id: int | str, sender: str, update_id: int | None, text: str,
                reserved: str | None = None, images: tuple[str, ...] = (), quiet: bool = False,
                burst: bool = False) -> Delivery:
        """Hand over a message's text and images. With reserved, the placeholder of that id is filled in place, and the
        delivery keeps the burst flag the placeholder was reserved with. Otherwise the message joins the back of the
        queue: behind a running turn (QUEUED, or FULL at QUEUE_MAX), or as the first message of an idle topic (STARTED).
        A quiet message behind a running turn is HANDED instead of QUEUED: the caller tells nobody and starts no runner,
        because the kept session will take it. A reserved id that is no longer queued (its place was released) is queued
        as new."""
        nid = _new_id()
        with self._lock:
            queue = self._queues.setdefault(thread_id, [])
            idx = self._index_of(queue, reserved) if reserved is not None else None
            if idx is not None:
                entry = queue[idx]
                if isinstance(entry.body, Placeholder):
                    entry.body.text = text
                    entry.body.images = images
                    entry.body.ev.set()
                    return Delivery(Admission.FILLED, entry.nid, idx + 1, entry.body.busy, entry.follower)
            if thread_id in self._busy:
                if len(queue) >= self._max:
                    return self._refuse(thread_id)
                queue.append(_Entry(nid, text, chat_id, sender, update_id, images, self._group(thread_id, nid, burst)))
                if quiet:
                    return Delivery(Admission.HANDED, nid, len(queue), burst=burst)
                return Delivery(Admission.QUEUED, nid, len(queue), burst=burst)
            queue.append(_Entry(nid, text, chat_id, sender, update_id, images, self._group(thread_id, nid, burst)))
            self._busy.add(thread_id)
            return Delivery(Admission.STARTED, nid, len(queue), burst=burst)

    def unreserve(self, thread_id: int | str, nid: str | None) -> bool:
        """Remove a placeholder that will never be filled (its download failed, or it timed out). Wakes a runner that
        waits on it. True when this call removed it; False for None, an unknown id or one already filled."""
        if nid is None:
            return False
        with self._lock:
            queue = self._queues.get(thread_id) or []
            for i, entry in enumerate(queue):
                if entry.nid == nid and isinstance(entry.body, Placeholder) and entry.body.text is None:
                    queue.pop(i)
                    entry.body.ev.set()
                    return True
        return False

    def to_front(self, thread_id: int | str, nid: str) -> bool:
        """Move a queued message to the front (⏭ Now and ⏩ Next), with every message of its group, in their order: the
        group runs as one turn, so it moves as one. False when it is not queued."""
        with self._lock:
            queue = self._queues.get(thread_id) or []
            idx = self._index_of(queue, nid)
            if idx is None:
                return False
            group = queue[idx].group
            moved = {entry.nid for entry in queue if entry.group == group} if group else {nid}
            queue[:] = [entry for entry in queue if entry.nid in moved] + [e for e in queue if e.nid not in moved]
            return True

    def head_nid(self, thread_id: int | str) -> str | None:
        """The nid of the topic's queue head, or None for an empty queue."""
        with self._lock:
            queue = self._queues.get(thread_id) or []
            return queue[0].nid if queue else None

    def drop(self, thread_id: int | str, nid: str) -> QueuedItem | None:
        """Remove a queued message (❌ Drop) and return it, or None when it is not queued. A placeholder is woken so
        the runner does not wait for it."""
        with self._lock:
            queue = self._queues.get(thread_id) or []
            idx = self._index_of(queue, nid)
            if idx is None:
                return None
            entry = queue.pop(idx)
            if isinstance(entry.body, Placeholder):
                entry.body.ev.set()
            return _snapshot(entry)

    def drain(self, thread_id: int | str) -> tuple[QueuedItem, ...]:
        """Removes every queued message of a topic (a topic removal) and returns copies of them, in queue order.
        Placeholders are woken so no runner waits on them."""
        with self._lock:
            queue = self._queues.pop(thread_id, None) or []
            self._groups.pop(thread_id, None)
            for entry in queue:
                if isinstance(entry.body, Placeholder):
                    entry.body.ev.set()
            return tuple(_snapshot(entry) for entry in queue)

    def clear(self, thread_id: int | str) -> int:
        """Removes every queued message of a topic. Returns how many were removed."""
        return len(self.drain(thread_id))

    def snapshot(self, thread_id: int | str) -> tuple[QueuedItem, ...]:
        """Copies of the queued messages of a topic, in queue order."""
        with self._lock:
            return tuple(_snapshot(entry) for entry in self._queues.get(thread_id) or [])

    def take_head(self, thread_id: int | str, keep_busy: bool = False) -> _Head:
        """The next turn for the runner.

        An unfilled placeholder at the head is left in place (WAIT, with that entry). Otherwise the head is taken with the
        later entries of its group, in order; a command head runs alone. The batch stops at a command of the group (it
        stays for the next turn) or at MERGE_MAX. When a member of the group is an unfilled placeholder, nothing is
        popped and the runner waits on it (WAIT with that entry, and lead: the head's nid), so the burst is taken whole.
        A placeholder of another group never blocks the batch. The taken entries are removed as one batch (READY) whose
        text is burst_text() of them.

        An empty queue ends the topic's busy state (EMPTY), unless keep_busy is set: the runner then hands the topic to a
        kept idle session, which stays busy until its next turn ends."""
        with self._lock:
            queue = self._queues.get(thread_id) or []
            if not queue:
                if not keep_busy:
                    self._busy.discard(thread_id)
                return _Head(_HeadKind.EMPTY)
            head = queue[0]
            if not _ready(head):
                return _Head(_HeadKind.WAIT, head, lead=head.nid)
            taken = [head]
            if head.group and not _stands_alone(_text_of(head)):
                for member in (entry for entry in queue[1:] if entry.group == head.group):
                    if len(taken) >= MERGE_MAX:
                        break
                    if not _ready(member):
                        return _Head(_HeadKind.WAIT, member, lead=head.nid)  # a burst media message: pop nothing
                    if _stands_alone(_text_of(member)):
                        break
                    taken.append(member)
            taken_ids = {entry.nid for entry in taken}
            queue[:] = [entry for entry in queue if entry.nid not in taken_ids]
            for entry in taken:
                if isinstance(entry.body, Placeholder):
                    entry.body.ev.set()
            text = burst_text([_text_of(entry) for entry in taken])
            images = tuple(image for entry in taken for image in _images_of(entry))
            return _Head(_HeadKind.READY, taken[0], text, images, tuple(taken), _text_of(taken[0]))

    def _group(self, thread_id: int | str, nid: str, burst: bool) -> str:
        """The group of a message being queued (call under the lock): the group of the topic's latest queued message
        for a burst follower, else a group of its own. It becomes the topic's latest group."""
        group = self._groups.get(thread_id, nid) if burst else nid
        self._groups[thread_id] = group
        return group

    def _refuse(self, thread_id: int | str) -> Delivery:
        """A message the full queue refused (call under the lock): it ends the chain, so a follower of it starts anew."""
        self._groups.pop(thread_id, None)
        return Delivery(Admission.FULL)

    @staticmethod
    def _index_of(queue: list[_Entry], nid: str | None) -> int | None:
        return next((i for i, entry in enumerate(queue) if entry.nid == nid), None)
