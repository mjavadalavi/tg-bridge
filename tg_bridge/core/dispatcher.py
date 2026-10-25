"""TopicQueues and Dispatcher: the per-topic message queue and its serial runner (C15).

Port of the live file's _queues, _dispatchers, _queue_lk, QUEUE_MAX, _Pending, _reserve_slot, _unreserve, _dispatch and
the queue part of _continue_message (tg-bridge.py:1983-2276, 3280-3330). A topic runs one turn at a time. Messages that
arrive while a turn runs wait in the topic's queue, in order, at most QUEUE_MAX of them. A media message reserves its
place at once as a Placeholder; its download and transcription fill it later, and the runner waits for it (at most
MEDIA_WAIT_SECONDS) when it reaches the head of the queue.

TopicQueues holds the queue data only. It starts no thread and sends nothing. Its lock guards the queues and the set of
busy topics, and no caller of it ever holds that lock while calling another service.

Dispatcher starts one daemon runner thread per idle topic. The runner marks the inbox row done when its turn starts,
sends the typing action, takes a slot, runs the TurnRunner, then sends the answer or the error and submits the title job.
A turn that a restart stopped keeps its inbox row done (recover() reports it). A turn that a shutdown refused before it
ran puts its row back to pending, so the next start replays it.
"""

import enum
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass

from tg_bridge.config import Config
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.slots import SlotPool
from tg_bridge.core.titles import TitleService
from tg_bridge.core.turn_runner import TurnRunner
from tg_bridge.core.turn_text import NO_OUTPUT, RESTART_TEXT
from tg_bridge.logging_ import Logger
from tg_bridge.store.inbox import InboxRepo
from tg_bridge.telegram.answers import AnswerSender
from tg_bridge.telegram.client import TelegramClient
from tg_bridge.telegram.messenger import Messenger

QUEUE_MAX = 5  # messages a topic may hold behind its running turn
MEDIA_WAIT_SECONDS = 300  # a queued voice/photo still not transcribed after this leaves its place (it is queued at the back)
ERROR_TEXT_CHARS = 200
SEND_ERROR_CHARS = 120
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
    RESERVED = "reserved"  # a media placeholder behind a running turn: no notice until it is filled
    FILLED = "filled"  # a placeholder was filled in place (its place in the order is kept)
    FULL = "full"  # the queue already holds QUEUE_MAX messages: nothing was queued


@dataclass(frozen=True)
class Delivery:
    """The outcome of one queue operation. nid is the id the Now/Next/Drop buttons carry; position is 1-based."""

    admission: Admission
    nid: str | None = None
    position: int = 0
    busy: bool = False  # FILLED: whether a turn was running when the placeholder was reserved

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


class _HeadKind(enum.Enum):
    EMPTY = "empty"  # nothing left: the topic is no longer busy
    WAIT = "wait"  # the head is a placeholder that is not filled yet
    READY = "ready"  # the head was taken off the queue and carries its text


@dataclass(frozen=True)
class _Head:
    kind: _HeadKind
    entry: _Entry | None = None
    text: str | None = None
    images: tuple[str, ...] = ()


def _new_id() -> str:
    return os.urandom(ID_BYTES).hex()


def _snapshot(entry: _Entry) -> QueuedItem:
    text = entry.body if isinstance(entry.body, str) else entry.body.text
    return QueuedItem(entry.nid, text, entry.chat_id, entry.sender, entry.update_id)


class TopicQueues:
    """The message queue of every topic. Thread-safe; holds data only, so it is tested without threads."""

    def __init__(self, max_items: int = QUEUE_MAX) -> None:
        self._max = max_items
        self._lock = threading.Lock()  # guards _queues and _busy
        self._queues: dict[int | str, list[_Entry]] = {}
        self._busy: set[int | str] = set()  # topics whose runner is running (or about to run)

    def busy(self, thread_id: int | str) -> bool:
        with self._lock:
            return thread_id in self._busy

    def reserve(self, thread_id: int | str, chat_id: int | str, sender: str, update_id: int | None) -> Delivery:
        """Put a placeholder at the back of the topic's queue, so a media message keeps its place while its download
        and transcription run. An idle topic is marked busy here, and the caller starts its runner (STARTED)."""
        nid = _new_id()
        with self._lock:
            queue = self._queues.setdefault(thread_id, [])
            if len(queue) >= self._max:
                return Delivery(Admission.FULL)
            busy = thread_id in self._busy
            queue.append(_Entry(nid, Placeholder(busy), chat_id, sender, update_id))
            if busy:
                return Delivery(Admission.RESERVED, nid, len(queue))
            self._busy.add(thread_id)
            return Delivery(Admission.STARTED, nid, len(queue))

    def deliver(self, thread_id: int | str, chat_id: int | str, sender: str, update_id: int | None, text: str,
                reserved: str | None = None, images: tuple[str, ...] = ()) -> Delivery:
        """Hand over a message's text and images. With reserved, the placeholder of that id is filled in place.
        Otherwise the message joins the back of the queue: behind a running turn (QUEUED, or FULL at QUEUE_MAX), or as
        the first message of an idle topic (STARTED). A reserved id that is no longer queued (its place was released)
        is queued as new."""
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
                    return Delivery(Admission.FILLED, entry.nid, idx + 1, entry.body.busy)
            if thread_id in self._busy:
                if len(queue) >= self._max:
                    return Delivery(Admission.FULL)
                queue.append(_Entry(nid, text, chat_id, sender, update_id, images))
                return Delivery(Admission.QUEUED, nid, len(queue))
            queue.append(_Entry(nid, text, chat_id, sender, update_id, images))
            self._busy.add(thread_id)
            return Delivery(Admission.STARTED, nid, len(queue))

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
        """Move a queued message to the front (⏭ Now and ⏩ Next). False when it is not queued."""
        with self._lock:
            queue = self._queues.get(thread_id) or []
            idx = self._index_of(queue, nid)
            if idx is None:
                return False
            queue.insert(0, queue.pop(idx))
            return True

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

    def snapshot(self, thread_id: int | str) -> tuple[QueuedItem, ...]:
        """Copies of the queued messages of a topic, in queue order."""
        with self._lock:
            return tuple(_snapshot(entry) for entry in self._queues.get(thread_id) or [])

    def take_head(self, thread_id: int | str) -> _Head:
        """The next message for the runner. An unfilled placeholder at the head is left in place (WAIT). Otherwise the
        head is removed and returned (READY). An empty queue ends the topic's busy state (EMPTY)."""
        with self._lock:
            queue = self._queues.get(thread_id) or []
            if not queue:
                self._busy.discard(thread_id)
                return _Head(_HeadKind.EMPTY)
            entry = queue[0]
            if isinstance(entry.body, Placeholder) and entry.body.text is None:
                return _Head(_HeadKind.WAIT, entry)
            queue.pop(0)
            if isinstance(entry.body, str):
                return _Head(_HeadKind.READY, entry, entry.body, entry.images)
            return _Head(_HeadKind.READY, entry, entry.body.text, entry.body.images)

    @staticmethod
    def _index_of(queue: list[_Entry], nid: str | None) -> int | None:
        return next((i for i, entry in enumerate(queue) if entry.nid == nid), None)


class Dispatcher:
    """Owns the topic queues and starts the serial runner of each idle topic. One instance per App.

    Entry points for the inbound handler: reserve (media placeholder), deliver (text or filled placeholder), unreserve.
    Entry points for the callback handlers: to_front (Now, Next), drop (Drop) and queue_snapshot (/queue, /status).
    """

    def __init__(
        self,
        *,
        turns: TurnRunner,
        slots: SlotPool,
        inbox: InboxRepo,
        messenger: Messenger,
        answers: AnswerSender,
        titles: TitleService,
        client: TelegramClient,
        live: LiveRegistry,
        config: Config,
        logger: Logger,
        media_wait_seconds: float = MEDIA_WAIT_SECONDS,
        start_thread: Callable[..., threading.Thread] = threading.Thread,
    ) -> None:
        self._turns = turns
        self._slots = slots
        self._inbox = inbox
        self._messenger = messenger
        self._answers = answers
        self._titles = titles
        self._client = client
        self._live = live
        self._config = config
        self._logger = logger
        self._media_wait = media_wait_seconds
        self._start_thread = start_thread
        self._queues = TopicQueues()

    def reserve(self, thread_id: int, topic: str, chat_id: int | str, sender: str, update_id: int | None) -> Delivery:
        """Reserve a media message's place in the topic's queue. Starts the runner when the topic is idle."""
        delivery = self._queues.reserve(thread_id, chat_id, sender, update_id)
        if delivery.admission is Admission.STARTED:
            self._start(thread_id, topic)
        return delivery

    def deliver(self, thread_id: int, topic: str, chat_id: int | str, sender: str, update_id: int | None, text: str,
                reserved: str | None = None, images: tuple[str, ...] = ()) -> Delivery:
        """Hand over a message's text and local image paths: a filled placeholder, a queued message, or the first
        message of an idle topic (which starts the runner). The images reach the turn's session as native image input
        when its backend takes them. Notices and buttons are the caller's to send (see Delivery.notify)."""
        delivery = self._queues.deliver(thread_id, chat_id, sender, update_id, text, reserved, images)
        if delivery.admission is Admission.STARTED:
            self._start(thread_id, topic)
        return delivery

    def unreserve(self, thread_id: int, nid: str | None) -> bool:
        return self._queues.unreserve(thread_id, nid)

    def to_front(self, thread_id: int, nid: str) -> bool:
        return self._queues.to_front(thread_id, nid)

    def drop(self, thread_id: int, nid: str) -> QueuedItem | None:
        """Remove a queued message. Its inbox row is marked done, so it never replays."""
        item = self._queues.drop(thread_id, nid)
        if item is not None:
            self._mark_done(item.update_id)
        return item

    def queue_snapshot(self, thread_id: int) -> tuple[QueuedItem, ...]:
        return self._queues.snapshot(thread_id)

    # the runner

    def _start(self, thread_id: int, topic: str) -> None:
        self._start_thread(target=self._run, args=(thread_id, topic), daemon=True,
                           name=f"tg-topic-{thread_id}").start()

    def _run(self, thread_id: int, topic: str) -> None:
        """Serial runner of one topic: turns run one at a time, queued messages follow in order. Returns when the queue
        is empty, when a turn was stopped by a restart or refused by a shutdown, or when the bridge is shutting down."""
        while not self._live.is_shutting_down():
            head = self._queues.take_head(thread_id)
            if head.kind is _HeadKind.EMPTY:
                return
            if head.kind is _HeadKind.WAIT:
                self._await_media(thread_id, head.entry)
                continue
            if head.entry is None or head.text is None:  # READY always carries both; guards the type only
                return
            if not self._run_entry(thread_id, topic, head.entry, head.text, head.images):
                return

    def _await_media(self, thread_id: int, entry: _Entry | None) -> None:
        """Wait for the head placeholder for at most the media wait. On timeout the place is released and the topic is
        told; the job still runs, and its late result is queued at the back by the inbound handler."""
        if entry is None or not isinstance(entry.body, Placeholder):
            return
        if entry.body.ev.wait(self._media_wait):
            return
        if self._queues.unreserve(thread_id, entry.nid):
            self._logger.log(f"thread {thread_id}: media message not transcribed after {self._media_wait}s "
                             "— released from the queue")
            self._messenger.send_text(
                entry.chat_id, thread_id,
                f"⚠️ a voice/photo is still being processed after {self._media_wait}s — "
                "the queue moved on; it will run as soon as it is ready.")

    def _run_entry(self, thread_id: int, topic: str, entry: _Entry, text: str, images: tuple[str, ...]) -> bool:
        """Run one queued message as a turn. False: the runner must stop (restart stop, shutdown refusal)."""
        chat_id = entry.chat_id
        try:
            self._mark_done(entry.update_id)  # the turn starts now: a restart from here is reported by recover(), not re-run
            self._typing(chat_id, thread_id)
            slot = self._slots.new_slot(chat_id, thread_id)
            if not slot.take():  # a shutdown came while the message waited for a slot: it replays on restart
                self._replay_later(entry.update_id)
                return False
            try:
                reply = self._turns.run(thread_id=thread_id, chat_id=chat_id, topic=topic, text=text,
                                        sender=entry.sender, slot=slot, images=images)
            finally:
                slot.give_back()
            if reply.err == RESTART_TEXT:  # a restart stopped this turn: recover() tells the topic, the row stays done
                return False
            if reply.answer is None and reply.err is None:  # a shutdown refused the spawn: the message replays on restart
                self._replay_later(entry.update_id)
                return False
            self._reply(chat_id, thread_id, topic, text, reply.answer, reply.err)
            return True
        except Exception as exc:
            self._logger.log(f"turn {topic}: {exc}")
            self._messenger.send_text(chat_id, thread_id, f"bridge error: {str(exc)[:ERROR_TEXT_CHARS]}")
            return True

    def _reply(self, chat_id: int | str, thread_id: int, topic: str, text: str, answer: str | None,
               err: str | None) -> None:
        if err:
            self._messenger.send_text(chat_id, thread_id, err)
        else:
            self._answers.send_answer(chat_id, thread_id, answer or NO_OUTPUT, self._config.answer_doc_threshold())
        if answer:  # the title call can take up to two minutes: it is queued after the answer, on its own worker
            self._titles.submit(chat_id, thread_id, topic, text, answer)

    def _typing(self, chat_id: int | str, thread_id: int) -> None:
        try:
            self._client.call("sendChatAction", chat_id=chat_id, action="typing", message_thread_id=thread_id)
        except Exception as exc:  # cosmetic only: the turn must run regardless
            self._logger.log(f"sendChatAction: {str(exc)[:SEND_ERROR_CHARS]}")

    def _mark_done(self, update_id: int | None) -> None:
        if update_id is not None:
            self._inbox.done(update_id)

    def _replay_later(self, update_id: int | None) -> None:
        if update_id is not None:
            self._inbox.mark_pending(update_id)
