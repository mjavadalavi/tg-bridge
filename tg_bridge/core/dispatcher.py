"""Dispatcher: the serial runner of each topic (C15). The queue data lives in tg_bridge.core.topic_queues.

Port of the live file's _dispatchers, _dispatch and the runner part of _continue_message (tg-bridge.py:2200-2276,
3280-3330). A topic runs one turn at a time. Messages that arrive while a turn runs wait in the topic's queue. A media
message's placeholder is waited for (at most MEDIA_WAIT_SECONDS) when it reaches the head of the queue.

Batches: before it takes a turn, the runner settles (core/arrivals): it waits while messages keep arriving, so a
forwarded batch is all queued. TopicQueues.take_head then hands over the head and the ready messages behind it as one
turn, with the merged text and images, and every inbox row of the batch is marked, replayed or dropped together.

Dispatcher starts one daemon runner thread per idle topic. The runner marks the inbox row done when its turn starts,
sends the typing action, takes a slot, runs the TurnRunner, then sends the answer or the error and submits the title
job. A turn that a restart stopped keeps its inbox row done (recover() reports it). A turn that a shutdown refused
before it ran puts its row back to pending, so the next start replays it.

Linger: while the LingerControl holds the topic's kept session, the runner stays on the topic (its queue keeps the
topic busy) and calls wait_once on each tick, which runs the session's self turns and checks its end conditions. A
message to an idle kept session is handed over quietly (HANDED) and wakes the tick. The two waits a head message can
block in (the media placeholder and the slot permit) are sliced while the topic is held, so the ticks keep running.

Threads: the inbound handler calls deliver and reserve on its own thread; each topic's runner thread runs _run, with the
hold ticks inside it.

Terminal: with a view folder, the head message waits for the terminal claims of the topic (core/terminal_wait). A
claim with a pane and a session uuid is attachable: when its pane is idle, the message runs as a turn in that terminal
session (TerminalTurn.attach set; no turn mark is taken, the process belongs to the terminal). A pane that is not ready
when the turn is about to paste (TerminalNotReady) sends the message back into that wait, with no reply. A claim without
them keeps the old wait. Otherwise a turn starts only when no claim is live: the check and the turn mark are one step
under the topic's claim lock (terminal_claim.begin_turn). The mark stays while the topic's process may run: after a
kept session's turn it stays until the hold ends (the tick that ends it clears it), and a turn that left no kept session
clears it when it ends (terminal_claim.end_turn). A begun turn keeps its mark through its slot wait: the hold ticks of
that wait never clear it, since the turn is about to spawn its process. Without a view folder the terminal is not
checked at all. The before_turn hook runs after the wait, before a turn the bridge spawns (not an attached one).

Lock map: none. TopicQueues and LingerService guard their own data; nothing is called under a lock here. The claim lock
is an fcntl file lock taken inside terminal_claim, and only file I/O runs under it.
"""

import math
import threading
import time
from collections.abc import Callable

from tg_bridge.agents.base import AttachSpec
from tg_bridge.clock import Clock, SystemClock
from tg_bridge.config import Config
from tg_bridge.core import terminal_claim
from tg_bridge.core.arrivals import Arrivals
from tg_bridge.core.linger import LINGER_CAP_TEXT, LINGER_EXIT_TEXT, POLL_SECONDS, HoldEnd, HoldResult, LingerControl
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.removals import Removals
from tg_bridge.core.slots import Slot, SlotPool
from tg_bridge.core.terminal_wait import STALL_SECONDS, TerminalWait
from tg_bridge.core.titles import TitleService
from tg_bridge.core.topic_queues import (
    Admission, Delivery, Placeholder, QueuedItem, TopicQueues, _Entry, _Head, _HeadKind,
)
from tg_bridge.core.turn_runner import TurnRunner
from tg_bridge.core.turn_text import NO_OUTPUT, RESTART_TEXT
from tg_bridge.logging_ import Logger
from tg_bridge.store.inbox import InboxRepo
from tg_bridge.telegram.answers import AnswerSender
from tg_bridge.telegram.client import TelegramClient
from tg_bridge.telegram.messenger import Messenger

MEDIA_WAIT_SECONDS = 300  # a queued voice/photo still not transcribed after this leaves its place (queued at the back)
ERROR_TEXT_CHARS = 200
SEND_ERROR_CHARS = 120
HOLD_FAILURE_LIMIT = 3  # failed hold ticks in a row before the held session is asked to close
HOLD_FAILURE_PAUSE_SECONDS = 1.0  # a failed hold tick waits this long before the next, so a broken tick does not spin
SECONDS_PER_MINUTE = 60


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
        linger: LingerControl | None = None,
        hold_failure_pause: float = HOLD_FAILURE_PAUSE_SECONDS,
        clock: Clock | None = None,
        poll_seconds: float = POLL_SECONDS,
        view_dir: str = "",  # the view folder of the terminal claims; '' = views off, the terminal is not checked
        removals: Removals | None = None,  # the marks of topics being removed, shared with TopicRemover
        pane_idle: Callable[[terminal_claim.Claim], bool] | None = None,  # the probe of an attachable claim's pane
        pane_reason: Callable[[terminal_claim.Claim], str] | None = None,  # why that pane is not idle (stall notice)
        stall_seconds: float = STALL_SECONDS,  # a pane not idle this long gets the stall notice
        before_turn: Callable[[int], None] | None = None,  # run before a turn the bridge spawns (not an attached one)
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
        self._before_turn = before_turn  # None: no hook; a raising hook is logged and the turn runs anyway
        self._media_wait = media_wait_seconds
        self._start_thread = start_thread
        self._linger = linger  # None: no session is kept, the runner never holds a topic
        self._hold_pause = hold_failure_pause
        self._clock = clock if clock is not None else SystemClock()  # the media wait's deadline
        self._arrivals = Arrivals(self._clock)  # the burst settle before each take
        self._poll = poll_seconds  # the slice of a media wait while the topic is held
        self._view_dir = view_dir
        self._queues = TopicQueues()
        self._removals = removals if removals is not None else Removals(self._clock)
        self._terminal = TerminalWait(
            view_dir=view_dir, removals=self._removals, shutting_down=live.is_shutting_down, clock=self._clock,
            logger=logger, messenger=messenger, poll_seconds=poll_seconds, pane_idle=pane_idle,
            holding=self._holding, tick=self._hold_tick, pane_reason=pane_reason, stall_seconds=stall_seconds)

    def busy(self, thread_id: int) -> bool:
        """True while the topic's runner runs: a message is queued, taken or in a turn, or a session is held."""
        return self._queues.busy(thread_id)

    def waiting_terminal(self, thread_id: int) -> bool:
        """True while the topic's head message waits for its terminal session (tg-bridge view) to be free."""
        return self._terminal.waiting(thread_id)

    def removing(self, thread_id: int) -> bool:
        """True while the topic is being removed (or was forgotten a moment ago): its messages are dropped."""
        return self._removals.active(thread_id)

    def reserve(self, thread_id: int, topic: str, chat_id: int | str, sender: str, update_id: int | None) -> Delivery:
        """Reserve a media message's place in the topic's queue. Starts the runner when the topic is idle. The delivery
        says whether the message is a burst follower (Delivery.burst), and the place keeps that flag."""
        burst = self._arrivals.mark(thread_id)
        if self._removals.active(thread_id):
            self._drop_removing(thread_id, update_id)
            return Delivery(Admission.DROPPED)
        delivery = self._queues.reserve(thread_id, chat_id, sender, update_id, quiet=self._quiet(thread_id),
                                        burst=burst)
        if delivery.admission is Admission.STARTED:
            self._start(thread_id, topic)
        return delivery

    def deliver(self, thread_id: int, topic: str, chat_id: int | str, sender: str, update_id: int | None, text: str,
                reserved: str | None = None, images: tuple[str, ...] = ()) -> Delivery:
        """Hand over a message's text and local image paths: a filled placeholder, a queued message, or the first
        message of an idle topic (which starts the runner). The images reach the turn's session as native image input
        when its backend takes them. Notices and buttons are the caller's to send (see Delivery.notify and
        Delivery.burst). A message to an idle kept session is HANDED: no notice, and the held runner is woken to take
        it."""
        burst = self._arrivals.mark(thread_id)
        if self._removals.active(thread_id):
            self._drop_removing(thread_id, update_id)
            return Delivery(Admission.DROPPED)
        delivery = self._queues.deliver(thread_id, chat_id, sender, update_id, text, reserved, images,
                                        quiet=self._quiet(thread_id), burst=burst)
        if delivery.admission is Admission.STARTED:
            self._start(thread_id, topic)
        elif delivery.admission is Admission.HANDED and self._linger is not None:
            self._linger.wake(thread_id)
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

    def clear_queue(self, thread_id: int) -> int:
        """Drops every queued message of a topic, as a topic removal does. Their inbox rows are marked done here, so a
        refused removal does not replay them at the next start. Returns how many were dropped."""
        items = self._queues.drain(thread_id)
        for item in items:
            self._mark_done(item.update_id)
        return len(items)

    def _drop_removing(self, thread_id: int, update_id: int | None) -> None:
        self._mark_done(update_id)
        self._logger.log(f"message for a topic being removed: dropped (thread {thread_id})")

    # the runner

    def _start(self, thread_id: int, topic: str) -> None:
        self._start_thread(target=self._run, args=(thread_id, topic), daemon=True,
                           name=f"tg-topic-{thread_id}").start()

    def _quiet(self, thread_id: int) -> bool:
        """True when the topic's kept session is idle with no self turn pending: its next message needs no notice."""
        return self._linger is not None and self._linger.idle_alive(thread_id)

    def _run(self, thread_id: int, topic: str) -> None:
        """Serial runner of one topic: turns run one at a time, queued messages follow in order. While the topic's kept
        session is held, the runner stays on the topic and ticks the hold instead of returning on an empty queue.
        Returns when the queue is empty and nothing is held, when a turn was stopped by a restart or refused by a
        shutdown, or when the bridge is shutting down."""
        failures = 0  # consecutive failed hold ticks of this runner
        while not self._live.is_shutting_down():
            self._arrivals.settle(thread_id)  # a forwarded batch still arriving is taken as one turn
            holding = self._holding(thread_id)
            head = self._queues.take_head(thread_id, keep_busy=holding)
            if head.kind is _HeadKind.EMPTY:
                if not holding:
                    return
                failures = self._hold_tick(thread_id, failures)
                continue
            if head.kind is _HeadKind.WAIT:
                self._await_media(thread_id, head.entry)
                continue
            if head.entry is None or head.text is None:  # READY always carries both; guards the type only
                return
            if not self._run_entry(thread_id, topic, head):
                return

    def _await_media(self, thread_id: int, entry: _Entry | None) -> None:
        """Wait for the head placeholder for at most the media wait. While the topic holds a kept session, the wait runs
        in slices of poll_seconds and a hold tick runs between them; when the hold ends, the rest is one plain wait. On
        timeout the place is released and the topic is told; the job still runs, and its late result is queued at the
        back by the inbound handler. The slices are real waits on the event; only the deadline reads the injected
        clock, so a FakeClock moves the deadline while the slices stay real."""
        if entry is None or not isinstance(entry.body, Placeholder):
            return
        ev = entry.body.ev
        deadline = self._clock.monotonic() + self._media_wait
        failures = 0
        while self._holding(thread_id):
            left = deadline - self._clock.monotonic()
            if left <= 0:
                break  # the media wait is over: time out below
            if ev.wait(min(self._poll, left)):
                return
            failures = self._hold_tick(thread_id, failures)
        else:  # the topic is not held (any more): the rest of the wait is plain
            if ev.wait(max(0.0, deadline - self._clock.monotonic())):
                return
        self._media_timed_out(thread_id, entry)

    def _media_timed_out(self, thread_id: int, entry: _Entry) -> None:
        if self._queues.unreserve(thread_id, entry.nid):
            self._logger.log(f"thread {thread_id}: media message not transcribed after {self._media_wait}s "
                             "— released from the queue")
            self._messenger.send_text(
                entry.chat_id, thread_id,
                f"⚠️ a voice/photo is still being processed after {self._media_wait}s — "
                "the queue moved on; it will run as soon as it is ready.")

    def _holding(self, thread_id: int) -> bool:
        return self._linger is not None and self._linger.holding(thread_id)

    def _take_slot(self, thread_id: int, slot: Slot) -> bool:
        """Wait for a slot permit. Without a held session this is the plain take(). While the topic is held, the wait is
        a series of timed takes of poll_seconds, so the slot waiter stays in the pool's queue like any other topic, and
        a hold tick runs between the slices while the topic is still held. When the hold ends, the timed takes go on
        without ticks. The topic is told once that it is queued, as take() does. False: shutdown came first, no permit
        taken."""
        if not self._holding(thread_id):
            return slot.take()
        notified = False
        failures = 0
        while not slot.take_for(self._poll):
            if self._slots.shutting_down():
                return False
            if not notified:
                notified = True
                self._slots.announce_queued(slot.chat_id, slot.thread_id)
            if self._holding(thread_id):
                failures = self._hold_tick(thread_id, failures, clear_mark=False)  # the turn is begun: its mark stays
        return True

    def _hold_tick(self, thread_id: int, failures: int, *, clear_mark: bool = True) -> int:
        """One tick of a topic whose kept session is held: the linger service runs a pending self turn and checks the
        end conditions. A cap or an exit with background work pending is told to the topic. A failed tick is logged and
        the runner goes on, since it owns the topic's queue; after HOLD_FAILURE_LIMIT failures in a row the held session
        is asked to close. Returns the count of consecutive failed ticks. clear_mark False (a tick inside a begun turn's
        slot wait) leaves the turn mark alone: the turn's own end in _run_entry clears it, once no process is left."""
        if self._linger is None:
            return 0
        try:
            result = self._linger.wait_once(thread_id)
        except Exception as exc:
            failures += 1
            self._logger.log(f"linger: thread {thread_id} hold tick failed ({failures} in a row): "
                             f"{str(exc)[:SEND_ERROR_CHARS]}")
            if failures >= HOLD_FAILURE_LIMIT:
                self._linger.request_release(thread_id)  # refused while a self turn runs: the next tick tries again
            if clear_mark:  # a tick that released the hold before it raised has left no process: the mark goes
                self._end_terminal(thread_id)
            time.sleep(self._hold_pause)
            return failures
        if result.end is not None and clear_mark:  # the hold ended and its process is gone: the mark goes with it
            self._end_terminal(thread_id)
        self._announce(thread_id, result)  # a lost notice is only logged: the session is already closed, so no failure
        return 0

    def _announce(self, thread_id: int, result: HoldResult) -> None:
        """Tells the topic why a kept session closed while background work was still pending. Nothing else is posted.
        A send that fails is logged and dropped; it never counts as a failed tick."""
        if result.pending <= 0:
            return
        if result.end is HoldEnd.CAP:
            minutes = math.ceil(self._config.linger_seconds() / SECONDS_PER_MINUTE)
            text = LINGER_CAP_TEXT.format(minutes=minutes)
        elif result.end is HoldEnd.EXITED:
            text = LINGER_EXIT_TEXT.format(pending=result.pending)
        else:
            return
        try:
            self._messenger.send_text(result.chat_id, thread_id, text)
        except Exception as exc:
            self._logger.log(f"linger: thread {thread_id} notice not sent: {str(exc)[:SEND_ERROR_CHARS]}")

    def _run_entry(self, thread_id: int, topic: str, head: _Head) -> bool:
        """Run one queued batch as a turn, once the terminal allows it (TerminalWait). False: the runner must stop
        (restart stop, shutdown refusal, shutdown while the terminal held the topic). A terminal that was not ready when
        the turn was to start (_run_turn returns None) sends the batch back into the terminal wait, without a reply.
        Before a turn the bridge spawns itself (attach None), the before_turn hook runs; an attached turn skips it."""
        while True:
            turn = self._terminal.wait(thread_id, head.entry.chat_id)
            if turn is None:  # no turn started: the rows are still pending for a restart
                return False
            if turn.attach is None and self._before_turn is not None:
                try:
                    self._before_turn(thread_id)
                except Exception as exc:  # the hook is best-effort: the turn runs regardless
                    self._logger.log(f"thread {thread_id}: before-turn hook failed: {str(exc)[:SEND_ERROR_CHARS]}")
            try:
                done = self._run_turn(thread_id, topic, head, turn.attach)
            finally:
                if turn.attach is None:  # an attached turn marks nothing: the terminal's process is not the bridge's
                    self._end_terminal(thread_id)
            if done is not None:
                return done

    def _end_terminal(self, thread_id: int) -> None:
        """Clears the topic's turn mark once no process of the topic is left. A kept session (the topic is held) keeps
        its mark until the hold is released, and a shutdown keeps it: the lifecycle stops the process, and the start-up
        sweep removes the mark."""
        if not self._view_dir or self._holding(thread_id) or self._live.is_shutting_down():
            return
        try:
            terminal_claim.end_turn(self._view_dir, thread_id, self._clock)
        except OSError as exc:
            self._logger.log(f"terminal: thread {thread_id} turn end not recorded: {str(exc)[:SEND_ERROR_CHARS]}")

    def _run_turn(self, thread_id: int, topic: str, head: _Head, attach: AttachSpec | None) -> bool | None:
        """The turn of one queued batch. False: the runner must stop (restart stop, shutdown refusal). None: the
        terminal session was not ready (busy, text typed in it, or gone): nothing ran, the rows are pending again and
        the batch goes back into the terminal wait. A topic removed before the turn spawns gets no turn: the batch is
        dropped here (checked before any write and after the slot). attach names the terminal session the turn runs in
        (None: the bridge spawns its own process). Every update_id of the batch is marked, replayed or dropped as one."""
        chat_id = head.entry.chat_id
        if self._removals.active(thread_id):
            self._batch(head, lambda uid: self._drop_removing(thread_id, uid))
            return True
        try:
            # the turn starts now: a restart from here is reported by recover(), not re-run
            self._batch(head, self._mark_done)
            self._typing(chat_id, thread_id)
            slot = self._slots.new_slot(chat_id, thread_id)
            if not self._take_slot(thread_id, slot):  # a shutdown came while the batch waited: it replays on restart
                self._batch(head, self._replay_later)
                return False
            if self._removals.active(thread_id):  # the topic was removed while the slot was awaited: nothing spawns
                slot.give_back()
                self._batch(head, lambda uid: self._drop_removing(thread_id, uid))
                return True
            try:
                reply = self._turns.run(thread_id=thread_id, chat_id=chat_id, topic=topic, text=head.text,
                                        sender=head.entry.sender, slot=slot, images=head.images, attach=attach)
            finally:
                slot.give_back()
            if reply.terminal_busy:  # nothing was pasted: the batch waits for the terminal again, no reply is sent
                self._batch(head, self._replay_later)
                self._logger.log(f"turn {topic}: terminal not ready, the message waits for it again")
                return None
            if reply.err == RESTART_TEXT:  # a restart stopped this turn: recover() tells the topic, the rows stay done
                return False
            # a shutdown refused the spawn: the batch replays on restart
            if reply.answer is None and reply.err is None:
                self._batch(head, self._replay_later)
                return False
            self._reply(chat_id, thread_id, topic, head.text, reply.answer, reply.err)
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

    @staticmethod
    def _batch(head: _Head, action: Callable[[int | None], None]) -> None:
        """Applies an inbox action to the update_id of every message of the head's batch."""
        for entry in head.entries:
            action(entry.update_id)
