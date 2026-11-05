"""Dispatcher and TopicQueues (STEP 44): port of the dispatch and queue scenarios of fakecli/test_durability.py,
test_restart_media.py, test_slot_release.py and test_modes.py (labels kept in the test docstrings).

A StubTurns stands in for TurnRunner: it records each run, can hold a run on a gate, and plays a scripted reply or
raises. Everything else is real: the SQLite inbox, the SlotPool, the Messenger over FakeTelegram, the AnswerSender,
and the Dispatcher's own runner threads. Every thread the dispatcher starts is recorded and joined in tearDown.
Every wait is an Event or a bounded poll, so a broken dispatcher fails in seconds instead of hanging.
"""

import threading
import time
import unittest

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.fakes.uploads import FakeUploader
from tests.support import TempHome
from tg_bridge.config import Config
from tg_bridge.core.arrivals import BURST_GAP_SECONDS
from tg_bridge.core.dispatcher import MEDIA_WAIT_SECONDS, Dispatcher
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.slots import SlotPool
from tg_bridge.core.topic_queues import QUEUE_MAX, Admission, Delivery, Placeholder, TopicQueues, burst_text
from tg_bridge.core.turn_kit import TurnReply
from tg_bridge.core.turn_state import TurnOutcome
from tg_bridge.core.turn_text import RESTART_TEXT
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.db import Database
from tg_bridge.store.inbox import STATE_DONE, STATE_PENDING, InboxRepo
from tg_bridge.telegram.answers import AnswerSender
from tg_bridge.telegram.messenger import Messenger

WAIT = 5.0
THREAD = 9301
OTHER_THREAD = 9302
CHAT = -100
TOPIC = "Durable"
SENDER = "42"


class ThreadLog:
    """A thread factory for the dispatcher that remembers every thread it made, so tearDown can join them."""

    def __init__(self) -> None:
        self.threads: list[threading.Thread] = []

    def __call__(self, **kwargs: object) -> threading.Thread:
        thread = threading.Thread(**kwargs)  # type: ignore[arg-type]
        self.threads.append(thread)
        return thread


class StubTurns:
    """Stands in for TurnRunner. A text may hold a gate the run waits on, a scripted TurnReply, or an exception."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.runs: list[tuple[int, str, str]] = []  # (thread_id, text, sender), in run order
        self.images: list[tuple[str, ...]] = []  # the images of each run, in run order
        self.attaches: list[object] = []  # the attach target of each run (None: the bridge's own process)
        self.gates: dict[str, threading.Event] = {}
        self.replies: dict[str, TurnReply | Exception] = {}
        self._started: dict[str, threading.Event] = {}

    def started(self, text: str) -> threading.Event:
        with self._lock:
            return self._started.setdefault(text, threading.Event())

    def attached(self, text: str) -> object:
        """The attach target the turn of text was given (None: the bridge's own process)."""
        with self._lock:
            return [a for (_, t, _), a in zip(self.runs, self.attaches) if t == text][-1]

    def texts_run(self, thread_id: int) -> list[str]:
        with self._lock:
            return [text for tid, text, _ in self.runs if tid == thread_id]

    def run(self, *, thread_id: int, chat_id: int, topic: str, text: str, sender: str,
            slot: object = None, images: tuple[str, ...] = (), attach: object = None) -> TurnReply:
        with self._lock:
            self.runs.append((thread_id, text, sender))
            self.images.append(tuple(images))
            self.attaches.append(attach)
        self.started(text).set()
        gate = self.gates.get(text)
        if gate is not None:
            gate.wait(WAIT)
        scripted = self.replies.get(text)
        if isinstance(scripted, Exception):
            raise scripted
        if scripted is not None:
            return scripted
        return TurnReply(f"answer to {text}", None, TurnOutcome.OK)

    def release_all(self) -> None:
        for gate in self.gates.values():
            gate.set()


class RecordingTitles:
    """Stands in for TitleService: records each submit."""

    def __init__(self) -> None:
        self.submits: list[tuple[object, ...]] = []

    def submit(self, chat_id: object, thread_id: object, name: str, first_msg: str, first_answer: str) -> None:
        self.submits.append((chat_id, thread_id, name, first_msg, first_answer))


def start_running(q: TopicQueues, thread_id: int = 1) -> None:
    """Start a turn the way the runner does: the first message leaves the queue and the topic stays busy."""
    assert q.deliver(thread_id, 1, "s", 1, "run").admission is Admission.STARTED
    head = q.take_head(thread_id)
    assert head.text == "run", head


def _joined(thread: threading.Thread) -> bool:
    thread.join(WAIT)
    return not thread.is_alive()


def wait_until(predicate, timeout: float = WAIT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class DispatcherCase(unittest.TestCase):
    """A Dispatcher wired to real storage and Messenger, with a StubTurns, under a TempHome."""

    media_wait: float = MEDIA_WAIT_SECONDS
    config_data: dict = {}

    def setUp(self) -> None:
        self._home = TempHome()
        self.home = self._home.__enter__().home
        self.clock = FakeClock()
        self.logger = Logger(f"{self._home.state}/bridge.log")
        self.tg = FakeTelegram(clock=self.clock)
        self.messenger = Messenger(self.tg, self.logger, self.clock)
        self.db = Database(Paths.from_home(self.home))
        self.inbox = InboxRepo(self.db, self.clock)
        self.live = LiveRegistry()
        self.slots = SlotPool(2, self.messenger, self.live.is_shutting_down, poll_seconds=0.01)
        self.turns = StubTurns()
        self.titles = RecordingTitles()
        self.uploads = FakeUploader()
        self.config = Config(dict(self.config_data), home=self.home, env={})
        self.answers = AnswerSender(self.tg, self.uploads, self.messenger, self.logger, self.clock)
        self.thread_log = ThreadLog()
        self.dispatcher = self._make_dispatcher()

    def _make_dispatcher(self, media_wait: float | None = None) -> Dispatcher:
        return Dispatcher(
            turns=self.turns, slots=self.slots, inbox=self.inbox, messenger=self.messenger, answers=self.answers,
            titles=self.titles, client=self.tg, live=self.live, config=self.config, logger=self.logger,
            media_wait_seconds=self.media_wait if media_wait is None else media_wait,
            start_thread=self.thread_log, clock=self.clock)

    def tearDown(self) -> None:
        self.turns.release_all()
        for thread in self.thread_log.threads:
            thread.join(WAIT)
        self.db.close()
        self._home.__exit__(None, None, None)

    def ingest(self, update_id: int, thread_id: int = THREAD) -> int:
        self.assertTrue(self.inbox.ingest(update_id, {"update_id": update_id}, thread_id=thread_id,
                                          chat_id=CHAT, sender=SENDER))
        return update_id

    def state(self, update_id: int) -> str | None:
        row = self.inbox.get(update_id)
        return row.state if row is not None else None

    def deliver(self, update_id: int, text: str, thread_id: int = THREAD, reserved: str | None = None) -> Delivery:
        return self.dispatcher.deliver(thread_id, TOPIC, CHAT, SENDER, update_id, text, reserved)

    def idle(self) -> bool:
        """True once every runner thread the dispatcher started has ended."""
        return all(_joined(thread) for thread in self.thread_log.threads)


class TopicQueuesUnit(unittest.TestCase):
    """The queue data alone: no threads, no storage."""

    def test_queue_constants(self) -> None:
        """2: media wait covers a worst-case download + transcription (300s)

        MEDIA_WAIT_SECONDS covers a worst-case download and transcription, QUEUE_MAX is 100.
        """
        self.assertEqual(QUEUE_MAX, 100)
        self.assertGreaterEqual(MEDIA_WAIT_SECONDS, 300)

    def test_queue_full_refuses_at_the_limit(self) -> None:
        """test_durability (queue full): a sixth message behind a running turn is refused, nothing is queued."""
        q = TopicQueues()
        start_running(q)
        for i in range(QUEUE_MAX):
            self.assertIs(q.deliver(1, 1, "s", 10 + i, f"q{i}").admission, Admission.QUEUED)
        self.assertIs(q.deliver(1, 1, "s", 99, "one too many").admission, Admission.FULL)
        self.assertEqual([item.text for item in q.snapshot(1)], [f"q{i}" for i in range(QUEUE_MAX)])

    def test_reserve_then_fill_keeps_the_place(self) -> None:
        """test_restart_media (late result): a placeholder keeps its place; filling it does not move it."""
        q = TopicQueues()
        start_running(q)
        reserved = q.reserve(1, 1, "s", 2)
        self.assertIs(reserved.admission, Admission.RESERVED)
        filled = q.deliver(1, 1, "s", 2, "voice text", reserved.nid)
        self.assertIs(filled.admission, Admission.FILLED)
        self.assertEqual(filled.position, 1)
        self.assertTrue(filled.notify)  # reserved while a turn ran, so the topic hears where it stands
        self.assertEqual(q.snapshot(1)[0].text, "voice text")

    def test_filled_placeholder_from_idle_topic_is_not_announced(self) -> None:
        q = TopicQueues()
        reserved = q.reserve(1, 1, "s", 2)
        self.assertIs(reserved.admission, Admission.STARTED)
        filled = q.deliver(1, 1, "s", 2, "voice", reserved.nid)
        self.assertIs(filled.admission, Admission.FILLED)
        self.assertFalse(filled.notify)

    def test_late_result_with_released_placeholder_is_queued_as_new(self) -> None:
        """3: late result queued at the back (the placeholder was released; the late text is queued, not dropped).

        3: user sees the queued notice for the late result (the delivery asks for the notice).
        """
        q = TopicQueues()
        start_running(q)
        self.assertFalse(q.unreserve(1, "gone"))
        delivery = q.deliver(1, 1, "s", 4242, "late voice text", reserved="gone")
        self.assertIs(delivery.admission, Admission.QUEUED)
        self.assertTrue(delivery.notify)
        self.assertEqual([item.text for item in q.snapshot(1)], ["late voice text"])

    def test_unreserve_removes_only_unfilled_placeholders(self) -> None:
        """Ported labels:
            5b: placeholder removed, dispatcher finished"""
        q = TopicQueues()
        start_running(q)
        reserved = q.reserve(1, 1, "s", 2)
        self.assertFalse(q.unreserve(1, None))
        self.assertTrue(q.unreserve(1, reserved.nid))
        self.assertEqual(q.snapshot(1), ())
        second = q.reserve(1, 1, "s", 3)
        q.deliver(1, 1, "s", 3, "filled", second.nid)
        self.assertFalse(q.unreserve(1, second.nid))  # already filled: left in the queue

    def test_to_front_and_drop(self) -> None:
        """test_buttons queue taps: Now and Next move an item to the front, Drop removes it and returns it."""
        q = TopicQueues()
        start_running(q)
        q.deliver(1, 1, "s", 2, "a")
        b = q.deliver(1, 1, "s", 3, "b")
        self.assertTrue(q.to_front(1, b.nid))
        self.assertEqual([item.text for item in q.snapshot(1)], ["b", "a"])
        self.assertFalse(q.to_front(1, "missing"))
        dropped = q.drop(1, b.nid)
        self.assertIsNotNone(dropped)
        self.assertEqual((dropped.text, dropped.update_id), ("b", 3))
        self.assertIsNone(q.drop(1, b.nid))

    def test_drop_wakes_a_placeholder_wait(self) -> None:
        """A runner that waits on a placeholder is woken when that place is dropped, and the place is gone."""
        q = TopicQueues()
        start_running(q)
        reserved = q.reserve(1, 1, "s", 2)
        head = q.take_head(1)  # the runner's view: an unfilled placeholder at the head
        self.assertIsNone(head.text)
        self.assertIsInstance(head.entry.body, Placeholder)
        self.assertFalse(head.entry.body.ev.is_set())
        self.assertIsNotNone(q.drop(1, reserved.nid))
        self.assertTrue(head.entry.body.ev.is_set())
        self.assertEqual(q.snapshot(1), ())

    def test_placeholder_text_is_none_until_filled(self) -> None:
        self.assertIsNone(Placeholder(busy=True).text)
        self.assertFalse(Placeholder(busy=False).ev.is_set())

    def test_delivery_notify_rules(self) -> None:
        """The inbound handler sends the queued notice for QUEUED and for a busy placeholder that is filled, only."""
        self.assertTrue(Delivery(Admission.QUEUED, "n", 1).notify)
        self.assertTrue(Delivery(Admission.FILLED, "n", 1, busy=True).notify)
        self.assertFalse(Delivery(Admission.FILLED, "n", 1, busy=False).notify)
        self.assertFalse(Delivery(Admission.STARTED, "n", 1).notify)
        self.assertFalse(Delivery(Admission.RESERVED, "n", 1).notify)
        self.assertFalse(Delivery(Admission.FULL).notify)


class DispatcherQueueScenarios(DispatcherCase):
    """test_durability (a)(c)(d): a queued text waits behind a running turn; drop; replay of a pending row."""

    def test_text_queued_behind_running_turn_stays_pending_then_runs_in_order(self) -> None:
        """test_durability (a): the first message is done once its turn starts; the queued one stays pending.

        Ported labels:
        a: first turn started
        a: first message done once its turn started (got done)
        a: message sits in the in-memory queue
        a: queued message stays pending (got pending)
        a: queued message dispatched after the first turn
        a: queued row done after dispatch (got done)
        a: turns ran in order
        """
        self.turns.gates["a1"] = threading.Event()
        u1, u2 = self.ingest(990001), self.ingest(990002)
        first = self.deliver(u1, "a1")
        self.assertIs(first.admission, Admission.STARTED)
        self.assertTrue(self.turns.started("a1").wait(WAIT))
        self.assertTrue(wait_until(lambda: self.state(u1) == STATE_DONE))
        queued = self.deliver(u2, "a2")
        self.assertIs(queued.admission, Admission.QUEUED)
        self.assertEqual(queued.position, 1)
        self.assertEqual(self.state(u2), STATE_PENDING)
        self.assertEqual([item.text for item in self.dispatcher.queue_snapshot(THREAD)], ["a2"])
        self.turns.gates["a1"].set()
        self.assertTrue(self.turns.started("a2").wait(WAIT))
        self.assertTrue(wait_until(lambda: self.state(u2) == STATE_DONE))
        self.assertTrue(self.idle())
        self.assertEqual(self.turns.texts_run(THREAD), ["a1", "a2"])

    def test_drop_marks_row_done_and_the_message_never_runs(self) -> None:
        """test_durability (c): a dropped queued message is done in the inbox and does not run.

        Ported labels:
        c: first turn started
        c: queued message pending before drop (got pending)
        c: dropped row done (got done)
        c: dropped message never ran
        """
        self.turns.gates["c1"] = threading.Event()
        u1, u2 = self.ingest(990020), self.ingest(990021)
        self.deliver(u1, "c1")
        self.assertTrue(self.turns.started("c1").wait(WAIT))
        self.deliver(u2, "c2")
        self.assertEqual(self.state(u2), STATE_PENDING)
        nid = self.dispatcher.queue_snapshot(THREAD)[0].nid
        dropped = self.dispatcher.drop(THREAD, nid)
        self.assertIsNotNone(dropped)
        self.assertEqual(self.state(u2), STATE_DONE)
        self.turns.gates["c1"].set()
        self.assertTrue(self.idle())
        self.assertEqual(self.turns.texts_run(THREAD), ["c1"])

    def test_now_and_next_move_the_queued_message_to_the_front(self) -> None:
        """test_buttons (now/jump): a moved message runs before the others that were queued ahead of it."""
        self.turns.gates["x1"] = threading.Event()
        self.deliver(self.ingest(1001), "x1")
        self.assertTrue(self.turns.started("x1").wait(WAIT))
        self.deliver(self.ingest(1002), "x2")
        third = self.deliver(self.ingest(1003), "x3")
        self.assertTrue(self.dispatcher.to_front(THREAD, third.nid or ""))
        self.assertEqual([item.text for item in self.dispatcher.queue_snapshot(THREAD)], ["x3", "x2"])
        self.turns.gates["x1"].set()
        self.assertTrue(self.idle())
        # x3 and x2 were both queued behind x1, so they run as one turn, x3 first
        self.assertEqual(self.turns.texts_run(THREAD), ["x1", burst_text(["x3", "x2"])])
        self.assertEqual(self.dispatcher.queue_snapshot(THREAD), ())

    def test_queue_full_refuses_the_sixth_message_and_the_rest_run_in_order(self) -> None:
        self.turns.gates["f1"] = threading.Event()
        self.deliver(self.ingest(2000), "f1")
        self.assertTrue(self.turns.started("f1").wait(WAIT))
        for i in range(QUEUE_MAX):
            self.assertIs(self.deliver(self.ingest(2001 + i), f"f{i + 2}").admission, Admission.QUEUED)
        refused_uid = self.ingest(2001 + QUEUE_MAX)
        self.assertIs(self.deliver(refused_uid, "f-extra").admission, Admission.FULL)
        self.assertEqual(self.state(refused_uid), STATE_PENDING)
        self.turns.gates["f1"].set()
        self.assertTrue(self.idle())
        # the 100 queued messages run in two batches of MERGE_MAX (50), in order
        self.assertEqual(self.turns.texts_run(THREAD), [
            "f1", burst_text([f"f{i}" for i in range(2, 52)]), burst_text([f"f{i}" for i in range(52, 102)])])

    def test_replayed_pending_row_runs_exactly_once(self) -> None:
        """test_durability (d): a pending row with an idle topic starts a runner; its row is done after it runs."""
        uid = self.ingest(990030)
        self.assertIs(self.deliver(uid, "d1").admission, Admission.STARTED)
        self.assertTrue(self.idle())
        self.assertEqual(self.turns.texts_run(THREAD), ["d1"])
        self.assertEqual(self.state(uid), STATE_DONE)

    def test_topics_are_independent(self) -> None:
        self.turns.gates["t1"] = threading.Event()
        self.deliver(self.ingest(3001), "t1", thread_id=THREAD)
        self.assertTrue(self.turns.started("t1").wait(WAIT))
        other = self.deliver(self.ingest(3002, OTHER_THREAD), "t2", thread_id=OTHER_THREAD)
        self.assertIs(other.admission, Admission.STARTED)  # not queued behind the other topic's turn
        self.assertTrue(self.turns.started("t2").wait(WAIT))
        self.turns.gates["t1"].set()
        self.assertTrue(self.idle())


class DispatcherMediaScenarios(DispatcherCase):
    """test_durability (b) and test_restart_media (2)(3): a media placeholder holds its place."""

    def test_voice_placeholder_holds_the_head_until_filled(self) -> None:
        """test_durability (b): the voice row stays pending while its placeholder is not filled, then runs.

        Ported labels:
            1b: transcribed text reached the dispatcher path
            b: voice row pending while the job is queued/running (got pending)
            b: voice row still pending while transcription is blocked
            b: transcribed voice message dispatched
            b: voice row done after dispatch (got done)
        """
        uid = self.ingest(990010)
        reserved = self.dispatcher.reserve(THREAD, TOPIC, CHAT, SENDER, uid)
        self.assertIs(reserved.admission, Admission.STARTED)
        time.sleep(0.1)
        self.assertEqual(self.state(uid), STATE_PENDING)
        self.assertEqual(self.turns.texts_run(THREAD), [])
        self.assertIs(self.deliver(self.ingest(990011), "after voice", reserved=reserved.nid).admission,
                      Admission.FILLED)
        self.assertTrue(self.idle())
        self.assertEqual(self.turns.texts_run(THREAD), ["after voice"])
        self.assertEqual(self.state(uid), STATE_DONE)

    def test_text_behind_a_placeholder_waits_for_it(self) -> None:
        """A text sent after a voice note waits behind the voice note's place, so the order of messages is kept:
        once the voice is filled, both are ready and run as one turn, the voice first.

        Ported labels:
            4a: voice and text both ran
            4b: voice transcript ran before the text (got ['hello there', 'after voice'])
        """
        voice = self.dispatcher.reserve(THREAD, TOPIC, CHAT, SENDER, self.ingest(4001))
        behind = self.deliver(self.ingest(4002), "typed second")
        self.assertIs(behind.admission, Admission.QUEUED)
        self.assertEqual(behind.position, 2)
        self.deliver(self.ingest(4003), "voice first", reserved=voice.nid)
        self.assertTrue(self.idle())
        self.assertEqual(self.turns.texts_run(THREAD), [burst_text(["voice first", "typed second"])])

    def test_media_timeout_releases_the_place_and_tells_the_topic(self) -> None:
        """2: dispatcher moved on after the placeholder timed out
        2: topic told the media is still processing
        2: placeholder released from the queue
        2: inbox row stays pending while the media job still runs

        test_restart_media (2): after the wait the place is released, the topic is told, the row stays pending.
        """
        self.dispatcher = self._make_dispatcher(media_wait=0.05)
        media = self.ingest(4100)
        self.dispatcher.reserve(THREAD, TOPIC, CHAT, SENDER, media)
        self.assertTrue(self.idle())
        self.assertTrue(any("still being processed" in text for text in self.tg.texts(THREAD)))
        self.assertEqual(self.dispatcher.queue_snapshot(THREAD), ())
        self.assertEqual(self.state(media), STATE_PENDING)
        self.assertEqual(self.turns.texts_run(THREAD), [])

    def test_late_result_after_release_still_runs(self) -> None:
        """3: late result queued at the back: the late result is not dropped; it runs as the next message.

        test_restart_media (3): the late result is not dropped; it runs as the next message.
        """
        self.dispatcher = self._make_dispatcher(media_wait=0.05)
        late_uid = self.ingest(4200)
        self.dispatcher.reserve(THREAD, TOPIC, CHAT, SENDER, self.ingest(4199))
        self.assertTrue(self.idle())
        self.assertIs(self.deliver(late_uid, "late voice text").admission, Admission.STARTED)
        self.assertTrue(self.idle())
        self.assertEqual(self.turns.texts_run(THREAD), ["late voice text"])
        self.assertEqual(self.state(late_uid), STATE_DONE)

    def test_filled_placeholder_while_a_turn_runs_is_announced(self) -> None:
        """test_restart_media (3) busy case: a voice reserved behind a turn is announced when its text arrives."""
        self.turns.gates["busy"] = threading.Event()
        self.deliver(self.ingest(4300), "busy")
        self.assertTrue(self.turns.started("busy").wait(WAIT))
        reserved = self.dispatcher.reserve(THREAD, TOPIC, CHAT, SENDER, self.ingest(4301))
        self.assertIs(reserved.admission, Admission.RESERVED)
        filled = self.deliver(self.ingest(4302), "voice done", reserved=reserved.nid)
        self.assertIs(filled.admission, Admission.FILLED)
        self.assertTrue(filled.notify)
        self.turns.gates["busy"].set()
        self.assertTrue(self.idle())
        self.assertEqual(self.turns.texts_run(THREAD), ["busy", "voice done"])

    def test_unreserve_of_unknown_placeholder_is_false(self) -> None:
        self.assertFalse(self.dispatcher.unreserve(THREAD, None))
        self.assertFalse(self.dispatcher.unreserve(THREAD, "nope"))


class GatedClock(FakeClock):
    """A FakeClock whose sleep waits for the test: the runner settles there while the test delivers the rest of a
    batch, so every message of the batch lands before the settle ends. Only the first sleep blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.released = threading.Event()

    def sleep(self, seconds: float) -> None:
        self.entered.set()
        self.released.wait(WAIT)
        super().sleep(seconds)


class BurstScenarios(DispatcherCase):
    """Batching: messages that arrive together run as one turn, and their inbox rows follow the batch."""

    def setUp(self) -> None:
        super().setUp()
        self.clock = GatedClock()
        self.dispatcher = self._make_dispatcher()

    def deliver_together(self, *messages: tuple[int, str]) -> None:
        """Hands over messages the way a forwarded batch arrives: the first starts the runner, which settles; the rest
        arrive during the settle, then the settle ends."""
        for update_id, _ in messages:
            self.ingest(update_id)
        self.deliver(messages[0][0], messages[0][1])
        self.assertTrue(self.clock.entered.wait(WAIT))
        for update_id, text in messages[1:]:
            self.deliver(update_id, text)
        self.clock.released.set()

    def test_forwarded_batch_runs_as_one_turn_and_every_row_is_done(self) -> None:
        self.deliver_together((8001, "bug one"), (8002, "bug two"), (8003, "bug three"))
        self.assertTrue(self.idle())
        self.assertEqual(self.turns.texts_run(THREAD), [burst_text(["bug one", "bug two", "bug three"])])
        self.assertTrue(self.turns.texts_run(THREAD)[0].startswith("[3 messages sent together"))
        self.assertEqual([self.state(uid) for uid in (8001, 8002, 8003)], [STATE_DONE] * 3)
        self.assertEqual(self.dispatcher.queue_snapshot(THREAD), ())

    def test_message_alone_starts_its_turn_after_the_gap_only(self) -> None:
        uid = self.ingest(8101)
        self.assertIs(self.deliver(uid, "alone").admission, Admission.STARTED)
        self.assertTrue(self.clock.entered.wait(WAIT))
        self.assertEqual(self.turns.texts_run(THREAD), [])  # still settling: the gap has not passed
        self.clock.released.set()
        self.assertTrue(self.idle())
        self.assertEqual(self.clock.sleeps, [BURST_GAP_SECONDS])
        self.assertEqual(self.turns.texts_run(THREAD), ["alone"])

        # a message that has waited in the queue for longer than the gap is not delayed once its turn ends
        self.turns.gates["slow"] = threading.Event()
        self.assertIs(self.deliver(self.ingest(8102), "slow").admission, Admission.STARTED)
        self.assertTrue(self.turns.started("slow").wait(WAIT))
        self.assertIs(self.deliver(self.ingest(8103), "later").admission, Admission.QUEUED)
        self.clock.advance(BURST_GAP_SECONDS + 1)
        self.turns.gates["slow"].set()
        self.assertTrue(self.idle())
        self.assertEqual(self.clock.sleeps, [BURST_GAP_SECONDS] * 2)  # one gap for "alone", one for "slow", none after
        self.assertEqual(self.turns.texts_run(THREAD), ["alone", "slow", "later"])

    def test_refused_batch_replays_every_row_on_restart(self) -> None:
        """A shutdown refused the spawn of a batch: every update_id of the batch goes back to pending."""
        self.turns.replies[burst_text(["r1", "r2"])] = TurnReply(None, None, TurnOutcome.RESTART)
        self.deliver_together((8201, "r1"), (8202, "r2"))
        self.assertTrue(self.idle())
        self.assertEqual([self.state(uid) for uid in (8201, 8202)], [STATE_PENDING] * 2)
        self.assertEqual(self.tg.texts(THREAD), [])


class DispatcherRunScenarios(DispatcherCase):
    """The turn part of the runner: answers, errors, restart stops, shutdown refusals and slots."""

    def test_answer_is_sent_and_the_title_is_submitted_after_it(self) -> None:
        uid = self.ingest(5001)
        self.deliver(uid, "hello")
        self.assertTrue(self.idle())
        self.assertTrue(any("answer to hello" in text for text in self.tg.texts(THREAD)))
        self.assertEqual(self.titles.submits, [(CHAT, THREAD, TOPIC, "hello", "answer to hello")])
        self.assertTrue(any(c.method == "sendChatAction" and c.params.get("action") == "typing"
                            for c in self.tg.calls))

    def test_long_answer_goes_out_as_a_document_past_the_config_threshold(self) -> None:
        self.config = Config({"answer_doc_threshold": 5}, home=self.home, env={})
        self.dispatcher = self._make_dispatcher()
        self.deliver(self.ingest(5002), "doc")
        self.assertTrue(self.idle())
        self.assertEqual(len(self.uploads.uploads), 1)

    def test_turn_error_text_is_sent_and_the_queue_moves_on(self) -> None:
        """A turn that raises is reported as a bridge error; the next queued message still runs."""
        self.turns.gates["boom"] = threading.Event()
        self.turns.replies["boom"] = RuntimeError("kaboom")
        self.deliver(self.ingest(5101), "boom")
        self.assertTrue(self.turns.started("boom").wait(WAIT))
        uid = self.ingest(5102)
        self.deliver(uid, "after")
        self.turns.gates["boom"].set()
        self.assertTrue(self.idle())
        self.assertTrue(any("bridge error: kaboom" in text for text in self.tg.texts(THREAD)))
        self.assertEqual(self.turns.texts_run(THREAD), ["boom", "after"])
        self.assertEqual(self.state(uid), STATE_DONE)

    def test_error_reply_is_sent_as_text(self) -> None:
        self.turns.replies["bad"] = TurnReply(None, "⏹ interrupted — session kept", TurnOutcome.INTERRUPTED)
        self.deliver(self.ingest(5201), "bad")
        self.assertTrue(self.idle())
        self.assertIn("⏹ interrupted — session kept", self.tg.texts(THREAD))
        self.assertEqual(self.titles.submits, [])

    def test_restart_stop_keeps_the_row_done_and_says_nothing(self) -> None:
        """1: dispatcher finished after the restart
        1: no error text sent to the topic during a restart (sent [])

        test_durability/test_restart_media (1): a restart stop sends no error text and leaves the row done.
        """
        self.turns.replies["long"] = TurnReply(None, RESTART_TEXT, TurnOutcome.RESTART)
        uid = self.ingest(5301)
        self.deliver(uid, "long")
        self.assertTrue(self.idle())
        self.assertEqual(self.tg.texts(THREAD), [])
        self.assertEqual(self.state(uid), STATE_DONE)

    def test_refused_spawn_during_shutdown_replays_on_restart(self) -> None:
        """A shutdown refused the spawn before any input was sent: the row goes back to pending."""
        self.turns.replies["refused"] = TurnReply(None, None, TurnOutcome.RESTART)
        uid = self.ingest(5401)
        self.deliver(uid, "refused")
        self.assertTrue(self.idle())
        self.assertEqual(self.state(uid), STATE_PENDING)
        self.assertEqual(self.tg.texts(THREAD), [])

    def test_shutdown_while_waiting_for_a_slot_leaves_the_row_pending(self) -> None:
        """test_slot_release (busy bridge) + shutdown: the queued message replays on restart and does not run now."""
        self.assertTrue(self.slots.acquire_permit(timeout=0))
        self.slots.acquire_permit(timeout=0)  # hold the second permit too, so the pool is full for this test
        uid = self.ingest(5501)
        self.deliver(uid, "waiting")
        self.assertTrue(wait_until(lambda: any("waiting for a free slot" in t for t in self.tg.texts(THREAD))))
        self.live.begin_shutdown()
        self.assertTrue(self.idle())
        self.assertEqual(self.state(uid), STATE_PENDING)
        self.assertEqual(self.turns.texts_run(THREAD), [])

    def test_busy_bridge_announces_once_and_runs_when_a_slot_frees(self) -> None:
        """test_slot_release 2: the topic is told once that it waits for a free slot, then the turn runs.

        Ported labels:
            busy bridge: the topic is told it is waiting for a free slot
            busy bridge: turn has not run yet
        """
        self.slots = SlotPool(1, self.messenger, self.live.is_shutting_down, poll_seconds=0.01)
        self.dispatcher = self._make_dispatcher()
        self.assertTrue(self.slots.acquire_permit(timeout=0))
        self.deliver(self.ingest(5601), "hi again")
        self.assertTrue(wait_until(lambda: any("waiting for a free slot" in t for t in self.tg.texts(THREAD))))
        time.sleep(0.05)
        self.assertEqual(self.turns.texts_run(THREAD), [])
        self.slots.release_permit()
        self.assertTrue(self.idle())
        self.assertEqual(self.turns.texts_run(THREAD), ["hi again"])
        self.assertEqual(sum("waiting for a free slot" in t for t in self.tg.texts(THREAD)), 1)
        self.assertEqual(self.slots.free(), 1)

    def test_queue_snapshot_is_a_copy(self) -> None:
        self.turns.gates["snap"] = threading.Event()
        self.deliver(self.ingest(5701), "snap")
        self.assertTrue(self.turns.started("snap").wait(WAIT))
        self.deliver(self.ingest(5702), "queued")
        snap = self.dispatcher.queue_snapshot(THREAD)
        self.assertEqual([item.text for item in snap], ["queued"])
        self.assertEqual(snap[0].update_id, 5702)
        self.turns.gates["snap"].set()
        self.assertTrue(self.idle())
        self.assertEqual(self.dispatcher.queue_snapshot(THREAD), ())


if __name__ == "__main__":
    unittest.main()
