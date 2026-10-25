"""TurnFeed: the live tool log of one turn in Telegram, as an EventListener (port of the sender half of C12 Turn).

The normalized events of a turn arrive from the TurnEventBus on the backend's reader thread. That thread only changes
the feed's state under a Condition and returns; a sender thread makes every Telegram call of the feed, so no two threads
ever post the same message. What the feed shows:

- the assistant's text (💬) as its own message, once something else follows it; the final answer is held back,
  because the turn result carries it;
- the tool calls between two texts in one tool-log message (feed/log.py): a header with the running count, elapsed
  time, idle time and the current state note (API retry, stream cut, compacting), then one line per tool in an
  expandable quote. A tool whose result failed gets ❌; a subagent's line is indented under its parent and counted;
- an open log carries the ⏹ Stop button (callback st:<thread>); a closed log has none.

Pacing: a send waits for flush_seconds since the last send, a log is edited at most once per flush_seconds, and its
header alone is refreshed every header_seconds. A log that is full freezes and the next tool call starts another one.
A permanent failure (HTTP 400, or LOG_GIVE_UP_FAILURES failed calls on one log) ends the feed for the rest of the turn:
the log still on screen loses its Stop button, and finish() gives it the summary if it can. The Telegram rules of a
single call live in feed/wire.py.

Locking: self._cv guards all feed state. The Telegram client and the parked() callable are never called while it is
held, and times come from the injected Clock, so tests move time with a FakeClock.
"""

import html
import math
import threading
from dataclasses import dataclass
from typing import Callable

from tg_bridge.agents.events import SessionStarted, StatusEvent, TextEvent, ToolEnd, ToolStart, TurnResult
from tg_bridge.clock import Clock
from tg_bridge.feed.format import running_header, short, status_text, waiting_header
from tg_bridge.feed.log import LOG_CHARS, LOG_LABEL_CHARS, LogSegment, ToolLine, line_text, lines_text, log_len
from tg_bridge.feed.wire import FeedWire, MessageGone, is_bad_request
from tg_bridge.logging_ import Logger
from tg_bridge.telegram.client import TelegramClient
from tg_bridge.telegram.keyboards import EMPTY_KEYBOARD, CallbackData, keyboard_json

FEED_TEXT_CHARS = 1500  # a 💬 message keeps at most this much of the model's text
LOG_GIVE_UP_FAILURES = 3  # failed calls on one log: the feed gives up for the rest of the turn
RETRY_PAUSE_SECONDS = 5.0  # after a failed send or edit, the log waits this long before it is tried again
POLL_SECONDS = 0.05  # the sender re-checks its queue this often at most, so a fake clock moves it promptly
CLOSE_JOIN_SECONDS = 20.0
FINISH_JOIN_SECONDS = 120.0
ERROR_TEXT_LIMIT = 120
TEXT_MARK = "💬 "
STOP_LABEL = "⏹ Stop"
STOP_PREFIX = "st"
ParkedCheck = Callable[[], bool]


@dataclass(slots=True)
class TextOp:
    """A 💬 text waiting to be sent. host is a tool-log message that had no tool yet (the waiting message): the text
    is edited into it, so the model's first words take the place of the waiting note instead of sitting under it."""

    body: str
    host: LogSegment | None = None


def _never_parked() -> bool:
    return False


def _poll_timeout(wait: float | None) -> float:
    return POLL_SECONDS if wait is None else max(0.001, min(wait, POLL_SECONDS))


def _log_body(header: str, lines: str) -> tuple[str, str]:
    """The HTML and plain versions of a log message: the header, then the lines in an expandable quote."""
    html_text = f"{html.escape(header)}\n<blockquote expandable>{html.escape(lines)}</blockquote>"
    return html_text, f"{header}\n{lines}"


class TurnFeed:
    """Live progress of one turn in one topic. Built per turn by the turn runner, which also calls finish() at the end.

    close() ends the turn's stream: the held text is released unless it is the answer, every log is closed, and the
    sender posts what is left before it stops. finish(header) queues the summary as the last log's header, or posts it
    itself once the sender has stopped. Once the feed has given up, nothing more is queued or posted except the final
    edit of the log that is still on screen.
    """

    def __init__(
        self,
        client: TelegramClient,
        logger: Logger,
        clock: Clock,
        chat_id: int | str,
        thread_id: int,
        *,
        flush_seconds: float = 3.0,
        ping_seconds: float = 60.0,
        header_seconds: float = 60.0,
        parked: ParkedCheck | None = None,
    ) -> None:
        self._logger = logger
        self._clock = clock
        self._thread_id = thread_id
        self._wire = FeedWire(client, logger, chat_id, thread_id)
        self._flush = flush_seconds
        self._ping = ping_seconds
        self._header_every = header_seconds
        self._parked: ParkedCheck = parked or _never_parked
        self._stop_markup = keyboard_json([[(STOP_LABEL, CallbackData(STOP_PREFIX, (thread_id,)))]])
        self._cv = threading.Condition()
        self._tools = 0  # tool calls seen, subagents included
        self._answer: str | None = None
        self._names: dict[str, tuple[LogSegment, ToolLine]] = {}  # tool id -> its log and line, for the ❌ mark
        self._ops: list[LogSegment | TextOp] = []  # waiting to be sent, in order
        self._logs: list[LogSegment] = []  # every log of the turn, in order
        self._held: str | None = None  # the latest top-level text, sent only once something else follows it
        self._state: str | None = None  # a retry or compacting note, until the next event
        self._started = self._last_event = clock.monotonic()
        self._last_send = -math.inf
        self._closed = False
        self._dead = False  # the feed gave up for this turn
        self._text_hosted = False  # the waiting message became a 💬 text, so finish() needs a message of its own
        self._exited = False  # the sender has stopped; set under _cv, so finish() knows who posts
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"feed-{thread_id}")
        self._thread.start()

    def start(self) -> None:
        """Turn start, called by the turn runner before the backend is started: the tool log opens now with the
        waiting note and the Stop button, so a slow model is visible before its first event."""
        with self._cv:
            if not self._closed:
                self._open_log()
                self._cv.notify_all()

    @property
    def tools(self) -> int:
        """Tool calls seen so far in this turn, subagents included (read by /status)."""
        with self._cv:
            return self._tools

    @property
    def gave_up(self) -> bool:
        """True once the feed has given up for this turn."""
        with self._cv:
            return self._dead

    def on_event(self, event: object) -> None:
        """One normalized event of the turn (reader thread). Events the tool log does not show are ignored."""
        now = self._clock.monotonic()
        with self._cv:
            if self._closed:
                return
            if isinstance(event, TextEvent):
                self._on_text(event, now)
            elif isinstance(event, ToolStart):
                self._on_tool_start(event, now)
            elif isinstance(event, ToolEnd):
                self._on_tool_end(event, now)
            elif isinstance(event, StatusEvent):
                self._on_status(event, now)
            elif isinstance(event, SessionStarted):
                self._clear_state()
            elif isinstance(event, TurnResult):
                self._answer = event.answer
            else:
                return
            self._cv.notify_all()

    def close(self) -> None:
        """End of the turn. Joins the sender, which posts whatever is left and then stops."""
        with self._cv:
            if self._held is not None and self._held.strip() != str(self._answer or "").strip():
                self._release()
            self._held = None
            self._close_open()
            self._closed = True
            self._cv.notify_all()
        self._thread.join(timeout=CLOSE_JOIN_SECONDS)

    def finish(self, header: str) -> None:
        """The turn's summary as the last log's header, without the Stop button. Call after close().

        While the sender runs, the summary is its last job, so it never overtakes queued texts and logs. Once the sender
        has stopped, this call posts it. After a dead feed it edits the newest log that has a message. When the waiting
        message was turned into a text, the summary goes into a new message, so every turn ends with one header.
        """
        with self._cv:
            if not self._logs and self._text_hosted and not self._dead:
                self._logs.append(LogSegment())
            if not self._logs:
                return
            seg = self._logs[-1]
            if self._dead:
                seg = next((s for s in reversed(self._logs) if s.mid is not None), None)
                if seg is None:
                    return
            seg.final, seg.closed, seg.dirty = header, True, True
            sender_runs = not self._exited
            if sender_runs and not self._dead:
                if not any(x is seg for x in self._ops):
                    self._ops.append(seg)
                self._cv.notify_all()
        if sender_runs:
            self._thread.join(timeout=FINISH_JOIN_SECONDS)
        if self._dead:
            self._final_edit(seg)
        elif not sender_runs:
            self._post_log(seg)

    # --- reader-thread side: state changes only, caller holds _cv

    def _on_text(self, event: TextEvent, now: float) -> None:
        self._clear_state()
        self._last_event = now
        if event.parent_id is None and event.text.strip():
            self._release()
            self._held = event.text

    def _on_tool_start(self, event: ToolStart, now: float) -> None:
        self._clear_state()
        self._last_event = now
        self._tools += 1
        line = ToolLine(label=short(event.label, LOG_LABEL_CHARS), parent=event.parent_id is not None)
        self._release()
        seg = self._open_log()
        if seg.lines and log_len(seg) + len(line_text(line)) > LOG_CHARS:
            seg.closed, seg.frozen, seg.dirty = True, True, True
            seg = self._open_log()
        seg.lines.append(line)
        seg.tools += 1
        seg.dirty = True
        self._names[event.tool_id] = (seg, line)

    def _on_tool_end(self, event: ToolEnd, now: float) -> None:
        self._clear_state()
        self._last_event = now
        entry = self._names.get(event.tool_id)
        if not event.ok and entry is not None:
            entry[1].failed = True
            entry[0].dirty = True

    def _on_status(self, event: StatusEvent, now: float) -> None:
        self._clear_state()
        note = status_text(event)
        if note is None:
            return
        self._last_event = now
        self._state = note
        self._open_log().dirty = True  # the note shows in the open log, or in a waiting one when there is none

    def _release(self) -> None:
        """The held text joins the queue. An open log with no tool yet is the waiting message: the text takes its
        place. Otherwise the open log ends first."""
        if self._held is None:
            return
        body = short(self._held, FEED_TEXT_CHARS)
        self._held = None
        seg = self._open_seg()
        if seg is not None and not seg.lines:
            self._logs.remove(seg)
            self._text_hosted = True
            self._ops[-1] = TextOp(body, seg)
            return
        self._close_open()
        self._ops.append(TextOp(body))

    def _open_seg(self) -> LogSegment | None:
        """The open log at the end of the queue, if there is one."""
        if self._ops and isinstance(self._ops[-1], LogSegment) and not self._ops[-1].closed:
            return self._ops[-1]
        return None

    def _open_log(self) -> LogSegment:
        """The open log at the end of the queue, or a new one."""
        seg = self._open_seg()
        if seg is None:
            seg = LogSegment()
            self._ops.append(seg)
            self._logs.append(seg)
        return seg

    def _close_open(self) -> None:
        seg = self._open_seg()
        if seg is not None:
            seg.closed = True
            seg.dirty = True

    def _mark_open(self) -> None:
        seg = self._open_seg()
        if seg is not None:
            seg.dirty = True

    def _clear_state(self) -> None:
        if self._state is not None:
            self._state = None
            self._mark_open()

    # --- sender thread side

    def _run(self) -> None:
        while True:
            parked = self._parked()  # outside the lock: it belongs to another service
            with self._cv:
                job, wait = self._pick(self._clock.monotonic(), parked)
                if job is None and not (self._closed and not self._ops):
                    self._cv.wait(_poll_timeout(wait))
                    continue
                if job is None:
                    self._exited = True
                    return
            kind, payload = job
            if kind == "text":
                self._post_text(payload)
                with self._cv:
                    self._ops.pop(0)
            else:
                self._post_log(payload)

    def _pick(self, now: float, parked: bool) -> tuple[tuple[str, object] | None, float | None]:
        """The next job (('text', TextOp) or ('log', segment)) and how long to wait when there is none yet."""
        if self._dead:  # the feed gave up: what is queued is dropped
            self._ops.clear()
            return None, None
        while self._ops:
            head = self._ops[0]
            if isinstance(head, TextOp):
                gap = self._last_send + self._flush - now
                return (("text", head), None) if gap <= 0 else (None, gap)
            if head.closed and not head.dirty:
                self._ops.pop(0)
                continue
            if now < head.next_try:
                return None, head.next_try - now
            if head.mid is None:  # a new message keeps to the send gap like any other send
                gap = self._last_send + self._flush - now
                return (("log", head), None) if gap <= 0 else (None, gap)
            if head.dirty and now - head.last_edit >= self._flush:
                return ("log", head), None
            if head.dirty:  # a closed log's last edit is due once the flush gap has passed
                return None, max(0.01, head.last_edit + self._flush - now)
            if now - head.last_edit >= self._header_every and self._header(head, now, parked) != head.shown:
                return ("log", head), None  # the header alone (elapsed, idle) is refreshed this often
            return None, 1.0
        return None, None

    def _header(self, seg: LogSegment, now: float, parked: bool) -> str:
        """The first line of a log message at this moment. Caller holds _cv."""
        base = seg.header_base()
        if base is not None:
            return base
        if not seg.lines:
            return waiting_header(self._state)
        return running_header(
            self._tools,
            now - self._started,
            now - self._last_event,
            ping_seconds=self._ping,
            parked=parked,
            state=self._state,
        )

    def _post_text(self, op: TextOp) -> None:
        now = self._clock.monotonic()
        with self._cv:
            self._last_send = now
        text = TEXT_MARK + op.body
        host = op.host
        if host is not None and host.mid is not None:  # the waiting message becomes the text, without its button
            try:
                self._wire.call("editMessageText", html.escape(text), text, EMPTY_KEYBOARD, message_id=host.mid)
                return
            except MessageGone:
                pass  # the waiting message is gone: the text is sent as a new message
            except Exception as exc:
                self._logger.log(f"feed {self._thread_id}: text edit failed: {str(exc)[:ERROR_TEXT_LIMIT]}")
                return
        try:
            self._wire.call("sendMessage", html.escape(text), text, None)
        except Exception as exc:
            self._logger.log(f"feed {self._thread_id}: text send failed: {str(exc)[:ERROR_TEXT_LIMIT]}")

    def _post_log(self, seg: LogSegment) -> None:
        """Send or edit one log message with its current text: from the sender, or from finish() after it stopped."""
        parked = self._parked()
        with self._cv:
            now = self._clock.monotonic()
            if seg.mid is None:
                self._last_send = now
            header = self._header(seg, now, parked)
            html_text, plain = _log_body(header, lines_text(seg))
            mid = seg.mid
            if mid is None:
                markup = None if seg.closed else self._stop_markup
            else:
                markup = EMPTY_KEYBOARD if seg.closed else self._stop_markup
            seg.dirty, seg.last_edit, seg.shown = False, now, header
        try:
            if mid is None:
                sent = self._wire.call("sendMessage", html_text, plain, markup)
                with self._cv:
                    seg.mid = sent["message_id"]
            else:
                self._wire.call("editMessageText", html_text, plain, markup, message_id=mid)
        except MessageGone:
            with self._cv:  # the message is gone: the next send starts a new one with the same lines
                seg.mid, seg.dirty, seg.next_try = None, True, 0.0
        except Exception as exc:
            self._on_log_failure(seg, exc, mid)

    def _on_log_failure(self, seg: LogSegment, exc: Exception, mid: int | None) -> None:
        kind = "send" if mid is None else "edit"
        self._logger.log(f"feed {self._thread_id}: log {kind} failed: {str(exc)[:ERROR_TEXT_LIMIT]}")
        give_up = False
        lift_mid: int | None = None
        with self._cv:
            seg.failures += 1
            if seg.failures >= LOG_GIVE_UP_FAILURES or is_bad_request(exc):
                seg.dirty = False  # a 400 never heals on a retry
                self._dead = True
                give_up = True
                lift_mid = seg.mid  # the message may still show the Stop button of its last good edit
            else:
                seg.dirty = True
                seg.next_try = self._clock.monotonic() + RETRY_PAUSE_SECONDS
        if give_up:
            self._logger.log(f"feed {self._thread_id}: gave up the feed of this turn after {seg.failures} failed attempt(s)")
        if lift_mid is not None:
            self._wire.lift_keyboard(lift_mid)

    def _final_edit(self, seg: LogSegment) -> None:
        """Best effort after the feed gave up: the summary and no keyboard on the log message. A 400 on the full body
        gets one more try with the header alone; any other failure is only logged."""
        parked = self._parked()
        with self._cv:
            mid = seg.mid
            header = self._header(seg, self._clock.monotonic(), parked)
            lines = lines_text(seg)
        full = _log_body(header, lines)
        header_only = (html.escape(header), header)
        for html_text, plain in (full, header_only):
            try:
                self._wire.call("editMessageText", html_text, plain, EMPTY_KEYBOARD, message_id=mid)
                return
            except MessageGone:
                return
            except Exception as exc:
                if not is_bad_request(exc):  # a 400 falls through to the header-only form
                    self._logger.log(f"feed {self._thread_id}: final edit failed: {str(exc)[:ERROR_TEXT_LIMIT]}")
                    return
        self._logger.log(f"feed {self._thread_id}: final edit refused even as header only")
