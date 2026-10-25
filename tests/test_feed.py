"""TurnFeed: the live tool log, ported from fakecli/test_feed.py (B-E, H-K, N, P-W) and fakecli/test_tool_log.py.

Scenarios that belong to other steps are left out here: the label rules (agents/claude/tools.py), the footer and header
texts (test_feed_format.py), the Stop tap (callbacks), the end-to-end runs through the fake CLI (turn runner), and the
early waiting message (step 33).

The feed runs on its real sender thread. Time comes from a FakeClock: a test moves it with advance(), and run_until_done()
moves it while close() or finish() waits for the sender, so flush gaps and back-offs pass without sleeping. The fake
Telegram is wrapped in a recorder, so each test can rebuild what the chat shows: texts, edits and keyboards.
"""

import html
import json
import os
import re
import threading
import time
import unittest
from dataclasses import dataclass
from typing import Any, Callable

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.agents.events import (
    ProcessExited,
    StatusEvent,
    TextEvent,
    ToolEnd,
    ToolStart,
    TurnResult,
    TurnStatus,
    UsageEvent,
)
from tg_bridge.feed.feed import TurnFeed
from tg_bridge.feed.log import LOG_CHARS
from tg_bridge.logging_ import Logger
from tg_bridge.telegram.keyboards import EMPTY_KEYBOARD

CHAT = -1001
THREAD = 9601
OTHER_THREAD = 9602
STOP_MARKUP = {"inline_keyboard": [[{"text": "⏹ Stop", "callback_data": f"st:{THREAD}"}]]}
NOT_MODIFIED = "Bad Request: message is not modified: specified new message content"
GONE = "Bad Request: message to edit not found"
THREAD_GONE = "Bad Request: message thread not found"
CANT_PARSE = "Bad Request: can't parse entities"
TOO_LONG = "Bad Request: message is too long"


@dataclass(frozen=True)
class Sent:
    """One call as the feed made it; result is None when the call failed."""

    method: str
    feed: bool
    params: dict[str, Any]
    result: Any


class RecordingTelegram:
    """A FakeTelegram behind a recorder. Scripted failures work as on the fake; every call is kept, failed or not."""

    def __init__(self, clock: FakeClock) -> None:
        self._fake = FakeTelegram(clock=clock)
        self._clock = clock
        self._lock = threading.Lock()
        self._calls: list[Sent] = []
        self._times: list[tuple[str, float]] = []

    @property
    def calls(self) -> tuple[Sent, ...]:
        with self._lock:
            return tuple(self._calls)

    def timed(self, method: str) -> list[float]:
        """The fake clock time of every call of method, in order."""
        with self._lock:
            return [at for name, at in self._times if name == method]

    def fail_next(self, method: str, code: int, description: str = "") -> None:
        self._fake.fail_next(method, code, description=description)

    def call(self, method: str, feed: bool = False, **params: Any) -> Any:
        try:
            result = self._fake.call(method, feed=feed, **params)
        except Exception:
            self._record(method, feed, params, None)
            raise
        self._record(method, feed, params, result)
        return result

    def _record(self, method: str, feed: bool, params: dict[str, Any], result: Any) -> None:
        with self._lock:
            self._calls.append(Sent(method, feed, dict(params), result))
            self._times.append((method, self._clock.now()))


def visible(tg: RecordingTelegram, thread_id: int = THREAD) -> list[tuple[str, Any]]:
    """What the topic shows, oldest message first: (text, reply_markup) after every applied edit. HTML is unescaped."""
    messages: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for c in tg.calls:
        if c.result is None:
            continue
        if c.method == "sendMessage" and str(c.params.get("message_thread_id")) == str(thread_id):
            mid = c.result["message_id"]
            messages[mid] = {"text": c.params["text"], "markup": c.params.get("reply_markup"),
                             "parse": c.params.get("parse_mode")}
            order.append(mid)
        elif c.method in ("editMessageText", "editMessageReplyMarkup") and c.params["message_id"] in messages:
            message = messages[c.params["message_id"]]
            if c.method == "editMessageText":
                message["text"], message["parse"] = c.params["text"], c.params.get("parse_mode")
            if "reply_markup" in c.params:
                message["markup"] = c.params["reply_markup"]
    out = []
    for mid in order:
        message = messages[mid]
        text = message["text"]
        if message["parse"] == "HTML":
            text = html.unescape(re.sub(r"</?blockquote[^>]*>", "", text))
        out.append((text, message["markup"]))
    return out


def texts(tg: RecordingTelegram, thread_id: int = THREAD) -> list[str]:
    return [text for text, _ in visible(tg, thread_id)]


def calls_of(tg: RecordingTelegram, method: str) -> list[Sent]:
    return [c for c in tg.calls if c.method == method]


def run_until_done(clock: FakeClock, action: Callable[[], None], step: float = 1.0, timeout: float = 20.0) -> bool:
    """Run action (close or finish) on its own thread while the fake clock moves on, so the sender's gaps pass."""
    worker = threading.Thread(target=action, daemon=True)
    worker.start()
    deadline = time.monotonic() + timeout
    while worker.is_alive() and time.monotonic() < deadline:
        clock.advance(step)
        time.sleep(0.002)
    worker.join(timeout=1.0)
    return not worker.is_alive()


def settle(clock: FakeClock, pred: Callable[[], bool], timeout: float = 5.0, step: float = 1.0) -> bool:
    """Move the fake clock until pred holds (the sender works in between)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        clock.advance(step)
        time.sleep(0.002)
    return pred()


def first_line(text: str) -> str:
    return text.split("\n")[0]


def read(i: int) -> ToolStart:
    return ToolStart(tool_id=f"r{i}", name="Read", label=f"Read f{i}.py")


def result(answer: str) -> TurnResult:
    return TurnResult(status=TurnStatus.OK, answer=answer)


class FeedCase(unittest.TestCase):
    """Every test gets its own HOME, clock, recorder and logger; feeds made here are closed at cleanup."""

    def setUp(self) -> None:
        home = TempHome()
        home.__enter__()
        self.addCleanup(home.__exit__, None, None, None)
        self.clock = FakeClock()
        self.tg = RecordingTelegram(self.clock)
        self.logger = Logger(os.path.join(home.state, "bridge.log"))

    def feed(self, thread_id: int = THREAD, **options: Any) -> TurnFeed:
        settings = {"flush_seconds": 0.3, "ping_seconds": 10**6, "header_seconds": 10**6}
        settings.update(options)
        feed = TurnFeed(self.tg, self.logger, self.clock, CHAT, thread_id, **settings)
        self.addCleanup(run_until_done, self.clock, feed.close)
        return feed

    def close(self, feed: TurnFeed) -> None:
        self.assertTrue(run_until_done(self.clock, feed.close), "close() did not return")


class TextAndSummaryOrder(FeedCase):
    def test_texts_and_tool_log_in_order_with_failed_and_subagent_lines(self) -> None:
        """Port of test_feed.py B: the text goes out first, the tools share one log, a subagent's line is indented,

        Ported labels:
        B: messages use HTML
        B: user text is HTML-escaped on the wire
        B: tool lines sit in an expandable quote
        B: text, then one log with the tool lines; failed tool marked (['💬 Let me look.', '🔧 3 tools\\n🔧 Read src/a.py\\n❌ Bash <b>&\\n  ↳ 🔧 Glob *.md'])
        B: the answer and subagent text are not shown; the retry notice is cleared by the next event
        B: the 💬 message has no Stop button (only the open log carries one)
        B: the finished log has no Stop button
        a failed tool is marked, and the answer and subagent text are never shown."""
        feed = self.feed()
        feed.on_event(TextEvent(text="Let me look."))
        feed.on_event(ToolStart(tool_id="r1", name="Read", label="Read src/a.py"))
        feed.on_event(ToolStart(tool_id="b1", name="Bash", label="Bash <b>&"))
        feed.on_event(ToolStart(tool_id="s1", name="Glob", label="Glob *.md", parent_id="b1"))
        feed.on_event(TextEvent(text="SUBAGENT TEXT", parent_id="b1"))
        feed.on_event(ToolEnd(tool_id="b1", ok=False, detail="boom"))
        feed.on_event(StatusEvent(kind="retry", attempt=2, max_attempts=10))
        feed.on_event(TextEvent(text="ok:allow"))
        feed.on_event(result("ok:allow"))
        self.close(feed)

        self.assertEqual(
            visible(self.tg),
            [("💬 Let me look.", None),
             ("🔧 3 tools\n🔧 Read src/a.py\n❌ Bash <b>&\n  ↳ 🔧 Glob *.md", None)],
        )
        wire = [c.params["text"] for c in self.tg.calls if c.result is not None]
        self.assertTrue(any("&lt;b&gt;&amp;" in t for t in wire), "user text is HTML-escaped on the wire")
        self.assertTrue(any("<blockquote expandable>" in t for t in wire), "tool lines sit in an expandable quote")
        self.assertFalse(any("SUBAGENT" in t or "ok:allow" in t for t in wire))
        self.assertTrue(all(c.params.get("parse_mode") == "HTML" for c in self.tg.calls
                            if c.method in ("sendMessage", "editMessageText")))
        self.assertTrue(all(c.feed for c in self.tg.calls if c.method in ("sendMessage", "editMessageText")))

    def test_sends_and_edits_keep_the_flush_gap_between_them(self) -> None:
        """Ported labels:
        B: never two sends closer than the flush gap
        D: edits respect the 0.3 s flush gap
        """
        feed = self.feed(flush_seconds=0.3)
        feed.on_event(read(0))
        self.assertTrue(settle(self.clock, lambda: len(self.tg.timed("sendMessage")) == 1))
        for i in range(1, 10):
            feed.on_event(read(i))
            self.clock.advance(0.1)
            time.sleep(0.01)
        feed.on_event(TextEvent(text="between"))
        for i in range(10, 20):
            feed.on_event(read(i))
        self.assertTrue(run_until_done(self.clock, feed.close, step=0.05))
        sends, edits = self.tg.timed("sendMessage"), self.tg.timed("editMessageText")
        self.assertGreater(len(sends), 1)
        self.assertGreater(len(edits), 1)
        self.assertTrue(all(b - a >= 0.3 - 1e-9 for a, b in zip(sends, sends[1:])))
        self.assertTrue(all(b - a >= 0.3 - 1e-9 for a, b in zip(edits, edits[1:])))

    def test_held_text_that_differs_from_the_answer_is_echoed_on_close(self) -> None:
        """Port of test_feed.py C.

        Ported labels:
        C: differing held text is echoed (['💬 Partial reply'])
        """
        feed = self.feed()
        feed.on_event(TextEvent(text="Partial reply"))
        feed.on_event(result("Something else"))
        self.close(feed)
        self.assertEqual(texts(self.tg), ["💬 Partial reply"])

    def test_text_between_tools_and_summary_on_the_last_log_only(self) -> None:
        """Port of test_tool_log.py B: texts and logs alternate in order; the summary goes on the last log.

        Ported labels:
        B: texts and logs in order, summary only on the last log
        B: no message keeps a Stop button once the turn has ended; the 💬 text never carries one
        B: the summary is not on an earlier log
        """
        feed = self.feed()
        feed.on_event(TextEvent(text="A"))
        feed.on_event(read(1))
        feed.on_event(TextEvent(text="B"))
        feed.on_event(ToolStart(tool_id="b1", name="Bash", label="Bash List"))
        feed.on_event(ToolStart(tool_id="g1", name="Glob", label="Glob *.md"))
        feed.on_event(result("done"))
        self.assertTrue(run_until_done(self.clock, feed.close))
        self.assertTrue(run_until_done(self.clock, lambda: feed.finish("✅ 2 tools · summary")))

        shown = visible(self.tg)
        self.assertEqual(
            [text for text, _ in shown],
            ["💬 A", "🔧 1 tools\n🔧 Read f1.py", "💬 B", "✅ 2 tools · summary\n🔧 Bash List\n🔧 Glob *.md"],
        )
        self.assertFalse(any("st:" in (markup or "") for _, markup in shown), "no message keeps a Stop button")
        self.assertTrue(all(markup is None for text, markup in shown if text.startswith("💬")))

    def test_summary_is_the_senders_last_job(self) -> None:
        """Port of test_tool_log.py S: every queued text and log is posted before the summary edit.

        Ported labels:
        S: texts and logs in order, the summary last
        S: each message was sent once (7 sends)
        S: the summary edit is the last call of the turn
        S: the sender has stopped once close and finish returned
        """
        feed = self.feed()
        for i in range(3):
            feed.on_event(read(i))
            feed.on_event(TextEvent(text=f"step {i}"))
        feed.on_event(read(9))
        self.close(feed)
        feed.finish("✅ FINAL")

        self.assertEqual(
            texts(self.tg),
            ["🔧 1 tools\n🔧 Read f0.py", "💬 step 0", "🔧 1 tools\n🔧 Read f1.py", "💬 step 1",
             "🔧 1 tools\n🔧 Read f2.py", "💬 step 2", "✅ FINAL\n🔧 Read f9.py"],
        )
        self.assertEqual(len([c for c in calls_of(self.tg, "sendMessage") if c.result]), 7)
        last = self.tg.calls[-1]
        self.assertEqual(last.method, "editMessageText")
        self.assertTrue(last.params["text"].startswith("✅ FINAL\n<blockquote"))
        self.assertFalse(feed._thread.is_alive(), "the sender is still running")

    def test_text_only_turn_posts_only_its_text_and_finish_posts_nothing_without_a_log(self) -> None:
        """Port of test_tool_log.py I and J.

        Ported labels:
        I: a text-only turn posts only its text
        """
        feed = self.feed()
        feed.on_event(TextEvent(text="just words"))
        feed.on_event(result("the answer"))
        self.close(feed)
        feed.finish("✅ 0 tools")
        self.assertEqual(texts(self.tg), ["💬 just words"])

    def test_no_log_means_finish_sends_nothing(self) -> None:
        """
        Ported labels:
        J: no log, no message
        """
        feed = self.feed()
        self.close(feed)
        feed.finish("✅ 0 tools")
        self.assertEqual(self.tg.calls, ())

    def test_text_message_is_sent_without_a_keyboard(self) -> None:
        """Port of test_tool_log.py R: the 💬 text has no reply_markup, the open log does.

        Ported labels:
        R: the 💬 text is sent without reply_markup
        """
        feed = self.feed()
        feed.on_event(TextEvent(text="step one"))
        feed.on_event(read(1))
        self.close(feed)
        feed.finish("✅ 1 tools")
        text_sends = [c for c in calls_of(self.tg, "sendMessage") if c.params["text"].startswith("💬")]
        self.assertEqual(len(text_sends), 1)
        self.assertNotIn("reply_markup", text_sends[0].params)


class TurnStateAndHeader(FeedCase):
    def test_running_header_shows_retry_and_compacting_then_clears_on_the_next_tool(self) -> None:
        """Port of test_tool_log.py A: count and elapsed time, the API retry note, compacting, the Stop button.

        Ported labels:
        A: API retry shown in the header
        A: closed log keeps a segment header
        A: compacting shown in the header
        A: the open log carries the Stop button
        A: the retry note is cleared by the next event
        """
        feed = self.feed()
        feed.on_event(read(1))
        self.assertTrue(settle(self.clock, lambda: len(visible(self.tg)) == 1))
        head = first_line(visible(self.tg)[0][0])
        self.assertRegex(head, r"^⚙️ working · 1 tools · \d+:\d\d$")
        self.assertEqual(visible(self.tg)[0][1], json.dumps(STOP_MARKUP))

        feed.on_event(StatusEvent(kind="retry", attempt=2, max_attempts=10))
        self.assertTrue(settle(self.clock, lambda: "🔁 API retry 2/10" in first_line(visible(self.tg)[0][0])))

        feed.on_event(read(2))
        self.assertTrue(settle(self.clock, lambda: first_line(visible(self.tg)[0][0]).startswith("⚙️ working · 2 tools")
                               and "API retry" not in first_line(visible(self.tg)[0][0])))

        feed.on_event(StatusEvent(kind="compacting"))
        self.assertTrue(settle(self.clock, lambda: "🗜 compacting" in first_line(visible(self.tg)[0][0])))

        self.close(feed)
        self.assertEqual(first_line(visible(self.tg)[0][0]), "🔧 2 tools")

    def test_idle_time_appears_in_the_header_after_the_ping_threshold(self) -> None:
        """Port of test_feed.py E: the idle note is shown once the turn has been quiet for ping_seconds.

        Ported labels:
        E: no separate still-working message any more
        E: the log header shows the idle time (['⚙️ working · 1 tools · 0:01 · ⏳ idle 0 min\\n🔧 Read a.py'])
        """
        feed = self.feed(ping_seconds=5, header_seconds=1)
        feed.on_event(read(1))
        self.assertTrue(settle(self.clock, lambda: any("⏳ idle 0 min" in t for t in texts(self.tg))))
        self.close(feed)
        self.assertFalse(any("still working" in t for t in texts(self.tg)))

    def test_idle_time_shows_again_once_the_card_is_answered(self) -> None:
        """C: the idle time shows once the card is answered: a parked turn shows no idle time, and after the card is
        answered (the turn is free again) the quiet time appears."""
        state = {"parked": True}
        feed = self.feed(THREAD, ping_seconds=5, header_seconds=1, parked=lambda: state["parked"])
        feed.on_event(read(1))
        for _ in range(3):
            self.clock.advance(1)
            time.sleep(0.02)
        self.assertFalse(any("idle" in t for t in texts(self.tg, THREAD)))
        state["parked"] = False
        self.assertTrue(settle(self.clock, lambda: any("⏳ idle" in t for t in texts(self.tg, THREAD))))
        self.close(feed)

    def test_parked_turn_is_not_idle(self) -> None:
        """A permission card is open: the turn waits on a tap, so the header never says idle."""
        parked = self.feed(OTHER_THREAD, ping_seconds=5, header_seconds=1, parked=lambda: True)
        free = self.feed(THREAD, ping_seconds=5, header_seconds=1, parked=lambda: False)
        parked.on_event(read(1))
        free.on_event(read(1))
        self.assertTrue(settle(self.clock, lambda: any("⏳ idle" in t for t in texts(self.tg, THREAD))))
        for _ in range(3):
            self.clock.advance(1)
            time.sleep(0.02)
        self.assertFalse(any("idle" in t for t in texts(self.tg, OTHER_THREAD)))
        self.close(parked)
        self.close(free)

    def test_tools_counter_counts_subagents_and_stops_after_close(self) -> None:
        feed = self.feed()
        feed.on_event(read(1))
        feed.on_event(ToolStart(tool_id="s1", name="Glob", label="Glob *.md", parent_id="r1"))
        feed.on_event(ToolStart(tool_id="s2", name="Glob", label="Glob *.py", parent_id="r1"))
        self.assertEqual(feed.tools, 3)
        self.close(feed)
        feed.on_event(read(9))
        self.assertEqual(feed.tools, 3)

    def test_events_the_log_does_not_show_change_nothing(self) -> None:
        feed = self.feed()
        feed.on_event(UsageEvent(input_tokens=5, output_tokens=2))
        feed.on_event(ProcessExited(returncode=0))
        self.close(feed)
        self.assertEqual((feed.tools, self.tg.calls), (0, ()))


class LogPacing(FeedCase):
    def test_ten_quick_reads_share_one_log_message(self) -> None:
        """Port of test_feed.py D: ten reads in a burst make one message, edited, every line kept.

        Ported labels:
        D: all ten reads in one log message, in order
        D: one message sent for the ten reads
        D: the log is edited a few times, not per call (0 edits)
        """
        feed = self.feed(flush_seconds=0.3)
        for i in range(10):
            feed.on_event(read(i))
        self.close(feed)
        self.assertEqual(texts(self.tg), ["🔧 10 tools\n" + "\n".join(f"🔧 Read f{i}.py" for i in range(10))])
        self.assertEqual(len(calls_of(self.tg, "sendMessage")), 1)
        self.assertLessEqual(len(calls_of(self.tg, "editMessageText")), 4)

    def test_thirty_reads_make_a_few_edits_and_the_last_edit_shows_them_all(self) -> None:
        """Port of test_tool_log.py D: one edit per flush gap, not one per call.

        Ported labels:
        D: the log message is sent first
        D: 30 calls cost 1 edits, not 30
        D: the last edit shows all thirty lines
        """
        feed = self.feed(flush_seconds=0.4)
        feed.on_event(read(0))
        self.assertTrue(settle(self.clock, lambda: len(visible(self.tg)) == 1))
        for i in range(1, 30):
            feed.on_event(read(i))
        self.close(feed)
        edits = calls_of(self.tg, "editMessageText")
        self.assertTrue(1 <= len(edits) <= 4, f"{len(edits)} edits for 30 calls")
        self.assertEqual(len(visible(self.tg)), 1)
        self.assertEqual(visible(self.tg)[0][0].count("🔧 Read"), 30)

    def test_a_second_send_waits_for_the_flush_gap(self) -> None:
        """Nothing moves the fake clock, so the text that follows the first log waits until it is advanced."""
        feed = self.feed(flush_seconds=3.0)
        feed.on_event(read(1))
        self.assertTrue(settle(self.clock, lambda: len(calls_of(self.tg, "sendMessage")) == 1))
        feed.on_event(TextEvent(text="a"))
        feed.on_event(read(2))
        time.sleep(0.2)
        self.assertEqual(len(calls_of(self.tg, "sendMessage")), 1)
        self.close(feed)
        self.assertEqual(texts(self.tg), ["🔧 1 tools\n🔧 Read f1.py", "💬 a", "🔧 1 tools\n🔧 Read f2.py"])

    def test_full_log_freezes_and_no_line_is_lost(self) -> None:
        """Port of test_tool_log.py C: a full log says 'continued ↓' and has no button; the next one starts.

        Ported labels:
        C: a long turn spans 2 log messages
        C: all sixty tool lines kept, in order
        C: every log stays under the length cap
        C: every message is under Telegram 4096 characters
        C: every tool line is cut to 120 characters
        C: frozen logs say continued ↓ and have no button
        C: the last log gets its plain segment header
        """
        feed = self.feed()
        for i in range(60):
            feed.on_event(ToolStart(tool_id=f"l{i}", name="Bash", label=f"Bash step {i} " + "x" * 200))
        self.close(feed)
        logs = texts(self.tg)
        self.assertGreaterEqual(len(logs), 2)
        self.assertTrue(all(len(t) <= LOG_CHARS + 100 for t in logs))
        self.assertTrue(all(len(t) <= 4096 for t in logs))
        frozen = [(t, m) for t, m in visible(self.tg)[:-1]]
        self.assertTrue(all(first_line(t).endswith("continued ↓") and "st:" not in (m or "") for t, m in frozen))
        lines = [line for t in logs for line in t.split("\n")[1:]]
        self.assertEqual(len(lines), 60)
        self.assertTrue(all(lines[i].startswith(f"🔧 Bash step {i} ") and len(lines[i]) <= 120 for i in range(60)))
        self.assertRegex(first_line(logs[-1]), r"^🔧 \d+ tools$")

    def test_long_label_is_cut_to_one_line_of_120_characters(self) -> None:
        """Port of test_tool_log.py E.

        Ported labels:
        E: long line cut to 120 with an ellipsis
        """
        feed = self.feed()
        feed.on_event(ToolStart(tool_id="long", name="Bash", label="Bash " + "y" * 400))
        self.close(feed)
        line = texts(self.tg)[0].split("\n")[1]
        self.assertEqual(len(line), 120)
        self.assertTrue(line.endswith("…"))

    def test_stop_button_is_on_the_open_log_from_its_first_message(self) -> None:
        """Port of test_tool_log.py K: sent with the button and HTML; a stopped turn ends without it.

        Ported labels:
        K: the log message was sent
        K: the first log message is sent with the Stop button and HTML
        K: a stopped turn ends its log with the stop header and no button

        Ported labels:
        I: the log header says stopped, without a Stop button (['💬 Let me check.', '⏹ stopped after 3 tools — session kept\\n❌ Read x.txt\\n🔧 Bash List files\\n  ↳ 🔧 Glob **/*.py', '💬 ok:allow'])
        """
        feed = self.feed()
        feed.on_event(read(1))
        self.assertTrue(settle(self.clock, lambda: len(calls_of(self.tg, "sendMessage")) == 1))
        first = calls_of(self.tg, "sendMessage")[0]
        self.assertEqual(json.loads(first.params["reply_markup"]), STOP_MARKUP)
        self.assertEqual(first.params["parse_mode"], "HTML")
        self.close(feed)
        feed.finish("⏹ stopped after 1 tools — session kept")
        text, markup = visible(self.tg)[-1]
        self.assertTrue(text.startswith("⏹ stopped after 1 tools — session kept\n"))
        self.assertNotIn("st:", markup)


class TelegramFailures(FeedCase):
    def test_not_modified_is_no_error_and_nothing_is_resent(self) -> None:
        """Port of test_tool_log.py F.

        Ported labels:
        F: the log message is sent
        F: an edit was attempted
        F: no second message for an unchanged edit
        """
        feed = self.feed()
        feed.on_event(read(1))
        self.assertTrue(settle(self.clock, lambda: len(calls_of(self.tg, "sendMessage")) == 1))
        self.tg.fail_next("editMessageText", 400, NOT_MODIFIED)
        feed.on_event(read(2))
        self.close(feed)
        self.assertEqual(len(calls_of(self.tg, "sendMessage")), 1)
        self.assertEqual(len([c for c in calls_of(self.tg, "editMessageText") if c.result is None]), 1)

    def test_deleted_message_starts_a_new_message_with_the_same_lines(self) -> None:
        """Port of test_tool_log.py G.

        Ported labels:
        G: a deleted log is sent again as a new message
        G: the new message carries both lines
        """
        feed = self.feed()
        self.tg.fail_next("editMessageText", 400, GONE)
        feed.on_event(read(1))
        self.assertTrue(settle(self.clock, lambda: len(calls_of(self.tg, "sendMessage")) == 1))
        feed.on_event(read(2))
        self.close(feed)
        self.assertEqual(len(calls_of(self.tg, "sendMessage")), 2)
        last_text, last_markup = visible(self.tg)[-1]
        self.assertEqual(last_text.count("🔧 Read"), 2)
        self.assertNotIn("st:", last_markup or "")

    def test_html_refused_by_the_server_goes_out_as_plain_text(self) -> None:
        """Port of test_tool_log.py H.

        Ported labels:
        H: refused HTML is resent as plain text
        """
        feed = self.feed()
        self.tg.fail_next("sendMessage", 400, CANT_PARSE)
        feed.on_event(read(1))
        self.close(feed)
        sends = [c for c in calls_of(self.tg, "sendMessage") if c.result is not None]
        self.assertEqual(len(sends), 1)
        self.assertNotIn("parse_mode", sends[0].params)
        self.assertIn("🔧 Read f1.py", sends[0].params["text"])
        self.assertNotIn("<blockquote", sends[0].params["text"])

    def test_topic_gone_takes_two_calls_and_no_retry_and_finish_posts_nothing(self) -> None:
        """Port of test_tool_log.py N.

        Ported labels:
        N: close returns at once after a dead feed
        N: finish after a dead feed posts nothing
        N: the log took two calls and no retry
        """
        feed = self.feed()
        self.tg.fail_next("sendMessage", 400, THREAD_GONE)
        self.tg.fail_next("sendMessage", 400, THREAD_GONE)
        feed.on_event(read(1))
        self.close(feed)
        self.assertEqual(len(self.tg.calls), 2)
        self.assertTrue(feed.gave_up)
        feed.finish("✅ 1 tools")
        self.assertEqual(len(self.tg.calls), 2)

    def test_dead_feed_drops_queued_lines_while_the_turn_stays_open(self) -> None:
        """Port of test_tool_log.py P: no call follows the death of the feed; close does not wait for it.

        Ported labels:
        P: close does not wait for a dead feed
        P: no call after the feed died, while the turn stayed open
        P: the first log fails for good
        """
        feed = self.feed()
        self.tg.fail_next("sendMessage", 400, THREAD_GONE)
        self.tg.fail_next("sendMessage", 400, THREAD_GONE)
        feed.on_event(read(1))
        self.assertTrue(settle(self.clock, lambda: len(self.tg.calls) == 2))
        for i in range(2, 6):
            feed.on_event(read(i))
        for _ in range(5):
            self.clock.advance(1)
            time.sleep(0.02)
        self.assertEqual(len(self.tg.calls), 2)
        self.close(feed)
        self.assertEqual(len(self.tg.calls), 2)

    def test_failures_are_counted_and_the_third_gives_up(self) -> None:
        """Port of test_tool_log.py Q: two failed edits leave the feed running, the third ends it.

        Ported labels:
        Q: two failures are counted, the feed goes on
        Q: the third failure in a row gives up the feed of the turn
        """
        feed = self.feed()
        feed.on_event(read(1))
        self.assertTrue(settle(self.clock, lambda: len(calls_of(self.tg, "sendMessage")) == 1))
        for _ in range(3):
            self.tg.fail_next("editMessageText", 500, "net down")
        feed.on_event(read(2))
        self.assertTrue(settle(self.clock, lambda: len(calls_of(self.tg, "editMessageText")) == 2))
        self.assertFalse(feed.gave_up)
        self.assertTrue(settle(self.clock, lambda: feed.gave_up))
        self.assertEqual(len(calls_of(self.tg, "editMessageText")), 3)
        self.close(feed)

    def test_give_up_removes_the_stop_button_and_finish_edits_the_log_on_screen(self) -> None:
        """Port of test_tool_log.py U: the open log loses its button once; finish() gives it the summary.

        Ported labels:
        U: three failed edits in a row give up the feed
        U: exactly three edits were tried
        U: the open log loses its Stop button once
        U: the log is sent
        U: the last log has no Stop button
        U: the last log shows the final header
        U: the sender has stopped
        """
        feed = self.feed()
        for _ in range(3):
            self.tg.fail_next("editMessageText", 500, "proxy flap")
        feed.on_event(read(1))
        self.assertTrue(settle(self.clock, lambda: len(calls_of(self.tg, "sendMessage")) == 1))
        posted = calls_of(self.tg, "sendMessage")[0].result["message_id"]
        feed.on_event(read(2))
        self.close(feed)

        self.assertTrue(feed.gave_up)
        lifts = calls_of(self.tg, "editMessageReplyMarkup")
        self.assertEqual([(c.params["message_id"], c.params["reply_markup"]) for c in lifts],
                         [(posted, EMPTY_KEYBOARD)])
        self.assertEqual(len(calls_of(self.tg, "editMessageText")), 3)
        self.assertTrue(run_until_done(self.clock, lambda: feed.finish("✅ 2 tools · done")))
        text, markup = visible(self.tg)[-1]
        self.assertTrue(text.startswith("✅ 2 tools · done"))
        self.assertEqual(markup, EMPTY_KEYBOARD)
        self.assertFalse(feed._thread.is_alive(), "the sender is still running")

    def test_final_edit_falls_back_to_the_header_alone_after_two_refusals(self) -> None:
        """Port of test_tool_log.py V: the full body is refused twice (HTML, then plain), the header alone is accepted.

        Ported labels:
        V: the feed gave up
        V: the final edit falls back to the header alone and drops the keyboard
        V: the log is sent
        """
        feed = self.feed()
        for _ in range(3):
            self.tg.fail_next("editMessageText", 500, "proxy flap")
        self.tg.fail_next("editMessageText", 400, TOO_LONG)
        self.tg.fail_next("editMessageText", 400, TOO_LONG)
        feed.on_event(read(1))
        self.assertTrue(settle(self.clock, lambda: len(calls_of(self.tg, "sendMessage")) == 1))
        feed.on_event(read(2))
        self.close(feed)
        self.assertTrue(feed.gave_up)
        edits_before = len(calls_of(self.tg, "editMessageText"))
        feed.finish("❌ 2 tools · failed")
        final = calls_of(self.tg, "editMessageText")[edits_before:]
        self.assertEqual(len(final), 3)
        self.assertEqual(final[-1].params["text"], "❌ 2 tools · failed")
        self.assertEqual(final[-1].params["reply_markup"], EMPTY_KEYBOARD)

    def test_finish_after_the_feed_died_edits_the_first_posted_log(self) -> None:
        """Port of test_tool_log.py W: a newer log that was never sent has no message; the summary goes to the first.

        Ported labels:
        W: three failed edits give up the feed
        W: the first log is posted
        W: nothing new was posted after the feed died
        W: the final header edits the first posted log
        W: the sender has stopped
        """
        feed = self.feed()
        for _ in range(3):
            self.tg.fail_next("editMessageText", 500, "proxy flap")
        feed.on_event(read(1))
        self.assertTrue(settle(self.clock, lambda: len(calls_of(self.tg, "sendMessage")) == 1))
        posted = calls_of(self.tg, "sendMessage")[0].result["message_id"]
        feed.on_event(read(2))
        self.assertTrue(settle(self.clock, lambda: feed.gave_up))
        feed.on_event(TextEvent(text="after the feed died"))
        feed.on_event(ToolStart(tool_id="b2", name="Bash", label="Bash List"))
        feed.on_event(result("done"))
        self.close(feed)
        sends_before = len(calls_of(self.tg, "sendMessage"))
        edits_before = len(calls_of(self.tg, "editMessageText"))
        self.assertTrue(run_until_done(self.clock, lambda: feed.finish("✅ 3 tools · done")))
        self.assertEqual(len(calls_of(self.tg, "sendMessage")), sends_before)
        final = calls_of(self.tg, "editMessageText")[edits_before:]
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0].params["message_id"], posted)
        self.assertTrue(final[0].params["text"].startswith("✅ 3 tools · done"))
        self.assertFalse(feed._thread.is_alive(), "the sender is still running")


if __name__ == "__main__":
    unittest.main()
