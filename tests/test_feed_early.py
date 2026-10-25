"""TurnFeed early status (step 33): the tool log opens at turn start with the waiting note and the Stop button.

The turn runner calls TurnFeed.start() before the backend starts. From then on: one sendMessage with the waiting text
before any event; the first tool line edits that same message; a retry or stream cut changes the header and keeps it
there until the next text or tool event; a turn without tools ends with that message edited to its final header; and a
text that comes before the first tool takes the place of the waiting message. The fixtures (RecordingTelegram, FakeClock,
settle, run_until_done) are the ones of test_feed.py.
"""

import json
import time
import unittest

from tests.test_feed import FeedCase, STOP_MARKUP, THREAD, calls_of, first_line, read, result, run_until_done, settle
from tests.test_feed import texts, visible
from tg_bridge.agents.events import StatusEvent, TextEvent
from tg_bridge.feed.format import WAITING_TEXT
from tg_bridge.telegram.keyboards import EMPTY_KEYBOARD


def header_now(tg: object) -> str:
    """The first line of the topic's newest message, as the chat shows it."""
    shown = visible(tg)  # type: ignore[arg-type]
    return first_line(shown[-1][0]) if shown else ""


def started(case: FeedCase, **options: object):
    """A feed whose turn has started, with the waiting message already sent."""
    feed = case.feed(**options)
    feed.start()
    assert settle(case.clock, lambda: len(calls_of(case.tg, "sendMessage")) == 1), "waiting message not sent"
    return feed


class WaitingMessage(FeedCase):
    def test_waiting_message_is_sent_at_turn_start_before_any_event(self) -> None:
        feed = self.feed()
        feed.start()
        self.assertTrue(settle(self.clock, lambda: len(calls_of(self.tg, "sendMessage")) == 1))
        self.assertEqual(self.tg.calls[0].method, "sendMessage")
        sent = calls_of(self.tg, "sendMessage")[0]
        self.assertEqual(first_line(sent.params["text"]), WAITING_TEXT)
        self.assertEqual(sent.params["parse_mode"], "HTML")
        self.assertEqual(json.loads(sent.params["reply_markup"]), STOP_MARKUP)
        self.assertEqual([first_line(t) for t in texts(self.tg)], [WAITING_TEXT])
        self.close(feed)

    def test_first_tool_edits_the_waiting_message_and_sends_no_second_one(self) -> None:
        feed = started(self)
        posted = calls_of(self.tg, "sendMessage")[0].result["message_id"]
        feed.on_event(read(1))
        self.assertTrue(settle(self.clock, lambda: header_now(self.tg).startswith("⚙️ working · 1 tools")))
        self.assertEqual(len(calls_of(self.tg, "sendMessage")), 1)
        edits = calls_of(self.tg, "editMessageText")
        self.assertTrue(edits and all(e.params["message_id"] == posted for e in edits))
        self.assertEqual(json.loads(visible(self.tg)[-1][1]), STOP_MARKUP)
        self.close(feed)
        self.assertEqual(texts(self.tg), ["🔧 1 tools\n🔧 Read f1.py"])
        self.assertEqual(visible(self.tg)[0][1], EMPTY_KEYBOARD)

    def test_without_start_the_first_tool_still_opens_the_log(self) -> None:
        feed = self.feed()
        feed.on_event(read(1))
        self.assertTrue(settle(self.clock, lambda: len(calls_of(self.tg, "sendMessage")) == 1))
        self.close(feed)
        self.assertEqual(len(calls_of(self.tg, "sendMessage")), 1)


class StatusInHeader(FeedCase):
    def test_retry_and_stream_cut_show_in_the_header_and_the_count_stays_visible(self) -> None:
        feed = started(self)
        feed.on_event(StatusEvent(kind="retry", attempt=2, max_attempts=10))
        self.assertTrue(settle(self.clock, lambda: header_now(self.tg) == f"{WAITING_TEXT} · 🔁 API retry 2/10"))
        for _ in range(5):  # quiet time passes: the note stays until a text or tool event says otherwise
            self.clock.advance(1)
            time.sleep(0.01)
        self.assertEqual(header_now(self.tg), f"{WAITING_TEXT} · 🔁 API retry 2/10")
        feed.on_event(StatusEvent(kind="stream_cut", attempt=1))
        self.assertTrue(settle(self.clock, lambda: header_now(self.tg) == f"{WAITING_TEXT} · 📡 stream cut, retrying 1"))
        self.assertEqual(len(calls_of(self.tg, "sendMessage")), 1)
        self.close(feed)

    def test_retry_note_clears_on_the_next_text_and_on_the_next_tool(self) -> None:
        feed = started(self)
        feed.on_event(StatusEvent(kind="retry", attempt=3, max_attempts=10))
        self.assertTrue(settle(self.clock, lambda: "API retry 3/10" in header_now(self.tg)))
        feed.on_event(TextEvent(text="still thinking"))
        self.assertTrue(settle(self.clock, lambda: header_now(self.tg) == WAITING_TEXT))
        feed.on_event(StatusEvent(kind="retry", attempt=4, max_attempts=10))
        self.assertTrue(settle(self.clock, lambda: "API retry 4/10" in header_now(self.tg)))
        feed.on_event(read(1))
        self.assertTrue(settle(self.clock, lambda: header_now(self.tg).startswith("⚙️ working · 1 tools")))
        self.assertNotIn("API retry", header_now(self.tg))
        self.close(feed)

    def test_retry_before_any_tool_opens_the_log_when_the_turn_had_none_open(self) -> None:
        feed = self.feed()
        feed.on_event(StatusEvent(kind="compacting"))
        self.assertTrue(settle(self.clock, lambda: header_now(self.tg) == f"{WAITING_TEXT} · 🗜 compacting"))
        self.close(feed)


class TurnEnds(FeedCase):
    def test_turn_without_tools_ends_with_the_waiting_message_edited_to_the_final_header(self) -> None:
        feed = started(self)
        posted = calls_of(self.tg, "sendMessage")[0].result["message_id"]
        feed.on_event(result("the answer"))
        self.assertTrue(run_until_done(self.clock, feed.close))
        self.assertTrue(run_until_done(self.clock, lambda: feed.finish("✅ 0 tools · 2s")))
        self.assertEqual(len(calls_of(self.tg, "sendMessage")), 1)
        final = calls_of(self.tg, "editMessageText")[-1]
        self.assertEqual(final.params["message_id"], posted)
        self.assertEqual(final.params["reply_markup"], EMPTY_KEYBOARD)
        self.assertEqual(first_line(visible(self.tg)[0][0]), "✅ 0 tools · 2s")

    def test_text_before_the_first_tool_takes_the_place_of_the_waiting_message(self) -> None:
        feed = started(self)
        posted = calls_of(self.tg, "sendMessage")[0].result["message_id"]
        feed.on_event(TextEvent(text="Let me look."))
        feed.on_event(read(1))
        self.close(feed)
        sends = calls_of(self.tg, "sendMessage")
        self.assertEqual(len(sends), 2)
        self.assertEqual(sends[1].params["text"].split("\n")[0], "🔧 1 tools")
        first_edit = calls_of(self.tg, "editMessageText")[0]
        self.assertEqual(first_edit.params["message_id"], posted)
        self.assertEqual(first_edit.params["reply_markup"], EMPTY_KEYBOARD)
        self.assertEqual([text for text, _ in visible(self.tg)],
                         ["💬 Let me look.", "🔧 1 tools\n🔧 Read f1.py"])

    def test_intermediate_text_before_the_answer_still_ends_with_one_final_header(self) -> None:
        feed = started(self)
        feed.on_event(TextEvent(text="Let me think."))
        feed.on_event(TextEvent(text="the answer"))
        feed.on_event(result("the answer"))
        self.assertTrue(run_until_done(self.clock, feed.close))
        self.assertTrue(run_until_done(self.clock, lambda: feed.finish("✅ 0 tools · 2s")))
        self.assertEqual(len(calls_of(self.tg, "sendMessage")), 2)
        self.assertEqual([first_line(t) for t in texts(self.tg)], ["💬 Let me think.", "✅ 0 tools · 2s"])
        self.assertNotIn("reply_markup", calls_of(self.tg, "sendMessage")[-1].params)

    def test_text_event_equal_to_the_answer_is_not_echoed_over_the_waiting_message(self) -> None:
        feed = started(self)
        feed.on_event(TextEvent(text="done"))
        feed.on_event(result("done"))
        self.assertTrue(run_until_done(self.clock, feed.close))
        self.assertTrue(run_until_done(self.clock, lambda: feed.finish("✅ 0 tools · 1s")))
        self.assertEqual(len(calls_of(self.tg, "sendMessage")), 1)
        self.assertEqual(first_line(visible(self.tg)[0][0]), "✅ 0 tools · 1s")


if __name__ == "__main__":
    unittest.main()
