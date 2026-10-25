"""messenger: port of the send, rename and buttons scenarios of fakecli/test_buttons.py and the chunking rules of
tg-bridge.py send_text (labels kept in docstrings).

Every test runs on FakeTelegram and FakeClock inside a TempHome, so nothing touches the network, the real clock or the
real state directory. Log lines are read back from a Logger file in the temp home.
"""

import json
import os
import threading
import unittest

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.logging_ import Logger
from tg_bridge.telegram.client import TelegramBadRequest, TelegramError
from tg_bridge.telegram.keyboards import EMPTY_KEYBOARD, CallbackData, CallbackDataError
from tg_bridge.telegram.messenger import (
    EDIT_TEXT_LIMIT,
    RENAME_INTERVAL_SECONDS,
    SEND_CHUNK,
    SEND_MAX_CHUNKS,
    TOPIC_NAME_LIMIT,
    TRUNCATED_NOTE,
    Messenger,
    chunk_text,
)

CHAT = -1001
THREAD = 9301


class Harness:
    """A Messenger wired to a fake Telegram, a fake clock and a log file inside one temp home."""

    def __init__(self, home: TempHome, start: float = 1000.0) -> None:
        self.clock = FakeClock(start)
        self.tg = FakeTelegram(clock=self.clock)
        self.log_path = os.path.join(home.state, "bridge.log")
        self.logger = Logger(self.log_path)
        self.messenger = Messenger(self.tg, self.logger, self.clock)

    def log_text(self) -> str:
        if not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, encoding="utf-8") as f:
            return f.read()

    def calls(self, method: str) -> list[dict]:
        return [c.params for c in self.tg.calls if c.method == method]

    def renames(self) -> list[str]:
        return [str(p["name"]) for p in self.calls("editForumTopic")]


class ChunkTextTests(unittest.TestCase):
    """The pure part of send_text: stripping, the '(empty)' placeholder, 3800-character pieces, 4 + a note."""

    def test_constants_match_the_old_helper(self) -> None:
        self.assertEqual(SEND_CHUNK, 3800)
        self.assertEqual(SEND_MAX_CHUNKS, 4)
        self.assertEqual(TRUNCATED_NOTE, "…(truncated)")

    def test_blank_text_becomes_the_placeholder(self) -> None:
        self.assertEqual(chunk_text("   \n "), ["(empty)"])

    def test_text_is_stripped_before_splitting(self) -> None:
        self.assertEqual(chunk_text("  hi  \n"), ["hi"])

    def test_exactly_one_chunk_at_the_limit(self) -> None:
        self.assertEqual(chunk_text("a" * SEND_CHUNK), ["a" * SEND_CHUNK])

    def test_one_character_over_the_limit_makes_two_chunks(self) -> None:
        self.assertEqual(chunk_text("a" * (SEND_CHUNK + 1)), ["a" * SEND_CHUNK, "a"])

    def test_four_chunks_need_no_note(self) -> None:
        text = "b" * (SEND_CHUNK * 4)
        self.assertEqual(chunk_text(text), ["b" * SEND_CHUNK] * 4)

    def test_fifth_chunk_is_replaced_by_the_truncated_note(self) -> None:
        text = "c" * (SEND_CHUNK * 5)
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 5)
        self.assertEqual(chunks[:4], ["c" * SEND_CHUNK] * 4)
        self.assertEqual(chunks[4], TRUNCATED_NOTE)


class SendTextTests(unittest.TestCase):
    """send_text sends each chunk to the topic; a failure is logged and stops the remaining chunks."""

    def test_short_text_is_one_send_to_the_topic(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.messenger.send_text(CHAT, THREAD, "hello")
            self.assertEqual(h.calls("sendMessage"), [
                {"chat_id": CHAT, "message_thread_id": THREAD, "text": "hello"},
            ])

    def test_blank_text_sends_the_placeholder(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.messenger.send_text(CHAT, THREAD, "  ")
            self.assertEqual(h.tg.texts(THREAD), ["(empty)"])

    def test_long_text_is_sent_in_order_without_loss(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            text = "".join(str(i % 10) for i in range(SEND_CHUNK * 3 + 5))
            h.messenger.send_text(CHAT, THREAD, text)
            self.assertEqual("".join(h.tg.texts(THREAD)), text)
            self.assertEqual(len(h.tg.texts(THREAD)), 4)

    def test_five_chunks_end_with_the_truncated_note(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.messenger.send_text(CHAT, THREAD, "d" * (SEND_CHUNK * 5))
            sent = h.tg.texts(THREAD)
            self.assertEqual(len(sent), 5)
            self.assertEqual(sent[-1], TRUNCATED_NOTE)

    def test_failed_chunk_is_logged_and_the_rest_are_not_sent(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.tg.fail_next("sendMessage", code=500, description="boom")
            h.messenger.send_text(CHAT, THREAD, "e" * (SEND_CHUNK * 3))
            self.assertEqual(len(h.calls("sendMessage")), 1)
            self.assertIn("sendMessage failed", h.log_text())

    def test_bad_request_is_logged_not_raised(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.tg.fail_next("sendMessage", code=400, description="can't parse")
            h.messenger.send_text(CHAT, THREAD, "f")
            self.assertIn("sendMessage failed", h.log_text())
            self.assertIn("can't parse", h.log_text())

    def test_log_line_is_bounded(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.tg.fail_next("sendMessage", code=500, description="x" * 500)
            h.messenger.send_text(CHAT, THREAD, "g")
            line = [ln for ln in h.log_text().splitlines() if "sendMessage failed" in ln][0]
            self.assertLess(len(line), 200)


class SendButtonsTests(unittest.TestCase):
    """send_buttons: the picker's message carries the keyboard; a failed send is logged and returns None."""

    def test_message_carries_the_keyboard_and_is_returned(self) -> None:
        """Old test_buttons A: keyboard rows are kept in the sent message."""
        with TempHome() as home:
            h = Harness(home)
            rows = [[("✓ opus", CallbackData("cm", ("9301", "mo", "opus"))), ("sonnet", "cm:9301:mo:sonnet")]]
            sent = h.messenger.send_buttons(CHAT, THREAD, "🧠 model: opus", rows)
            self.assertIsNotNone(sent)
            self.assertIn("message_id", sent or {})
            keyboards = h.tg.buttons(THREAD)
            self.assertEqual(
                [[b["text"] for b in row] for row in keyboards[0]],
                [["✓ opus", "sonnet"]],
            )
            callbacks = [b["callback_data"] for row in json.loads(h.calls("sendMessage")[0]["reply_markup"])["inline_keyboard"] for b in row]
            self.assertEqual(callbacks, ["cm:9301:mo:opus", "cm:9301:mo:sonnet"])

    def test_reply_markup_is_the_keyboard_json(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.messenger.send_buttons(CHAT, THREAD, "x", [[("Stop", "st:9301")]])
            self.assertEqual(
                h.calls("sendMessage")[0]["reply_markup"],
                '{"inline_keyboard": [[{"text": "Stop", "callback_data": "st:9301"}]]}',
            )

    def test_failed_send_is_logged_and_returns_none(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.tg.fail_next("sendMessage", code=500, description="down")
            self.assertIsNone(h.messenger.send_buttons(CHAT, THREAD, "x", [[("a", "b")]]))
            self.assertIn("sendButtons failed", h.log_text())

    def test_bad_request_on_buttons_is_logged_and_returns_none(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.tg.fail_next("sendMessage", code=400, description="bad markup")
            self.assertIsNone(h.messenger.send_buttons(CHAT, THREAD, "x", [[("a", "b")]]))
            self.assertIn("sendButtons failed", h.log_text())

    def test_oversized_callback_raises_and_sends_nothing(self) -> None:
        """The 64-byte guard is a programming error, so it is raised, not logged as a failed send."""
        with TempHome() as home:
            h = Harness(home)
            with self.assertRaises(CallbackDataError):
                h.messenger.send_buttons(CHAT, THREAD, "x", [[("big", "x" * 65)]])
            self.assertEqual(h.calls("sendMessage"), [])
            self.assertNotIn("sendButtons failed", h.log_text())


class EditTextTests(unittest.TestCase):
    """edit_text edits in place and lets the error reach the caller."""

    def test_edit_records_the_new_text(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            sent = h.messenger.send_buttons(CHAT, THREAD, "first", [[("a", "b")]])
            self.assertIsNotNone(sent)
            message_id = int(sent["message_id"]) if sent else 0
            h.messenger.edit_text(CHAT, message_id, "second")
            self.assertEqual(h.tg.edits(message_id), ["second"])

    def test_no_keyboard_key_when_none_is_given(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.messenger.edit_text(CHAT, 5, "text")
            params = h.calls("editMessageText")[0]
            self.assertNotIn("reply_markup", params)
            self.assertNotIn("parse_mode", params)

    def test_empty_keyboard_removes_the_buttons(self) -> None:
        """Old test_buttons M: the stop button is removed with the empty keyboard."""
        with TempHome() as home:
            h = Harness(home)
            h.messenger.edit_text(CHAT, 5, "done", reply_markup=EMPTY_KEYBOARD)
            self.assertEqual(h.calls("editMessageText")[0]["reply_markup"], '{"inline_keyboard": []}')

    def test_parse_mode_is_passed_through(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.messenger.edit_text(CHAT, 5, "<b>x</b>", parse_mode="HTML")
            self.assertEqual(h.calls("editMessageText")[0]["parse_mode"], "HTML")

    def test_feed_flag_is_recorded_on_the_call(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.messenger.edit_text(CHAT, 5, "feed", feed=True)
            h.messenger.edit_text(CHAT, 5, "card")
            flags = [c.feed for c in h.tg.calls if c.method == "editMessageText"]
            self.assertEqual(flags, [True, False])

    def test_long_text_is_cut_to_the_edit_limit(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.messenger.edit_text(CHAT, 5, "z" * (EDIT_TEXT_LIMIT + 50))
            self.assertEqual(len(h.calls("editMessageText")[0]["text"]), EDIT_TEXT_LIMIT)

    def test_refused_edit_raises_to_the_caller(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.tg.fail_next("editMessageText", code=500, description="gone")
            with self.assertRaises(TelegramError):
                h.messenger.edit_text(CHAT, 5, "x")

    def test_refused_edit_with_bad_request_raises_too(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.tg.fail_next("editMessageText", code=400, description="message is not modified")
            with self.assertRaises(TelegramBadRequest):
                h.messenger.edit_text(CHAT, 5, "x")


class DeleteTests(unittest.TestCase):
    """delete removes one message and lets a refusal reach the caller."""

    def test_delete_names_chat_and_message(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.messenger.delete(CHAT, 77)
            self.assertEqual(h.calls("deleteMessage"), [{"chat_id": CHAT, "message_id": 77}])

    def test_refused_delete_raises(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.tg.fail_next("deleteMessage", code=500, description="too old")
            with self.assertRaises(TelegramError):
                h.messenger.delete(CHAT, 77)


class RenameTopicTests(unittest.TestCase):
    """rename_topic: at most one rename per 15 s, never the same name twice unless forced, failures logged."""

    def test_interval_constant_is_fifteen_seconds(self) -> None:
        self.assertEqual(RENAME_INTERVAL_SECONDS, 15.0)

    def test_first_rename_goes_even_when_the_clock_reads_zero(self) -> None:
        """A monotonic clock may read 0 right after boot; the first rename must not be throttled by that."""
        with TempHome() as home:
            h = Harness(home, start=0.0)
            h.messenger.rename_topic(CHAT, THREAD, "first")
            self.assertEqual(h.renames(), ["first"])

    def test_same_name_is_never_sent_twice_unless_forced(self) -> None:
        """Old rename_topic: an unchanged name is skipped."""
        with TempHome() as home:
            h = Harness(home)
            h.messenger.rename_topic(CHAT, THREAD, "same")
            h.clock.advance(RENAME_INTERVAL_SECONDS * 10)
            h.messenger.rename_topic(CHAT, THREAD, "same")
            self.assertEqual(h.renames(), ["same"])

    def test_different_name_within_fifteen_seconds_is_throttled(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.messenger.rename_topic(CHAT, THREAD, "one")
            h.clock.advance(RENAME_INTERVAL_SECONDS - 1)
            h.messenger.rename_topic(CHAT, THREAD, "two")
            self.assertEqual(h.renames(), ["one"])

    def test_different_name_after_the_interval_goes(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.messenger.rename_topic(CHAT, THREAD, "one")
            h.clock.advance(RENAME_INTERVAL_SECONDS)
            h.messenger.rename_topic(CHAT, THREAD, "two")
            self.assertEqual(h.renames(), ["one", "two"])

    def test_force_bypasses_the_interval_and_the_same_name_check(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.messenger.rename_topic(CHAT, THREAD, "one")
            h.messenger.rename_topic(CHAT, THREAD, "one", force=True)
            h.messenger.rename_topic(CHAT, THREAD, "two", force=True)
            self.assertEqual(h.renames(), ["one", "one", "two"])

    def test_forced_rename_restarts_the_interval(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.messenger.rename_topic(CHAT, THREAD, "one", force=True)
            h.clock.advance(5)
            h.messenger.rename_topic(CHAT, THREAD, "two")
            self.assertEqual(h.renames(), ["one"])

    def test_name_is_cut_to_the_topic_limit(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.messenger.rename_topic(CHAT, THREAD, "n" * (TOPIC_NAME_LIMIT + 20))
            self.assertEqual(len(h.calls("editForumTopic")[0]["name"]), TOPIC_NAME_LIMIT)

    def test_rename_targets_the_topic_thread(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.messenger.rename_topic(CHAT, THREAD, "x")
            self.assertEqual(
                h.calls("editForumTopic")[0],
                {"chat_id": CHAT, "message_thread_id": THREAD, "name": "x"},
            )

    def test_failed_rename_is_logged_not_raised_and_still_throttles(self) -> None:
        with TempHome() as home:
            h = Harness(home)
            h.tg.fail_next("editForumTopic", code=500, description="nope")
            h.messenger.rename_topic(CHAT, THREAD, "one")
            self.assertIn("rename_topic", h.log_text())
            h.messenger.rename_topic(CHAT, THREAD, "two")
            self.assertEqual(h.renames(), ["one"])

    def test_each_messenger_keeps_its_own_throttle(self) -> None:
        """No module state: a second Messenger on the same client is not throttled by the first."""
        with TempHome() as home:
            h = Harness(home)
            other = Messenger(h.tg, h.logger, h.clock)
            h.messenger.rename_topic(CHAT, THREAD, "one")
            other.rename_topic(CHAT, THREAD, "two")
            self.assertEqual(h.renames(), ["one", "two"])

    def test_concurrent_renames_send_exactly_one(self) -> None:
        """rename_lock makes the check and the state update one step: eight racing callers, one rename."""
        with TempHome() as home:
            h = Harness(home)
            barrier = threading.Barrier(8)

            def race(index: int) -> None:
                barrier.wait()
                h.messenger.rename_topic(CHAT, THREAD, f"name-{index}")

            threads = [threading.Thread(target=race, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            self.assertEqual(len(h.renames()), 1)


if __name__ == "__main__":
    unittest.main()
