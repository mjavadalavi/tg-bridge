"""answers: port of the send checks of fakecli/test_answer.py (labels C, D, E, F, G, K kept in docstrings) plus the
document-failure and HTML-failure rules of tg-bridge.py send_answer / _send_answer_doc (C08).

Checks A, B and J are pure and live in test_render. H (no retry on 400) is the client's rule and lives in
test_telegram_client. I (the dispatcher wiring) is ported with the dispatcher in a later step.

Telegram is a FakeTelegram, uploads go to a FakeUploader, time is a FakeClock, and the log is a real Logger file
inside a TempHome. Nothing touches the network or the real state directory.
"""

import datetime
import os
import re
import unittest

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.fakes.uploads import FakeUploader
from tests.support import TempHome
from tg_bridge.logging_ import Logger
from tg_bridge.telegram.answers import DOC_THRESHOLD, AnswerSender
from tg_bridge.telegram.messenger import Messenger, chunk_text
from tg_bridge.telegram.render import ANSWER_CHUNK

CHAT = -100
THREAD = 9701
START = 1_700_000_000.0


class Harness:
    """An AnswerSender wired to fakes inside one temp home."""

    def __init__(self, home: TempHome) -> None:
        self.clock = FakeClock(START)
        self.tg = FakeTelegram(clock=self.clock)
        self.docs = FakeUploader()
        self.log_path = os.path.join(home.state, "bridge.log")
        self.logger = Logger(self.log_path)
        self.messenger = Messenger(self.tg, self.logger, self.clock)
        self.sender = AnswerSender(self.tg, self.docs, self.messenger, self.logger, self.clock)

    def html_sends(self) -> list[dict]:
        return [c.params for c in self.tg.calls if c.method == "sendMessage" and c.params.get("parse_mode") == "HTML"]

    def plain_sends(self) -> list[dict]:
        return [c.params for c in self.tg.calls if c.method == "sendMessage" and "parse_mode" not in c.params]

    def log_text(self) -> str:
        if not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, encoding="utf-8") as f:
            return f.read()


class AnswerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = Harness(self.home)


class ShortAnswerTests(AnswerTestCase):
    def test_c_short_answer_is_one_html_message(self) -> None:
        """C: one HTML message

        Ported labels:
            C: text is the converted answer
        """
        self.h.sender.send_answer(CHAT, THREAD, "First **para**.\n\nSecond para with `code`.")
        sends = self.h.html_sends()
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0]["text"], "First <b>para</b>.\n\nSecond para with <code>code</code>.")
        self.assertEqual(self.h.plain_sends(), [])
        self.assertEqual(self.h.docs.uploads, ())

    def test_blank_answer_becomes_the_placeholder(self) -> None:
        """C: a blank answer is sent as (empty)"""
        self.h.sender.send_answer(CHAT, THREAD, "  \n ")
        self.assertEqual([s["text"] for s in self.h.html_sends()], ["(empty)"])


class SplitAnswerTests(AnswerTestCase):
    def test_d_long_answer_splits_without_breaking_code(self) -> None:
        """D: split into 2-3 messages, each within 4000 characters, no code block split, no document

        Ported labels:
            D: every message within 4000 characters
            D: no code block split across messages
            D: no document for an answer that fits three messages
        """
        code = "```py\n" + "\n".join(f"value_{i} = {i}  # note" for i in range(260)) + "\n```"
        long_ans = "intro paragraph\n\n" + code + "\n\n" + ("filler text " * 120) + "\n\noutro"
        self.h.sender.send_answer(CHAT, THREAD, long_ans)
        sends = [s["text"] for s in self.h.html_sends()]
        self.assertTrue(1 < len(sends) <= 3, f"split into 2-3 messages ({len(sends)})")
        self.assertTrue(all(len(t) <= ANSWER_CHUNK for t in sends), "every message within 4000 characters")
        self.assertTrue(all(t.count("<pre>") == t.count("</pre>") for t in sends), "no code block split")
        self.assertEqual(self.h.docs.uploads, (), "no document for an answer that fits three messages")

    def test_k_long_four_backtick_block_is_split_cleanly(self) -> None:
        """K: long 4-backtick block split over messages, no stray backtick, last line kept

        Ported labels:
            K: no stray backtick in any split piece
            K: the last line of the block is kept
        """
        code4 = "````py\n" + "\n".join(f"value_{i} = {i}" for i in range(500)) + "\n````"
        self.h.sender.send_answer(CHAT, THREAD, code4)
        sends = [s["text"] for s in self.h.html_sends()]
        self.assertTrue(1 < len(sends) <= 3 and not self.h.docs.uploads, f"split over messages ({len(sends)})")
        self.assertTrue(all("`" not in t and len(t) <= ANSWER_CHUNK for t in sends), "no stray backtick")
        self.assertIn("value_499 = 499", "".join(sends))


class DocumentAnswerTests(AnswerTestCase):
    def test_e_above_threshold_goes_out_as_one_document(self) -> None:
        """E: above threshold -> one document, no messages; name, type, full body, caption ~900 characters

        Ported labels:
            E: caption is the first ~900 characters
            E: document holds the full answer
        """
        big = "word " * 1900  # 9500 characters
        self.h.sender.send_answer(CHAT, THREAD, big, doc_threshold=9000)
        self.assertEqual(self.h.html_sends(), [])
        self.assertEqual(self.h.plain_sends(), [])
        self.assertEqual(len(self.h.docs.uploads), 1)
        upload = self.h.docs.uploads[0]
        self.assertTrue(upload.filename.startswith("answer-") and upload.filename.endswith(".md"))
        self.assertEqual(upload.mime, "text/markdown")
        self.assertEqual(upload.data, big.strip().encode("utf-8"))
        self.assertTrue(len(upload.caption) <= 901 and upload.caption.startswith("word word"))

    def test_document_name_is_stamped_from_the_clock(self) -> None:
        """E: the document name is answer-<local time of the clock>.md

        Ported labels:
            E: file name and type (answer-20261024-195038.md)
        """
        self.h.sender.send_answer(CHAT, THREAD, "x" * (DOC_THRESHOLD + 1))
        expected = datetime.datetime.fromtimestamp(START).strftime("answer-%Y%m%d-%H%M%S.md")
        self.assertEqual(self.h.docs.uploads[0].filename, expected)
        self.assertRegex(expected, re.compile(r"^answer-\d{8}-\d{6}\.md$"))

    def test_fifteen_thousand_characters_go_out_as_a_document(self) -> None:
        """E: a 15000-character answer is one .md document holding all of it"""
        text = "abc def " * 1874 + "abc defx"  # 15000 characters, nothing for strip() to remove
        self.h.sender.send_answer(CHAT, THREAD, text)
        self.assertEqual(self.h.html_sends(), [])
        self.assertEqual(self.h.docs.uploads[0].data, text.strip().encode("utf-8"))
        self.assertEqual(len(self.h.docs.uploads[0].data), 15000)

    def test_f_more_than_three_messages_becomes_a_document_not_a_truncation(self) -> None:
        """F: does not fit three messages -> document, nothing truncated

        Ported labels:
            F: document holds the whole answer
        """
        four = "\n\n".join(f"para {i} " + "w" * 2100 for i in range(4))  # each paragraph needs its own message
        self.h.sender.send_answer(CHAT, THREAD, four)
        self.assertEqual(self.h.html_sends(), [])
        self.assertEqual(len(self.h.docs.uploads), 1, "does not fit three messages -> document")
        self.assertEqual(self.h.docs.uploads[0].data, four.encode("utf-8"), "document holds the whole answer")

    def test_failed_document_falls_back_to_plain_text(self) -> None:
        """document failure: the answer is sent as plain text through send_text, and the failure is logged"""
        body = "y" * 9500
        self.h.docs.fail_next("file too big for this chat")
        self.h.sender.send_answer(CHAT, THREAD, body)
        self.assertEqual(self.h.tg.texts(THREAD), chunk_text(body))
        self.assertIn("answer document failed: sendDocument failed (code 500): file too big for this chat", self.h.log_text())


class HtmlFailureTests(AnswerTestCase):
    def test_g_http_400_on_html_resends_the_chunk_as_plain_text(self) -> None:
        """G: HTML refused, plain resend of the same chunk

        Ported labels:
            G: plain text is the raw chunk
        """
        self.h.tg.fail_next("sendMessage", 400, description="can't parse entities")
        self.h.sender.send_answer(CHAT, THREAD, "BOOM **bad** html")
        sends = [c.params for c in self.h.tg.calls if c.method == "sendMessage"]
        self.assertEqual(len(sends), 2)
        self.assertEqual(sends[0].get("parse_mode"), "HTML")
        self.assertNotIn("parse_mode", sends[1])
        self.assertEqual(sends[1]["text"], "BOOM **bad** html")
        self.assertIn("answer HTML refused (400), sending plain text", self.h.log_text())

    def test_other_html_failure_logs_and_drops_the_rest(self) -> None:
        """a non-400 failure on the first chunk is logged and no later chunk is sent"""
        self.h.tg.fail_next("sendMessage", 500, description="server exploded")
        code = "\n\n".join(f"part {i} " + "z" * 2100 for i in range(3))  # three messages
        self.h.sender.send_answer(CHAT, THREAD, code)
        sends = [c for c in self.h.tg.calls if c.method == "sendMessage"]
        self.assertEqual(len(sends), 1)
        self.assertIn("answer send failed:", self.h.log_text())

    def test_failed_plain_resend_stops_the_answer(self) -> None:
        """a 400 followed by a failing plain resend logs once and stops"""
        self.h.tg.fail_next("sendMessage", 400)
        self.h.tg.fail_next("sendMessage", 500, description="down")
        code = "\n\n".join(f"part {i} " + "z" * 2100 for i in range(3))
        self.h.sender.send_answer(CHAT, THREAD, code)
        self.assertEqual(len([c for c in self.h.tg.calls if c.method == "sendMessage"]), 2)
        self.assertIn("answer send failed: sendMessage failed (code 500): down", self.h.log_text())


class DocThresholdTests(AnswerTestCase):
    def test_threshold_is_the_old_default(self) -> None:
        """the default document threshold is the live file's 9000 characters"""
        self.assertEqual(DOC_THRESHOLD, 9000)

    def test_answer_exactly_at_the_threshold_is_not_a_document(self) -> None:
        """an answer of exactly doc_threshold characters is still sent as messages"""
        text = "\n\n".join(["a" * 2998, "b" * 2999, "c" * 2999])  # 9000 characters in three paragraphs
        self.assertEqual(len(text), DOC_THRESHOLD)
        self.h.sender.send_answer(CHAT, THREAD, text)
        self.assertEqual(self.h.docs.uploads, ())
        self.assertEqual(len(self.h.html_sends()), 3)


if __name__ == "__main__":
    unittest.main()
