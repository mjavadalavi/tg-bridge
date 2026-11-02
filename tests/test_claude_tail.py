"""TranscriptTail (agents/claude/tail.py): following a growing Claude transcript.

Scenarios: lines written in several chunks, one of them split across two writes, arrive once and in order; a file that
appears after start() is picked up; a start_offset skips the lines already there; sidechain, blank, invalid and
non-object lines are skipped; an on_line exception is logged and does not stop later lines; a multi-byte character split
across writes and an invalid byte decode correctly; a line nested too deep for json does not lose the lines after it;
stop() drains the complete lines written before it, keeps the partial one buffered, is idempotent and works from inside
on_line; a second start() adds no thread; a truncated file restarts from 0 with one log line.
The poll interval is a few milliseconds, so every test finishes in well under a second.
"""

import json
import os
import tempfile
import threading
import time
import unittest

from tg_bridge.agents.claude.tail import TranscriptTail
from tg_bridge.logging_ import Logger

WAIT = 5.0
FAST_POLL = 0.005


def line(n: int, **extra: object) -> str:
    """One transcript line: a JSON object with a sequence number n, plus any extra fields."""
    return json.dumps({"type": "assistant", "n": n, **extra}) + "\n"


def wait_for(predicate, timeout: float = WAIT) -> bool:
    """Poll predicate until it is true or timeout passes. Returns the last value of predicate."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class TranscriptTailTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="tgb-v3-tail-")
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "session.jsonl")
        self.log_path = os.path.join(tmp.name, "bridge.log")
        self.logger = Logger(self.log_path)
        self.seen: list[dict] = []

    def append(self, text: str) -> None:
        """Append text to the transcript and flush it, the way the TUI writes it."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text)

    def write_bytes(self, data: bytes) -> None:
        """Append raw bytes to the transcript, for writes that split a character or carry invalid utf-8."""
        with open(self.path, "ab") as f:
            f.write(data)

    def numbers(self) -> list[int]:
        return [obj["n"] for obj in self.seen]

    @staticmethod
    def tail_threads() -> int:
        return sum(1 for thread in threading.enumerate() if thread.name == "transcript-tail")

    def read_log(self) -> str:
        if not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, encoding="utf-8") as f:
            return f.read()

    def make_tail(self, **kwargs: object) -> TranscriptTail:
        tail = TranscriptTail(self.path, self.seen.append, logger=self.logger, poll_seconds=FAST_POLL, **kwargs)
        self.addCleanup(tail.stop)
        return tail

    def test_chunks_including_a_split_line_arrive_once_and_in_order(self) -> None:
        self.append(line(1))
        tail = self.make_tail()
        tail.start()
        second = line(2)
        self.append(second[:10])  # a line split across two writes
        time.sleep(0.02)
        self.append(second[10:] + line(3))
        third = line(4)
        for piece in (third[:5], third[5:12], third[12:]):
            self.append(piece)
            time.sleep(0.01)
        self.assertTrue(wait_for(lambda: self.numbers() == [1, 2, 3, 4]))
        time.sleep(0.05)
        self.assertEqual(self.numbers(), [1, 2, 3, 4])
        total = len((line(1) + second + line(3) + third).encode("utf-8"))
        self.assertEqual(tail.offset, total)

    def test_file_created_after_start_is_followed(self) -> None:
        tail = self.make_tail()
        tail.start()
        time.sleep(0.03)
        self.assertEqual(self.seen, [])
        self.append(line(1))
        self.assertTrue(wait_for(lambda: self.numbers() == [1]))
        self.assertEqual(tail.offset, len(line(1).encode("utf-8")))

    def test_start_offset_skips_lines_already_in_the_file(self) -> None:
        self.append(line(1) + line(2))
        offset = TranscriptTail.size_of(self.path)
        self.append(line(3))
        tail = self.make_tail(start_offset=offset)
        tail.start()
        self.assertTrue(wait_for(lambda: self.numbers() == [3]))
        time.sleep(0.03)
        self.assertEqual(self.numbers(), [3])

    def test_size_of_is_zero_for_a_missing_file(self) -> None:
        self.assertEqual(TranscriptTail.size_of(self.path), 0)

    def test_blank_sidechain_invalid_and_non_object_lines_are_skipped(self) -> None:
        self.append(
            "\n"
            "   \n"
            "not json at all\n"
            "[1, 2]\n"
            "\"just a string\"\n"
            "42\n"
            + line(9, isSidechain=True)
            + line(1)
            + line(2, isSidechain=False)
        )
        tail = self.make_tail()
        tail.start()
        self.assertTrue(wait_for(lambda: self.numbers() == [1, 2]))
        time.sleep(0.03)
        self.assertEqual(self.numbers(), [1, 2])

    def test_handler_exception_is_logged_and_later_lines_still_arrive(self) -> None:
        def handler(obj: dict) -> None:
            self.seen.append(obj)
            if obj["n"] == 1:
                raise ValueError("boom")

        self.append(line(1) + line(2) + line(3))
        tail = TranscriptTail(self.path, handler, logger=self.logger, poll_seconds=FAST_POLL)
        self.addCleanup(tail.stop)
        tail.start()
        self.assertTrue(wait_for(lambda: self.numbers() == [1, 2, 3]))
        self.assertTrue(wait_for(lambda: "transcript line handler failed: boom" in self.read_log()))

    def test_stop_drains_complete_lines_and_is_idempotent(self) -> None:
        self.append(line(1))
        tail = TranscriptTail(self.path, self.seen.append, logger=self.logger, poll_seconds=30.0)
        tail.start()
        time.sleep(0.02)  # the thread is now waiting on its 30 second interval
        self.append(line(2) + '{"type": "assistant", "n": 3')  # the last line is still partial
        tail.stop()
        self.assertEqual(self.numbers(), [1, 2])
        complete = len((line(1) + line(2)).encode("utf-8"))
        self.assertEqual(tail.offset, complete)
        tail.stop()
        self.assertEqual(self.numbers(), [1, 2])
        self.assertEqual(tail.offset, complete)

    def test_stop_before_start_is_safe(self) -> None:
        self.append(line(1))
        tail = self.make_tail()
        tail.stop()
        self.assertEqual(self.numbers(), [1])
        tail.stop()
        self.assertEqual(self.numbers(), [1])

    def test_stop_from_inside_on_line_keeps_file_order_and_delivers_nothing_twice(self) -> None:
        self.append(line(1) + line(2) + line(3))
        tail: TranscriptTail

        def handler(obj: dict) -> None:
            self.seen.append(obj)
            if obj["n"] == 1:
                tail.stop()  # lines 2 and 3 are already in the file and must still arrive, in order

        tail = TranscriptTail(self.path, handler, logger=self.logger, poll_seconds=FAST_POLL)
        self.addCleanup(tail.stop)
        tail.start()
        self.assertTrue(wait_for(lambda: self.numbers() == [1, 2, 3]))
        time.sleep(0.05)
        self.assertEqual(self.numbers(), [1, 2, 3])
        tail.stop()
        self.assertEqual(self.numbers(), [1, 2, 3])

    def test_multibyte_character_split_across_writes_and_invalid_byte_are_decoded(self) -> None:
        raw = (json.dumps({"type": "assistant", "n": 1, "text": "سلام"}, ensure_ascii=False) + "\n").encode("utf-8")
        cut = raw.index("سلام".encode("utf-8")) + 1  # one byte into the first two-byte character
        tail = self.make_tail()
        tail.start()
        self.write_bytes(raw[:cut])
        time.sleep(0.02)
        self.write_bytes(raw[cut:] + b'{"type": "assistant", "n": 2, "text": "\xff"}\n')
        self.assertTrue(wait_for(lambda: self.numbers() == [1, 2]))
        self.assertEqual(self.seen[0]["text"], "سلام")
        self.assertEqual(self.seen[1]["text"], "�")

    def test_line_nested_too_deep_for_json_does_not_lose_the_lines_after_it(self) -> None:
        deep = "[" * 10000 + "]" * 10000
        self.append(line(1) + deep + "\n" + line(2))
        tail = self.make_tail()
        tail.start()
        self.assertTrue(wait_for(lambda: self.numbers() == [1, 2]))
        time.sleep(0.03)
        self.assertEqual(self.numbers(), [1, 2])

    def test_second_start_adds_no_thread_and_no_duplicate_lines(self) -> None:
        self.append(line(1))
        tail = self.make_tail()
        before = self.tail_threads()
        tail.start()
        tail.start()
        self.assertEqual(self.tail_threads(), before + 1)
        self.assertTrue(wait_for(lambda: self.numbers() == [1]))
        time.sleep(0.03)
        self.assertEqual(self.numbers(), [1])

    def test_truncated_file_restarts_from_zero_with_one_log_line(self) -> None:
        self.append(line(1, text="a fairly long line so the offset is clearly above the new size"))
        tail = self.make_tail()
        tail.start()
        self.assertTrue(wait_for(lambda: self.numbers() == [1]))
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(line(2))
        self.assertTrue(wait_for(lambda: self.numbers() == [1, 2]))
        time.sleep(0.03)
        self.assertEqual(self.numbers(), [1, 2])
        self.assertEqual(self.read_log().count("restarting from 0"), 1)


if __name__ == "__main__":
    unittest.main()
