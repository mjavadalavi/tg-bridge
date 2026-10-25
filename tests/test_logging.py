"""Logger and Clock: truncation rule of C03, redaction, thread safety, SystemClock and FakeClock.

Port of the log() behaviour of the live tg-bridge.py (C03): keep the last 300 lines once the file is over 200 KB.
Redaction is new in v3: bot tokens, api_key= values and user:password proxy credentials are masked.
"""

import os
import re
import threading
import time
import unittest

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.clock import Clock, SystemClock
from tg_bridge.logging_ import MAX_BYTES, REDACTED, Logger, redact

STAMP_LINE = re.compile(r"^\d{2}-\d{2} \d{2}:\d{2}:\d{2}Z .*$")
FAKE_TOKEN = "123456789:AAEhBP0av28bsoe9tqnjj8gqwBPa5Qy4K1x"


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


class LoggerWriteTests(unittest.TestCase):
    def test_line_has_utc_stamp_and_message(self) -> None:
        with TempHome() as home:
            path = os.path.join(home.state, "bridge.log")
            Logger(path).log("hello there")
            lines = _read_lines(path)
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], STAMP_LINE)
        self.assertTrue(lines[0].endswith(" hello there"))

    def test_non_string_message_is_str_ed(self) -> None:
        with TempHome() as home:
            path = os.path.join(home.state, "bridge.log")
            Logger(path).log(ValueError("boom"))
            self.assertTrue(_read_lines(path)[0].endswith(" boom"))

    def test_missing_directory_does_not_raise(self) -> None:
        with TempHome() as home:
            path = os.path.join(home.home, "no", "such", "dir", "bridge.log")
            Logger(path).log("still fine")  # OSError is swallowed, as in the live file
            self.assertFalse(os.path.exists(path))

    def test_each_logger_writes_only_to_its_own_path(self) -> None:
        with TempHome() as home:
            first = os.path.join(home.state, "a.log")
            second = os.path.join(home.state, "b.log")
            Logger(first).log("one")
            Logger(second).log("two")
            self.assertTrue(_read_lines(first)[0].endswith(" one"))
            self.assertTrue(_read_lines(second)[0].endswith(" two"))


class TruncationTests(unittest.TestCase):
    def _prefill(self, path: str, lines: int, width: int) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for i in range(lines):
                f.write(f"{i:06d}".ljust(width - 1, "x") + "\n")

    def test_file_at_the_limit_is_not_truncated(self) -> None:
        with TempHome() as home:
            path = os.path.join(home.state, "bridge.log")
            width = 1000
            self._prefill(path, MAX_BYTES // width, width)  # exactly 200 000 bytes
            self.assertEqual(os.path.getsize(path), MAX_BYTES)
            Logger(path).log("next")
            self.assertEqual(len(_read_lines(path)), MAX_BYTES // width + 1)

    def test_file_over_the_limit_keeps_last_300_lines_then_appends(self) -> None:
        with TempHome() as home:
            path = os.path.join(home.state, "bridge.log")
            self._prefill(path, 500, 500)  # 250 000 bytes, 500 lines
            self.assertGreater(os.path.getsize(path), MAX_BYTES)
            Logger(path).log("after truncation")
            lines = _read_lines(path)
        self.assertEqual(len(lines), 301)
        self.assertTrue(lines[0].startswith(f"{200:06d}"))
        self.assertTrue(lines[-2].startswith(f"{499:06d}"))
        self.assertTrue(lines[-1].endswith(" after truncation"))

    def test_small_file_is_appended_to_without_truncation(self) -> None:
        with TempHome() as home:
            path = os.path.join(home.state, "bridge.log")
            logger = Logger(path)
            for i in range(50):
                logger.log(f"line {i}")
            self.assertEqual(len(_read_lines(path)), 50)


class RedactionTests(unittest.TestCase):
    def test_bot_token_in_url_is_masked(self) -> None:
        text = f"getMe failed: https://api.telegram.org/bot{FAKE_TOKEN}/getMe"
        out = redact(text)
        self.assertNotIn(FAKE_TOKEN, out)
        self.assertNotIn("AAEhBP0av28", out)
        self.assertIn(REDACTED, out)

    def test_api_key_value_is_masked_and_name_kept(self) -> None:
        out = redact("soniox failed api_key=sk-live-abc123XYZ&retry=1")
        self.assertNotIn("sk-live-abc123XYZ", out)
        self.assertIn(f"api_key={REDACTED}&retry=1", out)

    def test_proxy_user_and_password_are_masked(self) -> None:
        out = redact("proxy: socks5://bridge:s3cret@127.0.0.1:10820 unreachable")
        self.assertNotIn("bridge:s3cret", out)
        self.assertNotIn("s3cret", out)
        self.assertIn(f"socks5://{REDACTED}@127.0.0.1:10820", out)

    def test_http_proxy_credentials_are_masked(self) -> None:
        out = redact("http://user:pass@127.0.0.1:8080")
        self.assertNotIn("user:pass", out)

    def test_plain_text_is_unchanged(self) -> None:
        text = "turn 12:30 done, 3 tools, cwd /home/x/project (ok)"
        self.assertEqual(redact(text), text)

    def test_logger_never_writes_the_token(self) -> None:
        with TempHome() as home:
            path = os.path.join(home.state, "bridge.log")
            Logger(path).log(f"egress failed for {FAKE_TOKEN} via socks5://u:p@h:1")
            with open(path, encoding="utf-8") as f:
                content = f.read()
        self.assertNotIn(FAKE_TOKEN, content)
        self.assertNotIn("u:p@", content)
        self.assertIn(REDACTED, content)


class ThreadSafetyTests(unittest.TestCase):
    def test_four_threads_by_200_lines_no_exception_no_torn_lines(self) -> None:
        with TempHome() as home:
            path = os.path.join(home.state, "bridge.log")
            logger = Logger(path)
            errors: list[BaseException] = []
            padding = "p" * 2000  # big enough that truncation happens while threads are writing

            def worker(n: int) -> None:
                try:
                    for i in range(200):
                        logger.log(f"thread {n} line {i} {padding}")
                except BaseException as exc:  # noqa: BLE001 - the test reports any failure
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)
            self.assertFalse(any(t.is_alive() for t in threads))
            self.assertEqual(errors, [])
            lines = _read_lines(path)
        self.assertLessEqual(len(lines), 800)
        self.assertTrue(lines)
        for line in lines:
            self.assertRegex(line, STAMP_LINE)

    def test_four_threads_small_lines_keep_every_line(self) -> None:
        with TempHome() as home:
            path = os.path.join(home.state, "bridge.log")
            logger = Logger(path)

            def worker(n: int) -> None:
                for i in range(200):
                    logger.log(f"t{n} i{i}")

            threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)
            lines = _read_lines(path)
        self.assertEqual(len(lines), 800)
        self.assertEqual(len({line.split(" ", 2)[2] for line in lines}), 800)


class SystemClockTests(unittest.TestCase):
    def test_satisfies_the_clock_protocol(self) -> None:
        self.assertIsInstance(SystemClock(), Clock)

    def test_now_and_monotonic_follow_the_real_clocks(self) -> None:
        clock = SystemClock()
        before = time.time()
        mono_before = clock.monotonic()
        self.assertGreaterEqual(clock.now(), before - 1)
        self.assertGreaterEqual(clock.monotonic(), mono_before)

    def test_sleep_really_waits_a_little(self) -> None:
        start = time.monotonic()
        SystemClock().sleep(0.01)
        self.assertGreaterEqual(time.monotonic() - start, 0.009)


class FakeClockTests(unittest.TestCase):
    def test_satisfies_the_clock_protocol(self) -> None:
        self.assertIsInstance(FakeClock(), Clock)

    def test_advance_moves_now_and_monotonic_together(self) -> None:
        clock = FakeClock(start=100.0)
        self.assertEqual(clock.now(), 100.0)
        clock.advance(5)
        self.assertEqual(clock.now(), 105.0)
        self.assertEqual(clock.monotonic(), 105.0)

    def test_sleep_does_not_block_but_advances_and_is_recorded(self) -> None:
        clock = FakeClock(start=0.0)
        started = time.monotonic()
        clock.sleep(30)
        clock.sleep(0.5)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(clock.monotonic(), 30.5)
        self.assertEqual(clock.sleeps, [30.0, 0.5])

    def test_negative_durations_are_rejected(self) -> None:
        clock = FakeClock()
        with self.assertRaises(ValueError):
            clock.advance(-1)
        with self.assertRaises(ValueError):
            clock.sleep(-1)


if __name__ == "__main__":
    unittest.main()
