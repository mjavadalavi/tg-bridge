"""PtyMirror (agents/pty_mirror.py): the file that `tg-bridge view` replays for one TUI session. Scenarios: start puts a
new file with the terminal reset first in place of the old one, write appends the raw text as UTF-8, a chunk that would
pass the cap restarts the file with the reset (a chunk that ends exactly at the cap is kept), close is repeatable and
ends the writes, a write into a file closed under it is dropped quietly, and an open or write failure is logged once and
the mirror stops without raising, and each new file is a new inode written under a temp name that os.replace moves into
place (0600 mode, no temp file left behind, a failed replace removes the temp file). Real files in a TempHome; a failing
file is a stand-in whose write raises ENOSPC."""

import os
import stat
import unittest
from unittest import mock

from tests.support import TempHome
from tg_bridge.agents import pty_mirror
from tg_bridge.agents.pty_mirror import MIRROR_MAX_BYTES, RESET, PtyMirror
from tg_bridge.logging_ import Logger


class FullDisk:
    """Stands in for the opened file. Every write fails the way a full disk does, and the writes are counted."""

    def __init__(self) -> None:
        self.writes = 0

    def write(self, data: bytes) -> int:
        self.writes += 1
        raise OSError(28, "No space left on device")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class PtyMirrorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.log_path = os.path.join(self.home.state, "bridge.log")
        self.logger = Logger(self.log_path)
        self.path = os.path.join(self.home.state, "7.pty")

    def new_mirror(self, path: str) -> PtyMirror:
        """A mirror whose file is closed at cleanup, so a test that ends early leaves no open file behind."""
        mirror = PtyMirror(path, self.logger)
        self.addCleanup(mirror.close)
        return mirror

    def read_bytes(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read()

    def mirror_lines(self) -> list[str]:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, encoding="utf-8") as f:
            return [line for line in f.read().splitlines() if "view mirror" in line]

    def test_start_replaces_the_file_and_writes_the_terminal_reset_first(self) -> None:
        with open(self.path, "wb") as f:
            f.write(b"output of the last process that must go")
        self.new_mirror(self.path).start()
        self.assertEqual(self.read_bytes(), b"\x1bc")

    def test_write_appends_the_raw_text_including_escape_codes_in_utf8(self) -> None:
        mirror = self.new_mirror(self.path)
        mirror.start()
        mirror.write("plain ")
        mirror.write("\x1b[1mبولد\x1b[0m\r\n")
        self.assertEqual(self.read_bytes(), b"\x1bcplain \x1b[1m" + "بولد".encode() + b"\x1b[0m\r\n")

    def test_a_chunk_that_would_pass_the_cap_restarts_the_file_with_the_reset(self) -> None:
        with mock.patch.object(pty_mirror, "MIRROR_MAX_BYTES", 10):
            mirror = self.new_mirror(self.path)
            mirror.start()
            mirror.write("abcdef")  # 2 + 6 = 8 bytes: under the cap, kept
            self.assertEqual(self.read_bytes(), b"\x1bcabcdef")
            mirror.write("ghijk")  # 8 + 5 = 13 bytes: over the cap, so the file starts again
            self.assertEqual(self.read_bytes(), b"\x1bcghijk")

    def test_a_chunk_that_ends_exactly_at_the_cap_is_kept(self) -> None:
        with mock.patch.object(pty_mirror, "MIRROR_MAX_BYTES", 10):
            mirror = self.new_mirror(self.path)
            mirror.start()
            mirror.write("abcdef")  # 2 + 6 = 8 bytes
            mirror.write("gh")  # 8 + 2 = 10 bytes: exactly the cap, so kept (a restart would leave only b"\x1bcgh")
            self.assertEqual(self.read_bytes(), b"\x1bcabcdefgh")

    def test_the_cap_is_four_mebibytes(self) -> None:
        self.assertEqual(MIRROR_MAX_BYTES, 4 * 1024 * 1024)

    def test_close_can_repeat_and_later_writes_do_nothing(self) -> None:
        mirror = self.new_mirror(self.path)
        mirror.start()
        mirror.write("a")
        mirror.close()
        mirror.close()
        mirror.write("b")
        self.assertEqual(self.read_bytes(), b"\x1bca")
        self.assertEqual(self.mirror_lines(), [])

    def test_a_missing_folder_is_logged_once_and_nothing_raises(self) -> None:
        path = os.path.join(self.home.state, "missing", "7.pty")
        mirror = self.new_mirror(path)
        mirror.start()
        mirror.write("text")
        mirror.write("more text")
        self.assertFalse(os.path.exists(path))
        self.assertEqual(len(self.mirror_lines()), 1)

    def test_a_write_failure_is_logged_once_and_the_mirror_stops(self) -> None:
        disk = FullDisk()
        with mock.patch.object(pty_mirror, "open", create=True, return_value=disk):
            mirror = self.new_mirror(self.path)
            mirror.start()  # the reset is the first write, and it fails
            mirror.write("text")
            mirror.write("more text")
        self.assertEqual(disk.writes, 1, "a stopped mirror must not write again")
        lines = self.mirror_lines()
        self.assertEqual(len(lines), 1)
        self.assertIn("No space left", lines[0])

    def test_each_start_is_a_new_inode_so_a_restarted_file_is_never_the_same_file(self) -> None:
        self.new_mirror(self.path).start()
        first = os.stat(self.path).st_ino
        second_process = self.new_mirror(self.path)
        second_process.start()
        self.assertNotEqual(os.stat(self.path).st_ino, first)
        self.assertEqual(self.read_bytes(), b"\x1bc")

    def test_a_restarted_file_is_longer_than_the_old_one_and_still_gets_its_writes(self) -> None:
        self.new_mirror(self.path).start()
        mirror = self.new_mirror(self.path)
        mirror.start()
        mirror.write("a much longer second run than the first one")
        self.assertEqual(self.read_bytes(), b"\x1bca much longer second run than the first one")

    def test_the_file_is_private_and_no_temp_file_is_left_beside_it(self) -> None:
        mirror = self.new_mirror(self.path)
        mirror.start()
        mirror.write("text")
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        self.assertEqual(sorted(os.listdir(self.home.state)), ["7.pty"])

    def test_a_failed_move_into_place_removes_the_temp_file_and_logs_once(self) -> None:
        mirror = self.new_mirror(self.path)
        with mock.patch.object(pty_mirror.os, "replace", side_effect=OSError(13, "Permission denied")):
            mirror.start()
            mirror.write("text")
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(os.listdir(self.home.state), ["bridge.log"])
        self.assertEqual(len(self.mirror_lines()), 1)

    def test_a_failed_open_of_the_temp_file_closes_its_descriptor_and_logs_once(self) -> None:
        mirror = self.new_mirror(self.path)
        with mock.patch.object(pty_mirror, "open", create=True, side_effect=OSError(28, "No space left on device")), \
                mock.patch.object(pty_mirror.os, "close", wraps=os.close) as close:
            mirror.start()
        self.assertEqual(close.call_count, 1, "the descriptor mkstemp made must not leak")
        self.assertEqual(os.listdir(self.home.state), ["bridge.log"])
        self.assertEqual(len(self.mirror_lines()), 1)

    def test_a_write_into_a_file_closed_under_it_is_dropped_without_a_log_line(self) -> None:
        mirror = self.new_mirror(self.path)
        mirror.start()
        mirror._file.close()  # what close() on another thread does to a write that is in progress
        mirror.write("late text")
        self.assertEqual(self.read_bytes(), b"\x1bc")
        self.assertEqual(self.mirror_lines(), [])



class OnRotateTests(unittest.TestCase):
    """on_rotate: called once per restart past the cap, after the new file is in place, never for the first file, and
    never when the restart fails; a callback that raises is logged and the chunk still lands."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.log_path = os.path.join(self.home.state, "bridge.log")
        self.logger = Logger(self.log_path)
        self.path = os.path.join(self.home.state, "7.pty")
        self.calls: list[int] = []

    def record(self) -> None:
        """The callback: records the size of the file at the time of the call, which shows the new file is in place."""
        self.calls.append(os.path.getsize(self.path))

    def test_on_rotate_is_called_once_per_restart_past_the_cap_and_not_for_the_first_file(self) -> None:
        with mock.patch.object(pty_mirror, "MIRROR_MAX_BYTES", 10):
            mirror = PtyMirror(self.path, self.logger, on_rotate=self.record)
            self.addCleanup(mirror.close)
            mirror.start()
            self.assertEqual(self.calls, [])
            mirror.write("abcdef")  # 2 + 6 = 8 bytes: under the cap
            self.assertEqual(self.calls, [])
            mirror.write("ghijk")   # 8 + 5 = 13 bytes: the first restart
            self.assertEqual(self.calls, [len(RESET)])  # the new file holds only the reset when it is called
            mirror.write("lmnop")   # 2 + 5 = 7, then 7 + 5 = 12: the second restart
            self.assertEqual(len(self.calls), 2)

    def test_a_write_that_does_not_restart_the_file_does_not_call_on_rotate(self) -> None:
        mirror = PtyMirror(self.path, self.logger, on_rotate=self.record)
        self.addCleanup(mirror.close)
        mirror.start()
        for _ in range(5):
            mirror.write("small chunk")
        self.assertEqual(self.calls, [])

    def test_no_callback_when_the_restart_fails(self) -> None:
        with mock.patch.object(pty_mirror, "MIRROR_MAX_BYTES", 10):
            mirror = PtyMirror(self.path, self.logger, on_rotate=self.record)
            self.addCleanup(mirror.close)
            mirror.start()
            mirror.write("abcdef")
            with mock.patch.object(pty_mirror.tempfile, "mkstemp", side_effect=OSError(28, "No space")):
                mirror.write("ghijk")
            self.assertEqual(self.calls, [])

    def test_a_failing_callback_is_logged_and_the_chunk_still_lands(self) -> None:
        def boom() -> None:
            raise RuntimeError("redraw broke")

        with mock.patch.object(pty_mirror, "MIRROR_MAX_BYTES", 10):
            mirror = PtyMirror(self.path, self.logger, on_rotate=boom)
            self.addCleanup(mirror.close)
            mirror.start()
            mirror.write("abcdef")
            mirror.write("ghijk")
            mirror.close()
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), RESET.encode("ascii") + b"ghijk")
        with open(self.log_path, encoding="utf-8") as f:
            self.assertIn("on_rotate", f.read())


if __name__ == "__main__":
    unittest.main()
