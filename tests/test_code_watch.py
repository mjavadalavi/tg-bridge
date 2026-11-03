"""The code watch (daemon/code_watch.py): a view pane learns that the tg_bridge code on disk changed since it started,
and execs itself again on it. The fingerprint is read through an injected function, so no real file is touched except in
the fingerprint test, which uses a temporary folder."""

import os
import sys
import tempfile
import unittest

from tests.fakes.clock import FakeClock
from tg_bridge.daemon.code_watch import CHECK_SECONDS, CodeWatch, fingerprint, reexec


class CodeWatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock(start=0.0)
        self.value = 100.0

    def watch(self) -> CodeWatch:
        return CodeWatch(self.clock, read=lambda: self.value)

    def test_unchanged_code_is_not_changed(self) -> None:
        watch = self.watch()
        self.clock.advance(CHECK_SECONDS)
        self.assertFalse(watch.changed())

    def test_newer_code_is_changed_once_the_check_interval_passed(self) -> None:
        watch = self.watch()
        self.value = 200.0
        self.assertFalse(watch.changed())  # inside the interval: the files are not read
        self.clock.advance(CHECK_SECONDS)
        self.assertTrue(watch.changed())

    def test_the_files_are_read_once_per_interval(self) -> None:
        reads = []
        watch = CodeWatch(self.clock, read=lambda: reads.append(1) or 1.0)
        for _ in range(10):
            self.clock.advance(CHECK_SECONDS / 10)
            watch.changed()
        self.assertEqual(len(reads), 2)  # the start fingerprint and one check


class FingerprintTests(unittest.TestCase):
    def test_newest_py_mtime_and_other_files_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            os.mkdir(os.path.join(root, "sub"))
            for name, mtime in (("a.py", 10.0), (os.path.join("sub", "b.py"), 20.0), ("c.txt", 99.0)):
                path = os.path.join(root, name)
                open(path, "w").close()
                os.utime(path, (mtime, mtime))
            self.assertEqual(fingerprint(root), 20.0)

    def test_an_empty_folder_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(fingerprint(root), 0.0)


class ReexecTests(unittest.TestCase):
    def test_execs_the_same_command_line(self) -> None:
        calls = []
        reexec(lambda path, argv: calls.append((path, list(argv))))
        self.assertEqual(calls, [(sys.executable, [sys.executable, *sys.orig_argv[1:]])])


if __name__ == "__main__":
    unittest.main()
