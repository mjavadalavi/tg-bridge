"""run_pane on a real terminal (tests/test_viewer.py covers the files): a pseudo-terminal from os.openpty stands in for the
tmux pane. While the pane runs its echo and line editing are off and unread input is flushed each round; after it ends,
or fails, the attributes are exactly what they were. Scenarios: the flags during a run, the restore after a crash, typed
keys that really reach the terminal and are dropped by the next round."""

import fcntl
import os
import struct
import termios
import time
import unittest

from tests.test_viewer import CrashClock, PaneCase, StepClock, THREAD
from tg_bridge.daemon.viewer import run_pane


class TerminalTests(PaneCase):
    """run_pane on a real pseudo-terminal (os.openpty): the terminal is quiet while the pane runs and as it was after."""

    def setUp(self) -> None:
        super().setUp()
        self.master, self.slave = os.openpty()
        self.addCleanup(os.close, self.master)
        self.addCleanup(os.close, self.slave)
        self.tty = os.fdopen(self.slave, "r", closefd=False)
        self.write_meta("print")

    def local_flags(self) -> int:
        return termios.tcgetattr(self.slave)[3]

    def pending_input(self) -> int:
        """The bytes waiting in the terminal's input queue (FIONREAD)."""
        return struct.unpack("i", fcntl.ioctl(self.slave, termios.FIONREAD, b"\0\0\0\0"))[0]

    def test_echo_and_line_editing_are_off_while_it_runs_and_restored_after(self) -> None:
        before = termios.tcgetattr(self.slave)
        self.assertTrue(before[3] & termios.ECHO and before[3] & termios.ICANON, "the test needs a terminal in cooked mode")
        seen: list[int] = []
        clock = StepClock([lambda: seen.append(self.local_flags() & (termios.ECHO | termios.ICANON))])
        run_pane(self.paths, THREAD, self.out, clock, stop_after=2, stdin=self.tty)
        self.assertEqual(seen, [0])
        self.assertEqual(termios.tcgetattr(self.slave), before)

    def test_the_attributes_come_back_when_the_pane_fails(self) -> None:
        before = termios.tcgetattr(self.slave)
        with self.assertRaises(RuntimeError):
            run_pane(self.paths, THREAD, self.out, CrashClock(), stop_after=2, stdin=self.tty)
        self.assertEqual(termios.tcgetattr(self.slave), before)

    def test_typed_input_is_flushed_every_round_and_never_piles_up(self) -> None:
        arrived: list[int] = []
        pending: list[int] = []

        def type_keys() -> None:
            os.write(self.master, b"typed\n")
            deadline = time.monotonic() + 2
            while self.pending_input() == 0 and time.monotonic() < deadline:  # the line discipline may read it later
                time.sleep(0.01)
            arrived.append(self.pending_input())

        def check_after_the_flush() -> None:
            pending.append(self.pending_input())

        run_pane(self.paths, THREAD, self.out, StepClock([type_keys, check_after_the_flush]), stop_after=2,
                 stdin=self.tty)
        self.assertEqual(arrived, [len(b"typed\n")])  # the keys really reached the terminal
        self.assertEqual(pending, [0])  # and the next round dropped them


if __name__ == "__main__":
    unittest.main()
