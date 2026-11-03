"""Process trees (core/process_tree.py): descendants() lists the pids below a pid, deepest first, and signal_tree()
sends one signal to the whole tree.

Covers a real shell tree (a shell with a background sleep), a three-level tree, a parent that catches TERM and waits for
its child (a launcher's shell), a dead pid, the /proc parser on a fake /proc (comm with spaces and parentheses, entries
that are not pids), a PermissionError on the pid itself, the caller's ancestry (a loop and an unreadable parent stop the
walk), and the protection that keeps the caller and its ancestors from being signalled. Every spawned process is started in its own session and its
group is killed in finally. No claude process is started.
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from tg_bridge.core import process_tree

WAIT_SECONDS = 5.0


def dead_pid() -> int:
    """The pid of a child that has exited and been reaped: no process has it now."""
    proc = subprocess.Popen([sys.executable, "-I", "-c", "pass"])
    proc.wait()
    return proc.pid


def is_running(pid: int) -> bool:
    """True while pid exists and is not a zombie (a zombie has ended but is not yet reaped)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            state = handle.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state not in ("Z", "X")


def ppid_of(pid: int) -> int:
    """The parent pid of pid, read from /proc like the module does."""
    with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
        return int(handle.read().rsplit(")", 1)[1].split()[1])


def wait_gone(pid: int) -> bool:
    """Polls until pid has ended, up to WAIT_SECONDS. True when it ended."""
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        if not is_running(pid):
            return True
        time.sleep(0.05)
    return not is_running(pid)


class TreeCase(unittest.TestCase):
    """Starts a shell tree in its own session and kills the whole session in cleanup."""

    def start(self, script: str) -> subprocess.Popen[bytes]:
        proc = subprocess.Popen(["sh", "-c", script], stdout=subprocess.PIPE, start_new_session=True)
        self.addCleanup(self.kill_session, proc)
        return proc

    @staticmethod
    def kill_session(proc: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()


class Descendants(TreeCase):
    def test_a_background_child_is_listed(self) -> None:
        shell = self.start("sleep 30 & echo $!; wait")
        sleeper = int(shell.stdout.readline())
        self.assertEqual(process_tree.descendants(shell.pid), [sleeper])

    def test_the_deepest_generation_comes_first(self) -> None:
        shell = self.start('sh -c "sleep 30 & echo \\$!; wait" & echo $!; wait')
        pids = {int(shell.stdout.readline()), int(shell.stdout.readline())}
        child = next(pid for pid in pids if ppid_of(pid) == shell.pid)
        grandchild = next(pid for pid in pids if pid != child)
        self.assertEqual(ppid_of(grandchild), child)
        self.assertEqual(process_tree.descendants(shell.pid), [grandchild, child])

    def test_a_pid_with_no_children_has_no_descendants(self) -> None:
        self.assertEqual(process_tree.descendants(os.getpid()), [])

    def test_a_dead_pid_has_no_descendants(self) -> None:
        self.assertEqual(process_tree.descendants(dead_pid()), [])

    def test_the_ancestry_holds_the_caller_and_its_parent(self) -> None:
        ancestry = process_tree._ancestry()
        self.assertIn(os.getpid(), ancestry)
        self.assertIn(ppid_of(os.getpid()), ancestry)

    def test_a_pid_that_is_not_positive_has_no_descendants(self) -> None:
        self.assertEqual(process_tree.descendants(0), [])
        self.assertEqual(process_tree.descendants(-1), [])


class FakeProc(unittest.TestCase):
    """The /proc parser on a fake /proc directory: comm may hold spaces and parentheses, and entries that are not pids
    or cannot be read are skipped."""

    def setUp(self) -> None:
        self.proc = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(process_tree, "PROC", self.proc))

    def write(self, name: str, text: str) -> None:
        os.makedirs(os.path.join(self.proc, name), exist_ok=True)
        with open(os.path.join(self.proc, name, "stat"), "w", encoding="utf-8") as handle:
            handle.write(text)

    def test_comm_with_spaces_and_parentheses_is_parsed_after_the_last_paren(self) -> None:
        self.write("100", "100 (my (odd) name) S 1 100 100 0 -1\n")
        self.write("101", "101 (sh) S 100 101 101 0 -1\n")
        self.write("102", "102 (sh) S 101 102 102 0 -1\n")
        self.write("103", "103 (other) S 1 103 103 0 -1\n")
        self.assertEqual(process_tree.descendants(100), [102, 101])

    def test_entries_that_are_not_pids_or_are_garbled_are_skipped(self) -> None:
        self.write("100", "100 (a) S 1 100\n")
        self.write("101", "101 (b) S 100 101\n")
        self.write("102", "garbled with no paren\n")
        self.write("self", "self (x) S 100 1\n")
        self.assertEqual(process_tree.descendants(100), [101])

    def test_ancestry_stops_at_a_loop_and_at_an_unreadable_parent(self) -> None:
        self.write("100", "100 (a) S 101 100\n")
        self.write("101", "101 (b) S 100 101\n")
        with mock.patch.object(process_tree.os, "getpid", return_value=100):
            self.assertEqual(process_tree._ancestry(), {100, 101})
        self.write("200", "200 (c) S 999 200\n")
        with mock.patch.object(process_tree.os, "getpid", return_value=200):
            self.assertEqual(process_tree._ancestry(), {200, 999}, "999 has no entry, so the walk stops there")


class SignalTree(TreeCase):
    def test_a_signal_ends_the_whole_tree(self) -> None:
        shell = self.start("sleep 30 & echo $!; wait")
        sleeper = int(shell.stdout.readline())
        self.assertTrue(process_tree.signal_tree(shell.pid, signal.SIGTERM))
        self.assertTrue(wait_gone(sleeper), "the background sleep survived")
        self.assertEqual(shell.wait(timeout=WAIT_SECONDS), -signal.SIGTERM)

    def test_a_catching_parent_ends_once_its_child_ends(self) -> None:
        # trap ':' catches TERM; a handler is reset on exec, so the child still dies from TERM. trap "" would
        # pass its ignore down to the child, which would survive the signal.
        shell = self.start("trap ':' TERM; sleep 30 & echo $!; wait; wait")
        sleeper = int(shell.stdout.readline())
        self.assertTrue(process_tree.signal_tree(shell.pid, signal.SIGTERM))
        self.assertTrue(wait_gone(sleeper), "the grandchild sleep survived")
        self.assertEqual(shell.wait(timeout=WAIT_SECONDS), 0)

    def test_the_caller_is_never_signalled(self) -> None:
        self.assertFalse(process_tree.signal_tree(os.getpid(), signal.SIGTERM))
        self.assertTrue(is_running(os.getpid()), "the test process was signalled")

    def test_a_descendant_that_is_an_ancestor_of_the_caller_is_skipped(self) -> None:
        shell = self.start("sleep 30 & echo $!; wait")
        sleeper = int(shell.stdout.readline())
        with mock.patch.object(process_tree, "_ancestry", return_value={os.getpid(), sleeper}):
            self.assertTrue(process_tree.signal_tree(shell.pid, signal.SIGTERM))
        self.assertTrue(is_running(sleeper), "the skipped descendant was signalled")
        self.assertEqual(shell.wait(timeout=WAIT_SECONDS), -signal.SIGTERM)

    def test_a_pid_in_the_ancestry_is_refused_without_signalling_anything(self) -> None:
        shell = self.start("sleep 30 & echo $!; wait")
        sleeper = int(shell.stdout.readline())
        with mock.patch.object(process_tree, "_ancestry", return_value={shell.pid}):
            self.assertFalse(process_tree.signal_tree(shell.pid, signal.SIGTERM))
        self.assertTrue(is_running(sleeper), "a descendant was signalled although the pid was refused")
        self.assertIsNone(shell.poll(), "the pid was signalled although it is an ancestor")

    def test_a_dead_pid_is_true(self) -> None:
        self.assertTrue(process_tree.signal_tree(dead_pid(), signal.SIGTERM))

    def test_a_pid_that_is_not_positive_is_refused(self) -> None:
        self.assertFalse(process_tree.signal_tree(0, signal.SIGTERM))
        self.assertFalse(process_tree.signal_tree(-1, signal.SIGTERM))

    def test_permission_error_on_the_pid_is_false(self) -> None:
        with mock.patch.object(process_tree, "descendants", return_value=[]), \
                mock.patch.object(process_tree.os, "kill", side_effect=PermissionError):
            self.assertFalse(process_tree.signal_tree(os.getpid(), signal.SIGTERM))

    def test_permission_error_on_a_descendant_is_skipped(self) -> None:
        sent: list[int] = []

        def kill(pid: int, sig: int) -> None:
            if pid == 11:
                raise PermissionError
            sent.append(pid)

        with mock.patch.object(process_tree, "descendants", return_value=[11, 12]), \
                mock.patch.object(process_tree.os, "kill", side_effect=kill):
            self.assertTrue(process_tree.signal_tree(10, signal.SIGTERM))
        self.assertEqual(sent, [12, 10], "the descendants come deepest first, then the pid")


if __name__ == "__main__":
    unittest.main()
