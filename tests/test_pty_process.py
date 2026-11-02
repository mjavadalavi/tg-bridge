"""PtyProcess: real child processes (stdlib python -I -c) on a pseudo-terminal.

Covers what the pty adds to the ProcessSession base: typed text and keys reach the child as bytes, the cursor query
ESC[6n gets its reply (also when the query is split across two reads), the screen keeps only the stripped text and
stays bounded, wait_for() succeeds, times out or returns early when the child dies, stop() empties the process group,
and ProcessExited is reported exactly once. The children are small -c scripts; a tty in raw mode lets them see bytes.
A child that stops reading must not hang a write: write() gives up after its timeout, and stop() still kills the child.
Each test runs under its own TempHome, and every test finishes in a few seconds.
"""

import json
import os
import re
import signal
import sys
import threading
import time
import unittest
from unittest import mock

from tests.fakes.backend import RecordingSink
from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.claude.tui_session import ClaudeTuiSession
from tg_bridge.agents.events import ProcessExited
from tg_bridge.agents.pty_process import PASTE_GAP_SECONDS, REDRAW_SETTLE_SECONDS, SCREEN_CHARS, PtyProcess
from tg_bridge.logging_ import Logger

WAIT = 15.0

# Raw mode: the child gets the bytes as typed, with no echo and no line editing.
RAW_ECHO_UNTIL_CR = (
    "import os, tty\n"
    "tty.setraw(0)\n"
    "os.write(1, b'ready\\n')\n"
    "buf = b''\n"
    "while b'\\r' not in buf:\n"
    "    buf += os.read(0, 1024)\n"
    "os.write(1, ('got ' + repr(buf) + '\\n').encode())\n"
)
# The query is split: ESC[ in one write, 6n in a later one. The child answers nothing itself; it reads the reply.
CURSOR_QUERY_SPLIT = (
    "import os, select, time, tty\n"
    "tty.setraw(0)\n"
    "os.write(1, b'ready\\n\\x1b[')\n"
    "time.sleep(0.3)\n"
    "os.write(1, b'6n')\n"
    "buf = b''\n"
    "while not buf.endswith(b'R'):\n"
    "    buf += os.read(0, 64)\n"
    "r, _, _ = select.select([0], [], [], 0.5)\n"
    "extra = os.read(0, 64) if r else b''\n"
    "os.write(1, ('reply ' + repr(buf) + ' extra ' + repr(extra) + '\\n').encode())\n"
)
KEYS_UNTIL_Z = (
    "import os, tty\n"
    "tty.setraw(0)\n"
    "os.write(1, b'ready\\n')\n"
    "buf = b''\n"
    "while b'Z' not in buf:\n"
    "    buf += os.read(0, 64)\n"
    "os.write(1, ('keys ' + repr(buf) + '\\n').encode())\n"
)
SIZE = "import os\ns = os.get_terminal_size(0)\nprint('size', s.lines, s.columns, flush=True)\n"
STRIPPED = "print('\\x1b[31mred\\x1b[0m text\\x1b]0;title\\x07 done\\x1b(B')\n"
BIG_OUTPUT = "print('a' * 20000 + 'END', flush=True)\n"
UTF8_SPLIT = (
    "import os, time\n"
    "os.write(1, b'\\xe2\\x82')\n"  # the first two bytes of the euro sign
    "time.sleep(0.3)\n"
    "os.write(1, b'\\xac\\n')\n"
)
MARKER_THEN_WAIT = "import time\nprint('marker-here', flush=True)\ntime.sleep(30)\n"
SLEEP_FOREVER = "import time\ntime.sleep(30)\n"
# Raw mode, so the kernel does not echo the paste back to the reader. Then never reads its stdin.
STOPPED_READING = "import os, time, tty\ntty.setraw(0)\nos.write(1, b'ready\\n')\ntime.sleep(60)\n"
READ_MEGABYTE_THEN_REPORT = (
    "import os, tty\n"
    "tty.setraw(0)\n"
    "os.write(1, b'ready\\n')\n"
    "n = 0\n"
    "while n < 1048576:\n"
    "    n += len(os.read(0, 65536))\n"
    "os.write(1, ('total %d\\n' % n).encode())\n"
)
ONE_MEGABYTE = "p" * (1024 * 1024)  # more than a pty buffer holds, so a write blocks once the child stops reading
QUICK_EXIT = "print('bye', flush=True)\n"
HOOK_TWO_LINES = "import time\nprint('one', flush=True)\ntime.sleep(0.2)\nprint('two', flush=True)\n"
EXIT_WITH_CODE = "import sys\nprint('done-x', flush=True)\nsys.exit(3)\n"
# The leader ignores SIGINT and SIGTERM, so the ladder must reach SIGKILL. The grandchild restores the default
# SIGTERM, so only the group sweep can end it.
STUBBORN_WITH_GRANDCHILD = (
    "import signal, subprocess, sys, time\n"
    "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "g = subprocess.Popen([sys.executable, '-I', '-c', "
    "'import signal, time\\nsignal.signal(signal.SIGTERM, signal.SIG_DFL)\\nwhile True: time.sleep(0.1)'])\n"
    "print('GRAND=%d' % g.pid, flush=True)\n"
    "while True:\n"
    "    time.sleep(0.1)\n"
)
STUBBORN = (
    "import signal, time\n"
    "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "print('ready', flush=True)\n"
    "while True:\n"
    "    time.sleep(0.1)\n"
)


def _python(code: str) -> list[str]:
    return [sys.executable, "-I", "-c", code]


def _pid_gone(pid: int) -> bool:
    """True when pid no longer runs. A zombie counts as gone: it has no code left to run."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return True
    return state in ("Z", "X")


def _group_has_members(pgid: int | None) -> bool:
    """Independent of the module under test: any non-zombie process still in group pgid."""
    if pgid is None:
        return False
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="utf-8") as f:
                fields = f.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if fields[0] != "Z" and int(fields[2]) == pgid:
            return True
    return False


class CommandPty(PtyProcess):
    """A test subclass: argv is given and every decoded chunk is recorded through the _on_output hook."""

    def __init__(self, argv: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._argv_value = list(argv)
        self.outputs: list[str] = []

    def _argv(self) -> list[str]:
        return list(self._argv_value)

    def _on_output(self, text: str) -> None:
        self.outputs.append(text)


class ClosingPty(CommandPty):
    """Closes the way ClaudeTuiSession does: the real _graceful_close, Esc then /exit, both on the bounded writes."""

    _graceful_close = ClaudeTuiSession._graceful_close


class PtyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = TempHome().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.clock = FakeClock()
        self.sessions: list[PtyProcess] = []
        self.addCleanup(self._stop_all)

    def _stop_all(self) -> None:
        for session in self.sessions:
            session.stop(grace=0.5)

    def _session(self, cls=CommandPty, argv=None, sink=None, name="p1", **extra) -> tuple[PtyProcess, RecordingSink]:
        sink = sink or RecordingSink()
        session = cls(
            argv=argv or _python("pass"), name=name, cwd=self.home.home,
            env={**os.environ, "TERM": "xterm-256color"}, errfile_dir=os.path.join(self.home.state, "turns"),
            sink=sink, logger=self.logger, clock=self.clock, **extra,
        )
        self.sessions.append(session)
        return session, sink

    def _exits(self, sink: RecordingSink) -> list[ProcessExited]:
        return [e for e in sink.events if isinstance(e, ProcessExited)]

    def _wait_exit(self, sink: RecordingSink) -> ProcessExited:
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ProcessExited), timeout=WAIT), "no ProcessExited")
        return self._exits(sink)[-1]

    def _wait_screen(self, session: PtyProcess, text: str) -> None:
        """Wait for text on the compact screen, so the spaces in text do not matter."""
        compact = re.sub(r"\s+", "", text)
        self.assertTrue(session.wait_for(lambda s: compact in s, timeout=WAIT), f"{text!r} never reached the screen")


class TypingTests(PtyTestCase):
    def test_type_text_reaches_the_child_as_one_bracketed_paste_then_cr(self) -> None:
        session, sink = self._session(argv=_python(RAW_ECHO_UNTIL_CR))
        session.start()
        self._wait_screen(session, "ready")
        self.assertTrue(session.type_text("hello world"))
        self.assertIn(PASTE_GAP_SECONDS, self.clock.sleeps)  # the pause between the paste and CR
        self._wait_screen(session, "got b'\\x1b[200~hello world\\x1b[201~\\r'")
        self._wait_exit(sink)

    def test_keys_reach_the_child_as_their_bytes(self) -> None:
        session, sink = self._session(argv=_python(KEYS_UNTIL_Z))
        session.start()
        self._wait_screen(session, "ready")
        self.assertTrue(session.press_esc())
        self.assertTrue(session.press_down())
        self.assertTrue(session.press_enter())
        self.assertTrue(session.write("Z"))
        self._wait_screen(session, "keys b'\\x1b\\x1b[B\\rZ'")
        self._wait_exit(sink)

    def test_writes_are_refused_before_start_and_after_stop(self) -> None:
        session, _ = self._session(argv=_python(SLEEP_FOREVER))
        self.assertFalse(session.write("early"))
        self.assertFalse(session.type_text("early"))
        self.assertFalse(session.press_enter())
        session.start()
        self.assertTrue(session.alive())
        session.stop(grace=0.5)
        self.assertFalse(session.write("late"))


class WriteTimeoutTests(PtyTestCase):
    def _bounded(self, session: PtyProcess, fn, seconds: float):
        """Run fn on a helper thread. A call still running after seconds fails the test instead of hanging the run."""
        box: list = []
        worker = threading.Thread(target=lambda: box.append(fn()), daemon=True)
        worker.start()
        worker.join(seconds)
        if worker.is_alive():
            try:  # kill the child, so a writer stuck on its pty can end, and the cleanup stop() can run
                os.killpg(session.pgid, signal.SIGKILL)
            except (OSError, TypeError):
                pass
            worker.join(WAIT)
            self.fail(f"call still running after {seconds}s")
        return box[0]

    def test_a_paste_larger_than_the_pty_buffer_gives_up_when_the_child_never_reads(self) -> None:
        session, _ = self._session(argv=_python(STOPPED_READING), write_timeout=0.5)
        session.start()
        self._wait_screen(session, "ready")
        started = time.monotonic()
        ok = self._bounded(session, lambda: session.write(ONE_MEGABYTE), WAIT)
        took = time.monotonic() - started
        self.assertFalse(ok)
        self.assertLess(took, 0.5 + 3.0, f"the write took {took:.2f}s")
        with open(self.logger.path, encoding="utf-8") as f:
            self.assertIn("write timed out", f.read())
        self.assertTrue(session.alive())  # the timeout does not kill the child by itself

    def test_stop_kills_the_child_while_a_write_is_stuck_on_a_full_buffer(self) -> None:
        session, _ = self._session(argv=_python(STOPPED_READING), write_timeout=10.0)
        session.start()
        self._wait_screen(session, "ready")
        writer = threading.Thread(target=session.write, args=(ONE_MEGABYTE,), daemon=True)
        writer.start()
        time.sleep(0.5)
        self.assertTrue(writer.is_alive())  # stuck: the child never reads
        started = time.monotonic()
        self._bounded(session, lambda: session.stop(grace=0.2), WAIT)
        took = time.monotonic() - started
        self.assertLess(took, 5.0, f"stop() took {took:.2f}s")
        self.assertFalse(session.alive())
        writer.join(WAIT)
        self.assertFalse(writer.is_alive())

    def test_stop_is_not_held_up_by_a_stuck_writer_during_the_close_step(self) -> None:
        session, _ = self._session(cls=ClosingPty, argv=_python(STOPPED_READING), write_timeout=10.0)
        session.start()
        self._wait_screen(session, "ready")
        writer = threading.Thread(target=session.write, args=(ONE_MEGABYTE,), daemon=True)
        writer.start()
        time.sleep(0.5)
        started = time.monotonic()
        self._bounded(session, lambda: session.stop(grace=0.2), WAIT)
        took = time.monotonic() - started
        self.assertLess(took, 5.0, f"stop() took {took:.2f}s")  # the Esc of the close gives up after its own 1 s
        self.assertFalse(session.alive())
        writer.join(WAIT)
        self.assertFalse(writer.is_alive())

    def test_a_write_waits_for_the_lock_no_longer_than_its_timeout(self) -> None:
        session, _ = self._session(argv=_python(STOPPED_READING), write_timeout=10.0)
        session.start()
        self._wait_screen(session, "ready")
        writer = threading.Thread(target=session.write, args=(ONE_MEGABYTE,), daemon=True)
        writer.start()
        time.sleep(0.5)
        started = time.monotonic()
        ok = self._bounded(session, lambda: session.write("x", timeout=0.3), WAIT)
        took = time.monotonic() - started
        self.assertFalse(ok)
        self.assertLess(took, 3.0, f"the write took {took:.2f}s")
        session.stop(grace=0.2)
        writer.join(WAIT)

    def test_a_large_paste_the_child_reads_arrives_whole(self) -> None:
        session, _ = self._session(argv=_python(READ_MEGABYTE_THEN_REPORT))
        session.start()
        self._wait_screen(session, "ready")
        self.assertTrue(session.write(ONE_MEGABYTE))
        self._wait_screen(session, "total1048576")


class CursorQueryTests(PtyTestCase):
    def test_the_cursor_query_gets_one_reply_even_when_split_across_reads(self) -> None:
        session, sink = self._session(argv=_python(CURSOR_QUERY_SPLIT))
        session.start()
        self._wait_screen(session, "reply b'\\x1b[1;1R' extra b''")
        self.assertNotIn("\x1b", session.screen())  # the query itself is stripped from the screen
        self._wait_exit(sink)


class ScreenTests(PtyTestCase):
    def test_ansi_codes_are_stripped_and_compact_screen_drops_all_whitespace(self) -> None:
        session, sink = self._session(argv=_python(STRIPPED))
        session.start()
        self._wait_exit(sink)
        screen = session.screen()
        self.assertIn("red text", screen)
        self.assertIn(" done", screen)
        self.assertNotIn("\x1b", screen)
        self.assertNotIn("title", screen)  # the OSC title is a sequence, not text
        self.assertNotIn("\r", screen)
        self.assertEqual(session.compact_screen(), "redtextdone")

    def test_the_child_sees_the_window_size_it_was_given(self) -> None:
        session, sink = self._session(argv=_python(SIZE))
        session.start()
        self._wait_exit(sink)
        self.assertIn("size50200", session.compact_screen())  # 50 rows, 200 columns

    def test_the_screen_keeps_only_the_newest_characters(self) -> None:
        session, sink = self._session(argv=_python(BIG_OUTPUT))
        session.start()
        self._wait_exit(sink)
        self.assertLessEqual(len(session.screen()), SCREEN_CHARS)
        self.assertTrue(session.screen().endswith("END\n"))

    def test_a_utf8_character_split_across_reads_decodes_without_replacement(self) -> None:
        session, sink = self._session(argv=_python(UTF8_SPLIT))
        session.start()
        self._wait_exit(sink)
        self.assertIn("\u20ac", session.screen())
        self.assertNotIn("\ufffd", session.screen())

    def test_on_output_sees_each_decoded_chunk(self) -> None:
        session, sink = self._session(argv=_python(HOOK_TWO_LINES))
        session.start()
        self._wait_exit(sink)
        self.assertIn("one", "".join(session.outputs))
        self.assertIn("two", "".join(session.outputs))


class WaitForTests(PtyTestCase):
    def test_wait_for_returns_true_once_the_text_is_on_the_screen(self) -> None:
        session, _ = self._session(argv=_python(MARKER_THEN_WAIT))
        session.start()
        started = time.monotonic()
        self.assertTrue(session.wait_for(lambda s: "marker-here" in s, timeout=WAIT))
        self.assertLess(time.monotonic() - started, WAIT)

    def test_wait_for_times_out_with_false(self) -> None:
        session, _ = self._session(argv=_python(SLEEP_FOREVER))
        session.start()
        started = time.monotonic()
        self.assertFalse(session.wait_for(lambda s: "never" in s, timeout=0.5))
        took = time.monotonic() - started
        self.assertTrue(0.4 <= took < 3.0, f"took {took:.2f}s")

    def test_wait_for_returns_false_early_when_the_child_died(self) -> None:
        session, sink = self._session(argv=_python(QUICK_EXIT))
        session.start()
        self._wait_exit(sink)
        started = time.monotonic()
        self.assertFalse(session.wait_for(lambda s: "never" in s, timeout=10))
        self.assertLess(time.monotonic() - started, 5.0)


class LifecycleTests(PtyTestCase):
    def test_start_twice_is_refused(self) -> None:
        session, sink = self._session(argv=_python(QUICK_EXIT))
        session.start()
        with self.assertRaises(RuntimeError):
            session.start()
        self._wait_exit(sink)

    def test_process_exited_is_reported_once_with_the_return_code_and_the_screen_tail(self) -> None:
        session, sink = self._session(argv=_python(EXIT_WITH_CODE))
        session.start()
        exited = self._wait_exit(sink)
        self.assertEqual(exited.returncode, 3)
        self.assertIn("done-x", exited.stderr_tail)
        self.assertEqual(session.stop(grace=0.5), 3)  # a later stop() does not report again
        self.assertEqual(len(self._exits(sink)), 1)
        turns = os.path.join(self.home.state, "turns")
        errfiles = [f for f in os.listdir(turns) if f.endswith(".err")]
        self.assertEqual(len(errfiles), 1)  # the screen tail survives the process
        with open(os.path.join(turns, errfiles[0]), encoding="utf-8") as f:
            self.assertIn("done-x", f.read())

    def test_stop_kills_a_child_that_ignores_sigint_and_sigterm_and_its_grandchild(self) -> None:
        session, sink = self._session(argv=_python(STUBBORN_WITH_GRANDCHILD))
        session.start()
        self.assertTrue(session.wait_for(lambda s: "GRAND=" in s, timeout=WAIT))
        grand = int(re.search(r"GRAND=(\d+)", session.compact_screen()).group(1))
        group = session.pgid
        self.assertFalse(_pid_gone(grand))  # it runs before stop()
        started = time.monotonic()
        rc = session.stop(grace=0.2)
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertIsNotNone(rc)
        self.assertLess(rc, 0)  # ended by a signal
        self.assertFalse(session.alive())
        self.assertTrue(_pid_gone(grand), "grandchild survived stop()")
        self.assertFalse(_group_has_members(group))
        self.assertEqual(self._wait_exit(sink).returncode, rc)
        self.assertEqual(len(self._exits(sink)), 1)

    def test_a_failing_hook_is_logged_and_the_reader_goes_on(self) -> None:
        class FailingOutput(CommandPty):
            def _on_output(self, text: str) -> None:
                raise ValueError("boom in output")

        session, sink = self._session(cls=FailingOutput, argv=_python(HOOK_TWO_LINES))
        session.start()
        exited = self._wait_exit(sink)
        self.assertEqual(exited.returncode, 0)
        self.assertIn("two", session.screen())
        with open(self.logger.path, encoding="utf-8") as f:
            self.assertIn("boom in output", f.read())

    def test_stop_before_start_returns_none(self) -> None:
        session, _ = self._session(argv=_python("pass"))
        self.assertIsNone(session.stop())
        self.assertTrue(session.end_gracefully(0.1))



WINCH_CHILD = (
    "import json, os, signal, time\n"
    "path = os.environ['WINCH_LOG']\n"
    "def log(*_):\n"
    "    size = os.get_terminal_size(0)\n"
    "    with open(path, 'a', encoding='utf-8') as f:\n"
    "        f.write(json.dumps([size.lines, size.columns]) + chr(10))\n"
    "signal.signal(signal.SIGWINCH, log)\n"
    "print('ready', flush=True)\n"
    "time.sleep(30)\n"
)


# A launcher like a teamclaude-style wrapper: it starts the real child in the same process group and does not forward
# SIGWINCH. So only a signal to the whole group reaches the grandchild.
WRAPPER_CHILD = (
    "import subprocess, sys\n"
    "sys.exit(subprocess.Popen([sys.executable, '-I', sys.argv[1]]).wait())\n"
)


class RedrawTests(PtyTestCase):
    """PtyProcess.redraw(): a child that logs each SIGWINCH with the size it reads gets one column narrower, then the
    real size, each with its own SIGWINCH, and redraw() returns before the steps run."""

    def _winch_session(self, cols: int = 200, wrapped: bool = False) -> tuple[PtyProcess, str]:
        script = os.path.join(self.home.home, "winch_child.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(WINCH_CHILD)
        log = os.path.join(self.home.home, "winch.jsonl")
        argv = [sys.executable, "-I", "-c", WRAPPER_CHILD, script] if wrapped else [sys.executable, "-I", script]
        session, _ = self._session(argv=argv, rows=50, cols=cols)
        session._env["WINCH_LOG"] = log
        session.start()
        self._wait_screen(session, "ready")
        return session, log

    def _winch_lines(self, log: str) -> list[list[int]]:
        if not os.path.exists(log):
            return []
        with open(log, encoding="utf-8") as f:
            return [json.loads(line) for line in f.read().splitlines()]

    def test_redraw_sends_the_narrow_size_then_the_real_size_each_with_a_sigwinch(self) -> None:
        session, log = self._winch_session()
        session.redraw()
        deadline = time.monotonic() + WAIT
        while len(self._winch_lines(log)) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(self._winch_lines(log), [[50, 199], [50, 200]])

    def test_a_grandchild_behind_a_wrapper_launcher_gets_both_sizes(self) -> None:
        session, log = self._winch_session(wrapped=True)
        session.redraw()
        deadline = time.monotonic() + WAIT
        while len(self._winch_lines(log)) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(self._winch_lines(log), [[50, 199], [50, 200]])

    def test_a_pty_narrower_than_two_columns_is_not_redrawn(self) -> None:
        session, log = self._winch_session(cols=1)
        session.redraw()
        time.sleep(2 * REDRAW_SETTLE_SECONDS + 0.4)
        self.assertEqual(self._winch_lines(log), [])
        self.assertFalse(session._redrawing)  # no worker was started

    def test_redraw_returns_at_once_and_the_steps_run_after_the_call(self) -> None:
        session, log = self._winch_session()
        started = time.monotonic()
        session.redraw()
        self.assertLess(time.monotonic() - started, REDRAW_SETTLE_SECONDS / 2, "redraw held the caller")
        self.assertEqual(self._winch_lines(log), [], "the steps ran before redraw returned")
        deadline = time.monotonic() + WAIT
        while len(self._winch_lines(log)) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(len(self._winch_lines(log)), 2)

    def test_redraw_after_the_child_is_gone_sends_nothing_and_does_not_raise(self) -> None:
        session, log = self._winch_session()
        session.stop(grace=0.5)
        before = self._winch_lines(log)
        errors: list = []
        with mock.patch.object(threading, "excepthook", side_effect=errors.append):  # the worker must not die silently
            session.redraw()
            time.sleep(2 * REDRAW_SETTLE_SECONDS + 0.4)
        self.assertEqual(errors, [])
        self.assertEqual(self._winch_lines(log), before)


if __name__ == "__main__":
    unittest.main()
