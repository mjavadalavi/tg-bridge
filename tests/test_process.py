"""ProcessSession: real child processes (stdlib python -I -c) through the Template Method hooks.

Covers the scenarios of the old TopicSession suite that the base owns: lines reach _on_line, a child that ignores
EOF and signals is killed inside the ladder, a grandchild in the group is gone after stop(), a read failure still
reports ProcessExited, and on_reader_thread() is true only on the reader thread. Each test runs under its own TempHome.
"""

import os
import sys
import threading
import time
import unittest
from collections.abc import Iterable

from tests.fakes.backend import RecordingSink
from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.events import ProcessExited
from tg_bridge.agents.process import KEEP_ERRFILES, ProcessSession
from tg_bridge.logging_ import Logger

WAIT = 15.0

ECHO_LINES = "print('alpha'); print('beta')"
STUBBORN = (
    "import signal, time\n"
    "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "print('ready', flush=True)\n"
    "while True:\n"
    "    time.sleep(0.1)\n"
)
GRANDCHILD = (
    "import subprocess, sys, time\n"
    "g = subprocess.Popen([sys.executable, '-I', '-c', 'import time\\nwhile True: time.sleep(0.1)'])\n"
    "print(g.pid, flush=True)\n"
    "while True:\n"
    "    time.sleep(0.1)\n"
)
ECHO_JSON_UNTIL_EOF = (
    "import sys\n"
    "for line in sys.stdin:\n"
    "    print('got ' + line.strip(), flush=True)\n"
)
STDERR_THEN_EXIT = "import sys; sys.stderr.write('boom-on-stderr\\n'); sys.stderr.flush()"


def _pid_gone(pid: int) -> bool:
    """True when pid no longer runs. A zombie counts as gone: it has no code left to run."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return True
    return state in ("Z", "X")


class CommandProcess(ProcessSession):
    """A test subclass: argv is given, every stdout line is recorded, and flags show where the hooks ran."""

    def __init__(self, argv: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._argv_value = list(argv)
        self.lines: list[str] = []
        self.thread_flags: list[bool] = []

    def _argv(self) -> list[str]:
        return list(self._argv_value)

    def _on_line(self, line: str) -> None:
        self.thread_flags.append(self.on_reader_thread())
        self.lines.append(line)


def _python(code: str) -> list[str]:
    return [sys.executable, "-I", "-c", code]


class ProcessTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = TempHome().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.clock = FakeClock()
        self.sessions: list[ProcessSession] = []
        self.addCleanup(self._stop_all)

    def _stop_all(self) -> None:
        for session in self.sessions:
            session.stop(grace=1)

    def _session(self, cls=CommandProcess, argv=None, sink=None, name="t1") -> tuple[ProcessSession, RecordingSink]:
        sink = sink or RecordingSink()
        session = cls(
            argv=argv or _python("pass"), name=name, cwd=self.home.home, env=dict(os.environ),
            errfile_dir=os.path.join(self.home.state, "turns"), sink=sink, logger=self.logger, clock=self.clock,
        )
        self.sessions.append(session)
        return session, sink

    def _exited(self, sink: RecordingSink) -> ProcessExited | None:
        found = [e for e in sink.events if isinstance(e, ProcessExited)]
        return found[-1] if found else None

    def _wait_exit(self, sink: RecordingSink) -> ProcessExited:
        self.assertTrue(sink.wait_for(lambda e: isinstance(e, ProcessExited), timeout=WAIT), "no ProcessExited")
        return self._exited(sink)

    def _wait_line(self, session: CommandProcess, text: str) -> None:
        deadline = time.monotonic() + WAIT
        while text not in session.lines and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertIn(text, session.lines)


class StdoutAndExitTests(ProcessTestCase):
    def test_start_makes_the_child_its_own_group_leader(self) -> None:
        """Ported labels:
        process started, pgid == pid
        """
        code = "import os, time\nprint(os.getpid(), flush=True)\ntime.sleep(30)\n"
        session, _ = self._session(argv=_python(code))
        session.start()
        pid = None
        deadline = time.monotonic() + WAIT
        while pid is None and time.monotonic() < deadline:
            for line in list(session.lines):
                if line.isdigit():
                    pid = int(line)
            time.sleep(0.02)
        self.assertIsNotNone(pid)
        self.assertTrue(session.alive())
        self.assertEqual(session.pgid, pid)

    def test_stdout_lines_reach_on_line_and_exit_is_reported(self) -> None:
        session, sink = self._session(argv=_python(ECHO_LINES))
        session.start()
        exited = self._wait_exit(sink)
        self.assertEqual(session.lines, ["alpha", "beta"])
        self.assertEqual(exited.returncode, 0)
        self.assertFalse(session.alive())

    def test_on_reader_thread_is_true_only_on_the_reader_thread(self) -> None:
        session, sink = self._session(argv=_python(ECHO_LINES))
        self.assertFalse(session.on_reader_thread())  # before start there is no reader thread
        session.start()
        self._wait_exit(sink)
        self.assertEqual(session.thread_flags, [True, True])  # both calls were inside _on_line
        self.assertFalse(session.on_reader_thread())  # the test thread is not the reader thread

    def test_write_sends_one_json_line_and_eof_ends_the_child_cleanly(self) -> None:
        session, sink = self._session(argv=_python(ECHO_JSON_UNTIL_EOF))
        self.assertFalse(session.write({"type": "early"}))  # no process yet
        session.start()
        self.assertTrue(session.write({"type": "user", "n": 1}))
        self._wait_line(session, 'got {"type": "user", "n": 1}')
        started = time.monotonic()
        rc = session.stop(grace=5)
        self.assertEqual(rc, 0)  # the child left on EOF, so no signal was needed
        self.assertLess(time.monotonic() - started, 4.0)
        self.assertEqual(self._wait_exit(sink).returncode, 0)

    def test_start_twice_is_refused(self) -> None:
        session, sink = self._session(argv=_python(ECHO_LINES))
        session.start()
        with self.assertRaises(RuntimeError):
            session.start()
        self._wait_exit(sink)


class ErrorHandlingTests(ProcessTestCase):
    def test_a_bad_line_is_logged_and_the_stream_goes_on(self) -> None:
        class FailsOnFirst(CommandProcess):
            def _on_line(self, line: str) -> None:
                self.lines.append(line)
                if line == "alpha":
                    raise ValueError("boom in line")

        session, sink = self._session(cls=FailsOnFirst, argv=_python(ECHO_LINES))
        session.start()
        exited = self._wait_exit(sink)
        self.assertEqual(session.lines, ["alpha", "beta"])
        self.assertEqual(exited.returncode, 0)
        with open(self.logger.path, encoding="utf-8") as f:
            self.assertIn("boom in line", f.read())

    def test_a_failing_read_is_logged_and_still_reports_process_exited(self) -> None:
        class ReadBreaks(CommandProcess):
            def _iter_stdout(self, proc) -> Iterable[str]:
                yield "first"
                raise RuntimeError("pipe broke")

        session, sink = self._session(cls=ReadBreaks, argv=_python("print('x')"))
        session.start()
        exited = self._wait_exit(sink)
        self.assertEqual(session.lines, ["first"])
        self.assertEqual(exited.returncode, 0)
        with open(self.logger.path, encoding="utf-8") as f:
            self.assertIn("pipe broke", f.read())

    def test_a_failing_sink_does_not_stop_the_reader(self) -> None:
        seen: list[object] = []

        def sink(event: object) -> None:
            seen.append(event)
            raise RuntimeError("sink broke")

        session, _ = self._session(argv=_python(ECHO_LINES), sink=sink)
        session.start()
        deadline = time.monotonic() + WAIT
        while not any(isinstance(e, ProcessExited) for e in seen) and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(any(isinstance(e, ProcessExited) for e in seen))
        self.assertEqual(session.lines, ["alpha", "beta"])


class KillLadderTests(ProcessTestCase):
    def test_a_child_that_ignores_eof_and_signals_is_killed_within_grace_plus_two(self) -> None:
        session, sink = self._session(argv=_python(STUBBORN))
        session.start()
        self._wait_line(session, "ready")
        group = session.pgid
        started = time.monotonic()
        rc = session.stop(grace=1)
        took = time.monotonic() - started
        self.assertLess(took, 1 + 2.0, f"stop() took {took:.2f}s")
        self.assertIsNotNone(rc)
        self.assertLess(rc, 0)  # ended by a signal
        self.assertFalse(session.alive())
        self.assertFalse(_group_has_members(group))
        self.assertEqual(self._wait_exit(sink).returncode, rc)

    def test_a_grandchild_in_the_group_is_gone_after_stop(self) -> None:
        session, sink = self._session(argv=_python(GRANDCHILD))
        session.start()
        deadline = time.monotonic() + WAIT
        grand = None
        while grand is None and time.monotonic() < deadline:
            for line in list(session.lines):
                if line.isdigit():
                    grand = int(line)
            time.sleep(0.02)
        self.assertIsNotNone(grand)
        self.assertFalse(_pid_gone(grand))  # it is running before stop()
        session.stop(grace=1)
        self.assertTrue(_pid_gone(grand), "grandchild survived stop()")
        self.assertFalse(_group_has_members(session.pgid))
        self._wait_exit(sink)


class StderrTests(ProcessTestCase):
    def test_stderr_goes_to_the_session_file_and_stderr_tail_reads_it(self) -> None:
        session, sink = self._session(argv=_python(STDERR_THEN_EXIT))
        session.start()
        self._wait_exit(sink)
        self.assertIn("boom-on-stderr", session.stderr_tail())
        files = [f for f in os.listdir(os.path.join(self.home.state, "turns")) if f.endswith(".err")]
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].startswith("t1-"))

    def test_only_the_newest_stderr_files_are_kept(self) -> None:
        turns = os.path.join(self.home.state, "turns")
        os.makedirs(turns, exist_ok=True)
        for i in range(KEEP_ERRFILES + 10):
            path = os.path.join(turns, f"old-{i:03d}.err")
            with open(path, "w", encoding="utf-8") as f:
                f.write("old\n")
            os.utime(path, (1000 + i, 1000 + i))
        session, sink = self._session(argv=_python("pass"))
        session.start()
        self._wait_exit(sink)
        names = sorted(os.listdir(turns))
        self.assertEqual(len(names), KEEP_ERRFILES)
        self.assertNotIn("old-000.err", names)  # the oldest went first
        self.assertIn(f"old-{KEEP_ERRFILES + 9:03d}.err", names)

    def test_stderr_tail_is_empty_before_start(self) -> None:
        session, _ = self._session(argv=_python("pass"))
        self.assertEqual(session.stderr_tail(), "")
        self.assertIsNone(session.stop())


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


if __name__ == "__main__":
    unittest.main()
