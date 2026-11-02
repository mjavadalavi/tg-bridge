"""`tg-bridge view-delete <thread>` (card D3): the request file, the wait for the daemon's answer, the printed and tmux
result, the timeout and the exit status.

The daemon is simulated by a clock whose sleep() plays it: it takes the request, acknowledges it with <thread>.deleting,
and answers `busy_steps` sleeps later. Without a daemon the plain FakeClock is used. The wait runs without real time. The
tmux display is an injected callable, so no test runs tmux.
"""

import contextlib
import io
import os
import unittest

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge import cli
from tg_bridge.core.delete_requests import FAILED_PREFIX, ack_path, request_path, result_path, write_atomic
from tg_bridge.daemon.view_delete import ACK_SECONDS, ANSWER_SECONDS, STILL_TEXT, TIMEOUT_TEXT
from tg_bridge.paths import Paths

THREAD = 9
ANSWERED = "deleted topic alpha"
REFUSED = f"{FAILED_PREFIX} the Telegram deletion failed: boom"


class DaemonClock(FakeClock):
    """A clock whose sleep() plays the daemon. The first sleep takes the request file and writes <thread>.deleting; the
    answer (None: never) is written at the sleep that comes busy_steps sleeps after that, and the acknowledgement goes."""

    def __init__(self, view_dir: str, answer: str | None, busy_steps: int = 0) -> None:
        super().__init__()
        self.view_dir = view_dir
        self.answer = answer
        self.busy_steps = busy_steps
        self.seen_request: str | None = None
        self.taken_at: int | None = None

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        path = request_path(self.view_dir, THREAD)
        if self.taken_at is None:
            if not os.path.exists(path):
                return
            with open(path, encoding="utf-8") as handle:
                self.seen_request = handle.read()
            os.remove(path)
            write_atomic(ack_path(self.view_dir, THREAD), "working")
            self.taken_at = len(self.sleeps)
            return
        if self.answer is None or len(self.sleeps) - self.taken_at < self.busy_steps:
            return
        write_atomic(result_path(self.view_dir, THREAD), self.answer)
        os.remove(ack_path(self.view_dir, THREAD))
        self.answer = None


class ViewDeleteCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.paths = Paths.from_home(self.home.home)
        os.makedirs(self.paths.view, exist_ok=True)
        self.shown: list[str] = []

    def run_delete(self, clock: FakeClock, environ: dict[str, str] | None = None, thread: str = str(THREAD)) -> int:
        return cli.view_delete_topic(self.paths, thread, environ or {}, clock, show=self.shown.append)


class ViewDeleteTests(ViewDeleteCase):
    def test_writes_the_request_then_prints_the_daemons_answer(self) -> None:
        clock = DaemonClock(self.paths.view, ANSWERED)
        status, out, err = self._capture(lambda: self.run_delete(clock))
        self.assertEqual((status, out.strip(), err), (0, ANSWERED, ""))
        self.assertTrue(clock.seen_request.startswith("requested "))
        self.assertFalse(os.path.exists(result_path(self.paths.view, THREAD)))  # the answer is read and removed
        self.assertEqual(self.shown, [])  # no TMUX in the environment: nothing is shown in tmux

    def test_a_refused_removal_exits_1(self) -> None:
        status, out, _ = self._capture(lambda: self.run_delete(DaemonClock(self.paths.view, REFUSED)))
        self.assertEqual((status, out.strip()), (1, REFUSED))

    def test_inside_tmux_the_answer_is_also_shown(self) -> None:
        clock = DaemonClock(self.paths.view, ANSWERED)
        self._capture(lambda: self.run_delete(clock, {"TMUX": "/tmp/tmux-1000/default,1,0"}))
        self.assertEqual(self.shown, [ANSWERED])

    def test_an_old_answer_is_not_taken_for_this_request(self) -> None:
        write_atomic(result_path(self.paths.view, THREAD), "deleted topic old")
        status, out, _ = self._capture(lambda: self.run_delete(DaemonClock(self.paths.view, ANSWERED)))
        self.assertEqual((status, out.strip()), (0, ANSWERED))
        self.assertFalse(os.path.exists(result_path(self.paths.view, THREAD)))

    def test_no_daemon_times_out_after_the_ack_window_and_removes_the_request(self) -> None:
        clock = FakeClock()
        status, out, _ = self._capture(lambda: self.run_delete(clock, {"TMUX": "x"}))
        self.assertEqual((status, out.strip()), (1, TIMEOUT_TEXT))
        self.assertEqual(len(clock.sleeps), int(ACK_SECONDS))  # one-second steps for 10 s
        self.assertFalse(os.path.exists(request_path(self.paths.view, THREAD)))
        self.assertEqual(self.shown, [TIMEOUT_TEXT])

    def test_a_daemon_that_takes_longer_than_the_ack_window_still_answers(self) -> None:
        """The removal takes 15 s after the acknowledgement: the CLI waits past the 10 s ack window and gets the answer."""
        clock = DaemonClock(self.paths.view, ANSWERED, busy_steps=15)
        status, out, _ = self._capture(lambda: self.run_delete(clock))
        self.assertEqual((status, out.strip()), (0, ANSWERED))
        self.assertGreater(len(clock.sleeps), int(ACK_SECONDS))
        self.assertFalse(os.path.exists(ack_path(self.paths.view, THREAD)))  # the ack goes with the answer

    def test_an_acknowledged_request_that_never_answers_is_still_deleting_not_failed(self) -> None:
        clock = DaemonClock(self.paths.view, None)
        status, out, _ = self._capture(lambda: self.run_delete(clock, {"TMUX": "x"}))
        self.assertEqual((status, out.strip()), (0, STILL_TEXT))
        self.assertEqual(len(clock.sleeps), 1 + int(ANSWER_SECONDS))  # the ack is seen at the first step, then 120 s
        self.assertEqual(self.shown, [STILL_TEXT])

    def test_stale_ack_and_answer_from_earlier_are_removed_before_the_request(self) -> None:
        """A daemon that crashed left .deleting behind: it must not put this request on the 120 s wait."""
        write_atomic(ack_path(self.paths.view, THREAD), "working")
        write_atomic(result_path(self.paths.view, THREAD), "deleted topic old")
        clock = FakeClock()
        status, out, _ = self._capture(lambda: self.run_delete(clock))
        self.assertEqual((status, out.strip()), (1, TIMEOUT_TEXT))
        self.assertEqual(len(clock.sleeps), int(ACK_SECONDS))
        self.assertFalse(os.path.exists(ack_path(self.paths.view, THREAD)))

    def test_a_thread_that_is_not_a_number_writes_no_request(self) -> None:
        status, _, err = self._capture(lambda: self.run_delete(FakeClock(), thread="9x"))
        self.assertEqual(status, 1)
        self.assertIn("'9x'", err)
        self.assertEqual(os.listdir(self.paths.view), [])

    def _capture(self, action):
        return _run_quietly(action)


def _run_quietly(action):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        result = action()
    return result, out.getvalue(), err.getvalue()


if __name__ == "__main__":
    unittest.main()
