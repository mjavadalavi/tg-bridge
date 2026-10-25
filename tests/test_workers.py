"""WorkerPool: port of the worker scenarios of fakecli/test_workers.py (labels kept in docstrings) plus the sender
ordering and title-queue rules of tg-bridge.py (_media_q x2, _out_q x1, _title_q x1).

Real threads, but every wait is an Event or a bounded join, so a broken pool fails in seconds instead of hanging.
Pool names are unique per test so thread-name checks never see another test's threads.
"""

import os
import threading
import time
import unittest
import uuid

from tests.support import TempHome
from tg_bridge.logging_ import Logger
from tg_bridge.core.workers import WorkerPool

WAIT = 5.0


def _unique_name(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def _threads_named(prefix: str) -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith(f"tg-{prefix}-")]


class _Harness:
    """A pool wired to a Logger inside a temp home. The log file is read back to check failure lines."""

    def __init__(self, home: TempHome, name: str, size: int) -> None:
        self.log_path = os.path.join(home.state, "bridge.log")
        self.pool = WorkerPool(name, size, Logger(self.log_path))
        self.name = name

    def log_text(self) -> str:
        if not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, encoding="utf-8") as f:
            return f.read()


class LazyStartTests(unittest.TestCase):
    def test_no_thread_before_the_first_submit(self) -> None:
        with TempHome() as home:
            h = _Harness(home, _unique_name("lazy"), 2)
            self.assertEqual(_threads_named(h.name), [])
            self.assertTrue(h.pool.join(0.0))  # an idle pool is drained at once

    def test_first_submit_starts_exactly_size_threads(self) -> None:
        with TempHome() as home:
            h = _Harness(home, _unique_name("lazy"), 2)
            for _ in range(5):
                h.pool.submit(lambda: None)
            self.assertTrue(h.pool.join(WAIT))
            self.assertEqual(sorted(t.name for t in _threads_named(h.name)),
                             [f"tg-{h.name}-0", f"tg-{h.name}-1"])

    def test_size_below_one_is_refused(self) -> None:
        with TempHome() as home:
            with self.assertRaises(ValueError):
                WorkerPool(_unique_name("bad"), 0, Logger(os.path.join(home.state, "bridge.log")))


class SubmitReturnsAtOnceTests(unittest.TestCase):
    def test_media_off_the_poll_thread(self) -> None:
        """1a: submit returns while the job is still running (the poll thread never waits for a download).

        Ported labels:
            1a: handle_message(voice) returned in 0.001s (< 0.3 s, job in background)
        """
        with TempHome() as home:
            h = _Harness(home, _unique_name("media"), 2)
            release, started = threading.Event(), threading.Event()

            def slow_job() -> None:
                started.set()
                release.wait(WAIT)

            t0 = time.monotonic()
            h.pool.submit(slow_job)
            self.assertLess(time.monotonic() - t0, 0.5)
            self.assertTrue(started.wait(WAIT), "the job never started")
            self.assertFalse(h.pool.join(0.0), "submit must not have waited for the job")
            release.set()
            self.assertTrue(h.pool.join(WAIT))

    def test_submit_passes_arguments_to_the_job(self) -> None:
        with TempHome() as home:
            h = _Harness(home, _unique_name("args"), 1)
            seen: list[tuple[object, ...]] = []
            h.pool.submit(lambda *a: seen.append(a), "chat", 7)
            self.assertTrue(h.pool.join(WAIT))
            self.assertEqual(seen, [("chat", 7)])


class ConcurrencyTests(unittest.TestCase):
    def test_media_pool_runs_two_jobs_at_once(self) -> None:
        """Two media workers: both jobs must be inside the barrier together, or the barrier times out."""
        with TempHome() as home:
            h = _Harness(home, _unique_name("pair"), 2)
            barrier = threading.Barrier(2, timeout=WAIT)
            results: list[str] = []
            lock = threading.Lock()

            def meet() -> None:
                barrier.wait()
                with lock:
                    results.append("met")

            h.pool.submit(meet)
            h.pool.submit(meet)
            self.assertTrue(h.pool.join(WAIT))
            self.assertEqual(results, ["met", "met"])

    def test_single_thread_pool_keeps_order_and_never_overlaps(self) -> None:
        """The ordered sender (size 1): replies go out in submission order, one at a time."""
        with TempHome() as home:
            h = _Harness(home, _unique_name("sender"), 1)
            order: list[int] = []
            active, peak = [0], [0]
            lock = threading.Lock()

            def job(i: int) -> None:
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                time.sleep(0.01)
                order.append(i)
                with lock:
                    active[0] -= 1

            for i in range(8):
                h.pool.submit(job, i)
            self.assertTrue(h.pool.join(WAIT))
            self.assertEqual(order, list(range(8)))
            self.assertEqual(peak[0], 1)

    def test_title_jobs_do_not_delay_media(self) -> None:
        """6a/6b: two slow title jobs queued on the title pool; a media job on its own pool still starts at once.

        Ported labels:
            6a: voice transcript reached the turn
            6b: voice turn started in 2.01s while title jobs sleep 3 s each (< 2.9 s)
            6c: both title jobs finished
        """
        with TempHome() as home:
            title = _Harness(home, _unique_name("title"), 1)
            media = _Harness(home, _unique_name("media"), 2)
            gate, media_done = threading.Event(), threading.Event()

            def slow_title() -> None:
                gate.wait(WAIT)

            title.pool.submit(slow_title)
            title.pool.submit(slow_title)
            media.pool.submit(media_done.set)
            self.assertTrue(media_done.wait(2.0), "media job waited behind the title jobs")
            self.assertFalse(title.pool.join(0.0), "the title jobs are still sleeping")
            gate.set()
            self.assertTrue(title.pool.join(WAIT))


class FailureAndJoinTests(unittest.TestCase):
    def test_a_failed_job_is_logged_and_the_pool_keeps_going(self) -> None:
        """A job that raises is logged with its exception; the next job still runs on the same thread."""
        with TempHome() as home:
            h = _Harness(home, _unique_name("fail"), 1)
            ran = threading.Event()

            def boom() -> None:
                raise RuntimeError("boom from job")

            h.pool.submit(boom)
            h.pool.submit(ran.set)
            self.assertTrue(h.pool.join(WAIT))
            self.assertTrue(ran.is_set())
            text = h.log_text()
            self.assertIn(f"{h.name} job failed", text)
            self.assertIn("boom from job", text)

    def test_join_times_out_while_a_job_is_running(self) -> None:
        with TempHome() as home:
            h = _Harness(home, _unique_name("join"), 1)
            release = threading.Event()
            h.pool.submit(release.wait, WAIT)
            self.assertFalse(h.pool.join(0.05))
            release.set()
            self.assertTrue(h.pool.join(WAIT))


if __name__ == "__main__":
    unittest.main()
