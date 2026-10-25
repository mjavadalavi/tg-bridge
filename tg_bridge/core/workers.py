"""WorkerPool(name, size, logger): lazy daemon threads over one FIFO queue (C16).

Port of the live file's media workers (_media_q + MEDIA_WORKERS = 2), the ordered sender (_out_q, one thread, so
replies keep their order and the poll thread never waits for the send budget) and the one-time title worker
(_title_q). Each pool owns its queue, its threads and its lock; nothing is a module global.

- Threads start on the first submit(), not at construction.
- size 1 gives strict FIFO order: a job starts only after the previous one returned.
- A job that raises is logged and the thread keeps going, so one bad job never stops the pool.
- join(timeout) waits until every submitted job has finished; it exists for tests.
"""

import functools
import queue
import threading
from collections.abc import Callable

from tg_bridge.logging_ import Logger


class WorkerPool:
    """Runs submitted callables on `size` daemon threads. Thread-safe: any thread may submit."""

    def __init__(self, name: str, size: int, logger: Logger) -> None:
        if size < 1:
            raise ValueError("a WorkerPool needs at least one thread")
        self.name = name
        self.size = size
        self._logger = logger
        self._queue: queue.Queue[Callable[[], object]] = queue.Queue()
        self._lock = threading.Lock()  # guards _started only
        self._started = False

    def submit(self, fn: Callable[..., object], *args: object) -> None:
        """Queue fn(*args) and return at once. The threads are started here on first use."""
        self._ensure_started()
        self._queue.put(functools.partial(fn, *args))

    def join(self, timeout: float | None = None) -> bool:
        """Wait until every submitted job has finished. True when the pool drained, False when the timeout ran out."""
        with self._queue.all_tasks_done:
            return self._queue.all_tasks_done.wait_for(lambda: self._queue.unfinished_tasks == 0, timeout)

    def _ensure_started(self) -> None:
        with self._lock:
            if self._started:
                return
            for i in range(self.size):
                threading.Thread(target=self._run, daemon=True, name=f"tg-{self.name}-{i}").start()
            self._started = True

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                job()
            except Exception as exc:
                self._logger.log(f"{self.name} job failed: {exc!r}")
            finally:
                self._queue.task_done()
