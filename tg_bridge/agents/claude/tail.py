"""Follows a growing Claude transcript for the interactive TUI transport (agents/claude/tui_session.py).

The TUI writes its transcript live, while a turn runs, to <config_dir>/projects/<encoded cwd>/<sid>.jsonl: one JSON
object per line. The bridge cannot read that file once and be done, so TranscriptTail polls it on a daemon thread and
hands every complete line to on_line in file order. A partial last line stays buffered until its newline arrives, so a
JSON object is never parsed half-written. Lines of subagents (isSidechain true), blank lines, invalid JSON and values
that are not objects are skipped, so the translator only sees the main conversation. The file may not exist yet when a
session starts, and it may be truncated or replaced; both are handled here and never raised into the caller.
"""

import json
import os
import threading
from collections.abc import Callable

from tg_bridge.logging_ import Logger

POLL_SECONDS = 0.2
STOP_TIMEOUT_SECONDS = 2.0
NEWLINE = b"\n"


class TranscriptTail:
    """Reports each complete JSON object line of one transcript file to on_line, starting at a byte offset.

    The poll runs on one daemon thread. The poll and the final drain of stop() share one re-entrant lock, so a line is
    never delivered twice. on_line runs while that lock is held, so offset and stop() wait for a running handler. A
    stop() called from inside on_line only sets the flag: the batch in progress is delivered in file order, then the
    poll ends.
    """

    def __init__(self, path: str, on_line: Callable[[dict], None], *, logger: Logger, start_offset: int = 0,
                 poll_seconds: float = POLL_SECONDS) -> None:
        self.path = path
        self._on_line = on_line
        self._logger = logger
        self._poll_seconds = poll_seconds
        self._read_pos = start_offset  # bytes read from the file so far, the partial line included
        self._partial = b""  # bytes after the last newline, waiting for the rest of their line
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._deliverer: int | None = None  # thread ident of the poll that is delivering a batch now

    @staticmethod
    def size_of(path: str) -> int:
        """The current size of a transcript, 0 when it does not exist yet. A resumed turn starts at this offset."""
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    @property
    def offset(self) -> int:
        """Bytes of complete lines consumed so far. The partial last line is not counted. Waits for a running on_line."""
        with self._lock:
            return self._read_pos - len(self._partial)

    def start(self) -> None:
        """Start the poll thread. A second call does nothing, so there is never more than one thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="transcript-tail", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = STOP_TIMEOUT_SECONDS) -> None:
        """Stop the poll thread, then drain the complete lines written before this call. Safe to call twice.

        The drain needs the lock, so a running on_line delays it. Called from inside on_line, stop() only sets the flag:
        the batch in progress finishes in file order and no second drain runs on top of it.
        """
        self._stop.set()
        if self._deliverer == threading.get_ident():
            return
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        self._safe_poll()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._safe_poll()
            self._stop.wait(self._poll_seconds)

    def _safe_poll(self) -> None:
        try:
            self._poll()
        except Exception as exc:  # a bad read must not end the tail
            self._logger.log(f"transcript tail {self.path} failed: {exc}")

    def _poll(self) -> None:
        with self._lock:
            try:
                f = open(self.path, "rb")
            except OSError:
                return  # the transcript does not exist yet
            with f:
                if os.fstat(f.fileno()).st_size < self._read_pos:
                    self._logger.log(f"transcript {self.path} is shorter than offset {self._read_pos}; "
                                     "restarting from 0")
                    self._read_pos = 0
                    self._partial = b""
                f.seek(self._read_pos)
                chunk = f.read()
            if not chunk:
                return
            self._read_pos += len(chunk)
            data = self._partial + chunk
            cut = data.rfind(NEWLINE)
            if cut < 0:
                self._partial = data
                return
            self._partial = data[cut + 1:]
            self._deliverer = threading.get_ident()
            try:
                for raw in data[:cut].split(NEWLINE):
                    self._deliver(raw)
            finally:
                self._deliverer = None

    def _deliver(self, raw: bytes) -> None:
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return
        try:
            obj = json.loads(text)
        except (ValueError, RecursionError):  # json raises RecursionError for a line nested too deep
            return
        if not isinstance(obj, dict) or obj.get("isSidechain") is True:
            return
        try:
            self._on_line(obj)
        except Exception as exc:  # one bad line must not stop the lines after it
            self._logger.log(f"transcript line handler failed: {exc}")
