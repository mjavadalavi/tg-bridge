"""ViewRecorder: one turn's readable mirror for `tg-bridge view` (a TurnEventBus listener, one per attempt).

Each topic has two files under the view folder (Paths.view, mode 0700): <thread>.feed holds plain terminal lines, one
per shown event, and <thread>.json holds the topic's meta (topic name, transport, launcher, state, updated epoch). The
recorder writes the header and state "running" when it is built, appends plain_line() output for every event, and
writes state "idle" at the turn's end (TurnResult or ProcessExited). finish() writes "idle" too when the attempt ended
without such an event (a launch that failed), so a turn never leaves "running" behind. mark_stale_views() does the same
at daemon start-up for the files a killed daemon left "running" or "lingering". set_state() writes another state (the
kept session of a topic shows "lingering" between its turns).

The feed is capped at FEED_MAX_BYTES: past the cap the newest half is kept, cut at a line break. Both files are written
atomically (a temp file and os.replace), so a reader never sees half a file. Every write is best-effort: an OSError or a
ValueError is logged once and never raised, so a full disk or an odd string cannot stop a turn. The meta is written
with ensure_ascii (a lone surrogate in a launcher name becomes an escape) and the feed is encoded with errors='replace'.

Threads: on_event runs on the backend reader thread; one lock inside the instance guards the files and the state.
"""

import datetime
import json
import os
import threading
from collections.abc import Callable

from tg_bridge.agents.events import AgentEvent, ProcessExited, TurnResult
from tg_bridge.clock import Clock
from tg_bridge.feed.format import plain_line
from tg_bridge.logging_ import Logger

FEED_SUFFIX = ".feed"
META_SUFFIX = ".json"
TMP_SUFFIX = ".tmp"
FEED_MAX_BYTES = 1024 * 1024  # 1 MiB: past this the newest half of the feed is kept
STATE_RUNNING = "running"
STATE_IDLE = "idle"
STATE_LINGERING = "lingering"
HEADER_FORMAT = "%Y-%m-%d %H:%M:%S"  # the local wall time in the header line
HEADER_TEMPLATE = "── {when} · {launcher} ──"
NO_CWD = ""  # the session cwd is not needed for the feed yet (see plain_line)
STALE_STATES = (STATE_RUNNING, STATE_LINGERING)  # the states a killed daemon can leave behind
WRITE_ERRORS = (OSError, ValueError)  # a failed file write, or a string that cannot be encoded or serialised

ViewFactory = Callable[[int, str, str, str], "ViewRecorder"]  # (thread_id, topic, transport, launcher) -> recorder


class ViewRecorder:
    """Writes one topic's feed and meta for the turn it was built for. Build one per attempt, then bus-publish to it."""

    def __init__(self, view_dir: str, thread_id: int, topic: str, transport: str, launcher: str,
                 clock: Clock, logger: Logger) -> None:
        self._stem = os.path.join(view_dir, str(thread_id))
        self._meta = {"topic": topic, "transport": transport, "launcher": launcher}
        self._clock = clock
        self._logger = logger
        self._lock = threading.Lock()
        self._warned = False
        self._state = STATE_RUNNING
        with self._lock:
            self._write_meta_locked(STATE_RUNNING)
            when = datetime.datetime.fromtimestamp(clock.now()).strftime(HEADER_FORMAT)
            self._append_locked(HEADER_TEMPLATE.format(when=when, launcher=launcher) + "\n")

    def on_event(self, event: AgentEvent) -> None:
        line = plain_line(event, NO_CWD)
        with self._lock:
            if line is not None:
                self._append_locked(line + "\n")
            if isinstance(event, (TurnResult, ProcessExited)):
                self._write_meta_locked(STATE_IDLE)

    def finish(self) -> None:
        """Marks the topic idle when this recorder still says running (a launch that failed, an attempt that ended
        early). A recorder that is already idle writes nothing."""
        with self._lock:
            if self._state == STATE_RUNNING:
                self._write_meta_locked(STATE_IDLE)

    def set_state(self, state: str) -> None:
        """Writes the topic's meta with another state, such as STATE_LINGERING, and keeps the other fields."""
        with self._lock:
            self._write_meta_locked(state)

    # the files (every method below runs under the lock)

    def _write_meta_locked(self, state: str) -> None:
        self._state = state
        meta = dict(self._meta, state=state, updated=self._clock.now())
        self._guarded(lambda: _atomic_write(self._stem + META_SUFFIX,
                                            json.dumps(meta, ensure_ascii=True).encode("utf-8")))

    def _append_locked(self, text: str) -> None:
        path = self._stem + FEED_SUFFIX
        data = text.encode("utf-8", errors="replace")  # a lone surrogate becomes '?' instead of an exception
        self._guarded(lambda: _append(path, data))
        self._guarded(lambda: _trim_if_big(path))

    def _guarded(self, write: Callable[[], None]) -> None:
        """Runs one file write; a write error is logged once for this recorder and then swallowed."""
        try:
            write()
        except WRITE_ERRORS as exc:
            if not self._warned:
                self._warned = True
                self._logger.log(f"view: cannot write {self._stem}: {exc!r}")


def mark_stale_views(view_dir: str, clock: Clock, logger: Logger) -> int:
    """Daemon start-up: every <thread>.json still saying 'running' or 'lingering' belongs to a turn or a kept session
    that a killed daemon never ended, so it is rewritten as 'idle'. Returns how many files changed. A bad or unreadable
    file is logged and skipped."""
    try:
        names = sorted(os.listdir(view_dir))
    except OSError as exc:
        logger.log(f"view: cannot list {view_dir}: {exc!r}")
        return 0
    changed = 0
    for name in names:
        if not name.endswith(META_SUFFIX):
            continue
        path = os.path.join(view_dir, name)
        try:
            with open(path, encoding="utf-8") as handle:
                meta = json.load(handle)
            if not isinstance(meta, dict) or meta.get("state") not in STALE_STATES:
                continue
            meta["state"] = STATE_IDLE
            meta["updated"] = clock.now()
            _atomic_write(path, json.dumps(meta, ensure_ascii=True).encode("utf-8"))
            changed += 1
        except WRITE_ERRORS as exc:
            logger.log(f"view: cannot clear the running state of {path}: {exc!r}")
    if changed:
        logger.log(f"view: {changed} topic(s) left running by a stopped daemon set to idle")
    return changed


def _append(path: str, data: bytes) -> None:
    with open(path, "ab") as handle:
        handle.write(data)


def _trim_if_big(path: str) -> None:
    """Keeps the newest half of the feed once it passes FEED_MAX_BYTES, starting at the first whole line of that half."""
    if os.path.getsize(path) <= FEED_MAX_BYTES:
        return
    with open(path, "rb") as handle:
        handle.seek(-(FEED_MAX_BYTES // 2), os.SEEK_END)
        tail = handle.read()
    newline = tail.find(b"\n")
    if newline != -1:
        tail = tail[newline + 1:]
    _atomic_write(path, tail)


def _atomic_write(path: str, data: bytes) -> None:
    """Writes data to path through a temp file and os.replace, so readers see the old or the new file, never half."""
    tmp = path + TMP_SUFFIX
    with open(tmp, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)
