"""DeleteRequests (card D3): the daemon side of `tg-bridge view-delete <thread>`.

The view's Delete topic menu item runs view-delete, which writes <paths.view>/<thread>.delete (atomically). This watcher
polls the view folder every POLL_SECONDS on a daemon thread. For each request it removes the request file first (so the
request is handled once), writes <thread>.deleting (the acknowledgement: view-delete then knows the daemon is working),
runs TopicRemover.remove(thread, delete_in_telegram=True), writes <thread>.deleted with a one-line human result and
removes <thread>.deleting. view-delete reads the result and removes it.

A name of the request shape (digits, then .delete) that is not canonical ('0012.delete', '0.delete') is removed and
logged: it is never acted on and never left behind. Other names are ignored. A request whose file is already gone is
skipped. A remover that raises still writes a 'could not delete' answer, so view-delete does not wait for the timeout.
stop() ends the thread; App.shutdown_sessions calls it before the sessions are stopped.
"""

import os
import re
import threading

from tg_bridge.core.topic_remover import (ACK_SUFFIX, ALREADY_GONE, DELETED, REQUEST_SUFFIX, RESULT_SUFFIX,
                                          TopicRemover)
from tg_bridge.logging_ import Logger
from tg_bridge.store.sessions import SessionRepo

POLL_SECONDS = 2.0
STOP_JOIN_SECONDS = 5.0
ERROR_TEXT_LIMIT = 200
TMP_SUFFIX = ".tmp"
FAILED_PREFIX = "could not delete:"  # view-delete reports a result that starts with this as a failure
DELETED_TEXT = "deleted topic {name}"
GONE_TEXT = "topic was already gone in Telegram — removed here"
REMOVED_TEXT = "removed topic {name} here"
REMOVED_PREFIX = "topic {thread} removed (telegram: "
REFUSED_PREFIX = "topic {thread} not removed: "
REQUEST_NAME = re.compile(r"([0-9]+)\.delete")  # the shape of a request file name; canonical or not, see request_thread
WORKING_TEXT = "working"  # the one line of the acknowledgement file
DISCARD_LOG = "view delete: {name} is not a canonical request name, removed"
ACK_FAILED_LOG = "view delete: thread {thread} acknowledgement not written: {error}"
ROUND_ERROR_LOG = "view delete: {error}"
FAILED_LOG = "view delete: thread {thread} failed: {error}"


def request_thread(name: str) -> int | None:
    """The thread of a canonical request file name ('12.delete' -> 12). None for every other name: a name of the request
    shape that is not canonical ('0012.delete', '0.delete') is discarded by the watcher, not acted on."""
    match = REQUEST_NAME.fullmatch(name)
    if match is None:
        return None
    digits = match.group(1)
    thread = int(digits)
    return thread if thread > 0 and str(thread) == digits else None


def ack_path(view_dir: str, thread: int) -> str:
    return os.path.join(view_dir, f"{thread}{ACK_SUFFIX}")


def remove_quietly(path: str) -> None:
    """Removes a file, a missing one is fine."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def request_path(view_dir: str, thread: int) -> str:
    return os.path.join(view_dir, f"{thread}{REQUEST_SUFFIX}")


def result_path(view_dir: str, thread: int) -> str:
    return os.path.join(view_dir, f"{thread}{RESULT_SUFFIX}")


def write_atomic(path: str, text: str) -> None:
    """Writes one line through a temp file and os.replace, so a reader sees the old or the new file, never half."""
    tmp = path + TMP_SUFFIX
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text.replace("\n", " ") + "\n")
    os.replace(tmp, path)


def result_text(thread: int, name: str, line: str) -> str:
    """The human answer for one removal, from the line TopicRemover.remove returned."""
    removed = REMOVED_PREFIX.format(thread=thread)
    if line.startswith(removed):
        outcome = line[len(removed):].rstrip(")")
        if outcome == DELETED:
            return DELETED_TEXT.format(name=name)
        if outcome == ALREADY_GONE:
            return GONE_TEXT
        return REMOVED_TEXT.format(name=name)
    refused = REFUSED_PREFIX.format(thread=thread)
    reason = line[len(refused):] if line.startswith(refused) else line
    return f"{FAILED_PREFIX} {reason}"


class DeleteRequests:
    """Takes the view's delete requests and answers each one with a result file."""

    def __init__(self, *, view_dir: str, sessions: SessionRepo, remover: TopicRemover, logger: Logger,
                 interval: float = POLL_SECONDS) -> None:
        self._view_dir = view_dir
        self._sessions = sessions
        self._remover = remover
        self._logger = logger
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Starts the polling thread. Nothing starts when the interval is 0, or when the thread already runs."""
        if self._interval <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="view-delete", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ends the polling thread; a request in progress is waited for only STOP_JOIN_SECONDS."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(STOP_JOIN_SECONDS)

    def run_once(self) -> None:
        """One poll of the view folder. A missing folder has no requests."""
        try:
            names = sorted(os.listdir(self._view_dir))
        except FileNotFoundError:
            return
        for name in names:
            if self._stop.is_set():
                return
            if REQUEST_NAME.fullmatch(name) is None:
                continue
            thread = request_thread(name)
            if thread is None:
                self._discard(name)
            else:
                self._handle(thread)

    def _discard(self, name: str) -> None:
        """A request-shaped name that is not canonical: removed and logged, so it is never left behind."""
        try:
            os.remove(os.path.join(self._view_dir, name))
        except FileNotFoundError:
            return
        self._logger.log(DISCARD_LOG.format(name=name))

    def _handle(self, thread: int) -> None:
        try:
            os.remove(request_path(self._view_dir, thread))  # first: a request is handled once
        except FileNotFoundError:
            return
        self._acknowledge(thread)
        record = self._sessions.get(thread)
        name = record.name if record is not None else str(thread)
        try:
            line = self._remover.remove(thread, True)
        except Exception as exc:  # the answer must still reach view-delete, so it does not wait for the timeout
            self._logger.log(FAILED_LOG.format(thread=thread, error=str(exc)[:ERROR_TEXT_LIMIT]))
            line = REFUSED_PREFIX.format(thread=thread) + str(exc)[:ERROR_TEXT_LIMIT]
        write_atomic(result_path(self._view_dir, thread), result_text(thread, name, line))
        remove_quietly(ack_path(self._view_dir, thread))

    def _acknowledge(self, thread: int) -> None:
        """Tells view-delete that the request is taken. A failed write is logged; the removal goes on regardless."""
        try:
            write_atomic(ack_path(self._view_dir, thread), WORKING_TEXT)
        except OSError as exc:
            self._logger.log(ACK_FAILED_LOG.format(thread=thread, error=str(exc)[:ERROR_TEXT_LIMIT]))

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.run_once()
            except Exception as exc:  # a failing poll is logged; the next one still runs
                self._logger.log(ROUND_ERROR_LOG.format(error=str(exc)[:ERROR_TEXT_LIMIT]))
