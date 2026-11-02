"""PtyMirror: the raw output of one TUI process, written to a file that `tg-bridge view` replays (the tui transport).

The file is <view>/<thread>.pty. start() writes a new file under a temp name, writes ESC c (a full terminal reset) into
it and moves it into place with os.replace. Every restart is a new inode, so a viewer that replays the file from its start
draws on a blank screen, and it notices the restart even when the new file is longer than the old one.
write() appends every raw chunk the reader gets, escape codes included. Past MIRROR_MAX_BYTES a new file replaces
the old one and starts with the reset again, so the viewer restarts on a blank screen and the file never grows without
limit. The new file holds only the reset and the chunks after it, so the TUI's earlier screen is missing from it; on_rotate,
when given, is called after each such restart (not for the first file), so the TUI can draw its whole screen again.

A failed write or open (a full disk, a missing folder) is logged once and the mirror stops; the turn is not affected.
Threads: write() runs on the PTY reader thread only. close() may come from another thread; a write that races it
fails quietly.
"""

import os
import tempfile
from collections.abc import Callable
from typing import BinaryIO

from tg_bridge.logging_ import Logger

MIRROR_MAX_BYTES = 4 * 1024 * 1024  # a new file replaces the old one once a write would pass this size
RESET = "\x1bc"                     # ESC c: a full terminal reset, first in every file
TEMP_SUFFIX = ".tmp"                # the name a new file has until os.replace moves it into place


class PtyMirror:
    """Appends one process's raw output to one file. A session owns one mirror."""

    def __init__(self, path: str, logger: Logger, on_rotate: Callable[[], None] | None = None) -> None:
        self._path = path
        self._logger = logger
        self._on_rotate = on_rotate  # called on the reader thread after a restart past the cap: it must return at once
        self._file: BinaryIO | None = None
        self._size = 0

    def start(self) -> None:
        """Creates a new file for the session with the reset as its first bytes, replacing any old file at the path.
        Nothing is written when the file cannot be opened."""
        self._open()

    def write(self, text: str) -> None:
        """Appends text, restarting the file first when it would pass MIRROR_MAX_BYTES. A no-op once stopped."""
        if self._file is None:
            return
        data = text.encode("utf-8", errors="replace")
        if self._size + len(data) > MIRROR_MAX_BYTES:
            self._open()  # the viewer sees the reset and starts again on a blank screen
            if self._file is not None and self._on_rotate is not None:
                self._rotated()
        self._append(data)

    def close(self) -> None:
        """Closes the file. Safe to call more than once."""
        file, self._file = self._file, None
        if file is not None:
            try:
                file.close()
            except OSError:
                pass

    def _open(self) -> None:
        """Starts a new file: the reset is written under a temp name, then os.replace moves it onto the path. The open
        handle follows the inode, so later writes land in the file the viewer sees. Caller: start() or write() past the cap."""
        self.close()
        file: BinaryIO | None = None
        temp = ""
        try:
            fd, temp = tempfile.mkstemp(dir=os.path.dirname(self._path) or ".",
                                        prefix=os.path.basename(self._path) + ".", suffix=TEMP_SUFFIX)
            try:
                file = open(fd, "wb")  # mkstemp makes it 0600, as the mirror always was
            except OSError:
                os.close(fd)  # the handle was not wrapped, so nothing else will close it
                raise
            file.write(RESET.encode("ascii"))
            file.flush()
            os.replace(temp, self._path)
        except OSError as e:
            self._fail(e)
            _discard(file, temp)
            return
        self._file = file  # kept open for the life of the session
        self._size = len(RESET)

    def _rotated(self) -> None:
        """Calls on_rotate. A failing callback is logged and does not stop the mirror."""
        try:
            self._on_rotate()
        except Exception as e:
            self._logger.log(f"view mirror {self._path} on_rotate: {e!r}")

    def _append(self, data: bytes) -> None:
        file = self._file
        if file is None:
            return
        try:
            file.write(data)
            file.flush()
            self._size += len(data)
        except OSError as e:
            self._fail(e)
        except ValueError:  # the file was closed by close() on another thread: the write is simply dropped
            pass

    def _fail(self, error: Exception) -> None:
        """Logs the failure and stops the mirror, so a broken file is not written to again."""
        self._logger.log(f"view mirror {self._path} stopped: {error!r}")
        self.close()


def _discard(file: BinaryIO | None, temp: str) -> None:
    """Closes a temp file that did not become the mirror's file and removes it. Both steps may find nothing to do."""
    if file is not None:
        try:
            file.close()
        except OSError:
            pass
    if temp:
        try:
            os.remove(temp)
        except OSError:
            pass
