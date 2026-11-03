"""CodeWatch: tells a long-lived view process that the tg_bridge code on disk changed since it started.

A view pane lives as long as its tmux window, often for days. Python keeps the code it imported at start, so a pane
started before an upgrade keeps the old code: its terminal claim then lacks the fields a newer daemon needs (pane,
uuid), and a Telegram message for the topic waits for the tab to close instead of going into it. The pane asks
changed() between rounds, when no CLI runs, and ends with RESTART_STATUS; its caller then execs it again on the new code
(reexec).

The fingerprint is the newest mtime of the package's .py files. It is read at most once per CHECK_SECONDS of the
clock's monotonic time, and the first fingerprint is taken when the watch is made.

Threads: the caller's thread only. Lock map: none.
"""

import os
import sys
from collections.abc import Callable, Sequence

from tg_bridge.clock import Clock

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # the tg_bridge folder
CHECK_SECONDS = 5.0
RESTART_STATUS = 75  # the pane's return when the code changed: exec it again (EX_TEMPFAIL)


def fingerprint(root: str = PACKAGE_DIR) -> float:
    """The newest mtime of the .py files under root, 0.0 when none can be read. A file that goes while it is read is
    skipped."""
    newest = 0.0
    for folder, _, names in os.walk(root):
        for name in names:
            if not name.endswith(".py"):
                continue
            try:
                newest = max(newest, os.stat(os.path.join(folder, name)).st_mtime)
            except OSError:
                continue
    return newest


class CodeWatch:
    """The code fingerprint taken at start and the test against it. read replaces the fingerprint function (tests)."""

    def __init__(self, clock: Clock, read: Callable[[], float] = fingerprint,
                 check_seconds: float = CHECK_SECONDS) -> None:
        self._clock = clock
        self._read = read
        self._check_seconds = check_seconds
        self._start = read()
        self._checked = clock.monotonic()

    def changed(self) -> bool:
        """True when the code on disk is newer than at start. Reads the files only once per check_seconds; False in
        between."""
        now = self._clock.monotonic()
        if now - self._checked < self._check_seconds:
            return False
        self._checked = now
        return self._read() > self._start


def reexec(execv: Callable[[str, Sequence[str]], object] = os.execv) -> None:
    """Replaces this process with the same command line (interpreter flags included) on the code now on disk."""
    execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])
