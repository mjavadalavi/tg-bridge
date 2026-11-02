"""Shared helpers of the viewer tests (test_viewer.py, test_viewer_reconcile.py): the session constants and the sync
command, the scripted tmux fake, the fake clocks of the sync and pane tests, and HomeCase, which gives each test a TempHome
with its Paths and a Config that has the default tui size. No test lives here.
"""

import contextlib
import os
import shlex
import sqlite3
import subprocess
import sys
import unittest
from collections.abc import Callable, Sequence

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.claude.hookbridge import ENTRY_SCRIPT
from tg_bridge.config import Config
from tg_bridge.daemon.viewer import (
    THREAD_OPTION,
    WINDOW_FORMAT,
    WINDOW_SEPARATOR,
)
from tg_bridge.paths import Paths


V3_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ENTRY = os.path.join(V3_DIR, ENTRY_SCRIPT)
SESSIONS_SCHEMA = ("CREATE TABLE sessions(thread_id INTEGER PRIMARY KEY, name TEXT, uuid TEXT, launcher TEXT, "
                   "cwd TEXT, created TEXT, title TEXT DEFAULT '')")
THREAD = 5
OTHER_THREAD = 7
SYNC_CMD = shlex.join([sys.executable, "-I", ENTRY, "view-sync"])


def pane_cmd(thread: int) -> str:
    return shlex.join([sys.executable, "-I", ENTRY, "view-pane", str(thread)])


def _done(stdout: str, code: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["tmux"], returncode=code, stdout=stdout, stderr="")


DEFAULT_SIZE = "200x50"  # the default tui size of HomeCase's Config, as WxH


class ScriptedTmux:
    """Records every tmux argv and keeps the windows of the session the way tmux does: the sync window @0, then one window
    per `existing` row (thread tag, size as WxH, name) and one per `tagged` thread (default size, empty name), and one more
    for each new-window. list-windows answers the rows in WINDOW_FORMAT; kill-window, rename-window and set-option of the
    thread tag change the rows, and a name is stored as tmux expands it ('##' is one '#'). has-session
    answers with has_session, list-windows fails with list_code, list-keys prints keys (and exits keys_code). The rows
    are joined with WINDOW_SEPARATOR; raw_listing, when given, is the whole list-windows output instead."""

    def __init__(self, has_session: int = 0, tagged: Sequence[int] = (), list_code: int = 0,
                 existing: Sequence[tuple[str, str, str]] = (), raw_listing: str | None = None,
                 keys: str = "", keys_code: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.has_session = has_session
        self.list_code = list_code
        self.raw_listing = raw_listing
        self.keys = keys            # what list-keys prints (the root MouseDown3Status binding, as tmux writes it)
        self.keys_code = keys_code  # list-keys exit status
        self.windows: dict[str, list[str]] = {"@0": ["", DEFAULT_SIZE, "sync"]}  # window id -> [thread, size, name]
        self.next_id = 1
        for thread in tagged:
            self._new_window(str(thread), DEFAULT_SIZE, "")
        for thread, size, name in existing:
            self._new_window(thread, size, name)

    def _new_window(self, thread: str, size: str, name: str) -> str:
        window_id = f"@{self.next_id}"
        self.next_id += 1
        self.windows[window_id] = [thread, size, name]
        return window_id

    def __call__(self, args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        if args[0] == "has-session":
            return _done("", self.has_session)
        if args[0] == "list-windows":
            if self.list_code:
                return _done("", self.list_code)
            if self.raw_listing is not None:
                return _done(self.raw_listing)
            rows = (f"{window_id}{WINDOW_SEPARATOR}{thread}{WINDOW_SEPARATOR}{size}{WINDOW_SEPARATOR}{name}\n"
                    for window_id, (thread, size, name) in self.windows.items())
            return _done("".join(rows))
        if args[0] == "list-keys":
            return _done(self.keys, self.keys_code)
        if args[0] == "new-session":
            return _done("@0\n")
        if args[0] == "new-window":
            name = args[args.index("-n") + 1].replace("##", "#")
            return _done(f"{self._new_window('', DEFAULT_SIZE, name)}\n")
        if args[0] == "kill-window":
            self.windows.pop(args[args.index("-t") + 1], None)
        if args[0] == "rename-window":
            self.windows[args[args.index("-t") + 1]][2] = args[-1].replace("##", "#")
        if args[0] == "set-option" and THREAD_OPTION in args:
            self.windows[args[args.index("-t") + 1]][0] = args[-1]
        return _done("")

    def named(self, command: str) -> list[list[str]]:
        return [call for call in self.calls if call[0] == command]


class StepClock(FakeClock):
    """A FakeClock whose sleep first runs the next step: each step changes the files between two polls."""

    def __init__(self, steps: Sequence[Callable[[], None]] = ()) -> None:
        super().__init__()
        self.steps = list(steps)

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        if self.steps:
            self.steps.pop(0)()


class InterruptClock(FakeClock):
    """A FakeClock whose first sleep is a Ctrl+C."""

    def sleep(self, seconds: float) -> None:
        raise KeyboardInterrupt


class CrashClock(FakeClock):
    """A FakeClock whose first sleep fails with an error that is not Ctrl+C."""

    def sleep(self, seconds: float) -> None:
        raise RuntimeError("crash")


class HomeCase(unittest.TestCase):
    """A TempHome for each test, with its Paths and a Config that has the default tui size (50 rows, 200 columns)."""

    def setUp(self) -> None:
        home = TempHome()
        home.__enter__()
        self.addCleanup(home.__exit__, None, None, None)
        self.home = home.home
        self.paths = Paths.from_home(home.home)
        self.config = Config({}, home=home.home, env={}, log=lambda _line: None)

    def write_sessions(self, rows: Sequence[tuple[int, str | None]]) -> None:
        with contextlib.closing(sqlite3.connect(self.paths.db)) as db:
            db.execute(SESSIONS_SCHEMA)
            db.executemany("INSERT INTO sessions(thread_id, name) VALUES (?, ?)", rows)
            db.commit()
