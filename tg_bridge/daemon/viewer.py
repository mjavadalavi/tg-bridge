"""Viewer: the `tg-bridge view` side (B3). One tmux session "tg-bridge" holds a read-only window per Telegram topic.

list_topics reads the sessions table with a read-only connection (Database would migrate). Viewer.ensure_session makes
the session with a "sync" window that runs view-sync. view-sync is the only creator of topic windows: run_sync loads the
config once, and each round reconciles the windows with the topics. It closes the extra windows of a topic that has two,
closes the windows of topics that are gone, renames the windows of renamed topics, adds a window for each topic that has
none, and makes each window follow the size of the terminal attached to it (window-size latest), once per window. A
round whose topics cannot be read changes no topic window.
run_pane follows the topic's files under Paths.view and copies them to the terminal. A tui topic replays <thread>.pty (the
raw output), a print topic follows <thread>.feed (plain lines). The windows are read-only: run_pane never reads the
keyboard, turns echo and line editing off on a terminal while it runs, and drops whatever input arrives, so typed keys and
the answers to the replayed TUI's queries never reach the screen. attach() replaces the process with `tmux attach`, or with
`tmux switch-client` inside tmux.

Threads: none of its own. Every call runs on the caller's thread, and the tmux runner is injected, so tests script tmux.
"""

import contextlib
import itertools
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import termios
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple, TextIO

from tg_bridge.agents.claude.hookbridge import ENTRY_SCRIPT
from tg_bridge.agents.pty_mirror import RESET
from tg_bridge.clock import Clock, SystemClock
from tg_bridge.config import TRANSPORT_TUI, Config
from tg_bridge.core.view_recorder import FEED_SUFFIX, META_SUFFIX
from tg_bridge.daemon.panes import TmuxRunner, run_tmux, seat_window_name
from tg_bridge.daemon.view_menu import DELETE_MARKER, MENU_FILE, MENU_KEY, menu_config, original_binding
from tg_bridge.paths import Paths

SESSION = "tg-bridge"            # the tmux session the views live in
SESSION_TARGET = "=tg-bridge"    # exact session match (has-session, list-windows, attach): tg-bridge2 never matches
SESSION_TARGET_COLON = "=tg-bridge:"  # the session's window list (new-window) and its options (set-option): tmux refuses "=name" there
SYNC_WINDOW = "sync"             # the window that runs view-sync
THREAD_OPTION = "@tg_thread"     # the window option that tags a topic's window with its thread id
WINDOW_SEPARATOR = "|"           # between the fields of a row: a tab comes out as "_" on a non-UTF-8 tmux server
WINDOW_FORMAT = "#{window_id}|#{@tg_thread}|#{window_width}x#{window_height}|#{window_name}"  # list-windows: one row each
WINDOW_ID_FORMAT = "#{window_id}"  # new-session / new-window: the id of the window they made, such as @3
TMUX_ENV = "TMUX"                # set when this process already runs inside tmux
PTY_SUFFIX = ".pty"              # the raw mirror of a tui topic, written by tui_session
SYNC_SECONDS = 5                 # how often view-sync runs a round
FOLLOW_SECONDS = 1               # how often view-pane reads the topic's files
TMUX_TIMEOUT = 10                # seconds for one tmux call
ECHO_OFF_MASK = termios.ECHO | termios.ICANON  # the local flags run_pane clears: no echo, no line editing
LOCAL_FLAGS = 3                  # index of the local-modes flags in termios.tcgetattr's list
TOPICS_QUERY = "SELECT thread_id, name FROM sessions WHERE name IS NOT NULL AND name <> '' ORDER BY thread_id"
SYNC_ADDED = "view: window {window} added"
WAITING = "waiting for the first turn of {name}…"

Topic = tuple[int, str]  # (thread_id, name)


class _Window(NamedTuple):
    """One row of the window listing: the window id, its thread tag ("" for the sync window), its size as WxH, its name."""

    window_id: str
    thread: str
    size: str
    name: str


def list_topics(paths: Paths) -> list[Topic] | None:
    """The named topics from the sessions table, read-only. None when the database cannot be read (no file, not SQLite,
    no sessions table), so a caller changes no window. An empty list means the table has no named topic."""
    uri = f"{Path(paths.db).as_uri()}?mode=ro"
    try:
        with contextlib.closing(sqlite3.connect(uri, uri=True)) as connection:
            rows = connection.execute(TOPICS_QUERY).fetchall()
    except sqlite3.Error:
        return None
    return [(int(thread_id), name) for thread_id, name in rows]


class Viewer:
    """Builds the tmux session and reconciles its windows with the topics. Holds the paths, the config (for the tui size
    of a new session), the tmux runner and the ids of the windows already set to follow the terminal's size."""

    def __init__(self, paths: Paths, config: Config, run: TmuxRunner = run_tmux) -> None:
        self._paths = paths
        self._config = config
        self._run = run
        self._following: set[str] = set()  # window ids whose window-size is already latest, in this process

    def ensure_session(self) -> None:
        """Makes the tg-bridge session with its sync window when it does not exist yet, with the status line at the bottom
        and the mouse on (a click picks a tab). An existing session keeps its windows; only the mouse option is set on it
        too, since a session made before that option existed never had it."""
        result = self._tmux(["has-session", "-t", SESSION_TARGET])
        if result is not None and result.returncode == 0:
            self._tmux(["set-option", "-t", SESSION_TARGET_COLON, "mouse", "on"])
            self._bind_delete_menu()
            return
        rows, cols = self._config.tui_size()
        made = self._tmux(["new-session", "-d", "-s", SESSION, "-n", SYNC_WINDOW, "-x", str(cols), "-y", str(rows),
                           "-P", "-F", WINDOW_ID_FORMAT, _command("view-sync")])
        if made is None or made.returncode != 0:
            return
        self._follow_size(made.stdout.strip())
        self._tmux(["set-option", "-t", SESSION_TARGET_COLON, "status-position", "bottom"])
        self._tmux(["set-option", "-t", SESSION_TARGET_COLON, "mouse", "on"])  # clicks pick a tab; the pane reads no mouse
        self._bind_delete_menu()

    def _bind_delete_menu(self) -> None:
        """Sources the tab menu file (view_menu) unless the root MouseDown3Status binding already has its Delete item. A
        listing that fails or has no such binding, and a file that cannot be written, change nothing: the view still works."""
        listing = self._tmux(["list-keys", "-T", "root", MENU_KEY])
        if listing is None or listing.returncode != 0 or DELETE_MARKER in listing.stdout:
            return
        original = original_binding(listing.stdout)
        config = None if original is None else menu_config(original, _command("view-delete"), SESSION)
        if config is None:
            return
        path = os.path.join(self._paths.view, MENU_FILE)
        try:
            os.makedirs(self._paths.view, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(config)
        except OSError:
            return
        self._tmux(["source-file", path])

    def sync_windows(self) -> list[str]:
        """One reconcile round. Windows are matched by their thread id tag, not by their name, because tmux expands the
        name and a non-UTF-8 server changes it. Returns the names of the windows it added. When the windows cannot be
        listed or parsed nothing changes; when the topics cannot be read, only the duplicates are closed and the windows
        set to follow the terminal."""
        listed = self._windows()
        if listed is None:
            return []  # the listing cannot be read: the windows are not touched
        windows = self._close_duplicates(listed)
        added: list[str] = []
        topics = list_topics(self._paths)
        if topics is not None:
            windows = self._close_removed(windows, {str(thread_id) for thread_id, _ in topics})
            self._rename_changed(windows, topics)
            added = self._add_missing(windows, topics)
        self._follow_windows(windows)
        return added

    def _windows(self) -> list[_Window] | None:
        """The windows of the session, one row each. None when tmux cannot list them or a row does not parse, so that a
        listing that is not the one asked for is never taken as "no window"."""
        listing = self._tmux(["list-windows", "-t", SESSION_TARGET, "-F", WINDOW_FORMAT])
        if listing is None or listing.returncode != 0:
            return None
        rows: list[_Window] = []
        for line in listing.stdout.splitlines():
            fields = line.split(WINDOW_SEPARATOR, 3)  # the name comes last, so it may hold any separator
            if len(fields) != 4:
                return None
            rows.append(_Window(*fields))
        return rows

    def _close_duplicates(self, windows: list[_Window]) -> list[_Window]:
        """Of the windows that share a thread tag, the one with the lowest window id stays and the others are closed.
        Windows without a tag (the sync window, any window of the user) are never closed."""
        kept: list[_Window] = []
        seen: set[str] = set()
        for window in sorted(windows, key=lambda item: int(item.window_id.lstrip("@"))):
            if not window.thread:
                kept.append(window)
            elif window.thread in seen:
                self._tmux(["kill-window", "-t", window.window_id])
            else:
                seen.add(window.thread)
                kept.append(window)
        return kept

    def _close_removed(self, windows: list[_Window], current: set[str]) -> list[_Window]:
        """Closes the tagged windows whose topic is not one of current (the thread ids of the sessions table)."""
        kept: list[_Window] = []
        for window in windows:
            if window.thread and window.thread not in current:
                self._tmux(["kill-window", "-t", window.window_id])
            else:
                kept.append(window)
        return kept

    def _rename_changed(self, windows: list[_Window], topics: Sequence[Topic]) -> None:
        """Renames the window of each topic whose name is not the topic's current seat name. On a non-UTF-8 tmux server a
        non-ASCII name shows as "_" and is renamed again each round: harmless, and fixing it needs the server's encoding."""
        tagged = {window.thread: window for window in windows if window.thread}
        for thread_id, name in topics:
            window = tagged.get(str(thread_id))
            if window is not None and window.name != seat_window_name(name):
                self._tmux(["rename-window", "-t", window.window_id, _tmux_name(name)])

    def _add_missing(self, windows: list[_Window], topics: Sequence[Topic]) -> list[str]:
        """Adds a window for each topic that has none. Returns the names of the windows it added."""
        tagged = {window.thread for window in windows if window.thread}
        added: list[str] = []
        for thread_id, name in topics:
            if str(thread_id) in tagged:
                continue
            made = self._tmux(["new-window", "-d", "-P", "-F", WINDOW_ID_FORMAT, "-t", SESSION_TARGET_COLON,
                               "-n", _tmux_name(name), _command("view-pane", str(thread_id))])
            if made is None or made.returncode != 0:
                continue
            window_id = made.stdout.strip()
            self._tmux(["set-option", "-w", "-t", window_id, THREAD_OPTION, str(thread_id)])
            self._follow_size(window_id)
            added.append(seat_window_name(name))
        return added

    def _follow_windows(self, windows: list[_Window]) -> None:
        """Makes each remaining window follow the terminal attached to it. The sync window is one of them, and so are
        windows made by an older version, which still have a fixed size."""
        for window in windows:
            self._follow_size(window.window_id)

    def _follow_size(self, window_id: str) -> None:
        """Sets window-size latest on a window, so tmux sizes it to the terminal attached to it and not to tui_size. Each
        window is set once per process: a window already done is skipped, and one whose option could not be set is tried
        again in the next round."""
        if window_id in self._following:
            return
        made = self._tmux(["set-option", "-w", "-t", window_id, "window-size", "latest"])
        if made is not None and made.returncode == 0:
            self._following.add(window_id)

    def _tmux(self, args: Sequence[str]) -> subprocess.CompletedProcess[str] | None:
        """One tmux call. None when tmux cannot run; a non-zero exit is returned for the caller to read."""
        try:
            return self._run(list(args), TMUX_TIMEOUT)
        except (OSError, subprocess.SubprocessError):
            return None


def run_sync(paths: Paths, load: Callable[[], Config], clock: Clock, stop_after: int | None = None,
             run: TmuxRunner = run_tmux) -> int:
    """view-sync: a round at once and then every SYNC_SECONDS, each printing one line per window it added. load is called
    once. One Viewer serves every round, so each window is set to follow the terminal once for the whole run. Returns 0."""
    viewer = Viewer(paths, load(), run)
    try:
        for _ in rounds(stop_after):
            for window in viewer.sync_windows():
                print(SYNC_ADDED.format(window=window), flush=True)
            clock.sleep(SYNC_SECONDS)
    except KeyboardInterrupt:
        pass
    return 0


def run_pane(paths: Paths, thread_id: int, out: TextIO | None = None, clock: Clock = SystemClock(),
             stop_after: int | None = None, stdin: TextIO | None = None) -> int:
    """view-pane: follows one topic until its window closes or Ctrl+C. Writes to out (stdout when omitted). When stdin
    (stdin when omitted) is a terminal, its echo is off and its input is flushed every round, and its attributes are
    restored on the way out. Returns 0."""
    follower = Follower(os.path.join(paths.view, str(thread_id)), sys.stdout if out is None else out)
    name = dict(list_topics(paths) or []).get(thread_id, str(thread_id))
    with _quiet_terminal(sys.stdin if stdin is None else stdin) as fd:
        try:
            for _ in rounds(stop_after):
                if fd is not None:
                    _drop_input(fd)  # unread input is dropped, never piles up
                follower.tick(name)
                clock.sleep(FOLLOW_SECONDS)
        except KeyboardInterrupt:
            pass
    return 0


@contextlib.contextmanager
def _quiet_terminal(stream: TextIO | None) -> Iterator[int | None]:
    """Turns echo and line editing off on stream while the block runs and restores the old attributes after it, even when
    the block raises. Yields the file descriptor, or None when stream is not a terminal (nothing is changed then)."""
    if stream is None or not stream.isatty():  # a detached process has no stdin
        yield None
        return
    fd = stream.fileno()
    saved = termios.tcgetattr(fd)
    quiet = list(saved)
    quiet[LOCAL_FLAGS] &= ~ECHO_OFF_MASK
    termios.tcsetattr(fd, termios.TCSANOW, quiet)
    try:
        yield fd
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSANOW, saved)
        except termios.error:  # the terminal closed while the pane ran: there is nothing left to restore
            pass


def _drop_input(fd: int) -> None:
    """Discards the input waiting on fd. A terminal that closed during the round has no input to drop."""
    try:
        termios.tcflush(fd, termios.TCIFLUSH)
    except termios.error:
        pass


def attach(execvp: Callable[[str, Sequence[str]], object] = os.execvp,
           env: Mapping[str, str] | None = None) -> None:
    """Replaces this process with `tmux attach -t =tg-bridge`, or with `tmux switch-client` when TMUX is set (already
    inside tmux). execvp and env (os.environ when omitted) are injectable for tests."""
    environ = os.environ if env is None else env
    if TMUX_ENV in environ:
        execvp("tmux", ["tmux", "switch-client", "-t", SESSION_TARGET])
    else:
        execvp("tmux", ["tmux", "attach", "-t", SESSION_TARGET])


class Follower:
    """Follows one topic's files for run_pane. Keeps the file it reads, its inode and how many of its bytes are written
    out. A file replaced (a new inode) or shorter than what was read, and a switch of suffix, each start the terminal
    again: RESET, then the file from its beginning."""

    def __init__(self, stem: str, out: TextIO) -> None:
        self._stem = stem
        self._out = out
        self._suffix = ""     # the file being followed: PTY_SUFFIX or FEED_SUFFIX
        self._inode: int | None = None  # the inode of that file at the last look, None before its first look
        self._offset = 0      # bytes of that file already written out
        self._waiting = False  # the waiting line has been printed

    def tick(self, name: str) -> None:
        """One look at the topic: prints the waiting line until the meta exists, then copies what the files gained."""
        meta = _read_meta(self._stem + META_SUFFIX)
        if meta is None:
            if not self._waiting:
                self._waiting = True
                self._say(WAITING.format(name=name))
            return
        suffix = PTY_SUFFIX if meta.get("transport") == TRANSPORT_TUI else FEED_SUFFIX
        if suffix != self._suffix:
            if self._suffix:  # a switch between the feed and the raw mirror: the screen is cleared before the replay
                self._restart()
            self._suffix, self._inode, self._offset = suffix, None, 0
        self._follow(self._stem + suffix, raw=suffix == PTY_SUFFIX)

    def _follow(self, path: str, raw: bool) -> None:
        try:  # open first: the inode and the size below are those of the file this handle reads, even if it is replaced
            handle = open(path, "rb")
        except OSError:  # the file of this kind is not written yet
            return
        with handle:
            status = os.fstat(handle.fileno())
            if self._inode is not None and (status.st_ino != self._inode or status.st_size < self._offset):
                # replaced (a new inode) or shorter than what was read: clear the screen, read the file from its start
                self._restart()
            self._inode = status.st_ino
            size = status.st_size
            if size == self._offset:
                return
            handle.seek(self._offset)
            data = handle.read(size - self._offset)
        if not raw:  # plain lines: only whole lines are written, a partial line waits for its newline
            data = data[:data.rfind(b"\n") + 1]
        self._offset += len(data)
        if data:
            self._emit(data)

    def _restart(self) -> None:
        """Clears the terminal and starts the followed file again from its beginning."""
        self._emit(RESET.encode("ascii"))
        self._offset = 0

    def _emit(self, data: bytes) -> None:
        """Writes bytes to the terminal as they are (the raw output keeps its escape codes)."""
        self._out.flush()
        buffer = getattr(self._out, "buffer", None)
        if buffer is None:
            self._out.write(data.decode("utf-8", errors="replace"))
            self._out.flush()
        else:
            buffer.write(data)
            buffer.flush()

    def _say(self, text: str) -> None:
        self._out.write(text + "\n")
        self._out.flush()


def _tmux_name(name: str) -> str:
    """The name tmux gets for a topic's window: the seat name with each '#' doubled, since tmux expands formats in names."""
    return seat_window_name(name).replace("#", "##")


def _command(*words: str) -> str:
    """The shell line a tmux window runs: this interpreter, -I, this checkout's tg-bridge.py and the given words."""
    return shlex.join([sys.executable, "-I", _entry_script(), *words])


def _entry_script() -> str:
    """tg-bridge.py in the checkout that holds this package. This file is tg_bridge/daemon/viewer.py, so the checkout
    is three directories up."""
    here = os.path.realpath(__file__)
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    return os.path.join(root, ENTRY_SCRIPT)


def _read_meta(path: str) -> dict | None:
    """The topic's meta as a dict, or None when the file is missing or not a JSON object."""
    try:
        with open(path, encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def rounds(stop_after: int | None) -> Iterable[int]:
    """The loop counter: endless when stop_after is None, else stop_after rounds (for tests)."""
    return itertools.count() if stop_after is None else range(stop_after)
