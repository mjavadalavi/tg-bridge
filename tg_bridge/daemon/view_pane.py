"""view-pane <thread>: the window of one topic in `tg-bridge view`, with three screens (spec: the three pane screens).

ready: a header (topic, launcher, folder, session id short), the last FEED_TAIL_LINES lines of <thread>.feed, and the
Enter line. Enter claims the topic (core/terminal_claim, under its flock) and runs the real CLI as a child of the pane,
on the pane's own tty: argv = the launcher's argv + --resume <uuid> when the topic has one, else --session-id <new
uuid>, --model when it has one and the permission flags of its mode (permission_args, the tui transport's mapping), and
--settings with one SessionStart hook (`tg-bridge hook session-start <thread> <pid>`) that moves the claim to the
session the tab is on after a /clear or /resume, and autoMemoryDirectory (the topic's memory folder, the one the
bridge's turns use). cwd is the topic's folder. The claim names the pane, the session uuid, the launcher's config folder
and the cwd (core/terminal_claim). The claim is taken unhooked: the SessionStart hook marks it hooked when it runs, so a
hook that never runs (disabled, moved checkout, import error) leaves the claim unhooked and the bridge keeps searching
for the session. When the CLI exits the pane records the session the tab ended on, the one it started with and the ones
it moved off (seen) as the topic's ended record, then removes the claim; the pane is ready again. Once the CLI runs, the
claim also names its child pid (set_child), so the bridge can end the tab's Claude Code.

Number keys 1..9 on ready change the topic's launcher (daemon/view_launcher): the same change as Telegram's /launcher,
under the topic's flock and refused while the topic is busy. The reply of the last change shows until the next key.

busy: the bridge runs a turn for the topic or keeps it lingering (meta state running or lingering). The pane shows the
live mirror of the topic's files (Follower, as run_pane does) with a top line. Enter does nothing.

Keys are read in raw mode, with ISIG off: Ctrl+C is a byte, and only Enter acts. While the CLI runs the terminal is cooked
again, and SIGINT is caught by the pane (the CLI gets its own default handler back on exec), so a Ctrl+C in the CLI
reaches the CLI and not the pane. The terminal attributes are restored when the pane ends.

The topic's row is read with a read-only connection each round, and the config is loaded through the caller's load, so
a launcher is resolved the way TurnRunner resolves it (core/topic_key). A topic whose launcher is not a Claude one is
shown with the reason, and Enter does nothing there.

Threads: the caller's thread only. Lock map: none here; the claim's flock is taken inside terminal_claim and never while
this module writes to the terminal.
"""

import contextlib
import errno
import os
import select
import signal
import sqlite3
import subprocess
import sys
import termios
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol, TextIO

from tg_bridge.agents.claude.session_hook import session_start_command, session_start_settings
from tg_bridge.agents.claude.tui_helpers import permission_args
from tg_bridge.clock import Clock, SystemClock
from tg_bridge.config import BACKEND_CLAUDE, Config, LauncherError, LauncherSpec
from tg_bridge.core import terminal_claim
from tg_bridge.core.topic_key import topic_cwd, topic_launcher, topic_mode, topic_model
from tg_bridge.core.turn_text import TIMEZONE
from tg_bridge.core.view_recorder import FEED_SUFFIX
from tg_bridge.daemon.code_watch import RESTART_STATUS
from tg_bridge.daemon.view_launcher import CHOICE_HINT_LINE, choice_line, claude_choices, switch_launcher
from tg_bridge.daemon.viewer import FOLLOW_SECONDS, Follower, rounds
from tg_bridge.paths import Paths, topic_memory
from tg_bridge.store.sessions import SessionRecord

FEED_TAIL_LINES = 30            # the ready screen shows the last lines of the feed
FEED_TAIL_BYTES = 65536         # how much of the feed's end is read to find those lines
ENTER_KEYS = (b"\r", b"\n")
DIGIT_KEYS = (b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8", b"9")  # the launcher keys, 1..9
CLEAR_SCREEN = "\x1b[2J\x1b[H"
SAVE_CURSOR = "\x1b7"
RESTORE_CURSOR = "\x1b8"
TOP_LINE = "\x1b[1;1H\x1b[2K"   # the first row, cleared; the busy line is drawn there after each replay
BUSY_LINE = "busy in Telegram — Enter opens Claude Code when it is idle"
READY_HINT = "Enter: open this session in Claude Code"
BUSY_MARK = "\x00busy"           # the screen last drawn was the busy one, so the ready screen is drawn again
NO_TOPIC_LINE = "no session row for this topic yet (or the database cannot be read)"
CONFIG_LINE = "the config cannot be read: Enter does nothing until it can"
BUSY_REFUSED_LINE = "busy in Telegram — try again when it is idle"
SWITCH_FAILED_LINE = "the launcher could not be changed: {detail}"
NO_BACKEND_LINE = "this topic runs on the {backend} backend: the terminal view opens Claude sessions only"
START_FAILED_LINE = "could not start the CLI: {detail}"
HEADER_LINE = "tg-bridge view · {name}"
DETAIL_LINE = "launcher: {launcher}   folder: {cwd}   session: {session}"
NEW_SESSION = "new"
SESSION_SHORT = 8               # the uuid's characters the header shows
CLAUDE_ENV_PREFIX = "CLAUDE"    # every CLAUDECODE, CLAUDE_CODE_*, CLAUDE_CONFIG_DIR, CLAUDE_PID, CLAUDE_PLUGIN_* name
BUSY_STATES = terminal_claim.BUSY_STATES


class InputClosed(Exception):
    """The pane's terminal has gone away (end of file, or EIO on a pty whose master closed): no key can arrive again."""


class KeySource(Protocol):
    """Where the pane's keys come from: read(seconds) waits up to seconds and returns the byte read, or b"" for none.
    read raises InputClosed when the input has ended for good."""

    def read(self, seconds: float) -> bytes: ...


RunChild = Callable[[Sequence[str], str, Mapping[str, str]], int]  # argv, cwd, environment -> exit status
ChildStart = Callable[[int], None]                                 # the CLI child's pid, once it runs


@dataclass(frozen=True)
class Topic:
    """What the pane shows and runs for its topic in one round. argv is None when the topic cannot be opened now, and
    problem then says why. name is the topic's name (its thread id when the row has none)."""

    name: str
    launcher: str
    cwd: str
    session: str
    argv: tuple[str, ...] | None
    problem: str
    choices: tuple[str, ...] = ()  # the configured Claude launchers the number keys pick from
    uuid: str = ""                 # the session the CLI runs: the topic's, or the new one a fresh topic starts with
    config_dir: str = ""           # the launcher's Claude config folder, expanded (LauncherSpec.config_dir)


SwitchFn = Callable[[int, str, str], str]  # thread, topic name, launcher name -> the reply text


def read_topic(paths: Paths, thread_id: int) -> SessionRecord | None:
    """The topic's sessions row, read with a read-only connection. None when there is no such row or the database cannot
    be read: the pane then says the topic has no session yet."""
    uri = f"{Path(paths.db).as_uri()}?mode=ro"
    try:
        with contextlib.closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT * FROM sessions WHERE thread_id=?", (thread_id,)).fetchall()
            return SessionRecord.from_row(rows[0]) if rows else None
    except (sqlite3.Error, IndexError):
        return None


def child_argv(config: Config, row: SessionRecord, launcher: LauncherSpec, session_uuid: str,
               settings: str = "") -> tuple[str, ...]:
    """The CLI's argv for the topic: the launcher's argv, --resume <uuid> when the topic has one, else --session-id
    <session_uuid> so the new session's id is known from the start, --model when the topic has one, the permission
    flags of its mode, and --settings <settings> when given. The settings install exactly one SessionStart hook and no
    other hook settings; there is no system prompt."""
    args = [*launcher.argv]
    if row.uuid:
        args += ["--resume", row.uuid]
    else:
        args += ["--session-id", session_uuid]
    model = topic_model(row)
    if model:
        args += ["--model", model]
    args += permission_args(topic_mode(config, row))
    if settings:
        args += ["--settings", settings]
    return tuple(args)


def child_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The CLI's environment, built the way the bridge builds its turns' (turn_runner._spec adds TZ to the daemon's
    environment, and the daemon has no CLAUDE* variable). Every CLAUDE* variable of the pane is dropped: a session the
    pane was started from would hand its config dir, pid or plugin paths to the CLI, which would then resume the topic
    in the wrong place. The bridge sets no config dir of its own; the launcher's command decides it."""
    env = {key: value for key, value in environ.items() if not key.startswith(CLAUDE_ENV_PREFIX)}
    env["TZ"] = TIMEZONE
    return env


def feed_tail(path: str) -> list[str]:
    """The last FEED_TAIL_LINES lines of the feed file, or [] when it is not there. A cut first line is dropped."""
    try:
        with open(path, "rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            handle.seek(max(0, size - FEED_TAIL_BYTES))
            data = handle.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > FEED_TAIL_BYTES:
        lines = lines[1:]
    return lines[-FEED_TAIL_LINES:]


def _no_start(pid: int) -> None:
    """The on_start of a run that needs no pid: nothing is done with it."""


def _run_child(argv: Sequence[str], cwd: str, env: Mapping[str, str], on_start: ChildStart = _no_start) -> int:
    """Runs the CLI on the pane's terminal and returns its exit status. on_start gets the child's pid once it runs, and
    0 once the CLI has exited. A SIGINT during the run is caught here, so the pane survives it while the CLI, which
    gets the default handler back on exec, takes it."""
    previous = signal.signal(signal.SIGINT, lambda _signum, _frame: None)
    try:
        child = subprocess.Popen(list(argv), cwd=cwd, env=dict(env))
        on_start(child.pid)
        return child.wait()
    finally:
        on_start(0)
        signal.signal(signal.SIGINT, previous)


class _Unattended:
    """The keys of a pane whose stdin is not a terminal: none arrive, and each wait is a clock sleep."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def read(self, seconds: float) -> bytes:
        self._clock.sleep(seconds)
        return b""


class _TerminalKeys:
    """The keys of a terminal: one byte, waited for up to the given seconds."""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def read(self, seconds: float) -> bytes:
        ready, _, _ = select.select([self._fd], [], [], seconds)
        if not ready:
            return b""
        try:
            data = os.read(self._fd, 1)
        except OSError as exc:  # a pty whose master closed reads as EIO: select keeps saying readable
            if exc.errno != errno.EIO:
                raise
            raise InputClosed() from exc
        if not data:  # b'' from a readable terminal is end of file; taken as no key, select would say readable at once
            raise InputClosed()
        return data


class _Terminal:
    """The pane's tty: raw (echo, line editing and signals off) while the pane waits for keys, cooked (the attributes it
    had at the start) while the CLI runs and when the pane ends. A stream that is not a terminal changes nothing."""

    def __init__(self, stream: TextIO | None) -> None:
        self._fd = stream.fileno() if stream is not None and stream.isatty() else None
        self._cooked = termios.tcgetattr(self._fd) if self._fd is not None else None
        self._raw = _raw_attributes(self._cooked) if self._cooked is not None else None

    @property
    def fd(self) -> int | None:
        return self._fd

    def raw(self) -> None:
        self._set(self._raw)

    def cooked(self) -> None:
        self._set(self._cooked)

    def _set(self, attributes: list | None) -> None:
        if self._fd is None or attributes is None:
            return
        try:
            termios.tcsetattr(self._fd, termios.TCSANOW, attributes)
        except termios.error:  # the terminal closed while the pane ran: nothing is left to set
            pass


def _raw_attributes(attributes: list) -> list:
    """The attributes with echo, line editing and signals (Ctrl+C) off, and one byte per read. Output processing stays on,
    so a newline still starts a new line."""
    raw = list(attributes)
    raw[3] &= ~(termios.ECHO | termios.ICANON | termios.ISIG)  # index 3: the local flags
    raw[6] = list(raw[6])                                       # index 6: the control characters
    raw[6][termios.VMIN] = 1
    raw[6][termios.VTIME] = 0
    return raw


class ViewPane:
    """The pane of one topic: the three screens, one round per FOLLOW_SECONDS. Holds the paths, the config loader, the
    clock, the output and the key source; the CLI runs through run, which tests may replace."""

    def __init__(self, paths: Paths, thread_id: int, load: Callable[[], Config], out: TextIO, clock: Clock,
                 keys: KeySource, run: RunChild | None = None, switch: SwitchFn | None = None,
                 new_uuid: Callable[[], str] | None = None) -> None:
        self._paths = paths
        self._thread = thread_id
        self._load = load
        self._out = out
        self._clock = clock
        self._keys = keys
        self._run = partial(_run_child, on_start=self._child_started) if run is None else run
        self._switch = partial(switch_launcher, paths, load) if switch is None else switch
        self._new_uuid = (lambda: str(uuid.uuid4())) if new_uuid is None else new_uuid  # a fresh topic's session id
        self._follower = Follower(os.path.join(paths.view, str(thread_id)), out)
        self._drawn = ""       # the last ready screen drawn, or BUSY_MARK
        self._notice = ""      # a line from the last attempt to open the CLI, shown on the ready screen
        self._reply = ""       # the reply of the last launcher change, shown until the next key

    def show(self) -> tuple[bool, Topic]:
        """Draws the screen the topic is in now. Returns (busy, the topic as it was read)."""
        topic = self._topic()
        busy = self._busy()
        if busy:
            self._draw_busy(topic)
        else:
            self._draw_ready(topic)
        return busy, topic

    def _topic(self) -> Topic:
        row = read_topic(self._paths, self._thread)
        if row is None:
            return Topic(str(self._thread), "", "", "", None, NO_TOPIC_LINE)
        name = row.name or str(self._thread)
        session = row.uuid[:SESSION_SHORT] if row.uuid else NEW_SESSION
        try:
            config = self._load()
        except (OSError, ValueError):
            return Topic(name, "", "", session, None, CONFIG_LINE)
        choices = claude_choices(config)
        try:
            launcher = topic_launcher(config, row, name)
        except LauncherError as exc:
            return Topic(name, "", "", session, None, str(exc), choices)
        cwd = topic_cwd(config, row, name)
        if launcher.backend != BACKEND_CLAUDE:
            problem = NO_BACKEND_LINE.format(backend=launcher.backend)
            return Topic(name, launcher.name, cwd, session, None, problem, choices)
        session_uuid = row.uuid or self._new_uuid()
        hook = session_start_command(self._thread, os.getpid())  # the pid take_claim gets too
        settings = session_start_settings(hook, memory_dir=topic_memory(self._paths.memory, self._thread))
        argv = child_argv(config, row, launcher, session_uuid, settings)
        return Topic(name, launcher.name, cwd, session, argv, "", choices, session_uuid, launcher.config_dir)

    def _busy(self) -> bool:
        """True while the bridge owns the topic: the meta state is running or lingering, or its turn mark is there (the
        meta is 'idle' between the result of a turn and the end of its process)."""
        if terminal_claim.meta_state(self._paths.view, self._thread) in BUSY_STATES:
            return True
        return terminal_claim.turn_marked(self._paths.view, self._thread)

    def _draw_ready(self, topic: Topic) -> None:
        feed = feed_tail(os.path.join(self._paths.view, f"{self._thread}{FEED_SUFFIX}"))
        lines = [HEADER_LINE.format(name=topic.name), "",
                 DETAIL_LINE.format(launcher=topic.launcher or "-", cwd=topic.cwd or "-", session=topic.session)]
        if topic.choices:
            lines += [choice_line(topic.choices, topic.launcher), CHOICE_HINT_LINE]
        lines += ["", *feed, "", topic.problem or READY_HINT]
        if self._notice:
            lines.append(self._notice)
        if self._reply:
            lines.append(self._reply)
        screen = "\n".join(lines)
        if screen != self._drawn:
            self._write(CLEAR_SCREEN + screen + "\n")
            self._drawn = screen

    def _draw_busy(self, topic: Topic) -> None:
        if self._drawn != BUSY_MARK:
            self._write(CLEAR_SCREEN)
            self._drawn = BUSY_MARK
        self._follower.tick(topic.name)
        self._write(SAVE_CURSOR + TOP_LINE + BUSY_LINE + RESTORE_CURSOR)

    def handle_key(self, key: bytes, busy: bool, topic: Topic, terminal: _Terminal) -> None:
        """Acts on one key of a round. Any key ends the reply line. Enter opens the CLI and a number key changes the
        launcher, both only on the ready screen: busy is a pre-check only, and the claim and the topic's flock refuse
        the same states again. The reply is not shown on the busy screen, so it is dropped there too."""
        if key or busy:
            self._reply = ""
        if busy or not key:
            return
        if key in ENTER_KEYS and topic.argv is not None:
            self.open_session(topic, terminal)
        elif key in DIGIT_KEYS:
            self.change_launcher(key, topic)

    def change_launcher(self, key: bytes, topic: Topic) -> None:
        """Switches the topic to the launcher numbered by the digit key, under the topic's flock. Refused while the topic is
        busy (the switch is not called then); a number with no launcher is ignored."""
        index = DIGIT_KEYS.index(key)
        if index >= len(topic.choices):
            return
        with terminal_claim.free_topic(self._paths.view, self._thread) as free:
            if not free:
                self._reply = BUSY_REFUSED_LINE
                return
            try:
                self._reply = self._switch(self._thread, topic.name, topic.choices[index])
            except (ValueError, OSError, sqlite3.Error, RuntimeError, KeyError) as exc:
                self._reply = SWITCH_FAILED_LINE.format(detail=exc)

    def open_session(self, topic: Topic, terminal: _Terminal) -> None:
        """Claims the topic and runs the CLI on the pane's tty. A claim refused by the bridge (running, lingering or another
        live claim) changes nothing. The claim is removed when the CLI exits, also when it could not start."""
        view_dir = self._paths.view
        pid = os.getpid()
        fields = dict(pane=os.environ.get("TMUX_PANE", ""), uuid=topic.uuid, config_dir=topic.config_dir, cwd=topic.cwd,
                      socket=os.environ.get("TMUX", "").split(",")[0])  # $TMUX = socket,server pid,session
        if not terminal_claim.take_claim(view_dir, self._thread, pid, self._clock.now(), **fields):
            return
        self._notice = ""
        ran = False
        try:
            terminal.cooked()
            self._run(topic.argv or (), topic.cwd, child_env(os.environ))
            ran = True
        except OSError as exc:
            self._notice = START_FAILED_LINE.format(detail=exc)
        finally:
            terminal.raw()
            try:  # the tab's live claim: a /clear or /resume in the tab moved its uuid and seen (SessionStart hook)
                claim = terminal_claim.own_claim(view_dir, self._thread, pid)
            except OSError:
                claim = None
            if ran:  # the session the CLI ran is recorded, so a topic without a session can adopt it (no DB write here)
                terminal_claim.write_ended(view_dir, self._thread, (claim.uuid if claim else "") or topic.uuid,
                                           topic.config_dir, topic.cwd, self._clock.now(), started_uuid=topic.uuid,
                                           seen=claim.seen if claim else ())
            terminal_claim.release_claim(view_dir, self._thread, pid)
            self._drawn = ""

    def _child_started(self, child: int) -> None:
        """Writes the CLI child's pid into the claim this pane took, so the bridge can end the tab's CLI. A claim that
        is gone or names another pid is left as it is."""
        with contextlib.suppress(OSError):
            terminal_claim.set_child(self._paths.view, self._thread, os.getpid(), child)

    def _write(self, text: str) -> None:
        self._out.write(text)
        self._out.flush()


def run_topic_pane(paths: Paths, thread_id: int, load: Callable[[], Config], out: TextIO | None = None,
                   clock: Clock = SystemClock(), stop_after: int | None = None, stdin: TextIO | None = None,
                   keys: KeySource | None = None, run: RunChild | None = None, switch: SwitchFn | None = None,
                   new_uuid: Callable[[], str] | None = None, code_changed: Callable[[], bool] | None = None) -> int:
    """view-pane <thread>: runs the pane until its window closes. out is stdout when omitted; stdin (stdin when omitted)
    is the terminal the keys come from, unless keys is given. Tests replace run (the CLI runner), switch (the launcher
    change, default view_launcher.switch_launcher), new_uuid (a fresh topic's session id, default uuid4) and stop_after
    (the rounds before the loop ends). code_changed (daemon/code_watch) is asked before each round, when no CLI runs:
    True ends the pane with RESTART_STATUS and the terminal restored, so the caller execs it on the new code. Returns 0
    otherwise."""
    terminal = _Terminal(sys.stdin if stdin is None else stdin)
    source = keys if keys is not None else (
        _TerminalKeys(terminal.fd) if terminal.fd is not None else _Unattended(clock))
    pane = ViewPane(paths, thread_id, load, sys.stdout if out is None else out, clock, source, run, switch, new_uuid)
    try:
        terminal.raw()
        for _ in rounds(stop_after):
            if code_changed is not None and code_changed():
                return RESTART_STATUS
            busy, topic = pane.show()
            pane.handle_key(source.read(FOLLOW_SECONDS), busy, topic, terminal)
    except InputClosed:  # the terminal is gone: no key comes again, so the pane ends instead of spinning on it
        pass
    finally:
        terminal.cooked()
    return 0
