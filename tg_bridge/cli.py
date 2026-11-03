"""Command line dispatch (STEP 54): the subcommands of the live file's main(), with the same argv and exit codes.

  (no subcommand) or daemon          the daemon: lock, App, start-up recovery, inbox replay, poll loop
  send <topic> <text>                one message to a known topic
  sendfile | send-file <topic> <path> [caption]
  type <pane> <text>                 type text into a live tmux pane (no App, no database)
  spawn <command> <name> <text>      open a live seat in a new tmux window (no App, no database)
  hook permission                    the TUI's PermissionRequest hook: stdin JSON to the session socket (no App)
  hook session-start <thread> <pid>  the view tab's SessionStart hook: moves the tab's claim to the session (no App)
  view                               opens the tg-bridge tmux session: one read-only window per topic (no App)
  view-sync / view-pane <thread>     the view's own helpers, run by its windows (no App)
  view-delete <thread>               the view's Delete topic: the daemon deletes the topic here and in Telegram (no App)
  anything else                      prints the spec (tg_bridge.__doc__) and exits 0

main() sets the umask to 0o077 first, as the live file did. A failure a command reports (unknown topic, a refused upload,
a pane that did not take the text, missing arguments) exits 1. The stop signal (SIGTERM or SIGINT) exits 0 after the
daemon has stopped its sessions. A SystemExit that carries a message prints it to stderr and exits 1, as Python does.
"""

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeVar

import tg_bridge
from tg_bridge.app import App
from tg_bridge.clock import Clock, SystemClock
from tg_bridge.config import Config
from tg_bridge.daemon.inbox import GROUP_KEY
from tg_bridge.daemon.lifecycle import acquire_lock, ignore_stop_signals, install_stop_handlers
from tg_bridge.daemon.panes import Panes
from tg_bridge.daemon.permission_hook import run_hook_client
from tg_bridge.daemon.session_hook import run_session_start_hook
from tg_bridge.daemon.view_delete import request_topic_delete, show_in_tmux
from tg_bridge.daemon.code_watch import RESTART_STATUS, CodeWatch, reexec
from tg_bridge.daemon.view_pane import run_topic_pane
from tg_bridge.daemon.viewer import Viewer, attach, run_sync
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.telegram.upload import UploadRefused

T = TypeVar("T")

UMASK = 0o077
DAEMON_START_LOG = "daemon start"
DAEMON_STOP_LOG = "daemon stop: {count} session(s) signalled"
UNKNOWN_TOPIC = "tg-bridge {command}: unknown topic {topic!r} or group not learned yet"
UPLOAD_REFUSED = "tg-bridge sendfile: {detail}"
VIEW_NO_TMUX = "tg-bridge view: tmux did not run: {detail}"
VIEW_THREAD_NOT_A_NUMBER = "tg-bridge view-pane: the thread must be a number, not {thread!r}"
VIEW_DELETE_NOT_A_NUMBER = "tg-bridge view-delete: the thread must be a number, not {thread!r}"
USAGE = MappingProxyType({
    "send": "usage: tg-bridge send <topic> <text>",
    "sendfile": "usage: tg-bridge sendfile <topic> <path> [caption]",
    "type": "usage: tg-bridge type <pane> <text>",
    "spawn": "usage: tg-bridge spawn <command> <name> <text>",
    "hook": "usage: tg-bridge hook permission | tg-bridge hook session-start <thread> <pid>",
    "view-pane": "usage: tg-bridge view-pane <thread>",
    "view-delete": "usage: tg-bridge view-delete <thread>",
})
# subcommand -> number of arguments it needs; extra arguments are ignored, as the live file ignored them
REQUIRED_ARGUMENTS = MappingProxyType({"send": 2, "sendfile": 2, "type": 2, "spawn": 3, "view-pane": 1,
                                       "view-delete": 1})


@dataclass(frozen=True)
class Runners:
    """What each subcommand does. main() wires the real ones; route() only picks one and passes it the arguments."""

    daemon: Callable[[], None]
    send: Callable[[str, str], int]
    sendfile: Callable[[str, str, str], int]
    type_text: Callable[[str, str], int]
    spawn: Callable[[str, str, str], int]
    hook_permission: Callable[[], int]
    hook_session_start: Callable[[str, str], int]
    view: Callable[[], int]
    view_sync: Callable[[], int]
    view_pane: Callable[[str], int]
    view_delete: Callable[[str], int]


def route(args: Sequence[str], runners: Runners) -> int:
    """Picks the subcommand named by args[0] and calls its runner with the arguments. Returns the exit status."""
    if not args or args[0] == "daemon":
        runners.daemon()
        return 0
    command, rest = args[0], list(args[1:])
    if command == "send":
        return _with_arguments(command, rest, lambda: runners.send(rest[0], rest[1]))
    if command in ("sendfile", "send-file"):
        caption = rest[2] if len(rest) > 2 else ""
        return _with_arguments("sendfile", rest, lambda: runners.sendfile(rest[0], rest[1], caption))
    if command == "type":
        return _with_arguments(command, rest, lambda: runners.type_text(rest[0], rest[1]))
    if command == "spawn":
        return _with_arguments(command, rest, lambda: runners.spawn(rest[0], rest[1], rest[2]))
    if command == "hook":
        if rest[:1] == ["permission"]:
            return runners.hook_permission()
        if rest[:1] == ["session-start"]:
            return _with_arguments("hook", rest[1:], lambda: runners.hook_session_start(rest[1], rest[2]),
                                   required=2)
        print(USAGE["hook"], file=sys.stderr)
        return 1
    if command == "view":
        return runners.view()
    if command == "view-sync":
        return runners.view_sync()
    if command == "view-pane":
        return _with_arguments(command, rest, lambda: runners.view_pane(rest[0]))
    if command == "view-delete":
        return _with_arguments(command, rest, lambda: runners.view_delete(rest[0]))
    sys.stdout.write(tg_bridge.__doc__)
    return 0


def _with_arguments(command: str, rest: Sequence[str], call: Callable[[], int], required: int | None = None) -> int:
    """Runs the subcommand when it has the arguments it needs; otherwise prints its usage line and exits 1."""
    if len(rest) < (REQUIRED_ARGUMENTS[command] if required is None else required):
        print(USAGE[command], file=sys.stderr)
        return 1
    return call()


def _topic_target(app: App, topic: str, command: str) -> tuple[int, str]:
    """The thread of a known topic and the group id the bridge answers in. SystemExit when either is unknown."""
    thread = app.sessions.thread_of(topic)
    group = app.kv.get(GROUP_KEY)
    if thread is None or not group:
        raise SystemExit(UNKNOWN_TOPIC.format(command=command, topic=topic))
    return thread, group


def send_to_topic(app: App, topic: str, text: str) -> int:
    """`send`: one plain message to a topic. Telegram failures are logged by the messenger, as the live file did."""
    thread, group = _topic_target(app, topic, "send")
    app.messenger.send_text(int(group), thread, text)
    return 0


def send_file_to_topic(app: App, topic: str, path: str, caption: str) -> int:
    """`sendfile`: a photo or a document into a topic. A refused local file is SystemExit with the reason."""
    thread, group = _topic_target(app, topic, "sendfile")
    try:
        app.uploader.send_file(int(group), thread, path, caption)
    except UploadRefused as exc:
        raise SystemExit(UPLOAD_REFUSED.format(detail=exc)) from None
    return 0


def type_into_pane(panes: Panes, pane: str, text: str) -> int:
    """`type`: prints the one-line result. Exit 0 when the text was typed, 1 otherwise."""
    ok, info = panes.type_to_pane(pane, text)
    print(info)
    return 0 if ok else 1


def spawn_in_tmux(panes: Panes, command: str, name: str, text: str) -> int:
    """`spawn`: opens a seat with the text as its first prompt, verbatim. Exit 0 when the seat took it, 1 otherwise."""
    ok, info, _pane = panes.open_seat(command, name, text)
    print(info)
    return 0 if ok else 1


def load_config(paths: Paths) -> Config:
    """The config of the daemon and of the short commands. Its warnings (a rejected launcher) go to the bridge log,
    the same file App logs to; a default Config.load would drop them."""
    return Config.load(paths, log=Logger(paths.log).log)


def run_daemon(paths: Paths) -> None:
    """The daemon. The lock is taken first: a second daemon exits with 'already running'. Runs until a stop signal
    raises SystemExit(0); the finally stops the live sessions (no new turn can start) and closes the database."""
    lock = acquire_lock(paths.lock)  # the lock lasts as long as this handle is open
    try:
        app = App.build(paths, load_config(paths), sweep_stale_views=True)  # the lock is held: no live turn exists yet
        try:
            _serve(app)
        finally:
            ignore_stop_signals()
            stopped = app.shutdown_sessions()
            app.logger.log(DAEMON_STOP_LOG.format(count=stopped))
            app.close()
    finally:
        lock.close()


def _serve(app: App) -> None:
    """The daemon's start-up order of the live cmd_daemon: handlers, command menu, recovery, replay, then the poll loop."""
    app.logger.log(DAEMON_START_LOG)
    install_stop_handlers(app.logger)
    app.lifecycle.register_commands()  # after the handlers: a stop during this network call still shuts down cleanly
    app.lifecycle.recover()
    app.inbox.replay_inbox()
    app.probe.start()  # the deleted-topic probe: its first round is 60 s later, on its own daemon thread
    app.delete_requests.start()  # the view's delete requests, polled every 2 s on their own daemon thread
    app.poller.run()


def _with_app(paths: Paths, action: Callable[[App], T]) -> T:
    """Builds the App for one short command, runs the action and closes the database again."""
    app = App.build(paths, load_config(paths))
    try:
        return action(app)
    finally:
        app.close()


def open_view(paths: Paths) -> int:
    """`view`: makes the tg-bridge tmux session when it is missing, then attaches to it. The sync window of the session
    adds the topic windows. A tmux that cannot run is a message and exit 1."""
    Viewer(paths, load_config(paths)).ensure_session()  # view-sync makes the windows: it is the only creator
    try:
        attach()
    except OSError as exc:
        print(VIEW_NO_TMUX.format(detail=exc), file=sys.stderr)
        return 1
    return 0


def view_topic_pane(paths: Paths, thread: str) -> int:
    """`view-pane <thread>`: the pane of one topic's window (the ready, claude and busy screens). A thread that is not a
    number is usage. When the tg_bridge code on disk changes, the pane execs itself again on it (daemon/code_watch)."""
    if not thread.isdigit():
        print(VIEW_THREAD_NOT_A_NUMBER.format(thread=thread), file=sys.stderr)
        return 1
    clock = SystemClock()
    status = run_topic_pane(paths, int(thread), lambda: load_config(paths), clock=clock,
                            code_changed=CodeWatch(clock).changed)
    if status == RESTART_STATUS:
        reexec()
    return status


def view_delete_topic(paths: Paths, thread: str, environ: Mapping[str, str], clock: Clock,
                      show: Callable[[str], None] = show_in_tmux) -> int:
    """`view-delete <thread>`: asks the daemon to delete the topic. Prints the daemon's one-line answer and, inside
    tmux (TMUX set), shows it there too. Exit 0 when the topic is gone (or was already gone), 1 otherwise."""
    if not thread.isdigit():
        print(VIEW_DELETE_NOT_A_NUMBER.format(thread=thread), file=sys.stderr)
        return 1
    ok, text = request_topic_delete(paths.view, int(thread), clock)
    print(text)
    if environ.get("TMUX"):
        show(text)
    return 0 if ok else 1


def _live_runners(paths: Paths) -> Runners:
    """The runners of the real command line: the App for the Telegram commands, tmux alone for type and spawn, and the
    view's files and tmux for the view commands."""
    panes = Panes(clock=SystemClock())
    return Runners(
        daemon=lambda: run_daemon(paths),
        send=lambda topic, text: _with_app(paths, lambda app: send_to_topic(app, topic, text)),
        sendfile=lambda topic, path, caption: _with_app(
            paths, lambda app: send_file_to_topic(app, topic, path, caption)
        ),
        type_text=lambda pane, text: type_into_pane(panes, pane, text),
        spawn=lambda command, name, text: spawn_in_tmux(panes, command, name, text),
        hook_permission=lambda: run_hook_client(sys.stdin, sys.stdout, os.environ),
        hook_session_start=lambda thread, pid: run_session_start_hook(sys.stdin, thread, pid, paths),
        view=lambda: open_view(paths),
        view_sync=lambda: run_sync(paths, lambda: load_config(paths), SystemClock()),
        view_pane=lambda thread: view_topic_pane(paths, thread),
        view_delete=lambda thread: view_delete_topic(paths, thread, os.environ, SystemClock()),
    )


def _exit_status(exc: SystemExit) -> int:
    """The status Python itself would exit with for this SystemExit: a message goes to stderr and means 1."""
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point. argv is the arguments after the program name (sys.argv[1:] when omitted)."""
    os.umask(UMASK)
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        return route(args, _live_runners(Paths.from_environment()))
    except SystemExit as exc:
        return _exit_status(exc)
