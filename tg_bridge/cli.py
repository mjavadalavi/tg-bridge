"""Command line dispatch (STEP 54): the subcommands of the live file's main(), with the same argv and exit codes.

  (no subcommand) or daemon          the daemon: lock, App, start-up recovery, inbox replay, poll loop
  send <topic> <text>                one message to a known topic
  sendfile | send-file <topic> <path> [caption]
  type <pane> <text>                 type text into a live tmux pane (no App, no database)
  spawn <command> <name> <text>      open a live seat in a new tmux window (no App, no database)
  anything else                      prints the spec (tg_bridge.__doc__) and exits 0

main() sets the umask to 0o077 first, as the live file did. A failure a command reports (unknown topic, a refused upload,
a pane that did not take the text, missing arguments) exits 1. The stop signal (SIGTERM or SIGINT) exits 0 after the
daemon has stopped its sessions. A SystemExit that carries a message prints it to stderr and exits 1, as Python does.
"""

import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeVar

import tg_bridge
from tg_bridge.app import App
from tg_bridge.clock import SystemClock
from tg_bridge.config import Config
from tg_bridge.daemon.inbox import GROUP_KEY
from tg_bridge.daemon.lifecycle import acquire_lock, ignore_stop_signals, install_stop_handlers
from tg_bridge.daemon.panes import Panes
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.telegram.upload import UploadRefused

T = TypeVar("T")

UMASK = 0o077
DAEMON_START_LOG = "daemon start"
DAEMON_STOP_LOG = "daemon stop: {count} session(s) signalled"
UNKNOWN_TOPIC = "tg-bridge {command}: unknown topic {topic!r} or group not learned yet"
UPLOAD_REFUSED = "tg-bridge sendfile: {detail}"
USAGE = MappingProxyType({
    "send": "usage: tg-bridge send <topic> <text>",
    "sendfile": "usage: tg-bridge sendfile <topic> <path> [caption]",
    "type": "usage: tg-bridge type <pane> <text>",
    "spawn": "usage: tg-bridge spawn <command> <name> <text>",
})
# subcommand -> number of arguments it needs; extra arguments are ignored, as the live file ignored them
REQUIRED_ARGUMENTS = MappingProxyType({"send": 2, "sendfile": 2, "type": 2, "spawn": 3})


@dataclass(frozen=True)
class Runners:
    """What each subcommand does. main() wires the real ones; route() only picks one and passes it the arguments."""

    daemon: Callable[[], None]
    send: Callable[[str, str], int]
    sendfile: Callable[[str, str, str], int]
    type_text: Callable[[str, str], int]
    spawn: Callable[[str, str, str], int]


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
    sys.stdout.write(tg_bridge.__doc__)
    return 0


def _with_arguments(command: str, rest: Sequence[str], call: Callable[[], int]) -> int:
    """Runs the subcommand when it has the arguments it needs; otherwise prints its usage line and exits 1."""
    if len(rest) < REQUIRED_ARGUMENTS[command]:
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
        app = App.build(paths, load_config(paths))
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
    app.poller.run()


def _with_app(paths: Paths, action: Callable[[App], T]) -> T:
    """Builds the App for one short command, runs the action and closes the database again."""
    app = App.build(paths, load_config(paths))
    try:
        return action(app)
    finally:
        app.close()


def _live_runners(paths: Paths) -> Runners:
    """The runners of the real command line: the App for the Telegram commands, tmux alone for type and spawn."""
    panes = Panes(clock=SystemClock())
    return Runners(
        daemon=lambda: run_daemon(paths),
        send=lambda topic, text: _with_app(paths, lambda app: send_to_topic(app, topic, text)),
        sendfile=lambda topic, path, caption: _with_app(
            paths, lambda app: send_file_to_topic(app, topic, path, caption)
        ),
        type_text=lambda pane, text: type_into_pane(panes, pane, text),
        spawn=lambda command, name, text: spawn_in_tmux(panes, command, name, text),
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
