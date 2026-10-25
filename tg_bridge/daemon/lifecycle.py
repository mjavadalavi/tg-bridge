"""Lifecycle: start-up recovery, shutdown of the live sessions, the process lock, stop signals and the command menu.

Port of the start-up and shutdown parts of the live file: recover (tg-bridge.py:3582-3610), shutdown_sessions
(3611-3628), _on_stop_signal (3629-3634), register_commands (3667-3673) and the flock and signal set-up of cmd_daemon
(3674-3690). The rules that keep a restart and a stop safe:

- recover runs once at start-up, before the inbox replay. A turn marker (active:<thread>) left in kv by the previous
  process gets a restart notice in its topic and is removed; the session id is kept, so "continue" resumes the turn.
  Every ask still pending was never answered and its turn is gone: PermissionService.recover_cards voids it and closes
  its card.
- shutdown_sessions sets the shutdown flag first, so no turn can start after its snapshot of the live sessions. Then it
  stops every live session in parallel, inside one time budget (measured on the injected clock). A session whose process
  group is still alive when the budget is spent gets SIGKILL, so no orphan survives. The active markers of the turns
  that end are kept on purpose: the next start tells their topics about the restart.
- acquire_lock takes an exclusive, non-blocking flock. The returned handle must stay open for the life of the process.
- install_stop_handlers makes SIGTERM and SIGINT raise SystemExit(0). Raising here also interrupts a blocking long poll.
  ignore_stop_signals is set once the daemon is stopping, so a second signal cannot cut the shutdown short.
- register_commands publishes the command menu (setMyCommands). Best effort: a failure is logged and the bridge runs on.

Threads: shutdown_sessions starts one daemon thread per session and joins them. Signal handlers and the lock are process
wide by nature, so the functions that touch them are plain module functions and must run on the main thread.
"""

import fcntl
import json
import os
import signal
import threading
from dataclasses import dataclass
from types import FrameType
from typing import IO, Any, Protocol, runtime_checkable

from tg_bridge.agents.base import AgentSession
from tg_bridge.bot.commands.base import CommandRegistry
from tg_bridge.clock import Clock
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.permissions import PermissionService
from tg_bridge.core.turn_runner import ACTIVE_PREFIX
from tg_bridge.logging_ import Logger
from tg_bridge.store.kv import KvRepo
from tg_bridge.telegram.client import TelegramClient
from tg_bridge.telegram.messenger import Messenger

SHUTDOWN_BUDGET_SECONDS = 20.0  # the whole shutdown, all sessions together
SHUTDOWN_GRACE_SECONDS = 5.0  # each session's own grace before its kill ladder
ERROR_TEXT_LIMIT = 150
RESTART_NOTICE = (
    "⚠️ the bridge restarted while this turn was running. "
    'The session is kept — send "continue" to pick up where it left off.'
)
ALREADY_RUNNING = "tg-bridge: already running"
SIGNAL_EXIT_CODE = 0
STOP_LOG = "daemon stop (signal {signum})"
COMMANDS_FAILED_LOG = "setMyCommands failed: {error}"
SHUTDOWN_FAILED_LOG = "shutdown: {error}"


@runtime_checkable
class ProcessGroupSession(Protocol):
    """A session whose child runs in its own process group (agents.process.ProcessSession). Other sessions are not killed
    by group here: they have no group to kill."""

    @property
    def pgid(self) -> int | None: ...


@dataclass(frozen=True)
class Recovery:
    """What start-up recovery did: the turn markers it cleared, the restart notices it sent, the asks it voided."""

    markers: int
    notices: int
    asks: int


def _group_alive(pgid: int) -> bool:
    """True while any process still belongs to process group pgid (the leader may already be gone)."""
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _thread_of(key: str) -> int | None:
    """The topic id named by an active:<thread> key, or None when the key is malformed."""
    try:
        return int(key[len(ACTIVE_PREFIX):])
    except ValueError:
        return None


def _marker_info(value: str) -> dict[str, Any]:
    """The JSON object stored in a marker. Anything else reads as {} (the notice is then skipped)."""
    try:
        info = json.loads(value or "{}")
    except ValueError:
        return {}
    return info if isinstance(info, dict) else {}


class Lifecycle:
    """Start-up recovery and shutdown of the daemon. One per App; holds no state of its own between calls."""

    def __init__(
        self,
        *,
        live: LiveRegistry,
        permissions: PermissionService,
        kv: KvRepo,
        messenger: Messenger,
        telegram: TelegramClient,
        commands: CommandRegistry,
        clock: Clock,
        logger: Logger,
    ) -> None:
        self._live = live
        self._permissions = permissions
        self._kv = kv
        self._messenger = messenger
        self._telegram = telegram
        self._commands = commands
        self._clock = clock
        self._logger = logger

    def recover(self) -> Recovery:
        """Start-up, before the replay: restart notices for the turns that were running, then the asks left pending."""
        markers = notices = 0
        for key, value in self._kv.items_with_prefix(ACTIVE_PREFIX):
            thread = _thread_of(key)
            chat = _marker_info(value).get("chat_id")
            if thread is not None and chat is not None:
                self._messenger.send_text(chat, thread, RESTART_NOTICE)
                notices += 1
            self._kv.delete(key)
            markers += 1
        asks = self._permissions.recover_cards()
        return Recovery(markers=markers, notices=notices, asks=asks)

    def shutdown_sessions(self, budget: float = SHUTDOWN_BUDGET_SECONDS, grace: float = SHUTDOWN_GRACE_SECONDS) -> int:
        """Stops every live session in parallel within budget seconds, then SIGKILLs any group still alive.
        Returns how many sessions were live when the shutdown began."""
        sessions = self._live.begin_shutdown()
        workers = [
            threading.Thread(target=self._stop_one, args=(session, grace), name="shutdown-stop", daemon=True)
            for session in sessions
        ]
        for worker in workers:
            worker.start()
        deadline = self._clock.monotonic() + budget
        for worker in workers:
            worker.join(max(0.0, deadline - self._clock.monotonic()))
        self._kill_straggler_groups(sessions)
        return len(sessions)

    def register_commands(self) -> bool:
        """Publishes the command menu from the registry. True when Telegram accepted it; a failure is logged only."""
        menu = [{"command": word, "description": text} for word, text in self._commands.bridge_commands()]
        try:
            self._telegram.call("setMyCommands", commands=json.dumps(menu))
        except Exception as exc:
            self._logger.log(COMMANDS_FAILED_LOG.format(error=str(exc)[:ERROR_TEXT_LIMIT]))
            return False
        return True

    def _stop_one(self, session: AgentSession, grace: float) -> None:
        """Runs on a shutdown thread. A failing stop is logged; the other sessions are still stopped."""
        try:
            session.stop(grace=grace)
        except Exception as exc:
            self._logger.log(SHUTDOWN_FAILED_LOG.format(error=str(exc)[:ERROR_TEXT_LIMIT]))

    def _kill_straggler_groups(self, sessions: list[AgentSession]) -> None:
        """SIGKILL the process group of every session whose group survived the budget."""
        for session in sessions:
            if not isinstance(session, ProcessGroupSession):
                continue
            pgid = session.pgid
            if pgid is None or not _group_alive(pgid):
                continue
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def acquire_lock(path: str) -> IO[str]:
    """Takes the exclusive daemon lock at path. SystemExit when another bridge holds it. Keep the handle open: the lock
    lasts as long as the file is open."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    handle = open(path, "a", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise SystemExit(ALREADY_RUNNING)
    return handle


def install_stop_handlers(logger: Logger) -> None:
    """SIGTERM and SIGINT log the stop and raise SystemExit(0). Main thread only (a signal handler rule of Python)."""

    def _on_stop(signum: int, frame: FrameType | None) -> None:
        logger.log(STOP_LOG.format(signum=signum))
        raise SystemExit(SIGNAL_EXIT_CODE)

    signal.signal(signal.SIGTERM, _on_stop)
    signal.signal(signal.SIGINT, _on_stop)


def ignore_stop_signals() -> None:
    """Makes SIGTERM and SIGINT do nothing, so a second signal cannot interrupt the shutdown. Main thread only."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
