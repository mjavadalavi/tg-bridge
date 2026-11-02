"""The launcher change of the view pane (daemon/view_pane): the number keys of the ready screen.

choices are the configured Claude launchers in config order, numbered from 1 (at most CHOICE_LIMIT, the number keys).
choice_line draws them under the pane's detail line, the current one marked. switch_launcher is the change itself: the
same LauncherSwitch.set that Telegram's /launcher calls, so the topic changes for Telegram too. It opens the bridge
database for the one call and closes it. The pane holds the topic's flock around the call (terminal_claim.free_topic);
this module takes no lock.

Threads: the pane's thread only. Lock map: none.
"""

import os
from collections.abc import Callable, Sequence

from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.clock import SystemClock
from tg_bridge.config import BACKEND_CLAUDE, Config
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.launcher_switch import LauncherSwitch
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.db import Database
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.sessions import HistoryRepo, SessionRepo

CHOICE_LIMIT = 9                # the number keys 1..9
CHOICE_SEPARATOR = "   "
CHOICE_MARK = " ✓"
CHOICE_HINT_LINE = "1-9: change the launcher (Telegram uses it too)"
NO_LOG = os.devnull             # the switch's log lines go nowhere: the pane shows the reply instead


def claude_choices(config: Config) -> tuple[str, ...]:
    """The names of the configured Claude launchers, in config order, at most CHOICE_LIMIT."""
    names = [name for name, spec in config.launcher_specs().items() if spec.configured and spec.backend == BACKEND_CLAUDE]
    return tuple(names[:CHOICE_LIMIT])


def choice_line(choices: Sequence[str], current: str) -> str:
    """The launchers numbered, the current one marked: '  1 iclaude   2 oclaude ✓   3 claude'."""
    parts = [f"{number} {name}{CHOICE_MARK if name == current else ''}" for number, name in enumerate(choices, start=1)]
    return "  " + CHOICE_SEPARATOR.join(parts)


def switch_launcher(paths: Paths, load: Callable[[], Config], thread: int, name: str, picked: str) -> str:
    """Switches topic `thread` (named `name`) to launcher `picked`, as /launcher does, and returns the reply text.
    Raises ValueError when picked is not a visible launcher, OSError when the config cannot be read, sqlite3.Error when
    the database cannot be written."""
    config = load()
    db = Database(paths)
    try:
        clock = SystemClock()
        logger = Logger(NO_LOG)
        sessions = SessionRepo(db, clock)
        history = HistoryRepo(db, clock)
        registry = build_default_registry(config, logger, clock)
        handoff = HandoffService(KvRepo(db), registry, logger)
        switch = LauncherSwitch(config, sessions, history, handoff, registry)
        _kind, reply = switch.set(thread, name, None, picked)
    finally:
        db.close()
    return reply
