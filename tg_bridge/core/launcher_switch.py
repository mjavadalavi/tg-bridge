"""LauncherSwitch: /launcher and its picker taps, with the session carried over (port of the live set_topic_launcher).

The reply texts are the live ones, so a topic sees the same words in both files. The rules, decided from the topic's
current launcher (the one its turns run on now) and the launcher picked:

- the same launcher: nothing is stored ('same').
- a turn is running: nothing is stored ('busy'), the owner is told to /stop first.
- same backend (Claude to Claude, or Codex to Codex): when both launchers share one config folder the session id is
  kept; otherwise the transcript is copied into the new folder and the id is kept (BackendRegistry's backend does the
  copy through carry_history, then HistoryRepo.adopt_session records it); with nothing to copy the id moves to
  prev_uuid and the next message starts a fresh session.
- a different backend (Claude to Codex or back): ids and transcripts are not interchangeable, so nothing is copied.
  The old session is read into a handoff digest (HandoffService.prepare), the id moves to prev_uuid, and the digest is
  delivered to the next fresh session.

sessions.launcher is never written here: it keeps a Claude command, and the launcher pick (sessions.launcher_pick) plus
sessions.backend carry the choice. set() runs outside any lock of its own; the caller decides the busy state from the
LiveRegistry it holds, as the live file does.
"""

import os

from tg_bridge.agents.registry import BackendRegistry
from tg_bridge.config import Config, LauncherError, LauncherSpec
from tg_bridge.core.handoff import HandoffService
from tg_bridge.store.sessions import HistoryRepo, SessionRecord, SessionRepo

KIND_CHANGED = "changed"
KIND_SAME = "same"
KIND_BUSY = "busy"

SAME_TEXT = "🚀 launcher: {name} — already the launcher, nothing changed"
BUSY_TEXT = "a turn is running — send /stop first, then try again."
CONTINUED_TEXT = "🚀 launcher: {name} — continuing the same conversation"
COPIED_TEXT = "🚀 launcher: {name} — continuing the same conversation (history copied)"
NOT_FOUND_TEXT = "🚀 launcher: {name} — previous history not found, the next message starts a fresh session"
FRESH_TEXT = "🚀 launcher: {name} — the next message starts a fresh session"
HANDOFF_TEXT = "🚀 {name} — continuing with a summary of the previous conversation"


def _same_dir(a: str, b: str) -> bool:
    """True when two config folders are the same folder (after ~ expansion and symlink resolution)."""
    return bool(a) and bool(b) and os.path.realpath(os.path.expanduser(a)) == os.path.realpath(os.path.expanduser(b))


class LauncherSwitch:
    """Stores a launcher pick and settles the topic's session for it. Built once by App, used by /launcher and taps.

    The service holds no state of its own: every decision reads the rows it is given at call time.
    """

    def __init__(self, config: Config, sessions: SessionRepo, history: HistoryRepo, handoff: HandoffService,
                 registry: BackendRegistry) -> None:
        self._config = config
        self._sessions = sessions
        self._history = history
        self._handoff = handoff
        self._registry = registry

    def set(self, thread: int, name: str, live: object | None, new: str) -> tuple[str, str]:
        """Switches topic `thread` (configured as topic `name`) to launcher `new`. Returns (kind, reply).

        kind is KIND_CHANGED, KIND_SAME (nothing stored) or KIND_BUSY (a turn runs; nothing stored). `live` is the
        topic's running session, or None. ValueError when `new` is not a visible launcher: the caller checks names
        before calling.
        """
        target = self._config.launcher_specs().get(new)
        if target is None:
            raise ValueError(f"unknown launcher {new!r}")
        row = self._sessions.get(thread)
        current = self._current(row, name)
        if current is not None and current.name == new:
            return KIND_SAME, SAME_TEXT.format(name=new)
        if live is not None:
            return KIND_BUSY, BUSY_TEXT
        uuid = row.uuid if row else ""
        cwd = row.cwd if row else ""
        has_history = bool(uuid and cwd)
        if current is not None and current.backend != target.backend:
            return self._switch_backend(thread, current, target, uuid, cwd, has_history)
        return self._switch_same_backend(thread, current, target, uuid, cwd, has_history)

    def _current(self, row: SessionRecord | None, name: str) -> LauncherSpec | None:
        """The visible launcher the topic runs on now, or None when its command matches no visible launcher."""
        pick = row.launcher_pick if row else ""
        stored = row.launcher if row else ""
        topic_launcher = str(self._config.topic(name).get("launcher") or "")
        try:
            spec = self._config.resolve_launcher(pick, topic_launcher, stored)
        except LauncherError:
            return None
        return spec if spec.configured else None

    def _switch_same_backend(self, thread: int, current: LauncherSpec | None, target: LauncherSpec, uuid: str,
                             cwd: str, has_history: bool) -> tuple[str, str]:
        same_folder = current is not None and has_history and _same_dir(current.config_dir, target.config_dir)
        copied = (not same_folder and current is not None and has_history
                  and self._registry.get(current.backend).carry_history(current, target, cwd, uuid))
        self._store_pick(thread, target)
        if same_folder or copied:
            self._history.adopt_session(thread, uuid, target.command, cwd, target.backend)
        else:
            self._sessions.new_session(thread)
        self._handoff.discard(thread)
        if same_folder:
            return KIND_CHANGED, CONTINUED_TEXT.format(name=target.name)
        if copied:
            return KIND_CHANGED, COPIED_TEXT.format(name=target.name)
        if has_history:
            return KIND_CHANGED, NOT_FOUND_TEXT.format(name=target.name)
        return KIND_CHANGED, FRESH_TEXT.format(name=target.name)

    def _switch_backend(self, thread: int, current: LauncherSpec, target: LauncherSpec, uuid: str, cwd: str,
                        has_history: bool) -> tuple[str, str]:
        prepared = self._handoff.prepare(thread, current.backend, current, uuid or None, cwd)
        self._store_pick(thread, target)
        self._sessions.new_session(thread)
        if prepared:
            return KIND_CHANGED, HANDOFF_TEXT.format(name=target.name)
        if has_history:
            return KIND_CHANGED, NOT_FOUND_TEXT.format(name=target.name)
        return KIND_CHANGED, FRESH_TEXT.format(name=target.name)

    def _store_pick(self, thread: int, target: LauncherSpec) -> None:
        self._sessions.set_launcher_pick(thread, target.name)
        self._sessions.set_backend(thread, target.backend)
