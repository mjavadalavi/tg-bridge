"""Picker taps: the cm:<thread>:<kind>:<arg> buttons of /model, /mode, /launcher, /resume, /rules and /cwd (C19 taps).

Each tap makes the same change as its typed command, then edits the SAME message to show the new state with the check
mark moved. The rules, which are the live file's:

- mo (model) and md (mode): the arg is the model name or the index into the backend's mode list, read at tap time, so the
  button applies what its label showed. A stored model 'default' clears the topic's model. A running turn switches at
  once, on a daemon thread, because set_model and set_live_mode can wait for the CLI.
- la (launcher): LauncherSwitch decides the carry-over or the fresh session; a busy turn refuses and leaves the message.
- rs (resume): SessionCommands.resume_tap, the same rules as /resume <uuid>.
- rf / ra (rules): one rule by its number, or all of the topic's rules; the list is redrawn.
- cw / cs / ck (folders): the /cwd button, a search-card button, and the keep button of a new topic's card. Each is
  the /cwd <path> change; the card is edited to show the result.

Refusals (a missing session, a stale index, a turn running) are a toast and never change anything. The rule for the
arg is the live one: every kind but mo expects a number.
"""

import os

from tg_bridge.agents.registry import BackendRegistry
from tg_bridge.bot.callbacks.router import CallbackRouter, Tap
from tg_bridge.bot.commands.info import launcher_state
from tg_bridge.bot.commands.session import SessionCommands
from tg_bridge.bot.live_actions import apply_model, spawn
from tg_bridge.bot.pickers import (
    KIND_LAUNCHER,
    KIND_MODE,
    KIND_MODEL,
    KIND_RESUME,
    KIND_RULE_ALL,
    KIND_RULE_DELETE,
    launcher_picker,
    mode_picker,
    model_picker,
    rules_picker,
)
from tg_bridge.config import Config, tilde
from tg_bridge.core.cwd import BUSY_TEXT as FOLDER_BUSY_TEXT
from tg_bridge.core.cwd import KEEP_KIND, PICKER_KIND, SEARCH_KIND, CwdService, cwd_reply
from tg_bridge.core.launcher_switch import KIND_BUSY, LauncherSwitch
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.permissions import PermissionService
from tg_bridge.core.topic_backend import topic_backend, topic_models
from tg_bridge.logging_ import Logger
from tg_bridge.store.rules import RuleRepo
from tg_bridge.store.sessions import SessionRecord, SessionRepo

NO_SESSION_TEXT = "no session in this topic"
NOT_IN_LIST_TEXT = "not in the list anymore"
MODEL_SET_TEXT = "🧠 model set to {model}"
MODE_SET_TEXT = "🛡 mode set to {mode}"
DEFAULT_MODEL = "default"
MODE_TAP_BY = "mode button"          # who the permission service names when it allows open cards on bypass
RULE_REMOVED_TEXT = "🗑 rule removed"
RULE_GONE_TEXT = "no such rule anymore"
ALL_RULES_REMOVED_TEXT = "🗑 all rules removed"
NO_RULES_LEFT_TEXT = "no rules left"


class PickerTaps:
    """The handlers of the picker kinds. Every collaborator is injected; the object keeps no state of its own."""

    def __init__(self, *, config: Config, sessions: SessionRepo, live: LiveRegistry, backends: BackendRegistry,
                 permissions: PermissionService, launchers: LauncherSwitch, stops: SessionCommands,
                 rules: RuleRepo, folders: CwdService, logger: Logger) -> None:
        self._config = config
        self._sessions = sessions
        self._live = live
        self._backends = backends
        self._permissions = permissions
        self._launchers = launchers
        self._stops = stops
        self._rules = rules
        self._folders = folders
        self._logger = logger

    def register(self, router: CallbackRouter) -> None:
        """Adds the picker kinds to the router: model, mode, launcher, resume, rules (one and all), and the three folder kinds."""
        router.on_picker(KIND_MODEL, self._model)
        router.on_picker(KIND_MODE, self._mode)
        router.on_picker(KIND_LAUNCHER, self._launcher)
        router.on_picker(KIND_RESUME, self._resume)
        router.on_picker(KIND_RULE_DELETE, self._rule)
        router.on_picker(KIND_RULE_ALL, self._all_rules)
        router.on_picker(PICKER_KIND, self._folder)
        router.on_picker(SEARCH_KIND, self._folder_search)
        router.on_picker(KEEP_KIND, self._folder_keep)

    # --- helpers

    @staticmethod
    def _thread(tap: Tap) -> int:
        return int(tap.args[0])  # the router checked that the thread is a number

    @staticmethod
    def _number(tap: Tap) -> int | None:
        """The arg as an int, or None when it is not a number (the live rule for every kind but mo)."""
        return int(tap.args[2]) if tap.args[2].isdigit() else None

    def _row(self, tap: Tap, thread: int) -> SessionRecord | None:
        row = self._sessions.get(thread)
        if row is None:
            tap.toast(NO_SESSION_TEXT)
        return row

    def _backend(self, row: SessionRecord):
        """The backend the topic's turns run on now, resolved from its launcher (see topic_backend)."""
        return topic_backend(self._config, self._backends, row, row.name)

    def _edit_rules(self, tap: Tap, thread: int) -> None:
        picker = rules_picker(thread, self._rules.for_thread(thread))
        tap.edit(picker.text, picker.rows)

    # --- /model and /mode

    def _model(self, tap: Tap) -> None:
        thread, arg = self._thread(tap), tap.args[2]
        row = self._row(tap, thread)
        if row is None:
            return
        backend, models = topic_models(self._config, self._backends, row, row.name)
        if arg not in models:
            tap.toast(NOT_IN_LIST_TEXT)
            return
        stored = "" if arg.lower() == DEFAULT_MODEL else arg
        self._sessions.set_model(thread, stored)
        running = self._live.running(thread)
        if running is not None:
            spawn(apply_model, running.session, stored or DEFAULT_MODEL, self._logger)
        picker = model_picker(thread, models, stored or None, backend.capabilities.live_model)
        tap.toast(MODEL_SET_TEXT.format(model=stored or DEFAULT_MODEL))
        tap.edit(picker.text, picker.rows)

    def _mode(self, tap: Tap) -> None:
        thread, index = self._thread(tap), self._number(tap)
        row = self._row(tap, thread)
        if row is None:
            return
        backend = self._backend(row)
        modes = backend.modes()
        if index is None or index >= len(modes):
            tap.toast(NOT_IN_LIST_TEXT)
            return
        key = modes[index].key
        self._sessions.set_mode(thread, key)
        if self._live.running(thread) is not None:
            spawn(self._permissions.set_live_mode, thread, key, by=MODE_TAP_BY)
        picker = mode_picker(thread, modes, key, backend.capabilities.live_mode)
        tap.toast(MODE_SET_TEXT.format(mode=key))
        tap.edit(picker.text, picker.rows)

    # --- /launcher and /resume

    def _launcher(self, tap: Tap) -> None:
        thread, index = self._thread(tap), self._number(tap)
        row = self._row(tap, thread)
        if row is None:
            return
        specs = list(self._config.launcher_specs().values())
        if index is None or index >= len(specs):
            tap.toast(NOT_IN_LIST_TEXT)
            return
        running = self._live.running(thread)
        kind, reply = self._launchers.set(thread, row.name, running.session if running is not None else None,
                                          specs[index].name)
        tap.toast(reply)
        if kind != KIND_BUSY:  # the same message now shows the launcher in use, with its check mark
            current = self._sessions.get(thread) or row
            name, command = launcher_state(self._config, current, current.name)
            tap.edit(reply, launcher_picker(thread, specs, name, command).rows)

    def _resume(self, tap: Tap) -> None:
        thread, history_id = self._thread(tap), self._number(tap)
        if history_id is None:
            return
        outcome = self._stops.resume_tap(thread, history_id)
        tap.toast(outcome.toast)
        if outcome.picker is not None:
            tap.edit(outcome.picker.text, outcome.picker.rows)

    # --- /rules

    def _rule(self, tap: Tap) -> None:
        thread, rule_id = self._thread(tap), self._number(tap)
        if rule_id is None:
            return
        removed = self._rules.delete(thread, rule_id)
        tap.toast(RULE_REMOVED_TEXT if removed else RULE_GONE_TEXT)
        self._edit_rules(tap, thread)

    def _all_rules(self, tap: Tap) -> None:
        thread = self._thread(tap)
        if self._number(tap) is None:
            return
        removed = self._rules.delete_all(thread)
        tap.toast(ALL_RULES_REMOVED_TEXT if removed else NO_RULES_LEFT_TEXT)
        self._edit_rules(tap, thread)

    # --- /cwd: the picker button, the search card and the keep button of a new topic's card

    def _folder(self, tap: Tap) -> None:
        thread, path_id = self._thread(tap), self._number(tap)
        if path_id is None or self._row(tap, thread) is None:
            return
        path = self._folders.path_of(path_id)
        if path is None or not os.path.isdir(path):
            tap.toast(NOT_IN_LIST_TEXT)
            return
        if self._live.running(thread) is not None:
            tap.toast(FOLDER_BUSY_TEXT)
            return
        changed = self._folders.apply(thread, path)
        tap.toast(cwd_reply(tilde(path, self._config.home), changed))
        text, rows = self._folders.picker(self._config, thread, path)
        tap.edit(text, rows)

    def _folder_search(self, tap: Tap) -> None:
        thread, index = self._thread(tap), self._number(tap)
        if index is None:
            return
        outcome = self._folders.search_tap(self._config, thread, tap.message_id, index)
        tap.toast(outcome.toast)
        if outcome.card_text is not None:
            tap.edit(outcome.card_text)

    def _folder_keep(self, tap: Tap) -> None:
        thread = self._thread(tap)
        if self._number(tap) is None:
            return
        outcome = self._folders.keep_tap(thread, tap.message_id)
        tap.toast(outcome.toast)
        if outcome.card_text is not None:
            tap.edit(outcome.card_text)
