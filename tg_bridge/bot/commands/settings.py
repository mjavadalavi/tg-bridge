"""Settings commands: /mode, /model, /launcher and /cwd (C19 settings part, STEP 48).

Port of the settings branches of handle_command in tg-bridge.py (set_topic_mode, set_topic_model, set_topic_launcher
and the /cwd branch with apply_cwd). The reply texts are the live ones. The rules:

- /mode and /model with no argument send their picker (the backend's modes, the models its config lists). A picker
  with no buttons (a backend with no model list) is sent as plain text instead.
- /mode and /model with an argument store the value in sessions. A running turn switches at once: the mode goes
  through PermissionService.set_live_mode, the model through session.set_model. Both calls can wait for the CLI, so
  they run on a daemon thread and the command replies at once. A backend that cannot switch inside a turn says so in
  the reply (the change applies from the next message).
- /launcher is settled by LauncherSwitch (carry-over, handoff or a fresh session). This module only checks the name.
- /cwd is refused while a turn runs. A folder that exists becomes the topic's folder through CwdService.apply; a path
  that is not a folder is said aloud, never searched; a bare name starts a folder search through CwdService.

The commands run on the poll thread, so every reply goes through CommandContext (the ordered out-worker).
"""

import os

from tg_bridge.agents.base import AgentBackend
from tg_bridge.agents.registry import BackendRegistry
from tg_bridge.bot.commands.base import Command, CommandContext, CommandRegistry
from tg_bridge.bot.commands.info import launcher_state
from tg_bridge.bot.live_actions import apply_model, spawn
from tg_bridge.bot.pickers import launcher_picker, mode_picker, model_picker
from tg_bridge.config import Config
from tg_bridge.core.cwd import CwdService, cwd_reply
from tg_bridge.core.launcher_switch import LauncherSwitch
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.permissions import PermissionService
from tg_bridge.core.topic_backend import topic_backend, topic_models
from tg_bridge.logging_ import Logger
from tg_bridge.store.sessions import SessionRecord, SessionRepo

NO_SESSION_TEXT = "no session in this topic"
BUSY_TEXT = "a turn is running — send /stop first, then try again."
MODE_USAGE_TEXT = "usage: /mode bypass|acceptEdits|default|plan"
MODE_SET_TEXT = "🛡 mode set to {mode}"
MODEL_SET_TEXT = "🧠 model set to {model}"
DEFAULT_MODEL = "default"
CLEAR_WORD = "default"          # /model default clears the topic's model (any case)
LIVE_SUFFIX = " — applied to the running turn"
NEXT_MESSAGE_SUFFIX = " — applies from your next message"
MODE_LIVE_BY = "mode command"   # who the permission service names when it allows open cards on bypass
NOT_A_DIRECTORY_TEXT = "not a directory: {path}"
UNKNOWN_LAUNCHER_TEXT = 'unknown launcher "{name}". choose one of: {names} (or /launcher alone for buttons)'
MODE_HELP = "show or set mode: bypass, acceptEdits, default, plan"
MODEL_HELP = "set this topic model, or default"
LAUNCHER_HELP = "pick the launcher for this topic"
CWD_HELP = "set the working directory for the next turn: buttons, /cwd <path>, or /cwd <folder name> to search"


class SettingsCommands:
    """The /mode, /model, /launcher and /cwd commands. Every collaborator is injected; the object keeps no state."""

    def __init__(self, *, config: Config, sessions: SessionRepo, live: LiveRegistry, backends: BackendRegistry,
                 permissions: PermissionService, launchers: LauncherSwitch, cwd: CwdService,
                 logger: Logger) -> None:
        self._config = config
        self._sessions = sessions
        self._live = live
        self._backends = backends
        self._permissions = permissions
        self._launchers = launchers
        self._folders = cwd
        self._logger = logger

    def register(self, registry: CommandRegistry) -> None:
        """Adds /mode, /model, /launcher and /cwd to the registry."""
        registry.register(Command("mode", MODE_HELP, self._mode))
        registry.register(Command("model", MODEL_HELP, self._model))
        registry.register(Command("launcher", LAUNCHER_HELP, self._launcher))
        registry.register(Command("cwd", CWD_HELP, self._cwd))

    # --- helpers

    def _backend_of(self, row: SessionRecord) -> AgentBackend:
        """The backend the topic's turns run on now, resolved from its launcher (see topic_backend)."""
        return topic_backend(self._config, self._backends, row, row.name)

    @staticmethod
    def _live_suffix(running: bool, can_switch_live: bool) -> str:
        """The reply suffix for a change made while a turn runs: applied at once, or from the next message."""
        if not running:
            return ""
        return LIVE_SUFFIX if can_switch_live else NEXT_MESSAGE_SUFFIX

    @staticmethod
    def _mode_key(backend: AgentBackend, word: str) -> str | None:
        """The mode key a typed word names: its label or its key, any case. None for an unknown word."""
        lowered = word.lower()
        for choice in backend.modes():
            if lowered in (choice.label.lower(), choice.key.lower()):
                return choice.key
        return None

    # --- /mode

    def _mode(self, ctx: CommandContext, args: str) -> None:
        row = self._sessions.get(ctx.thread_id)
        if row is None:
            ctx.say(NO_SESSION_TEXT)
            return
        backend = self._backend_of(row)
        running = self._live.running(ctx.thread_id) is not None
        if not args:
            current = row.mode or self._config.default_mode()
            picker = mode_picker(ctx.thread_id, backend.modes(), current, backend.capabilities.live_mode)
            ctx.say_buttons(picker.text, picker.rows)
            return
        mode = self._mode_key(backend, args)
        if mode is None:
            ctx.say(MODE_USAGE_TEXT)
            return
        self._sessions.set_mode(ctx.thread_id, mode)
        if running:
            spawn(self._permissions.set_live_mode, ctx.thread_id, mode, by=MODE_LIVE_BY)
        suffix = self._live_suffix(running, backend.capabilities.live_mode)
        ctx.say(MODE_SET_TEXT.format(mode=mode) + suffix)

    # --- /model

    def _model(self, ctx: CommandContext, args: str) -> None:
        row = self._sessions.get(ctx.thread_id)
        if row is None:
            ctx.say(NO_SESSION_TEXT)
            return
        backend, models = topic_models(self._config, self._backends, row, row.name)
        if not args:
            picker = model_picker(ctx.thread_id, models, row.model or None, backend.capabilities.live_model)
            if picker.rows:
                ctx.say_buttons(picker.text, picker.rows)
            else:
                ctx.say(picker.text)
            return
        model = "" if args.lower() == CLEAR_WORD else args
        self._sessions.set_model(ctx.thread_id, model)
        running = self._live.running(ctx.thread_id)
        if running is not None:
            spawn(apply_model, running.session, model or DEFAULT_MODEL, self._logger)
        suffix = self._live_suffix(running is not None, backend.capabilities.live_model)
        ctx.say(MODEL_SET_TEXT.format(model=model or DEFAULT_MODEL) + suffix)

    # --- /launcher

    def _launcher(self, ctx: CommandContext, args: str) -> None:
        row = self._sessions.get(ctx.thread_id)
        if row is None:
            ctx.say(NO_SESSION_TEXT)
            return
        name, command = launcher_state(self._config, row, ctx.topic)
        if not args:
            specs = list(self._config.launcher_specs().values())
            picker = launcher_picker(ctx.thread_id, specs, name, command)
            ctx.say_buttons(picker.text, picker.rows)
            return
        names = self._config.launcher_choices()
        if args not in names:
            ctx.say(UNKNOWN_LAUNCHER_TEXT.format(name=args, names=", ".join(names)))
            return
        running = self._live.running(ctx.thread_id)
        session = running.session if running is not None else None
        _, reply = self._launchers.set(ctx.thread_id, ctx.topic, session, args)
        ctx.say(reply)

    # --- /cwd

    def _cwd(self, ctx: CommandContext, args: str) -> None:
        row = self._sessions.get(ctx.thread_id)
        if row is None:
            ctx.say(NO_SESSION_TEXT)
            return
        if self._live.running(ctx.thread_id) is not None:  # the running process still holds the old session
            ctx.say(BUSY_TEXT)
            return
        if not args:
            text, rows = self._folders.picker(self._config, ctx.thread_id, row.cwd)
            if rows:
                ctx.say_buttons(text, rows)
            else:
                ctx.say(text)
            return
        path = os.path.realpath(os.path.expanduser(args))
        if os.path.isdir(path):
            ctx.say(cwd_reply(path, self._folders.apply(ctx.thread_id, path)))
        elif args.startswith(("/", "~")):  # a path, not a name: say so instead of searching for it
            ctx.say(NOT_A_DIRECTORY_TEXT.format(path=path))
        else:  # a name, not a path: the search runs on its worker; the card follows as buttons
            self._folders.start_search(ctx.chat_id, ctx.thread_id, self._config, args)
