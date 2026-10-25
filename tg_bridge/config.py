"""Config wrapper over dict, LauncherSpec{name,backend,command,argv}, defaults.

Config reads config.json once and never writes it. Every key the bridge reads has a typed getter here, with the
same default the live file uses. The launcher table is built once, at construction: each entry is parsed into a
LauncherSpec, and a rejected entry is logged with one line and hidden. Defaults are functions, not module
constants, so no mutable state is shared between Config objects.
"""

import copy
import json
import os
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from tg_bridge.paths import Paths

BACKEND_CLAUDE = "claude"
BACKEND_CODEX = "codex"
KNOWN_BACKENDS = (BACKEND_CLAUDE, BACKEND_CODEX)
SECTION_CLAUDE = "launchers"
SECTION_AGENT = "agent_launchers"
DEFAULT_MODE = "bypassPermissions"
CLAUDE_CONFIG_DIR = "~/.claude"
CODEX_CONFIG_DIR = "~/.codex"
MODEL_CHOICES = ("opus", "sonnet", "haiku", "fable", "default")
EMPTY_LAUNCHER_MESSAGE = "the launcher is empty: set it in config (topics or launchers)"

LogFn = Callable[[str], None]


def _discard(message: str) -> None:
    """Default log sink: the line is dropped."""


class LauncherError(ValueError):
    """A launcher entry or command that cannot be used. A ValueError, so callers that catch ValueError still do."""


def expand_home(path: str, home: str) -> str:
    """Expand a leading ~ against the given home directory, as os.path.expanduser does for HOME."""
    if path == "~":
        return home
    if path.startswith("~/"):
        return os.path.join(home, path[2:])
    return path


def tilde(path: str, home: str) -> str:
    """A path for a button label: under the home directory it reads ~/..., anything else is unchanged."""
    if path == home or path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def split_command(command: str) -> tuple[str, ...]:
    """argv of a launcher command, shlex-split so 'myclaude run' is two words. LauncherError when it is blank."""
    try:
        argv = shlex.split(command or "")
    except ValueError as exc:
        raise LauncherError(f"the command cannot be split: {exc}") from exc
    if not argv:
        raise LauncherError(EMPTY_LAUNCHER_MESSAGE)
    return tuple(argv)


def default_launchers() -> dict[str, Any]:
    """The Claude launchers used when config has no 'launchers' value: name -> {cmd, config_dir}."""
    return {
        "claude": {"cmd": "claude", "config_dir": "~/.claude"},
    }


def default_config() -> dict[str, Any]:
    """The config used when config.json does not exist. A fresh dict on every call."""
    return {
        "launchers": default_launchers(),
        "allow_user_ids": [],
        "max_turn_seconds": 1800,
        "poll_seconds": 50,
        "max_output_chunks": 4,
        "default_mode": DEFAULT_MODE,
        "topics": {},
    }


def default_codex_home(home: str, env: Mapping[str, str]) -> str:
    """The Codex home folder: $CODEX_HOME when it is set, else ~/.codex."""
    return expand_home(env.get("CODEX_HOME") or CODEX_CONFIG_DIR, home)


@dataclass(frozen=True)
class LauncherSpec:
    """One launcher the topics can run. `argv` is the split command; `config_dir` is expanded.

    For a Claude launcher config_dir is the Claude config folder (transcripts live under projects/). For a Codex
    launcher it is the Codex home (rollouts live under sessions/).

    `configured` is False for an unnamed launcher built from a bare command that matches no visible entry (see
    Config.resolve_launcher). Such a spec runs like any Claude launcher, but the /launcher logic treats it as no
    configured launcher at all, as the live launcher_state does (its name is None).
    """

    name: str
    backend: str
    command: str
    argv: tuple[str, ...]
    config_dir: str
    configured: bool = True

    @classmethod
    def parse(cls, name: str, value: Any, *, section: str, home: str, codex_home: str) -> "LauncherSpec":
        """Validate one config entry. `section` is SECTION_CLAUDE ('launchers') or SECTION_AGENT ('agent_launchers').

        Raises LauncherError for a bad entry; the message never includes the value itself.
        """
        if section == SECTION_CLAUDE:
            backend, command, config_dir = _parse_claude_entry(name, value)
        elif section == SECTION_AGENT:
            backend, command, config_dir = _parse_agent_entry(name, value, codex_home)
        else:
            raise LauncherError(f"unknown config section {section!r}")
        if not command.strip():
            raise LauncherError(f"launcher {name!r} has an empty command")
        try:
            argv = split_command(command)
        except LauncherError as exc:
            raise LauncherError(f"launcher {name!r}: {exc}") from exc
        return cls(name=name, backend=backend, command=command, argv=argv,
                   config_dir=expand_home(config_dir, home))


def _text(name: str, field: str, value: Any) -> str:
    if not isinstance(value, str):
        raise LauncherError(f"launcher {name!r}: {field} must be a string")
    return value


def _parse_claude_entry(name: str, value: Any) -> tuple[str, str, str]:
    """(backend, command, config_dir) of a 'launchers' value: a command string or a {cmd, config_dir} object."""
    if isinstance(value, str):
        return BACKEND_CLAUDE, value, CLAUDE_CONFIG_DIR
    if not isinstance(value, Mapping):
        raise LauncherError(f"launcher {name!r}: a launchers entry is a command string or a cmd object")
    if "backend" in value:
        raise LauncherError(f"launcher {name!r}: launchers cannot set a backend, use agent_launchers")
    if "cmd" not in value:
        raise LauncherError(f"launcher {name!r}: a launchers object needs a cmd key")
    command = _text(name, "cmd", value["cmd"])
    config_dir = _text(name, "config_dir", value.get("config_dir") or CLAUDE_CONFIG_DIR)
    return BACKEND_CLAUDE, command, config_dir


def _parse_agent_entry(name: str, value: Any, codex_home: str) -> tuple[str, str, str]:
    """(backend, command, config_dir) of an 'agent_launchers' object: {backend, command, models, config_dir}."""
    if not isinstance(value, Mapping):
        raise LauncherError(f"launcher {name!r}: an agent_launchers entry must be an object")
    backend = value.get("backend")
    if backend not in KNOWN_BACKENDS:
        raise LauncherError(f"launcher {name!r}: backend must be one of: {', '.join(KNOWN_BACKENDS)}")
    command = _text(name, "command", value.get("command") or "")
    default_dir = codex_home if backend == BACKEND_CODEX else CLAUDE_CONFIG_DIR
    config_dir = _text(name, "config_dir", value.get("config_dir") or default_dir)
    return backend, command, config_dir


class Config:
    """Read access to config.json: typed getters, the launcher table and the precedence rule for a topic's launcher.

    The data is deep-copied on construction and never written back. `home` anchors ~ expansion and the default
    search roots; `env` supplies CODEX_HOME (os.environ when omitted). `log` receives one line per rejected launcher.
    """

    def __init__(self, data: Mapping[str, Any] | None = None, *, home: str,
                 env: Mapping[str, str] | None = None, log: LogFn = _discard) -> None:
        self._data: dict[str, Any] = copy.deepcopy(dict(data or {}))
        self._home = home
        self._codex_home = default_codex_home(home, os.environ if env is None else env)
        self._log = log
        self._specs: dict[str, LauncherSpec] = {}
        self._agent_models: dict[str, tuple[str, ...]] = {}
        self._build_launchers()

    @classmethod
    def load(cls, paths: Paths, *, env: Mapping[str, str] | None = None, log: LogFn = _discard) -> "Config":
        """Read paths.config. A missing file means the defaults, as the live file does. Nothing is written."""
        if os.path.exists(paths.config):
            with open(paths.config, encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError(f"{paths.config} must hold a JSON object")
        else:
            data = default_config()
        return cls(data, home=paths.home, env=env, log=log)

    @property
    def home(self) -> str:
        return self._home

    @property
    def codex_home(self) -> str:
        return self._codex_home

    # typed getters, same keys and defaults as the live file

    def allow_user_ids(self) -> list[Any]:
        return list(self._data.get("allow_user_ids") or [])

    def max_turn_seconds(self) -> float:
        return self._data.get("max_turn_seconds", 1800)

    def hard_cap_seconds(self) -> float:
        return self._data.get("hard_cap_seconds", 21600)

    def poll_seconds(self) -> float:
        return self._data.get("poll_seconds", 50)

    def max_output_chunks(self) -> int:
        return self._data.get("max_output_chunks", 4)

    def progress_flush_seconds(self) -> float:
        return self._data.get("progress_flush_seconds", 3)

    def idle_ping_seconds(self) -> float:
        return self._data.get("idle_ping_seconds", 60)

    def answer_doc_threshold(self) -> int:
        return self._data.get("answer_doc_threshold", 9000)

    def default_mode(self) -> str:
        """The permission mode a topic starts in: config default_mode, else bypassPermissions."""
        return str(self._data.get("default_mode") or DEFAULT_MODE)

    def topics(self) -> dict[str, Any]:
        found = self._data.get("topics")
        return dict(found) if isinstance(found, Mapping) else {}

    def default_topic(self) -> dict[str, Any] | None:
        found = self._data.get("default_topic")
        return dict(found) if isinstance(found, Mapping) and found else None

    def topic(self, name: str) -> dict[str, Any]:
        """The configured entry of a topic, else default_topic, else {} (the live turn lookup)."""
        found = self.topics().get(name) or self.default_topic() or {}
        return dict(found) if isinstance(found, Mapping) else {}

    def spawn(self) -> dict[str, Any]:
        found = self._data.get("spawn")
        return dict(found) if isinstance(found, Mapping) else {}

    def models(self) -> tuple[str, ...]:
        """The /model buttons: config 'models', else the built-in list."""
        return tuple(str(m) for m in self._data.get("models") or MODEL_CHOICES)

    def cwd_shortcuts(self) -> tuple[str, ...]:
        return tuple(str(p) for p in self._data.get("cwd_shortcuts") or [])

    def search_roots(self) -> tuple[str, ...]:
        """Where /cwd name searches look: config 'search_roots', default ~/workspace and the home directory."""
        found = self._data.get("search_roots")
        if found:
            return tuple(str(p) for p in found)
        return (os.path.join(self._home, "workspace"), self._home)

    def egress_proxy(self) -> str | None:
        value = self._data.get("egress_proxy")
        return str(value) if value else None

    # launchers

    def launcher_specs(self) -> dict[str, LauncherSpec]:
        """Every visible launcher, name -> spec: the 'launchers' section first, then the agent launchers."""
        return dict(self._specs)

    def launcher_choices(self) -> dict[str, str]:
        """name -> command of the visible launchers (the /launcher buttons)."""
        return {name: spec.command for name, spec in self._specs.items()}

    def launcher_config_dir(self, name: str) -> str | None:
        """The expanded config folder of launcher `name`, or None when it is not visible."""
        spec = self._specs.get(name)
        return spec.config_dir if spec else None

    def launcher_models(self, name: str) -> tuple[str, ...] | None:
        """The 'models' list of an agent launcher, or None when it has none."""
        return self._agent_models.get(name)

    def resolve_launcher(self, pick: str, topic_launcher: str, stored: str) -> LauncherSpec:
        """The launcher a turn runs: the stored pick while it is still in config, else the topic's configured
        launcher, else the one stored with its session. A topic or stored value is a command; it matches the
        visible launcher with that command, and an unmatched command runs as an unnamed Claude launcher whose
        name is the command. LauncherError when all three are empty.
        """
        if pick and pick in self._specs:
            return self._specs[pick]
        for command in (topic_launcher, stored):
            if command:
                return self._spec_for_command(command)
        raise LauncherError(EMPTY_LAUNCHER_MESSAGE)

    def _spec_for_command(self, command: str) -> LauncherSpec:
        for spec in self._specs.values():
            if spec.command == command:
                return spec
        argv = split_command(command)
        return LauncherSpec(name=command, backend=BACKEND_CLAUDE, command=command, argv=argv,
                            config_dir=expand_home(CLAUDE_CONFIG_DIR, self._home), configured=False)

    def _build_launchers(self) -> None:
        claude_section = self._data.get(SECTION_CLAUDE)
        if not isinstance(claude_section, Mapping) or not claude_section:
            claude_section = default_launchers()
        for name, value in claude_section.items():
            self._add_entry(str(name), value, SECTION_CLAUDE)
        agent_section = self._data.get(SECTION_AGENT)
        if not isinstance(agent_section, Mapping):
            return
        for name, value in agent_section.items():
            key = str(name)
            if key in {str(k) for k in claude_section}:
                self._log(f"config: agent launcher {key!r} hidden: the name is already a launchers entry")
                continue
            spec = self._add_entry(key, value, SECTION_AGENT)
            models = value.get("models") if spec and isinstance(value, Mapping) else None
            if isinstance(models, list):
                self._agent_models[key] = tuple(str(m) for m in models)

    def _add_entry(self, name: str, value: Any, section: str) -> LauncherSpec | None:
        try:
            spec = LauncherSpec.parse(name, value, section=section, home=self._home, codex_home=self._codex_home)
        except LauncherError as exc:
            self._log(f"config: launcher {name!r} hidden: {exc}")
            return None
        self._specs[name] = spec
        return spec
