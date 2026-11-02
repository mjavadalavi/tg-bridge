"""Config wrapper over dict, LauncherSpec{name,backend,command,argv}, defaults.

Config reads config.json once and never writes it. Every key the bridge reads has a typed getter here, with the
same default the live file uses. The launcher table is built once, at construction: each entry is parsed into a
LauncherSpec, and a rejected entry is logged with one line and hidden. Defaults are functions, not module
constants, so no mutable state is shared between Config objects.
"""

import copy
import json
import math
import os
import re
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from tg_bridge.paths import Paths

BACKEND_CLAUDE = "claude"
BACKEND_CODEX = "codex"
KNOWN_BACKENDS = (BACKEND_CLAUDE, BACKEND_CODEX)
TRANSPORT_PRINT = "print"
TRANSPORT_TUI = "tui"
TRANSPORTS = (TRANSPORT_PRINT, TRANSPORT_TUI)
SECTION_CLAUDE = "launchers"
SECTION_AGENT = "agent_launchers"
DEFAULT_MODE = "bypassPermissions"
CLAUDE_CONFIG_DIR = "~/.claude"
CODEX_CONFIG_DIR = "~/.codex"
MODEL_CHOICES = ("opus", "sonnet", "haiku", "fable", "default")
EMPTY_LAUNCHER_MESSAGE = "the launcher is empty: set it in config (topics or launchers)"
DEFAULT_TUI_SIZE = (50, 200)  # (rows, cols) of the TUI terminal; the config key is "COLSxROWS", default 200x50
TUI_SIZE_MAX = 1000          # each side of tui_size, in cells
DEFAULT_LINGER_SECONDS = 1800       # how long a session stays open after a user turn while background work runs; 0 = off
DEFAULT_LINGER_MAX_SESSIONS = 3     # how many such sessions may be open at once; 0 = none is kept
DEFAULT_TOPIC_PROBE_SECONDS = 600   # how often the daemon checks idle topics for a Telegram deletion; 0 = off
_TUI_SIZE = re.compile(r"([0-9]{1,4})x([0-9]{1,4})")

LogFn = Callable[[str], None]


def _non_negative(value: Any, default: float) -> Any:
    """The value when it is a finite number (not a bool) and not negative, else the default. json accepts NaN and
    Infinity, which are numbers to Python, so they are refused here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return default
    if isinstance(value, float) and not math.isfinite(value):
        return default
    return value


def _discard(message: str) -> None:
    """Default log sink: the line is dropped."""


def parse_tui_size(value: object) -> tuple[int, int] | None:
    """(rows, cols) from a 'COLSxROWS' string such as '200x50'. None when it is not one, or a side is 0 or too big."""
    match = _TUI_SIZE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        return None
    cols, rows = int(match[1]), int(match[2])
    if not (1 <= cols <= TUI_SIZE_MAX and 1 <= rows <= TUI_SIZE_MAX):
        return None
    return rows, cols


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

    `transport` is TRANSPORT_PRINT ('print' runs `claude -p` over stream-json) or TRANSPORT_TUI ('tui' runs the
    interactive TUI in a pseudo-terminal, for services that refuse non-interactive use). Only a Claude launcher may
    set it; the default is print.
    """

    name: str
    backend: str
    command: str
    argv: tuple[str, ...]
    config_dir: str
    configured: bool = True
    transport: str = TRANSPORT_PRINT

    @classmethod
    def parse(cls, name: str, value: Any, *, section: str, home: str, codex_home: str) -> "LauncherSpec":
        """Validate one config entry. `section` is SECTION_CLAUDE ('launchers') or SECTION_AGENT ('agent_launchers').

        Raises LauncherError for a bad entry; the message never includes the value itself.
        """
        if section == SECTION_CLAUDE:
            backend, command, config_dir, transport = _parse_claude_entry(name, value)
        elif section == SECTION_AGENT:
            backend, command, config_dir, transport = _parse_agent_entry(name, value, codex_home)
        else:
            raise LauncherError(f"unknown config section {section!r}")
        if not command.strip():
            raise LauncherError(f"launcher {name!r} has an empty command")
        try:
            argv = split_command(command)
        except LauncherError as exc:
            raise LauncherError(f"launcher {name!r}: {exc}") from exc
        return cls(name=name, backend=backend, command=command, argv=argv,
                   config_dir=expand_home(config_dir, home), transport=transport)


def _text(name: str, field: str, value: Any) -> str:
    if not isinstance(value, str):
        raise LauncherError(f"launcher {name!r}: {field} must be a string")
    return value


def _transport(name: str, value: Mapping[str, Any]) -> str:
    """The 'transport' key of an object entry: print when absent, else one of TRANSPORTS.

    The message names the allowed values but never the value itself, as the other errors do.
    """
    if "transport" not in value:
        return TRANSPORT_PRINT
    transport = value["transport"]
    if not isinstance(transport, str) or transport not in TRANSPORTS:
        raise LauncherError(f"launcher {name!r}: transport must be one of: {', '.join(TRANSPORTS)}")
    return transport


def _parse_claude_entry(name: str, value: Any) -> tuple[str, str, str, str]:
    """(backend, command, config_dir, transport) of a 'launchers' value: a command string or a {cmd, config_dir,
    transport} object. A command string is always print."""
    if isinstance(value, str):
        return BACKEND_CLAUDE, value, CLAUDE_CONFIG_DIR, TRANSPORT_PRINT
    if not isinstance(value, Mapping):
        raise LauncherError(f"launcher {name!r}: a launchers entry is a command string or a cmd object")
    if "backend" in value:
        raise LauncherError(f"launcher {name!r}: launchers cannot set a backend, use agent_launchers")
    if "cmd" not in value:
        raise LauncherError(f"launcher {name!r}: a launchers object needs a cmd key")
    command = _text(name, "cmd", value["cmd"])
    config_dir = _text(name, "config_dir", value.get("config_dir") or CLAUDE_CONFIG_DIR)
    return BACKEND_CLAUDE, command, config_dir, _transport(name, value)


def _parse_agent_entry(name: str, value: Any, codex_home: str) -> tuple[str, str, str, str]:
    """(backend, command, config_dir, transport) of an 'agent_launchers' object:
    {backend, command, models, config_dir, transport}. A transport other than print is for claude only."""
    if not isinstance(value, Mapping):
        raise LauncherError(f"launcher {name!r}: an agent_launchers entry must be an object")
    backend = value.get("backend")
    if backend not in KNOWN_BACKENDS:
        raise LauncherError(f"launcher {name!r}: backend must be one of: {', '.join(KNOWN_BACKENDS)}")
    transport = _transport(name, value)
    if backend != BACKEND_CLAUDE and transport != TRANSPORT_PRINT:
        raise LauncherError(f"launcher {name!r}: only a claude launcher can set transport")
    command = _text(name, "command", value.get("command") or "")
    default_dir = codex_home if backend == BACKEND_CODEX else CLAUDE_CONFIG_DIR
    config_dir = _text(name, "config_dir", value.get("config_dir") or default_dir)
    return backend, command, config_dir, transport


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
        self._tui_size = self._read_tui_size()
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

    def linger_seconds(self) -> float:
        """Seconds a session stays open after a user turn while background work is pending; 0 turns it off."""
        return _non_negative(self._data.get("linger_seconds"), DEFAULT_LINGER_SECONDS)

    def linger_max_sessions(self) -> int:
        """How many lingering sessions may be open at once; 0 means none is kept."""
        value = _non_negative(self._data.get("linger_max_sessions"), DEFAULT_LINGER_MAX_SESSIONS)
        return value if isinstance(value, int) else DEFAULT_LINGER_MAX_SESSIONS

    def topic_probe_seconds(self) -> float:
        """Seconds between two checks of idle topics for a topic deleted in Telegram; 0 turns the check off."""
        return _non_negative(self._data.get("topic_probe_seconds"), DEFAULT_TOPIC_PROBE_SECONDS)

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

    def tui_size(self) -> tuple[int, int]:
        """(rows, cols) of the terminal the tui transport draws on: config 'tui_size' as 'COLSxROWS', else 200x50."""
        return self._tui_size

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

    def _read_tui_size(self) -> tuple[int, int]:
        """The parsed tui_size. An invalid value is logged once here and replaced by the default."""
        raw = self._data.get("tui_size")
        if raw is None:
            return DEFAULT_TUI_SIZE
        parsed = parse_tui_size(raw)
        if parsed is None:
            self._log(f"config: tui_size {raw!r} ignored: use COLSxROWS such as 200x50; the default is used")
            return DEFAULT_TUI_SIZE
        return parsed

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
