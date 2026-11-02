"""The launcher, cwd, mode and model a topic resolves to: the one source for a turn's plan and for linger.

TurnRunner._plan takes its values from these helpers, and LingerService compares a kept process's TopicKey with the
TopicKey the topic resolves to now. Both sides use the same precedence, so a turn and the linger check cannot disagree:
launcher = the stored pick, else the topic's launcher, else the one stored with the session; cwd = the session's cwd,
else the topic's cwd, else the home folder; mode = the session's mode, else the default mode; model = the session's
model, or None when it is empty.

Pure functions only: no I/O, no clock, no locks. They raise LauncherError when no launcher resolves.

Threads: any (the functions only read the config and the row they are given). Lock map: none.
"""

from dataclasses import dataclass

from tg_bridge.config import Config, LauncherSpec
from tg_bridge.store.sessions import SessionRecord


@dataclass(frozen=True)
class TopicKey:
    """What a turn of one topic would run with. Two equal keys mean a kept process can serve the next turn."""

    launcher_name: str
    launcher_command: str
    cwd: str
    mode: str
    model: str | None


def topic_launcher(config: Config, row: SessionRecord, topic: str) -> LauncherSpec:
    """The launcher a turn of this topic runs. LauncherError when no launcher is configured or stored."""
    cfgt = config.topic(topic)
    return config.resolve_launcher(row.launcher_pick, str(cfgt.get("launcher") or ""), row.launcher)


def topic_cwd(config: Config, row: SessionRecord, topic: str) -> str:
    """The working folder of a turn of this topic: the session's cwd, else the topic's cwd, else the home folder."""
    return row.cwd or str(config.topic(topic).get("cwd") or "") or config.home


def topic_mode(config: Config, row: SessionRecord) -> str:
    """The permission mode of a turn: the session's mode, else the configured default."""
    return row.mode or config.default_mode()


def topic_model(row: SessionRecord) -> str | None:
    """The model of a turn: the session's model, or None (the CLI default) when it is empty."""
    return row.model or None


def topic_key(config: Config, row: SessionRecord, topic: str) -> TopicKey:
    """The TopicKey a turn of this topic would run with now. LauncherError when no launcher resolves."""
    launcher = topic_launcher(config, row, topic)
    return TopicKey(launcher_name=launcher.name, launcher_command=launcher.command,
                    cwd=topic_cwd(config, row, topic), mode=topic_mode(config, row), model=topic_model(row))
