"""Topic launcher and backend lookups for the bot layer (pickers, taps, /status, the photo path of inbound messages).

These resolve the launcher the way TurnRunner._plan does: the stored launcher_pick while it is still in config, else
the topic's configured launcher, else the launcher stored with the session. The backend is the one that launcher names.
sessions.backend is not used here: only LauncherSwitch and /resume write it, so after a config change (or a v2
'/launcher' that set only launcher_pick) it can name a backend the turns no longer run on.

When no launcher resolves at all (LauncherError) or the launcher's backend is not registered, the lookups fall back to
the stored backend column, so a display never fails where it used to work. TurnRunner still refuses such a turn.
"""

from tg_bridge.agents.base import AgentBackend
from tg_bridge.agents.registry import BackendRegistry, UnknownBackend
from tg_bridge.config import BACKEND_CLAUDE, Config, LauncherError, LauncherSpec
from tg_bridge.store.sessions import SessionRecord


def topic_launcher(config: Config, row: SessionRecord, topic: str) -> LauncherSpec | None:
    """The launcher a turn of this topic runs on, or None when config and the row name no launcher."""
    configured = str(config.topic(topic).get("launcher") or "")
    try:
        return config.resolve_launcher(row.launcher_pick, configured, row.launcher)
    except LauncherError:
        return None


def topic_backend(config: Config, backends: BackendRegistry, row: SessionRecord, topic: str) -> AgentBackend:
    """The backend the topic's turns run on now (the same one TurnRunner picks)."""
    spec = topic_launcher(config, row, topic)
    if spec is not None:
        try:
            return backends.for_launcher(spec)
        except UnknownBackend:
            pass
    return backends.get(row.backend or BACKEND_CLAUDE)


def topic_models(config: Config, backends: BackendRegistry, row: SessionRecord,
                 topic: str) -> tuple[AgentBackend, tuple[str, ...]]:
    """(backend, models) for the /model picker and its tap: the models of the topic's launcher when its config lists
    them (agent_launchers 'models'), else the backend's own list. The tap checks its name against the same tuple."""
    backend = topic_backend(config, backends, row, topic)
    spec = topic_launcher(config, row, topic)
    listed = config.launcher_models(spec.name) if spec is not None else None
    return backend, (listed if listed is not None else tuple(backend.models(config)))
