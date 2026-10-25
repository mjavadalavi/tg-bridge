"""The default BackendRegistry: which adapters exist and how they are configured from the outside.

App calls build_default_registry() once. Each factory captures its own configuration as values (the title argv here),
never through a module global. A backend that is not implemented yet is simply not registered, so asking for it
raises UnknownBackend. Codex is registered here with no title argv: titles always use the Claude launcher.
"""

from collections.abc import Callable, Mapping

from tg_bridge.agents.base import AgentBackend
from tg_bridge.agents.claude.backend import ClaudeBackend
from tg_bridge.agents.codex.backend import CodexBackend
from tg_bridge.agents.registry import BackendRegistry
from tg_bridge.clock import Clock
from tg_bridge.config import BACKEND_CLAUDE, BACKEND_CODEX, Config, LauncherError, split_command
from tg_bridge.logging_ import Logger

TITLE_LAUNCHER_ENV = "TG_TITLE_LAUNCHER"
DEFAULT_TITLE_LAUNCHER = "claude"   # the launcher used only for one-time topic titles


def title_launcher_from_env(env: Mapping[str, str]) -> str:
    """The title launcher named by TG_TITLE_LAUNCHER, else the default. Read once by the caller, never here."""
    return env.get(TITLE_LAUNCHER_ENV, DEFAULT_TITLE_LAUNCHER)


def _title_argv(title_launcher: str, logger: Logger) -> tuple[str, ...]:
    """The split title launcher; a blank or unsplittable one gives () (auto-titles off). A bad value costs only the
    titles, never the daemon start (live reads TG_TITLE_LAUNCHER inside its title try/except)."""
    if not title_launcher.strip():
        return ()
    try:
        return split_command(title_launcher)
    except LauncherError as exc:
        logger.log(f"title launcher {title_launcher!r} ignored, auto-titles off: {exc}")
        return ()


def _claude_factory(title_argv: tuple[str, ...]) -> Callable[[Config, Logger, Clock], AgentBackend]:
    def factory(config: Config, logger: Logger, clock: Clock) -> AgentBackend:
        return ClaudeBackend(logger=logger, clock=clock, title_argv=title_argv)
    return factory


def _codex_factory(config: Config, logger: Logger, clock: Clock) -> AgentBackend:
    return CodexBackend(logger=logger, clock=clock)


def build_default_registry(config: Config, logger: Logger, clock: Clock, *,
                           title_launcher: str = DEFAULT_TITLE_LAUNCHER) -> BackendRegistry:
    """A registry with every implemented backend registered. A blank title_launcher turns auto-titles off."""
    registry = BackendRegistry(config, logger, clock)
    title_argv = _title_argv(title_launcher, logger)
    registry.register(BACKEND_CLAUDE, _claude_factory(title_argv))
    registry.register(BACKEND_CODEX, _codex_factory)
    return registry
