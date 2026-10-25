"""BackendRegistry (DESIGN section 3): maps a backend name to its factory and builds each backend once.

Factories take the Config, Logger and Clock the registry was built with, so nothing is read from a module global.
A backend is built on first use and then kept in this registry's own dict, under its own lock. The registry is
created once by App and passed to the services that need it. A factory runs under the registry lock, so it must
not call back into the registry.
"""

import threading
from collections.abc import Callable

from tg_bridge.agents.base import AgentBackend
from tg_bridge.clock import Clock
from tg_bridge.config import Config, LauncherSpec
from tg_bridge.logging_ import Logger

BackendFactory = Callable[[Config, Logger, Clock], AgentBackend]


class UnknownBackend(KeyError):
    """No factory is registered under that name. A KeyError, so a plain lookup still catches it."""


class BackendRegistry:
    """Name -> factory, plus one built instance per name. Safe to use from several threads."""

    def __init__(self, config: Config, logger: Logger, clock: Clock) -> None:
        self._config = config
        self._logger = logger
        self._clock = clock
        self._factories: dict[str, BackendFactory] = {}
        self._built: dict[str, AgentBackend] = {}
        self._lock = threading.Lock()

    def register(self, name: str, factory: BackendFactory) -> None:
        """Add a factory. A name can be registered once; a second registration is a ValueError."""
        with self._lock:
            if name in self._factories:
                raise ValueError(f"backend {name!r} is already registered")
            self._factories[name] = factory

    def get(self, name: str) -> AgentBackend:
        """The backend for a name, built on first use. UnknownBackend when no factory is registered."""
        with self._lock:
            backend = self._built.get(name)
            if backend is not None:
                return backend
            factory = self._factories.get(name)
            if factory is None:
                raise UnknownBackend(name)
            backend = factory(self._config, self._logger, self._clock)
            self._built[name] = backend
            return backend

    def for_launcher(self, spec: LauncherSpec) -> AgentBackend:
        """The backend a launcher's `backend` field names."""
        return self.get(spec.backend)

    def names(self) -> tuple[str, ...]:
        """Registered names in registration order."""
        with self._lock:
            return tuple(self._factories)
