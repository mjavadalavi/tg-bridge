"""BackendRegistry: register, get (built once), for_launcher, names, and the UnknownBackend error."""

import os
import unittest

from tests.fakes.backend import FakeBackend
from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.registry import BackendRegistry, UnknownBackend
from tg_bridge.config import BACKEND_CODEX, Config, LauncherSpec
from tg_bridge.logging_ import Logger


def _launcher(backend: str, name: str = "x") -> LauncherSpec:
    return LauncherSpec(name=name, backend=backend, command=backend, argv=(backend,), config_dir="/nonexistent")


class BackendRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.config = Config({}, home=self.home.home)
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.clock = FakeClock()
        self.registry = BackendRegistry(self.config, self.logger, self.clock)

    def test_names_empty_before_registration(self) -> None:
        self.assertEqual(self.registry.names(), ())

    def test_register_then_get_returns_the_built_backend(self) -> None:
        self.registry.register("fake", lambda cfg, log, clk: FakeBackend())
        backend = self.registry.get("fake")
        self.assertIsInstance(backend, FakeBackend)
        self.assertEqual(self.registry.names(), ("fake",))

    def test_factory_receives_the_injected_config_logger_and_clock(self) -> None:
        seen: list[tuple[object, object, object]] = []

        def factory(cfg, log, clk):
            seen.append((cfg, log, clk))
            return FakeBackend()

        self.registry.register("fake", factory)
        self.registry.get("fake")
        self.assertEqual(seen, [(self.config, self.logger, self.clock)])

    def test_get_builds_each_backend_once(self) -> None:
        calls: list[int] = []

        def factory(cfg, log, clk):
            calls.append(1)
            return FakeBackend()

        self.registry.register("fake", factory)
        first = self.registry.get("fake")
        second = self.registry.get("fake")
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)

    def test_for_launcher_uses_the_launcher_backend_field(self) -> None:
        backend = FakeBackend()
        self.registry.register("fake", lambda cfg, log, clk: backend)
        self.assertIs(self.registry.for_launcher(_launcher("fake", name="alt")), backend)

    def test_unknown_name_raises_unknown_backend(self) -> None:
        with self.assertRaises(UnknownBackend) as caught:
            self.registry.get("nope")
        self.assertIn("nope", str(caught.exception))

    def test_unknown_backend_is_a_key_error(self) -> None:
        with self.assertRaises(KeyError):
            self.registry.get("nope")

    def test_for_launcher_with_unregistered_backend_raises(self) -> None:
        self.registry.register("fake", lambda cfg, log, clk: FakeBackend())
        with self.assertRaises(UnknownBackend):
            self.registry.for_launcher(_launcher(BACKEND_CODEX))

    def test_duplicate_registration_is_refused(self) -> None:
        self.registry.register("fake", lambda cfg, log, clk: FakeBackend())
        with self.assertRaises(ValueError):
            self.registry.register("fake", lambda cfg, log, clk: FakeBackend())

    def test_names_keep_registration_order(self) -> None:
        for name in ("codex", "claude", "fake"):
            self.registry.register(name, lambda cfg, log, clk: FakeBackend())
        self.assertEqual(self.registry.names(), ("codex", "claude", "fake"))

    def test_two_registries_do_not_share_built_backends(self) -> None:
        self.registry.register("fake", lambda cfg, log, clk: FakeBackend())
        other = BackendRegistry(self.config, self.logger, self.clock)
        other.register("fake", lambda cfg, log, clk: FakeBackend())
        self.assertIsNot(self.registry.get("fake"), other.get("fake"))


if __name__ == "__main__":
    unittest.main()
