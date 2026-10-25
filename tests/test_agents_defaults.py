"""The default registry builder (agents/defaults.py): what is registered, how the title launcher is chosen."""

import os
import unittest

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.claude.backend import ClaudeBackend
from tg_bridge.agents.codex.backend import CodexBackend
from tg_bridge.agents.defaults import (
    DEFAULT_TITLE_LAUNCHER,
    TITLE_LAUNCHER_ENV,
    build_default_registry,
    title_launcher_from_env,
)
from tg_bridge.agents.registry import UnknownBackend
from tg_bridge.config import BACKEND_CLAUDE, BACKEND_CODEX, Config, LauncherSpec
from tg_bridge.logging_ import Logger


class DefaultRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.config = Config({}, home=self.home.home)
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.clock = FakeClock()

    def test_claude_and_codex_are_registered(self) -> None:
        registry = build_default_registry(self.config, self.logger, self.clock)
        self.assertEqual(registry.names(), (BACKEND_CLAUDE, BACKEND_CODEX))
        self.assertIsInstance(registry.get(BACKEND_CLAUDE), ClaudeBackend)
        self.assertIsInstance(registry.get(BACKEND_CODEX), CodexBackend)

    def test_an_unregistered_backend_is_still_unknown(self) -> None:
        registry = build_default_registry(self.config, self.logger, self.clock)
        with self.assertRaises(UnknownBackend):
            registry.get("gemini")

    def test_a_launcher_resolves_to_the_one_built_claude_backend(self) -> None:
        registry = build_default_registry(self.config, self.logger, self.clock)
        spec = LauncherSpec(name="wrap", backend=BACKEND_CLAUDE, command="wrap run", argv=("wrap", "run"),
                            config_dir=self.home.home)
        self.assertIs(registry.for_launcher(spec), registry.get(BACKEND_CLAUDE))

    def test_each_registry_builds_its_own_backend(self) -> None:
        first = build_default_registry(self.config, self.logger, self.clock)
        second = build_default_registry(self.config, self.logger, self.clock)
        self.assertIsNot(first.get(BACKEND_CLAUDE), second.get(BACKEND_CLAUDE))

    def test_blank_title_launcher_turns_auto_titles_off(self) -> None:
        backend = build_default_registry(self.config, self.logger, self.clock, title_launcher="  ").get(BACKEND_CLAUDE)
        self.assertIsNone(backend.one_shot("prompt", timeout=1.0))

    def test_an_unsplittable_title_launcher_turns_auto_titles_off_and_does_not_stop_the_build(self) -> None:
        """A stray quote in TG_TITLE_LAUNCHER costs only the auto-title (live reads it inside its title try/except)."""
        registry = build_default_registry(self.config, self.logger, self.clock, title_launcher='alt "x')
        self.assertIsNone(registry.get(BACKEND_CLAUDE).one_shot("prompt", timeout=1.0))
        with open(os.path.join(self.home.state, "bridge.log"), encoding="utf-8") as handle:
            self.assertIn("title launcher", handle.read())

    def test_title_launcher_defaults_and_follows_the_environment_value(self) -> None:
        self.assertEqual(title_launcher_from_env({}), DEFAULT_TITLE_LAUNCHER)
        self.assertEqual(title_launcher_from_env({TITLE_LAUNCHER_ENV: "cheap seat"}), "cheap seat")


if __name__ == "__main__":
    unittest.main()
