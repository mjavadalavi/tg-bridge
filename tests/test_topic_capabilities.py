"""topic_capabilities: the capability flags of a topic's turns, resolved the way the turn runner resolves its launcher.

A Claude topic whose launcher is tui gets the tui flags, a print launcher gets the print flags, and a topic whose
launcher resolves to nothing falls back to the backend's plain flags (the same fallback topic_backend has). The bot
layer reads its live and image flags through this function, so a tui topic's pickers and replies match its turns.
"""

import os
import unittest

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.claude.backend import CLAUDE_CAPABILITIES, CLAUDE_TUI_CAPABILITIES
from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.config import Config
from tg_bridge.core.topic_backend import topic_capabilities
from tg_bridge.logging_ import Logger
from tg_bridge.store.sessions import SessionRecord

PRINT_TOPIC = "PRINT"
BARE_TOPIC = "BARE"
PRINT_CMD = "wrap run"
TUI_CMD = "tui run"


def row_for(topic: str, launcher_pick: str = "") -> SessionRecord:
    return SessionRecord(thread_id=7, name=topic, uuid="", launcher="", cwd="", created="", title="", mode="",
                         model="", prev_uuid="", last_active="", launcher_pick=launcher_pick, backend="")


class TopicCapabilitiesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        data = {
            "launchers": {
                "wrap": {"cmd": PRINT_CMD, "config_dir": os.path.join(self.home.home, "cfg-a")},
                "tui": {"cmd": TUI_CMD, "config_dir": os.path.join(self.home.home, "cfg-b"), "transport": "tui"},
            },
            "topics": {PRINT_TOPIC: {"launcher": PRINT_CMD, "cwd": self.home.home}},
        }
        self.config = Config(data, home=self.home.home, env={})
        logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.backends = build_default_registry(self.config, logger, FakeClock(), title_launcher="")

    def test_a_topic_picked_onto_a_tui_launcher_has_the_tui_flags(self) -> None:
        flags = topic_capabilities(self.config, self.backends, row_for(PRINT_TOPIC, "tui"), PRINT_TOPIC)
        self.assertIs(flags, CLAUDE_TUI_CAPABILITIES)
        self.assertFalse(flags.live_mode)
        self.assertFalse(flags.live_model)

    def test_a_topic_on_a_print_launcher_has_the_print_flags(self) -> None:
        flags = topic_capabilities(self.config, self.backends, row_for(PRINT_TOPIC), PRINT_TOPIC)
        self.assertIs(flags, CLAUDE_CAPABILITIES)

    def test_a_topic_with_no_resolvable_launcher_falls_back_to_the_backend_flags(self) -> None:
        flags = topic_capabilities(self.config, self.backends, row_for(BARE_TOPIC), BARE_TOPIC)
        self.assertIs(flags, self.backends.get("claude").capabilities)


if __name__ == "__main__":
    unittest.main()
