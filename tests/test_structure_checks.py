"""Structure-check tooling (STEP 55): tests for tests/check_layers.py and tests/check_globals.py, and the
real-package checks of check_layers and check_globals.
"""

import os
import tempfile
import textwrap
import unittest

from tests import check_globals, check_layers

V3_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def _write_tree(root: str, files: dict[str, str]) -> None:
    for relative, source in files.items():
        path = os.path.join(root, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(textwrap.dedent(source))


class RealPackageTests(unittest.TestCase):
    """The shipped package passes the layer, size and globals rules (acceptance of STEP 55)."""

    def test_package_has_no_import_direction_violations(self) -> None:
        self.assertEqual(check_layers.scan(os.path.join(V3_DIR, "tg_bridge")), [])

    def test_package_has_no_import_cycles_and_no_oversized_module(self) -> None:
        problems = [line for line in check_layers.scan(os.path.join(V3_DIR, "tg_bridge")) if "cycle" in line or "lines" in line]
        self.assertEqual(problems, [])

    def test_package_has_no_module_level_mutable_state(self) -> None:
        self.assertEqual(check_globals.scan(), [])


class LayerRuleTests(unittest.TestCase):
    """check_layers flags what the design forbids and nothing the design allows."""

    def _scan(self, files: dict[str, str]) -> list[str]:
        with tempfile.TemporaryDirectory() as root:
            _write_tree(root, {f"tg_bridge/{name}": source for name, source in files.items()})
            return check_layers.scan(os.path.join(root, "tg_bridge"))

    def test_clean_tree_has_no_problems(self) -> None:
        self.assertEqual(self._scan({
            "__init__.py": "__version__ = '3'\n",
            "config.py": "import tg_bridge.paths\n",
            "paths.py": "X = 1\n",
            "store/kv.py": "from tg_bridge.store.db import Database\nfrom tg_bridge import __version__\n",
            "store/db.py": "from tg_bridge.clock import Clock\n",
            "agents/base.py": "from tg_bridge.agents.events import Event\nfrom tg_bridge.config import Config\n",
            "agents/events.py": "class Event: pass\n",
            "core/turn.py": "from tg_bridge.agents.base import Session\nfrom tg_bridge.feed.format import short\n",
            "feed/format.py": "from tg_bridge.agents.events import Event\n",
            "bot/inbound.py": "from tg_bridge.core.turn import Session\n",
            "daemon/poll.py": "from tg_bridge.bot.inbound import Session\n",
            "app.py": "from tg_bridge.daemon.poll import Session\n",
            "cli.py": "from tg_bridge.app import Session\n",
        }), [])

    def test_agents_may_not_import_telegram_or_store(self) -> None:
        problems = self._scan({
            "agents/bad.py": "from tg_bridge.telegram.client import Client\n",
            "agents/worse.py": "import tg_bridge.store.kv\n",
        })
        self.assertEqual(len(problems), 2)
        self.assertIn("may not import 'telegram.client'", problems[0])
        self.assertIn("agents/worse.py:1", problems[1])

    def test_core_may_not_import_bot_or_daemon(self) -> None:
        problems = self._scan({
            "core/bad.py": "def f():\n    from tg_bridge.bot.inbound import Handler\n",
            "core/worse.py": "from tg_bridge.daemon import lifecycle\n",
        })
        self.assertEqual(len(problems), 2)
        self.assertTrue(all("core/" in line for line in problems))

    def test_feed_may_not_import_core(self) -> None:
        self.assertEqual(len(self._scan({"feed/bad.py": "from tg_bridge.core.live import Live\n"})), 1)

    def test_relative_import_is_resolved_before_it_is_checked(self) -> None:
        problems = self._scan({"agents/rel.py": "from ..telegram import client\n"})
        self.assertEqual(len(problems), 1)
        self.assertIn("'telegram'", problems[0])

    def test_leaf_modules_are_importable_from_every_unit(self) -> None:
        self.assertEqual(self._scan({
            "store/kv.py": "from tg_bridge.logging_ import Logger\nfrom tg_bridge.config import Config\n",
            "telegram/client.py": "from tg_bridge.clock import Clock\nfrom tg_bridge.paths import Paths\n",
        }), [])

    def test_import_cycle_is_reported_once_as_its_members(self) -> None:
        problems = self._scan({
            "core/a.py": "from tg_bridge.core import b\n",
            "core/b.py": "def f():\n    from tg_bridge.core.a import g\n",
        })
        self.assertEqual(problems, ["import cycle: core.a -> core.b"])

    def test_module_over_the_line_limit_is_reported(self) -> None:
        source = "\n".join(f"x{i} = {i}" for i in range(check_layers.MAX_LINES + 1)) + "\n"
        problems = self._scan({"store/big.py": source})
        self.assertEqual(problems, [f"store/big.py:1: {check_layers.MAX_LINES + 1} lines (limit {check_layers.MAX_LINES})"])

    def test_module_at_the_line_limit_passes(self) -> None:
        source = "\n".join(f"x{i} = {i}" for i in range(check_layers.MAX_LINES)) + "\n"
        self.assertEqual(self._scan({"store/edge.py": source}), [])


class GlobalsRuleTests(unittest.TestCase):
    def test_module_level_list_is_a_violation(self) -> None:
        self.assertTrue(check_globals.check_source("CACHE_ITEMS = []\n"))

    def test_uppercase_frozen_constant_is_allowed(self) -> None:
        self.assertEqual(check_globals.check_source("LIMITS = (1, 2)\nPATTERN = 'x'\n"), [])


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
