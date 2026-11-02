"""Paths: every location is derived from the home directory passed in, under a TempHome."""

import os
import unittest

from tests.support import TempHome
from tg_bridge.paths import Paths


class PathsTests(unittest.TestCase):
    def test_paths_are_under_the_state_directory_of_the_home(self) -> None:
        with TempHome() as home:
            paths = Paths.from_home(home.home)
            self.assertEqual(paths.home, home.home)
            self.assertEqual(paths.state, home.state)
            self.assertEqual(paths.token, os.path.join(home.state, "token"))
            self.assertEqual(paths.soniox_key, os.path.join(home.state, "soniox-key"))
            self.assertEqual(paths.config, os.path.join(home.state, "config.json"))
            self.assertEqual(paths.db, os.path.join(home.state, "tg-bridge.db"))
            self.assertEqual(paths.spool, os.path.join(home.state, "spool"))
            self.assertEqual(paths.inbox, os.path.join(home.state, "inbox"))
            self.assertEqual(paths.log, os.path.join(home.state, "bridge.log"))
            self.assertEqual(paths.lock, os.path.join(home.state, "lock"))
            self.assertEqual(paths.turns, os.path.join(home.state, "turns"))
            self.assertEqual(paths.view, os.path.join(home.state, "view"))

    def test_from_environment_follows_home_at_call_time(self) -> None:
        with TempHome() as home:
            self.assertEqual(Paths.from_environment().state, home.state)

    def test_construction_creates_nothing_on_disk(self) -> None:
        with TempHome() as home:
            paths = Paths.from_home(os.path.join(home.home, "fresh"))
            self.assertFalse(os.path.exists(paths.state))

    def test_paths_are_frozen(self) -> None:
        with TempHome() as home:
            paths = Paths.from_home(home.home)
            with self.assertRaises(AttributeError):
                paths.config = "/elsewhere"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
