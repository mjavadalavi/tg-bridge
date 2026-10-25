"""The shared live actions (bot/live_actions.py): the model switch and the background call that /model, the model
picker and the mode taps all use. One implementation, so a failure is logged the same way from every entry point."""

import time
import unittest
from unittest import mock

from tg_bridge.bot.callbacks.picker_taps import PickerTaps
from tg_bridge.bot.commands.settings import SettingsCommands
from tg_bridge.bot.live_actions import apply_model, spawn


class ApplyModelTests(unittest.TestCase):
    def test_switches_the_running_session_to_the_model(self) -> None:
        session = mock.Mock()
        logger = mock.Mock()
        apply_model(session, "opus", logger)
        session.set_model.assert_called_once_with("opus")
        logger.log.assert_not_called()

    def test_a_failed_switch_is_logged_and_not_raised(self) -> None:
        session = mock.Mock()
        session.set_model.side_effect = RuntimeError("no such model")
        logger = mock.Mock()
        apply_model(session, "bogus", logger)
        logger.log.assert_called_once_with("set_model bogus failed: no such model")


class SpawnTests(unittest.TestCase):
    def test_runs_the_call_on_a_daemon_thread(self) -> None:
        seen = []
        spawn(lambda value, other=None: seen.append((value, other, time.monotonic())), 1, other=2)
        deadline = time.monotonic() + 2
        while not seen and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual([(v, o) for v, o, _ in seen], [(1, 2)])


class SharedHelperTests(unittest.TestCase):
    def test_the_typed_and_the_tap_paths_do_not_keep_their_own_copies(self) -> None:
        for owner in (SettingsCommands, PickerTaps):
            self.assertFalse(hasattr(owner, "_apply_model"), owner.__name__)
            self.assertFalse(hasattr(owner, "_spawn"), owner.__name__)


if __name__ == "__main__":
    unittest.main()
