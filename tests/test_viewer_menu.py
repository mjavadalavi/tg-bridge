"""The tab menu of the tg-bridge view session (card D4, tg_bridge/daemon/view_menu.py). The fake cases script tmux with
ScriptedTmux: the file ensure_session writes and sources, the marker that stops a second sourcing, the listing failures.
The real case starts a private tmux server (TMUX_TMPDIR in a scratch directory, no config), sources the generated file on
it and lists the binding back: tmux must accept the file and the binding must carry view-delete and the original menu.
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Sequence

from tests.viewer_support import ENTRY, HomeCase, ScriptedTmux
from tg_bridge.daemon.view_menu import (DELETE_MARKER, MENU_FILE, MENU_KEY, _delete_item, menu_config,
                                        original_binding)
from tg_bridge.daemon.viewer import SESSION, Viewer

FAKE_LISTING = ('bind-key -T root MouseDown3Status       display-menu -T "#[align=centre]#{window_index}:#{window_name}" '
                '-t = -x W -y W "#{?#{>:#{session_windows},1},,-}Swap Left" l { swap-window -t :-1 } '
                'Kill X { kill-window } Rename n { command-prompt -F -I "#W" { rename-window -t "#{window_id}" "%%" } }\n')
FAKE_ORIGINAL = FAKE_LISTING.split("MouseDown3Status", 1)[1].strip()  # what the else-branch must carry, verbatim
VIEW_DELETE_CMD = shlex.join([sys.executable, "-I", ENTRY, "view-delete"])
MARKED_LISTING = ('bind-key -T root MouseDown3Status display-menu -T "x" -t = -x W -y W "Delete" d '
                  '{ run-shell "view-delete #{@tg_thread}" }\n')  # parses as a menu: only the marker stops a sourcing


class SourcingTmux(ScriptedTmux):
    """A ScriptedTmux that, like tmux, shows the sourced file's binding in list-keys from then on."""

    def __call__(self, args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        result = super().__call__(args, timeout)
        if args[0] == "source-file":
            with open(args[1], encoding="utf-8") as handle:
                self.keys = handle.read()
        return result


class MenuFileTests(HomeCase):
    def viewer(self, tmux: ScriptedTmux) -> Viewer:
        return Viewer(self.paths, self.config, run=tmux)

    def menu_path(self) -> str:
        return os.path.join(self.paths.view, MENU_FILE)

    def test_the_first_call_sources_a_file_with_the_session_guard_the_delete_item_and_the_original_menu(self) -> None:
        tmux = ScriptedTmux(has_session=1, keys=FAKE_LISTING)
        self.viewer(tmux).ensure_session()
        self.assertEqual(tmux.named("source-file"), [["source-file", self.menu_path()]])
        with open(self.menu_path(), encoding="utf-8") as handle:
            text = handle.read()
        self.assertTrue(text.startswith(f"bind-key -T root {MENU_KEY} if-shell -F '#{{==:#{{session_name}},{SESSION}}}' {{ "))
        self.assertIn('"Delete topic" d { if-shell -F "#{@tg_thread}" { confirm-before -p "Delete this topic here and in '
                      'Telegram? (y/n)" { run-shell -b "', text)
        self.assertIn(f"{VIEW_DELETE_CMD} #{{@tg_thread}}\" }} }} }} '' \"#{{?", text)  # the item, a separator, the menu
        self.assertTrue(text.endswith(f"}} {{ {FAKE_ORIGINAL} }}\n"))
        self.assertIn(' "#{?#{>:#{session_windows},1},,-}Swap Left" l { swap-window -t :-1 }', text)

    def test_a_second_call_with_the_marker_in_the_binding_does_not_source_again(self) -> None:
        tmux = SourcingTmux(has_session=0, keys=FAKE_LISTING)
        viewer = self.viewer(tmux)
        viewer.ensure_session()
        viewer.ensure_session()
        self.assertEqual(len(tmux.named("source-file")), 1)
        self.assertIn(DELETE_MARKER, tmux.keys)

    def test_a_binding_that_already_has_the_marker_is_not_sourced(self) -> None:
        tmux = ScriptedTmux(has_session=1, keys=MARKED_LISTING)
        self.viewer(tmux).ensure_session()
        self.assertEqual(tmux.named("source-file"), [])
        self.assertFalse(os.path.exists(self.menu_path()))

    def test_a_failed_list_keys_sources_nothing(self) -> None:
        tmux = ScriptedTmux(has_session=1, keys_code=1)
        self.viewer(tmux).ensure_session()
        self.assertEqual(tmux.named("source-file"), [])
        self.assertFalse(os.path.exists(self.menu_path()))

    def test_an_empty_listing_sources_nothing(self) -> None:
        tmux = ScriptedTmux(has_session=1, keys="")
        self.viewer(tmux).ensure_session()
        self.assertEqual(tmux.named("source-file"), [])

    def test_a_binding_that_is_not_a_display_menu_is_left_alone(self) -> None:
        tmux = ScriptedTmux(has_session=1, keys="bind-key -T root MouseDown3Status run-shell foo\n")
        self.viewer(tmux).ensure_session()
        self.assertEqual(tmux.named("source-file"), [])

    def test_the_listing_asks_for_the_root_table_of_the_right_click_on_the_status_line(self) -> None:
        tmux = ScriptedTmux(has_session=1, keys=FAKE_LISTING)
        self.viewer(tmux).ensure_session()
        self.assertEqual(tmux.named("list-keys"), [["list-keys", "-T", "root", MENU_KEY]])

    def test_no_window_name_is_substituted_into_the_delete_item(self) -> None:
        """tmux substitutes formats before it parses a command: a window or topic name there could break the item or run
        a shell command. The item must carry only the thread id format; the original menu items are copied as they are."""
        item = _delete_item(VIEW_DELETE_CMD)
        self.assertNotIn("window_name", item)
        self.assertEqual(set(re.findall(r"#\{[^}]*\}", item)), {"#{@tg_thread}"})
        text = menu_config(original_binding(FAKE_LISTING), VIEW_DELETE_CMD, SESSION)
        self.assertIsNotNone(text)
        self.assertIn(item, text)

    def test_the_item_does_nothing_for_an_untagged_window_and_the_else_branch_is_the_original(self) -> None:
        text = menu_config(original_binding(FAKE_LISTING), VIEW_DELETE_CMD, SESSION)
        self.assertIsNotNone(text)
        self.assertIn('if-shell -F "#{@tg_thread}" {', text)
        self.assertEqual(text.count("{ " + FAKE_ORIGINAL + " }"), 1)


class RealTmuxMenuTests(unittest.TestCase):
    """tmux itself reads the generated file: a private server with no config, the file sourced, the binding listed back."""

    def setUp(self) -> None:
        if shutil.which("tmux") is None:
            self.skipTest("tmux is not installed")
        self.tmpdir = tempfile.mkdtemp(prefix="tgbd")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.env = {key: value for key, value in os.environ.items() if key != "TMUX"}
        self.env["TMUX_TMPDIR"] = self.tmpdir
        self.addCleanup(self.tmux, "kill-server")

    def tmux(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["tmux", "-f", "/dev/null", *args], env=self.env, capture_output=True, text=True,
                              timeout=20)

    def test_tmux_accepts_the_file_and_lists_the_delete_item_with_the_original_menu(self) -> None:
        self.assertEqual(self.tmux("new-session", "-d", "-s", "probe", "sleep 600").returncode, 0)
        listing = self.tmux("list-keys", "-T", "root", MENU_KEY).stdout
        original = original_binding(listing)
        self.assertIsNotNone(original)
        text = menu_config(original, VIEW_DELETE_CMD, SESSION)
        self.assertIsNotNone(text)
        path = os.path.join(self.tmpdir, MENU_FILE)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        sourced = self.tmux("source-file", path)
        self.assertEqual(sourced.returncode, 0, sourced.stderr)
        bound = self.tmux("list-keys", "-T", "root", MENU_KEY).stdout
        self.assertIn(DELETE_MARKER, bound)
        self.assertIn("Delete topic", bound)
        self.assertIn("if-shell", bound)
        self.assertIn("Kill X { kill-window }", bound)  # the original items are still in the menu


if __name__ == "__main__":
    unittest.main()
