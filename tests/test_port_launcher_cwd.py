"""Port of fakecli/test_launcher_cwd.py: the cwd scenarios that the other suites do not state in these words (the
typed search returns at once and a slow search still answers, a stranger's tap on a search card, the new-topic card
with the real folder finder). The other cwd labels are in test_cwd, test_cmd_settings, test_callback_taps,
test_cmd_session, test_inbound, test_repos_sessions and test_turn_runner, and are listed in their docstrings.
"""

import os
import threading
import time
import unittest

from tests.callback_support import CHAT, THREAD, TOPIC
from tests.port_launcher_support import JOIN_SECONDS, LauncherCwdPortCase
from tg_bridge.config import Config, tilde
from tg_bridge.core.cwd import CwdService
from tg_bridge.core.cwd_search import find_dirs
from tg_bridge.core.handoff import HandoffService
from tg_bridge.store.inbox import CwdRepo


class CwdSearchPort(LauncherCwdPortCase):
    def mkdir(self, *parts: str) -> str:
        path = os.path.join(self.base, *parts)
        os.makedirs(path, exist_ok=True)
        return os.path.realpath(path)

    def test_a_typed_search_returns_at_once_and_a_slow_search_still_answers(self) -> None:
        """Ported labels:
        C: /cwd <name> returns at once, the search runs elsewhere
        H: the slow search still answers
        """
        gate = threading.Event()
        self.finder.gate = gate
        started = time.monotonic()
        self.assertTrue(self.commands.dispatch(self.deps, CHAT, THREAD, TOPIC, "/cwd slow-search"))
        self.assertLess(time.monotonic() - started, 0.5, "the typed /cwd waited for the search")
        self.assertEqual(self.sent_buttons(), [])
        self.assertEqual(self.telegram.texts(THREAD), [])
        gate.set()
        self.assertTrue(self.search.join(JOIN_SECONDS) and self.out.join(JOIN_SECONDS))
        self.assertEqual(self.last_text(), 'no folder matching "slow-search" — send a full path')

    def test_a_stranger_cannot_apply_a_folder_from_a_search_card(self) -> None:
        """Ported labels:
        G: a non-allowed tapper is refused
        """
        self.finder.result = [self.mkdir("alpha-dir")]
        self.run_cmd("/cwd alpha")
        self.tap(f"cm:{THREAD}:cs:0", tapper=7)
        self.assertEqual(self.answers(), ["not allowed"])
        self.assertEqual(self.row().cwd, self.project)

    def test_the_new_topic_card_lists_the_exact_folder_first(self) -> None:
        """Ported labels:
        I1: exact folder first (['~/l3search-xk0vk4s7/alpha-tool'])
        I1: up to 4 folder buttons (3)
        """
        root = self.mkdir("l3search-xk0vk4s7")
        for part in ("alpha-tool", "alpha-tool-old", "budget-alpha-tool"):
            os.makedirs(os.path.join(root, part))
        cfg = Config({"search_roots": [root], "allow_user_ids": [42], "topics": {}, "spawn": {"budget": "/bin/true"},
                      "default_topic": {"cwd": self.project}}, home=self.base, env={})
        handoff = HandoffService(self.kv, self.backends, self.logger)
        real = CwdService(sessions=self.sessions, kv=self.kv, paths=CwdRepo(self.db), messenger=self.messenger,
                          live=self.live, search_pool=self.search, out_pool=self.out, logger=self.logger,
                          clock=self.clock, handoff=handoff, finder=find_dirs)
        real.suggest_for_new_topic(cfg, CHAT, THREAD, "🚀 alpha-tool 🔥")
        self.assertTrue(self.search.join(JOIN_SECONDS) and self.out.join(JOIN_SECONDS))
        labels = self.card_labels()
        self.assertEqual(labels[0], tilde(os.path.join(root, "alpha-tool"), self.base))
        self.assertEqual(len(labels) - 1, 3, "three folder buttons, then the keep button")
        self.assertTrue(labels[-1].startswith("🏠 keep "))


if __name__ == "__main__":
    unittest.main()
