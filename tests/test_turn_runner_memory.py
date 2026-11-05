"""TurnRunner gives each Claude topic its own auto-memory folder: <memory root>/<thread id>, created 0700 before the
launch and named in the SessionSpec. A Codex topic gets none, and a runner built without a memory root names none.
Runs on FakeBackend (tests/fakes/backend.py), no process is started.
"""

import os
import stat
import unittest

from tests.fakes.backend import FAKE_CAPABILITIES, FakeBackend
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, ProbingBackend, TurnHarness
from tg_bridge.agents.events import TextEvent, TurnResult, TurnStatus

OK = TurnResult(TurnStatus.OK, answer="ok", duration_ms=5, num_turns=1, subtype="success")
CODEX_LAUNCHER = "codexy"


class TopicMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")
        self.memory_root = os.path.join(self.home.state, "memory")
        self.folder = os.path.join(self.memory_root, str(THREAD))

    def run_one_turn(self, backend: FakeBackend, *, registry_name: str = "claude", memory_dir: str) -> FakeBackend:
        runner = self.h.runner(self.h.registry_for(backend, name=registry_name), memory_dir=memory_dir)
        runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        self.assertEqual(len(backend.specs), 1)
        return backend

    def test_a_claude_topic_is_named_its_own_folder_under_the_memory_root(self) -> None:
        backend = self.run_one_turn(FakeBackend([[TextEvent(text="x"), OK]], capabilities=FAKE_CAPABILITIES),
                                    memory_dir=self.memory_root)
        self.assertEqual(backend.specs[0].memory_dir, self.folder)
        self.assertTrue(os.path.isdir(self.folder))
        self.assertEqual(stat.S_IMODE(os.stat(self.folder).st_mode), 0o700)

    def test_the_folder_exists_before_the_session_is_opened(self) -> None:
        backend = ProbingBackend([[TextEvent(text="x"), OK]], capabilities=FAKE_CAPABILITIES)
        backend.probe = lambda: "folder" if os.path.isdir(self.folder) else "missing"
        self.run_one_turn(backend, memory_dir=self.memory_root)
        self.assertEqual(backend.probed, "folder")

    def test_a_codex_topic_is_named_no_memory_folder_and_none_is_created(self) -> None:
        self.h.set_config(agent_launchers={CODEX_LAUNCHER: {"backend": "codex", "command": "codex"}})
        self.h.sessions.set_launcher_pick(THREAD, CODEX_LAUNCHER)
        backend = self.run_one_turn(FakeBackend([[TextEvent(text="x"), OK]], capabilities=FAKE_CAPABILITIES),
                                    registry_name="codex", memory_dir=self.memory_root)
        self.assertEqual(backend.specs[0].memory_dir, "")
        self.assertFalse(os.path.exists(self.folder))

    def test_without_a_memory_root_the_spec_names_no_folder(self) -> None:
        backend = self.run_one_turn(FakeBackend([[TextEvent(text="x"), OK]], capabilities=FAKE_CAPABILITIES),
                                    memory_dir="")
        self.assertEqual(backend.specs[0].memory_dir, "")
        self.assertFalse(os.path.exists(self.memory_root))


if __name__ == "__main__":
    unittest.main()
