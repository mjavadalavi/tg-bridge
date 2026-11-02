"""A fresh session started in a view tab (terminal_handoff.adopt_ended, TurnRunner): the pane writes its session when the CLI
exits; the next normal (spawned) turn adopts that uuid only when the topic has no stored session, and resumes it. A stored
uuid is never replaced. The record is read and removed on every normal turn. A FakeBackend records the spec it was given."""

import os
import time
import unittest

from tests.fakes.backend import FakeBackend
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, TurnHarness
from tg_bridge.agents.events import TurnResult, TurnStatus
from tg_bridge.core import terminal_claim

OLD_UUID = "11111111-2222-3333-4444-555555555555"
NEW_UUID = "66666666-7777-8888-9999-000000000000"
OK = TurnResult(TurnStatus.OK, answer="ok", duration_ms=5, num_turns=1, subtype="success")


class AdoptEndedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self._stop_live_turns)
        self.addCleanup(self.h.db.close)
        self.view = os.path.join(self.home.home, "view")
        os.makedirs(self.view)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")
        self.backend = FakeBackend([[OK], [OK]])
        self.runner = self.h.runner(self.h.registry_for(self.backend), view_dir=self.view)

    def _stop_live_turns(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)

    def leave_record(self, uuid: str) -> None:
        """What the view pane writes when its CLI exits in the tab."""
        terminal_claim.write_ended(self.view, THREAD, uuid, self.home.home, self.home.home, time.time())

    def record_exists(self) -> bool:
        return os.path.exists(os.path.join(self.view, f"{THREAD}.session"))

    def spawned_resume_id(self) -> str | None:
        self.assertEqual(len(self.backend.specs), 1, "the turn did not spawn one session")
        return self.backend.specs[0].resume_id

    def test_a_topic_without_a_session_resumes_the_session_its_tab_left(self) -> None:
        self.h.sessions.set_uuid(THREAD, "")
        self.leave_record(NEW_UUID)
        reply = self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        self.assertEqual(reply.answer, "ok")
        self.assertEqual(self.spawned_resume_id(), NEW_UUID, "the tab's session was not resumed")
        self.assertEqual(self.h.sessions.get(THREAD).uuid, NEW_UUID)
        self.assertIsNotNone(self.h.history.find(THREAD, NEW_UUID), "the adopted session has no history row")
        self.assertFalse(self.record_exists(), "the record was not removed")

    def test_a_stored_session_is_kept_and_the_tab_session_is_not_adopted(self) -> None:
        self.h.sessions.set_uuid(THREAD, OLD_UUID)
        self.leave_record(NEW_UUID)
        self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        self.assertEqual(self.spawned_resume_id(), OLD_UUID)
        self.assertEqual(self.h.sessions.get(THREAD).uuid, OLD_UUID, "the stored session was replaced")
        self.assertFalse(self.record_exists(), "the record was not removed")
        with open(self.h.log_path, encoding="utf-8") as handle:
            self.assertIn(f"terminal session {NEW_UUID} not adopted: topic already has {OLD_UUID}", handle.read())

    def test_no_record_changes_nothing(self) -> None:
        self.h.sessions.set_uuid(THREAD, "")
        self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        self.assertIsNone(self.spawned_resume_id(), "a topic with no record and no session starts a new one")
        self.assertEqual(self.h.sessions.get(THREAD).uuid, self.backend.specs[0].new_id)


if __name__ == "__main__":
    unittest.main()
