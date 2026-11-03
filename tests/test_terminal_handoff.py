"""A session run in a view tab (terminal_handoff.adopt_ended, TurnRunner): the pane writes its session when the CLI
exits; the next normal (spawned) turn adopts that uuid when the topic has no stored session, and resumes it. A stored
uuid is replaced only when it is a session the tab ran (started from it or moved off it) and the tab ended on another
(a /clear in the tab). The record is read and removed on every normal turn. A FakeBackend records the spec it was
given."""

import os
import time
import unittest

from tests.fakes.backend import FakeBackend
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, TurnHarness
from tg_bridge.agents.events import TurnResult, TurnStatus
from tg_bridge.core import terminal_claim, terminal_handoff
from tg_bridge.logging_ import Logger

OLD_UUID = "11111111-2222-3333-4444-555555555555"
NEW_UUID = "66666666-7777-8888-9999-000000000000"
THIRD_UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
FOURTH_UUID = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
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
        self.plan_dir = self.h.config.resolve_launcher("", "topic-cmd", "").config_dir  # the topic's config folder

    def _stop_live_turns(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)

    def leave_record(self, uuid: str, started_uuid: str = "", seen: tuple[str, ...] = (), config_dir: str = "") -> None:
        """What the view pane writes when its CLI exits in the tab. The tab ran under the topic's config folder unless
        config_dir names another one."""
        terminal_claim.write_ended(self.view, THREAD, uuid, config_dir or self.plan_dir, self.home.home, time.time(),
                                   started_uuid=started_uuid, seen=seen)

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

    def test_a_tab_that_moved_off_the_stored_session_hands_its_last_session_to_the_topic(self) -> None:
        self.h.sessions.set_uuid(THREAD, OLD_UUID)
        self.leave_record(NEW_UUID, started_uuid=OLD_UUID)
        reply = self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        self.assertEqual(reply.answer, "ok")
        self.assertEqual(self.spawned_resume_id(), NEW_UUID, "the tab's session after /clear was not resumed")
        self.assertEqual(self.h.sessions.get(THREAD).uuid, NEW_UUID)
        self.assertIsNotNone(self.h.history.find(THREAD, NEW_UUID), "the adopted session has no history row")
        self.assertFalse(self.record_exists(), "the record was not removed")

    def test_a_tab_that_moved_twice_after_a_telegram_turn_hands_over_its_last_session(self) -> None:
        self.h.sessions.set_uuid(THREAD, NEW_UUID)
        self.leave_record(THIRD_UUID, started_uuid=OLD_UUID, seen=(OLD_UUID, NEW_UUID))
        reply = self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        self.assertEqual(reply.answer, "ok")
        self.assertEqual(self.spawned_resume_id(), THIRD_UUID, "the tab's last session was not resumed")
        self.assertEqual(self.h.sessions.get(THREAD).uuid, THIRD_UUID)
        self.assertIsNotNone(self.h.history.find(THREAD, THIRD_UUID), "the adopted session has no history row")
        self.assertFalse(self.record_exists(), "the record was not removed")

    def test_a_stored_session_outside_the_tab_lineage_is_kept(self) -> None:
        self.h.sessions.set_uuid(THREAD, THIRD_UUID)
        self.leave_record(FOURTH_UUID, started_uuid=OLD_UUID, seen=(OLD_UUID, NEW_UUID))
        self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        self.assertEqual(self.spawned_resume_id(), THIRD_UUID)
        self.assertEqual(self.h.sessions.get(THREAD).uuid, THIRD_UUID, "the stored session was replaced")
        self.assertFalse(self.record_exists(), "the record was not removed")
        with open(self.h.log_path, encoding="utf-8") as handle:
            self.assertIn(f"terminal session {FOURTH_UUID} not adopted: topic already has {THIRD_UUID}", handle.read())

    def test_a_tab_started_from_another_session_is_not_adopted(self) -> None:
        self.h.sessions.set_uuid(THREAD, OLD_UUID)
        self.leave_record(NEW_UUID, started_uuid=THIRD_UUID)
        self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        self.assertEqual(self.spawned_resume_id(), OLD_UUID)
        self.assertEqual(self.h.sessions.get(THREAD).uuid, OLD_UUID, "the stored session was replaced")
        self.assertFalse(self.record_exists(), "the record was not removed")
        with open(self.h.log_path, encoding="utf-8") as handle:
            self.assertIn(f"terminal session {NEW_UUID} not adopted: topic already has {OLD_UUID}", handle.read())

    def test_a_tab_session_from_the_same_config_folder_written_with_another_spelling_is_adopted(self) -> None:
        self.h.sessions.set_uuid(THREAD, "")
        self.leave_record(NEW_UUID, config_dir=self.plan_dir + os.sep)
        self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        self.assertEqual(self.spawned_resume_id(), NEW_UUID, "the same folder under another spelling was not adopted")
        self.assertEqual(self.h.sessions.get(THREAD).uuid, NEW_UUID)

    def test_a_tab_session_from_another_config_folder_is_not_adopted(self) -> None:
        other = os.path.join(self.home.home, "other-claude")
        self.h.sessions.set_uuid(THREAD, OLD_UUID)
        self.leave_record(NEW_UUID, started_uuid=OLD_UUID, config_dir=other)
        self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        self.assertEqual(self.spawned_resume_id(), OLD_UUID)
        self.assertEqual(self.h.sessions.get(THREAD).uuid, OLD_UUID, "the stored session was replaced")
        self.assertFalse(self.record_exists(), "the record was not removed")
        with open(self.h.log_path, encoding="utf-8") as handle:
            self.assertIn(f"terminal session {NEW_UUID} not adopted: it ran under {other}, "
                          f"the topic's launcher uses {self.plan_dir}", handle.read())

    def test_no_record_changes_nothing(self) -> None:
        self.h.sessions.set_uuid(THREAD, "")
        self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        self.assertIsNone(self.spawned_resume_id(), "a topic with no record and no session starts a new one")
        self.assertEqual(self.h.sessions.get(THREAD).uuid, self.backend.specs[0].new_id)


class MoveClaimTests(unittest.TestCase):
    """move_claim: an attached turn moves the live claim only while it still names the uuid the turn started from."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.view = os.path.join(self.home.home, "view")
        os.makedirs(self.view)
        self.logger = Logger(os.path.join(self.home.home, "bridge.log"))
        self.pid = os.getpid()  # a live pid: the claim is not stale
        self.assertTrue(terminal_claim.take_claim(self.view, THREAD, self.pid, time.time(), uuid=OLD_UUID))

    def claim_uuid(self) -> str:
        claim = terminal_claim.live_claim(self.view, THREAD)
        self.assertIsNotNone(claim, "the claim is gone")
        return claim.uuid

    def test_move_claim_does_not_overwrite_a_newer_uuid(self) -> None:
        terminal_claim.update_uuid(self.view, THREAD, self.pid, NEW_UUID)  # the SessionStart hook wrote Z
        terminal_handoff.move_claim(self.view, THREAD, self.pid, THIRD_UUID, self.logger, expected=OLD_UUID)
        self.assertEqual(self.claim_uuid(), NEW_UUID, "the daemon's move overwrote the hook's newer uuid")

    def test_move_claim_moves_when_the_claim_still_names_the_start_uuid(self) -> None:
        terminal_handoff.move_claim(self.view, THREAD, self.pid, THIRD_UUID, self.logger, expected=OLD_UUID)
        self.assertEqual(self.claim_uuid(), THIRD_UUID)

    def test_move_claim_leaves_a_hooked_claim_alone(self) -> None:
        os.remove(terminal_claim.claim_path(self.view, THREAD))  # replace the setUp claim by a hooked one
        self.assertTrue(terminal_claim.take_claim(self.view, THREAD, self.pid, time.time(), uuid=OLD_UUID, hooked=True))
        terminal_handoff.move_claim(self.view, THREAD, self.pid, NEW_UUID, self.logger, expected=OLD_UUID)
        self.assertEqual(self.claim_uuid(), OLD_UUID, "the daemon rewrote the session of a hooked claim")
        self.assertTrue(terminal_claim.live_claim(self.view, THREAD).hooked)


if __name__ == "__main__":
    unittest.main()
