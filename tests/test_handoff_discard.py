"""Handoff digest discard (review fix): a pending digest left by a cross-backend switch is removed when the topic's
session binding changes outside that switch, so a later fresh session never gets an old conversation's summary.

Scenarios: claude -> codex stores the digest, then /new (through the command registry) and the next turn's system rules
carry no digest; claude -> codex, /resume <claude uuid>, /new, and again no digest; claude -> codex, then a same-backend
launcher switch (codex -> codexb) and no digest on the next turn; the /stop-and-new background half (finish_new) and a
/cwd change. The Claude transcript is the handoff fixture, read by the real ClaudeBackend; the turns run on FakeBackend.
"""

import os
import shutil
import unittest

from tests.fakes.backend import FAKE_CAPABILITIES, FakeBackend
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, TurnHarness
from tg_bridge.agents.claude.history import encoded_cwd
from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.agents.events import TurnResult, TurnStatus
from tg_bridge.bot.commands.base import CommandDeps, CommandRegistry
from tg_bridge.bot.commands.session import SessionCommands
from tg_bridge.core.cwd import CwdService
from tg_bridge.core.handoff import HEADER, HandoffService, handoff_key
from tg_bridge.core.launcher_switch import FRESH_TEXT, KIND_CHANGED, LauncherSwitch
from tg_bridge.core.topic_queues import TopicQueues
from tg_bridge.core.turn_text import format_rules
from tg_bridge.core.workers import WorkerPool
from tg_bridge.store.inbox import CwdRepo
from tg_bridge.store.kv import KvRepo

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
CLAUDE_TRANSCRIPT = os.path.join(FIXTURES, "claude_transcripts", "handoff_session.jsonl")
CLAUDE_SID = "1f2e3d4c-aaaa-4bbb-8ccc-000000000003"
OLD_SID = "1f2e3d4c-aaaa-4bbb-8ccc-000000000004"
CLAUDE_CMD = "fake-cmd"
OK = TurnResult(TurnStatus.OK, answer="ok", duration_ms=5, num_turns=1, subtype="success")


class DiscardCase(unittest.TestCase):
    """A harness with two Codex launchers and one Claude launcher; the topic starts on the Claude launcher with a
    session that has a transcript on disk, so claude -> codex produces a real digest."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.base = os.path.realpath(self.home.home)
        self.project = os.path.join(self.base, "project")
        self.cfg_claude = os.path.join(self.base, "claude-a")
        os.makedirs(self.project, exist_ok=True)
        self.h = TurnHarness(self.base, config_data={
            "launchers": {"fake": {"cmd": CLAUDE_CMD, "config_dir": self.cfg_claude}},
            "agent_launchers": {
                "codex": {"backend": "codex", "command": "codex", "config_dir": os.path.join(self.base, "cx")},
                "codexb": {"backend": "codex", "command": "codex-b", "config_dir": os.path.join(self.base, "cxb")},
            },
            "topics": {TOPIC: {"launcher": CLAUDE_CMD, "cwd": self.project}},
        })
        self.addCleanup(self._stop_live_turns)
        self.addCleanup(self.h.db.close)
        self.registry = build_default_registry(self.h.config, self.h.logger, self.h.clock, title_launcher="")
        self.kv = KvRepo(self.h.db)
        self.handoff = HandoffService(self.kv, self.registry, self.h.logger)
        self.switch = LauncherSwitch(self.h.config, self.h.sessions, self.h.history, self.handoff, self.registry)
        self.out = WorkerPool("out", 1, self.h.logger)
        self.addCleanup(self.out.join, 5)
        self.stops = SessionCommands(
            config=self.h.config, sessions=self.h.sessions, history=self.h.history, live=self.h.live,
            backends=self.registry, messenger=self.h.messenger, clock=self.h.clock, logger=self.h.logger,
            handoff=self.handoff, stop_wait=0.6)
        self.commands = CommandRegistry()
        self.stops.register(self.commands)
        self.deps = CommandDeps(
            config=self.h.config, sessions=self.h.sessions, turns=self.h.turns, rules=self.h.rules, kv=self.kv,
            live=self.h.live, queues=TopicQueues(), backends=self.registry, messenger=self.h.messenger, out=self.out)
        self.h.sessions.ensure(THREAD, TOPIC, CLAUDE_CMD, self.project, "bypassPermissions")
        self.h.sessions.set_launcher_pick(THREAD, "fake")
        self.h.sessions.set_uuid(THREAD, CLAUDE_SID)
        self.h.history.note_session(THREAD, CLAUDE_SID, "earlier question", self.project, CLAUDE_CMD, "claude")
        self.put_claude_transcript(CLAUDE_SID)

    def _stop_live_turns(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)

    def put_claude_transcript(self, sid: str) -> None:
        folder = os.path.join(self.cfg_claude, "projects", encoded_cwd(os.path.realpath(self.project)))
        os.makedirs(folder, exist_ok=True)
        shutil.copyfile(CLAUDE_TRANSCRIPT, os.path.join(folder, sid + ".jsonl"))

    def switch_to_codex(self) -> None:
        kind, reply = self.switch.set(THREAD, TOPIC, None, "codex")
        self.assertEqual(kind, KIND_CHANGED)
        self.assertIsNotNone(self.kv.get(handoff_key(THREAD)), "the claude -> codex switch stores a digest")

    def dispatch(self, text: str) -> None:
        self.assertTrue(self.commands.dispatch(self.deps, CHAT, THREAD, TOPIC, text), text)
        self.assertTrue(self.out.join(5), "the reply did not go out")

    def first_turn_rules(self, name: str) -> tuple[str | None, str]:
        """Runs one turn on a FakeBackend registered as `name` and returns (resume_id, system_rules) of its spec."""
        backend = FakeBackend([[OK]], capabilities=FAKE_CAPABILITIES)
        runner = self.h.runner(self.h.registry_for(backend, name=name))
        reply = runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester")
        self.assertEqual(reply.answer, "ok")
        spec = backend.specs[0]
        return spec.resume_id, spec.system_rules

    def rules(self) -> str:
        return format_rules(TOPIC, "tester")


class NewSessionDiscardTests(DiscardCase):
    def test_switch_to_codex_then_new_leaves_no_digest_for_the_next_turn(self) -> None:
        self.switch_to_codex()
        self.dispatch("/new")
        self.assertIsNone(self.kv.get(handoff_key(THREAD)))
        resume_id, system_rules = self.first_turn_rules("codex")
        self.assertIsNone(resume_id)
        self.assertEqual(system_rules, self.rules())
        self.assertNotIn(HEADER, system_rules)

    def test_switch_to_codex_then_clear_alias_leaves_no_digest(self) -> None:
        self.switch_to_codex()
        self.dispatch("/clear")
        self.assertIsNone(self.kv.get(handoff_key(THREAD)))

    def test_stop_and_new_background_half_leaves_no_digest(self) -> None:
        """The stop-and-new button ends in finish_new(), which starts a fresh session once no turn runs."""
        self.switch_to_codex()
        self.stops.finish_new(THREAD, CHAT, None)
        self.assertIsNone(self.kv.get(handoff_key(THREAD)))


class ResumeThenNewTests(DiscardCase):
    def test_switch_to_codex_then_resume_claude_then_new_leaves_no_digest(self) -> None:
        self.switch_to_codex()
        self.h.history.note_session(THREAD, OLD_SID, "older question", self.project, CLAUDE_CMD, "claude")
        self.put_claude_transcript(OLD_SID)
        self.dispatch(f"/resume {OLD_SID}")
        self.assertEqual(self.h.sessions.get(THREAD).uuid, OLD_SID)
        self.assertIsNone(self.kv.get(handoff_key(THREAD)), "a resume binds the session, so the digest goes")
        self.dispatch("/new")
        resume_id, system_rules = self.first_turn_rules("claude")
        self.assertIsNone(resume_id)
        self.assertEqual(system_rules, self.rules())
        self.assertNotIn(HEADER, system_rules)


class SameBackendSwitchTests(DiscardCase):
    def test_switch_to_codex_then_same_backend_switch_leaves_no_digest(self) -> None:
        self.switch_to_codex()
        kind, reply = self.switch.set(THREAD, TOPIC, None, "codexb")
        self.assertEqual((kind, reply), (KIND_CHANGED, FRESH_TEXT.format(name="codexb")))
        self.assertIsNone(self.kv.get(handoff_key(THREAD)))
        resume_id, system_rules = self.first_turn_rules("codex")
        self.assertIsNone(resume_id)
        self.assertNotIn(HEADER, system_rules)


class FolderChangeTests(DiscardCase):
    def test_switch_to_codex_then_a_folder_change_leaves_no_digest(self) -> None:
        self.switch_to_codex()
        other = os.path.join(self.base, "other")
        os.makedirs(other, exist_ok=True)
        service = CwdService(sessions=self.h.sessions, kv=self.kv, paths=CwdRepo(self.h.db),
                             messenger=self.h.messenger, live=self.h.live,
                             search_pool=WorkerPool("search", 1, self.h.logger), out_pool=self.out,
                             logger=self.h.logger, clock=self.h.clock, handoff=self.handoff,
                             finder=lambda term, roots, deadline: [])
        self.assertTrue(service.apply(THREAD, os.path.realpath(other)))
        self.assertIsNone(self.kv.get(handoff_key(THREAD)))
        self.assertNotIn(HEADER, self.first_turn_rules("codex")[1])


if __name__ == "__main__":
    unittest.main()
