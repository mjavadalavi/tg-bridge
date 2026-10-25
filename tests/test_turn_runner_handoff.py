"""TurnRunner handoff injection (STEP 39): the pending handoff digest goes into the system rules of a fresh session only.

Scenarios: the first fresh turn's spec carries the digest after the RULES text and the second turn's spec does not; the
digest leaves kv only once SessionStarted arrives; a failed start or a failed open keeps the kv row for the next try; a
resumed session never gets the digest and never removes it; a fresh relaunch after a missing resume gets it. Two end to
end runs check the argv of the Claude fake and the turn/start input of the Codex fake (no developerInstructions on the
wire: the rules travel with the turn, DESIGN section 4).
"""

import dataclasses
import json
import os
import shlex
import sys
import unittest
from unittest import mock

import tests.fakes as fakes_package
from tests.fakes.backend import FAKE_CAPABILITIES, FakeBackend, FakeSession
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, TurnHarness, join_turn, run_async
from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.agents.events import TurnResult, TurnStatus
from tg_bridge.core.handoff import HEADER, handoff_key
from tg_bridge.core.turn_state import TurnOutcome
from tg_bridge.core.turn_text import format_rules

FAKES = os.path.dirname(os.path.abspath(fakes_package.__file__))
FAKE_CLAUDE = os.path.join(FAKES, "fake_claude.py")
FAKE_CODEX = os.path.join(FAKES, "fake_codex.py")
DIGEST = HEADER + "User: Fix the login bug\nAgent: On it"
STORED = "stored-session-0039"
OK = TurnResult(TurnStatus.OK, answer="ok", duration_ms=5, num_turns=1, subtype="success")
FAILED = [TurnResult(TurnStatus.ERROR, answer=None, error="No conversation found", resume_failed=True)]


class HandoffInjectionCase(unittest.TestCase):
    """A harness with a FakeBackend (no processes) and a pending digest for the topic."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self._stop_live_turns)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")
        self.h.kv.set(handoff_key(THREAD), DIGEST)

    def _stop_live_turns(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)

    def start(self, scripts, *, capabilities=FAKE_CAPABILITIES) -> None:
        self.backend = FakeBackend(scripts, capabilities=capabilities)
        self.runner = self.h.runner(self.h.registry_for(self.backend))

    def turn(self, text: str = "hello"):
        return self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text=text, sender="tester")

    def pending(self) -> str | None:
        return self.h.kv.get(handoff_key(THREAD))

    def rules(self) -> str:
        return format_rules(TOPIC, "tester")


class FreshSessionTests(HandoffInjectionCase):
    def test_first_fresh_turn_carries_the_digest_and_the_second_turn_does_not(self) -> None:
        self.start([[OK], [OK]])
        self.turn()
        first = self.backend.specs[0]
        self.assertEqual((first.resume_id, first.system_rules), (None, self.rules() + "\n\n" + DIGEST))
        self.assertIsNone(self.pending(), "SessionStarted removes the digest from kv")
        self.turn("again")
        second = self.backend.specs[1]
        self.assertEqual(second.resume_id, first.new_id)
        self.assertEqual(second.system_rules, self.rules())
        self.assertNotIn(DIGEST, second.system_rules)

    def test_a_failed_start_keeps_the_digest_for_the_next_try(self) -> None:
        # no preassigned id: a failed start stores no session id, so the next try is a fresh session again
        self.start([[OK], [OK]], capabilities=dataclasses.replace(FAKE_CAPABILITIES, preassigned_id=False))
        with mock.patch.object(FakeSession, "start", side_effect=RuntimeError("boom")):
            reply = self.turn()
        self.assertEqual((reply.answer, reply.err, reply.outcome), (None, "launcher failed: boom", TurnOutcome.ERROR))
        self.assertEqual(self.pending(), DIGEST)
        self.turn("retry")
        self.assertEqual(self.backend.specs[1].system_rules, self.rules() + "\n\n" + DIGEST)
        self.assertIsNone(self.pending())

    def test_a_failed_open_keeps_the_digest_for_the_next_try(self) -> None:
        self.start([[OK]])
        with mock.patch.object(FakeBackend, "open_session", side_effect=RuntimeError("no spawn")):
            reply = self.turn()
        self.assertEqual(reply.err, "launcher failed: no spawn")
        self.assertEqual(self.pending(), DIGEST)

    def test_a_fresh_relaunch_after_a_missing_resume_takes_the_digest(self) -> None:
        self.start([FAILED, [OK]])
        self.h.sessions.set_uuid(THREAD, STORED)
        self.turn()
        resumed, relaunch = self.backend.specs
        self.assertEqual((resumed.resume_id, resumed.system_rules), (STORED, self.rules()))
        self.assertEqual((relaunch.resume_id, relaunch.new_id), (None, STORED))
        self.assertEqual(relaunch.system_rules, self.rules() + "\n\n" + DIGEST)
        self.assertIsNone(self.pending())


class ResumedSessionTests(HandoffInjectionCase):
    def test_a_resumed_session_never_gets_the_digest_and_keeps_it(self) -> None:
        self.start([[OK]])
        self.h.sessions.set_uuid(THREAD, STORED)
        self.turn()
        spec = self.backend.specs[0]
        self.assertEqual((spec.resume_id, spec.system_rules), (STORED, self.rules()))
        self.assertEqual(self.pending(), DIGEST, "a resumed start does not consume the digest")


class ClaudeEndToEndTests(unittest.TestCase):
    """The Claude adapter with fake_claude behind a wrapper that records its argv (NUL separated)."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.argfile = os.path.join(self.home.home, "claude-argv.bin")
        wrapper = os.path.join(self.home.home, "claude-argv.sh")
        with open(wrapper, "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\n"
                         f"printf '%s\\0' \"$@\" > {shlex.quote(self.argfile)}\n"
                         f"exec {shlex.quote(sys.executable)} -I {shlex.quote(FAKE_CLAUDE)} \"$@\"\n")
        os.chmod(wrapper, 0o755)
        self.h = TurnHarness(self.home.home, dict(launchers={"fake": {"cmd": wrapper,
                                                                       "config_dir": self.home.home}}),
                             base_env={"PATH": os.environ.get("PATH", ""), "FAKE_NOWAIT": "1"}, poll_seconds=0.01)
        self.addCleanup(self._stop_live_turns)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")
        self.h.sessions.set_launcher_pick(THREAD, "fake")
        self.h.kv.set(handoff_key(THREAD), DIGEST)

    def _stop_live_turns(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)

    def _argv(self) -> list[str]:
        with open(self.argfile, "rb") as handle:
            return handle.read().decode("utf-8").split("\0")[:-1]

    def _turn(self, text: str):
        runner = self.h.runner(build_default_registry(self.h.config, self.h.logger, self.h.clock, title_launcher=""))
        thread, box = run_async(runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text=text, sender="tester")
        join_turn(self.h.clock, thread)
        return box[0]

    def test_the_first_fresh_claude_launch_carries_the_digest_in_argv(self) -> None:
        reply = self._turn("hello")
        self.assertEqual((reply.err, reply.outcome), (None, TurnOutcome.OK))
        argv = self._argv()
        self.assertIn("--session-id", argv)
        self.assertEqual(argv[argv.index("--append-system-prompt") + 1], format_rules(TOPIC, "tester") + "\n\n" + DIGEST)
        self.assertIsNone(self.h.kv.get(handoff_key(THREAD)))

    def test_the_second_claude_launch_resumes_without_the_digest(self) -> None:
        self._turn("hello")
        self.assertEqual(self._turn("again").outcome, TurnOutcome.OK)
        argv = self._argv()
        self.assertIn("--resume", argv)
        self.assertNotIn(DIGEST, " ".join(argv))


class CodexEndToEndTests(unittest.TestCase):
    """The Codex adapter with fake_codex: the rules (and so the digest) are the leading text of turn/start input."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.logfile = os.path.join(self.home.home, "codex-frames.jsonl")
        command = f"{sys.executable} -I {FAKE_CODEX}"
        self.h = TurnHarness(self.home.home, dict(agent_launchers={"fakecodex": {
                                 "backend": "codex", "command": command, "config_dir": self.home.home}}),
                             base_env={"PATH": os.environ.get("PATH", ""), "FAKE_CODEX_MODE": "normal",
                                       "FAKE_CODEX_ANSWER": "codex answer", "FAKE_CODEX_LOG": self.logfile},
                             poll_seconds=0.01)
        self.addCleanup(self._stop_live_turns)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")
        self.h.sessions.set_launcher_pick(THREAD, "fakecodex")
        self.h.kv.set(handoff_key(THREAD), DIGEST)

    def _stop_live_turns(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)

    def _frames(self, method: str) -> list[dict]:
        with open(self.logfile, encoding="utf-8") as handle:
            frames = [json.loads(line)["in"] for line in handle if line.strip()]
        return [frame["params"] for frame in frames if frame.get("method") == method]

    def test_the_digest_reaches_the_first_codex_turn_and_the_thread_start_has_no_developer_instructions(self) -> None:
        runner = self.h.runner(build_default_registry(self.h.config, self.h.logger, self.h.clock, title_launcher=""))
        thread, box = run_async(runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hi", sender="tester")
        join_turn(self.h.clock, thread)
        self.assertEqual((box[0].err, box[0].answer), (None, "codex answer"))
        (start,) = self._frames("thread/start")
        self.assertNotIn("developerInstructions", start)
        (turn,) = self._frames("turn/start")
        self.assertEqual(turn["input"][0]["text"], f"<tg-bridge-rules>\n{format_rules(TOPIC, 'tester')}\n\n{DIGEST}\n</tg-bridge-rules>")
        self.assertIsNone(self.h.kv.get(handoff_key(THREAD)))


if __name__ == "__main__":
    unittest.main()
