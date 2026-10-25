"""Port of fakecli/test_run_turn_v2.py step 4 (the cap) on the real Claude adapter: a turn whose CLI never answers stops
at max_turn_seconds with the capped notice, its turns row is 'capped', the turn returns in about the cap time, the
session leaves the registry, and no child of this process is left running the fake CLI.

The FakeBackend version of the same cap is test_turn_runner.test_cap_stops_the_turn_with_the_notice.
"""

import os
import sys
import time
import unittest

import tests.fakes as fakes_package
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, TurnHarness, join_turn, run_async
from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.core.turn_state import TurnOutcome

FAKES = os.path.dirname(os.path.abspath(fakes_package.__file__))
FAKE_CLAUDE = os.path.join(FAKES, "fake_claude.py")
CAP_SECONDS = 3
CAPPED_TEXT = '⏱ stopped after 3s of work. The session is kept — send "continue" to pick up where it left off.'


def _fake_claude_children() -> list[int]:
    """The pids of the children of this process whose command line runs the fake CLI."""
    found = []
    me = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="ascii") as handle:
                ppid = int(handle.read().rsplit(")", 1)[1].split()[1])
            with open(f"/proc/{entry}/cmdline", "rb") as handle:
                cmdline = handle.read().decode("utf-8", "replace")
        except (OSError, ValueError, IndexError):
            continue
        if ppid == me and FAKE_CLAUDE in cmdline:
            found.append(int(entry))
    return found


class ClaudeCapPort(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        command = f"{sys.executable} -I {FAKE_CLAUDE}"
        self.h = TurnHarness(self.home.home, dict(launchers={"fake": {"cmd": command, "config_dir": self.home.home}},
                                                  max_turn_seconds=CAP_SECONDS),
                             base_env={"PATH": os.environ.get("PATH", ""), "FAKE_NOWAIT": "0", "FAKE_NEVER": "1"},
                             poll_seconds=0.01)
        self.addCleanup(self._stop_live_turns)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")
        self.h.sessions.set_launcher_pick(THREAD, "fake")

    def _stop_live_turns(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)

    def test_a_turn_that_never_answers_stops_at_the_cap_and_leaves_no_process(self) -> None:
        """4: cap: process group is gone; cap: returned in 2.1s (about the cap time); cap: turns row status capped;
        cap: ⏱ notice returned; session removed from registry after turn."""
        runner = self.h.runner(build_default_registry(self.h.config, self.h.logger, self.h.clock, title_launcher=""))
        started = time.monotonic()
        thread, box = run_async(runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="long job", sender="tester")
        join_turn(self.h.clock, thread)
        elapsed = time.monotonic() - started
        reply = box[0]
        self.assertEqual((reply.answer, reply.err, reply.outcome), (None, CAPPED_TEXT, TurnOutcome.CAPPED))
        self.assertLess(elapsed, 10, f"the turn took {elapsed:.1f}s of real time; the cap did not stop it")
        row = self.h.db.query("SELECT status FROM turns WHERE thread_id=? ORDER BY id DESC", (THREAD,))[0]
        self.assertEqual(row["status"], "capped")
        self.assertIsNone(self.h.live.running(THREAD))
        self.assertEqual(_fake_claude_children(), [], "the fake CLI is still running after the cap")


if __name__ == "__main__":
    unittest.main()
