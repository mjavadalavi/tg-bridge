"""Port of the real-process scenarios of fakecli/test_review_fixes.py (labels 2) on the real Claude adapter.

A fake CLI that exits without a result while a grandchild keeps its stdout open must end the turn as a process error
within a few seconds (the dead-process rule), not at the cap and not at an end of stream that never comes. The turn
row is 'error', the stderr tail is in the reason, and the grandchild is still running when the turn ends.

The garbage-line and subagent scenarios of the same suite are in test_claude_session.py and test_turn_runner.py.
"""

import os
import sys
import time
import unittest

import tests.fakes as fakes_package
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, TurnHarness, join_turn, run_async
from tg_bridge.agents.defaults import build_default_registry

FAKES = os.path.dirname(os.path.abspath(fakes_package.__file__))
FAKE_CLAUDE = os.path.join(FAKES, "fake_claude.py")
DIE_ERROR = "process ended without an answer (rc=3)"


class ClaudeProcessPort(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.pidfile = os.path.join(self.home.state, "die-grandchild.pid")
        command = f"{sys.executable} -I {FAKE_CLAUDE}"
        env = {"PATH": os.environ.get("PATH", ""), "FAKE_NOWAIT": "0", "FAKE_DIE": "1",
               "FAKE_DIE_PIDFILE": self.pidfile}
        self.h = TurnHarness(self.home.home, dict(launchers={"fake": {"cmd": command, "config_dir": self.home.home}},
                                                  max_turn_seconds=60),
                             base_env=env, poll_seconds=0.01)
        self.addCleanup(self._cleanup)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")
        self.h.sessions.set_launcher_pick(THREAD, "fake")

    def _cleanup(self) -> None:
        for session in self.h.live.begin_shutdown():
            session.stop(grace=1)
        if os.path.exists(self.pidfile):
            with open(self.pidfile, encoding="utf-8") as handle:
                _kill_quietly(int(handle.read()))
        self.h.db.close()

    def test_a_process_that_exits_while_a_grandchild_holds_stdout_ends_the_turn_as_an_error(self) -> None:
        """Ported labels:
        2: reported as process ended, rc=3
        2: stderr tail included
        2: returned in 5.0s (grandchild still alive, no EOF)
        2: turns row status error
        """
        runner = self.h.runner(build_default_registry(self.h.config, self.h.logger, self.h.clock, title_launcher=""))
        started = time.monotonic()
        thread, box = run_async(runner, thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="die", sender="tester")
        join_turn(self.h.clock, thread)
        elapsed = time.monotonic() - started
        reply = box[0]
        self.assertIsNone(reply.answer)
        self.assertIsNotNone(reply.err)
        self.assertTrue(reply.err.startswith(DIE_ERROR), f"got {reply.err!r}")
        self.assertIn("fake-die-stderr", reply.err)
        self.assertLess(elapsed, 6.5, f"the turn took {elapsed:.1f}s; a grandchild that keeps stdout open must not hold it")
        with open(self.pidfile, encoding="utf-8") as handle:
            grandchild = int(handle.read())
        self.assertTrue(_alive(grandchild), "the grandchild is still running: no end of stream came")
        row = self.h.db.query("SELECT status FROM turns WHERE thread_id=? ORDER BY id DESC", (THREAD,))[0]
        self.assertEqual(row["status"], "error")


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _kill_quietly(pid: int) -> None:
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass


if __name__ == "__main__":
    unittest.main()
