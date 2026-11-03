"""The SessionStart hook client (daemon/session_hook.py): a live claim of the tab's pid moves to the new session.

The claim is a real file taken by take_claim with os.getpid(), so it is live. Another pid (the parent of the test
process, alive) or no claim at all changes nothing. Bad input exits 0 and leaves the claim alone. Nothing is printed,
since a SessionStart hook's stdout becomes model context. No claude process is started.
"""

import contextlib
import io
import json
import os
import unittest

from tg_bridge.core import terminal_claim
from tg_bridge.daemon.session_hook import run_session_start_hook
from tg_bridge.paths import Paths
from tests.support import TempHome

THREAD = 5150
OLD_UUID = "0f3c9a7e-1b2d-4c5e-8f60-7a8b9c0d1e2f"
NEW_UUID = "5b1e7d20-9a4c-4f3e-b6d8-2c7a1e9f0b43"


class SessionHookCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.paths = Paths.from_home(self.home.home)
        os.makedirs(self.paths.view, mode=0o700)

    def take(self, pid: int) -> None:
        self.assertTrue(terminal_claim.take_claim(self.paths.view, THREAD, pid, 1.0, uuid=OLD_UUID))

    def claim_uuid(self) -> str:
        claim = terminal_claim.read_claim(self.paths.view, THREAD)
        self.assertIsNotNone(claim)
        return claim.uuid

    def log_text(self) -> str:
        with open(self.paths.log, encoding="utf-8") as handle:
            return handle.read()


class SessionHookTest(SessionHookCase):
    def test_a_session_start_moves_the_claim_held_by_that_pid_and_logs_it(self) -> None:
        self.take(os.getpid())
        stdin = io.StringIO(json.dumps({"session_id": NEW_UUID, "source": "clear", "transcript_path": "/x"}))

        code = run_session_start_hook(stdin, str(THREAD), str(os.getpid()), self.paths)

        self.assertEqual(code, 0)
        self.assertEqual(self.claim_uuid(), NEW_UUID)
        self.assertIn(f"terminal: thread {THREAD} tab session now {NEW_UUID} (clear)", self.log_text())

    def test_the_hook_run_marks_the_claim_hooked(self) -> None:
        self.take(os.getpid())
        self.assertFalse(terminal_claim.read_claim(self.paths.view, THREAD).hooked)
        stdin = io.StringIO(json.dumps({"session_id": OLD_UUID, "source": "startup"}))

        self.assertEqual(run_session_start_hook(stdin, str(THREAD), str(os.getpid()), self.paths), 0)

        claim = terminal_claim.read_claim(self.paths.view, THREAD)
        self.assertEqual((claim.uuid, claim.seen, claim.hooked), (OLD_UUID, (), True))

    def test_another_pid_or_no_claim_changes_nothing_and_still_exits_0(self) -> None:
        other_pid = os.getppid()  # alive, and not the pid that holds the claim
        self.take(os.getpid())
        stdin = io.StringIO(json.dumps({"session_id": NEW_UUID, "source": "resume"}))

        code = run_session_start_hook(stdin, str(THREAD), str(other_pid), self.paths)

        self.assertEqual(code, 0)
        self.assertEqual(self.claim_uuid(), OLD_UUID)
        self.assertIn(f"terminal: thread {THREAD} tab session {NEW_UUID} not moved: no live claim of pid {other_pid}",
                      self.log_text())

        os.remove(terminal_claim.claim_path(self.paths.view, THREAD))
        stdin = io.StringIO(json.dumps({"session_id": NEW_UUID, "source": "resume"}))

        code = run_session_start_hook(stdin, str(THREAD), str(os.getpid()), self.paths)

        self.assertEqual(code, 0)
        self.assertIsNone(terminal_claim.read_claim(self.paths.view, THREAD))

    def test_bad_input_exits_0_and_leaves_the_claim_alone(self) -> None:
        self.take(os.getpid())
        cases = [
            (io.StringIO("not json {"), str(THREAD), str(os.getpid())),
            (io.StringIO(json.dumps({"source": "clear"})), str(THREAD), str(os.getpid())),
            (io.StringIO(json.dumps({"session_id": ""})), str(THREAD), str(os.getpid())),
            (io.StringIO(json.dumps({"session_id": NEW_UUID})), "abc", str(os.getpid())),
            (io.StringIO(json.dumps({"session_id": NEW_UUID})), str(THREAD), "x1"),
        ]
        for stdin, thread, pid in cases:
            with self.subTest(thread=thread, pid=pid):
                self.assertEqual(run_session_start_hook(stdin, thread, pid, self.paths), 0)
                self.assertEqual(self.claim_uuid(), OLD_UUID)

    def test_nothing_is_printed(self) -> None:
        self.take(os.getpid())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            run_session_start_hook(io.StringIO(json.dumps({"session_id": NEW_UUID, "source": "clear"})),
                                   str(THREAD), str(os.getpid()), self.paths)
            run_session_start_hook(io.StringIO("not json"), str(THREAD), str(os.getpid()), self.paths)

        self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
