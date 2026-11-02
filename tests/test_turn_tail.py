"""The stderr tail of a failed turn: a launcher whose tail is a raw terminal screen (Capabilities.readable_tail False, the
tui transport) keeps it out of the reply and writes it to the bridge log; a print launcher keeps it in the reply. The
resume refusal is found in the tail for both. Fakes only: no process, no Telegram."""

import dataclasses
import unittest

from tests.fakes.backend import FAKE_CAPABILITIES
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, ProbingBackend, TurnHarness
from tg_bridge.agents.events import TurnResult, TurnStatus
from tg_bridge.core.turn_text import NO_CONVERSATION

STORED = "11111111-2222-3333-4444-555555555555"
RESULT_LINE = "❌ turn_duration: API Error: Connection lost mid-response"
FAILED = TurnResult(TurnStatus.ERROR, answer=None, subtype="turn_duration",
                    error="API Error: Connection lost mid-response")
REFUSED = TurnResult(TurnStatus.ERROR, answer=None, subtype="turn_duration", error="API Error: exit")
DONE = TurnResult(TurnStatus.OK, answer="ok:allow", duration_ms=5, num_turns=1, subtype="success")
# what a TUI session's PTY screen looks like in the tail: redrawn frames, spinner glyphs, a diff, the /exit menu
SCREEN = "\n".join([
    "\x1b[2K\x1b[1A\x1b[2K⠋ Thinking… (12s · esc to interrupt)",
    "⏺ Edit(src/app.py)",
    "  │ - old_line = 1",
    "  │ + new_line = 2",
    "\x1b[2K⠙ Thinking… (13s · esc to interrupt)",
    "  /exit to quit · /model to switch",
])
PRINT_STDERR = "boom-on-stderr-line"


class TailCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", self.home.home, "bypassPermissions")

    def start(self, scripts, *, capabilities=FAKE_CAPABILITIES, stderr: str = "") -> None:
        self.backend = ProbingBackend(scripts, capabilities=capabilities, stderr=stderr)
        self.runner = self.h.runner(self.h.registry_for(self.backend))

    def turn(self):
        return self.runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester", slot=None)


class TuiTailTests(TailCase):
    TUI = dataclasses.replace(FAKE_CAPABILITIES, readable_tail=False)

    def test_tui_failure_reply_has_the_error_line_and_no_tail(self) -> None:
        self.start([[FAILED]], capabilities=self.TUI, stderr=SCREEN)
        reply = self.turn()
        self.assertEqual(reply.err, RESULT_LINE)
        self.assertNotIn("/exit", reply.err)
        self.assertNotIn("Thinking", reply.err)

    def test_tui_tail_goes_to_the_log_with_the_thread_id(self) -> None:
        self.start([[FAILED]], capabilities=self.TUI, stderr=SCREEN)
        self.turn()
        lines = self.h.log_lines()
        header = [i for i, line in enumerate(lines) if f"thread {THREAD}: process output, kept out of the reply" in line]
        self.assertEqual(len(header), 1)
        logged = "\n".join(lines[header[0]:])
        self.assertIn("/exit to quit", logged)
        self.assertIn("│ + new_line = 2", logged)
        self.assertNotIn("\x1b", logged)  # the escape sequences are stripped for the log

    def test_tui_resume_refusal_is_found_in_the_tail(self) -> None:
        # the tui's own result carries no resume flag; the NO_CONVERSATION line of its screen still starts the relaunch
        refused_screen = f"{SCREEN}\n{NO_CONVERSATION} with id {STORED}"
        self.start([[REFUSED], [DONE]], capabilities=self.TUI, stderr=refused_screen)
        self.h.sessions.set_uuid(THREAD, STORED)
        reply = self.turn()
        self.assertIsNone(reply.err)
        self.assertIn(f"the earlier conversation {STORED} was not found in", reply.answer)
        self.assertEqual(len(self.backend.specs), 2)


class PrintTailTests(TailCase):
    def test_print_failure_keeps_the_tail_in_the_reply(self) -> None:
        self.start([[FAILED]], stderr=PRINT_STDERR)
        reply = self.turn()
        self.assertEqual(reply.err, f"{RESULT_LINE}\n{PRINT_STDERR}")
        self.assertFalse(any("kept out of the reply" in line for line in self.h.log_lines()))

    def test_print_resume_refusal_is_detected_as_before(self) -> None:
        self.start([[REFUSED], [DONE]], stderr=NO_CONVERSATION)
        self.h.sessions.set_uuid(THREAD, STORED)
        reply = self.turn()
        self.assertIn(f"the earlier conversation {STORED} was not found in", reply.answer)


if __name__ == "__main__":
    unittest.main()
