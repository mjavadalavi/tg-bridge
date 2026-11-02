"""The folder trust dialog at session level, with the fake TUI: a dialog split right after its bare cursor glyph must be
settled before it is read, or the glyph reads as a ready prompt and the dialog is never accepted.
"""

import time
import unittest

from tests.fakes.clock import FakeClock
from tests.test_claude_tui_session import TuiTestCase, only_result
from tg_bridge.agents.events import SessionStarted

SPLIT_AFTER_GLYPH = {"FAKE_TUI_TRUST_GLYPH_SPLIT": "1"}


class RealTimeClock(FakeClock):
    """A FakeClock that really sleeps each length: the trust dialog settle window then lasts as long as it does live."""

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        time.sleep(seconds)


class TrustDialogSessionTests(TuiTestCase):
    def test_a_dialog_split_right_after_its_cursor_glyph_is_accepted_and_the_turn_completes(self) -> None:
        clock = RealTimeClock()  # the second half comes 0.3 s after the glyph, inside the settle window
        sink, session = self.open("text", clock=clock, trust=True, extra_env=SPLIT_AFTER_GLYPH)
        session.start()
        self.assertIn("accepted the folder trust dialog", self.log_text())
        self.assertIn(0.8, clock.sleeps)  # one Down was pressed, then the cursor was read again
        self.assertTrue(session.alive())
        self.assertIsInstance(sink.events[0], SessionStarted)
        session.send_user("hello")
        self.wait_result(sink)
        self.assertEqual(only_result(sink).answer, "ANSWER")


if __name__ == "__main__":
    unittest.main()
