"""Port of the dispatcher part of fakecli/test_answer.py (labels I): a turn's answer leaves as HTML, an error text
leaves as plain text. The converter, the splitting and the document rules are in test_render and test_answers.

Uses the DispatcherCase harness of test_dispatcher: a real AnswerSender over FakeTelegram, a StubTurns for the runner.
"""

import unittest

from tests.test_dispatcher import THREAD, DispatcherCase
from tg_bridge.core.turn_runner import TurnReply
from tg_bridge.core.turn_state import TurnOutcome


class PortAnswerScenarios(DispatcherCase):
    def _send_params(self) -> list[dict]:
        return [c.params for c in self.tg.calls if c.method == "sendMessage"]

    def test_turn_answer_goes_out_as_html(self) -> None:
        """I: answer sent as HTML: the markdown of a turn's answer is sent with parse_mode HTML."""
        self.turns.replies["hi"] = TurnReply("**hi** there", None, TurnOutcome.OK)
        self.deliver(self.ingest(7001), "hi")
        self.assertTrue(self.idle())
        self.assertTrue(any(p.get("parse_mode") == "HTML" and p.get("text") == "<b>hi</b> there"
                            for p in self._send_params()), self._send_params())

    def test_turn_error_text_goes_out_as_plain_text(self) -> None:
        """I: error text keeps send_text: a turn's error text is sent without any parse mode, unchanged."""
        message = "❌ error_during_execution: boom"
        self.turns.replies["bad"] = TurnReply(None, message, TurnOutcome.INTERRUPTED)
        self.deliver(self.ingest(7101), "bad")
        self.assertTrue(self.idle())
        sent = [p for p in self._send_params() if p.get("text") == message]
        self.assertEqual(len(sent), 1)
        self.assertNotIn("parse_mode", sent[0])


if __name__ == "__main__":
    unittest.main()
