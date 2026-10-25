"""Deny reasons the bridge sends when it, not the owner, refuses an ask (port of the live file's three
answer_permission messages). Each case asserts the reason reaches the session answer; the Claude control_response
text for the same reasons is asserted in test_claude_session.py. A Deny tap by the owner carries no reason, so the
model keeps seeing DENY_MESSAGE.
"""

from tests.test_permissions import PermissionCase, bash_request, question_request
from tg_bridge.agents.events import Decision
from tg_bridge.core.permissions import CLOSED_TURN_MESSAGE, NO_QUESTIONS_MESSAGE, NOT_POSTED_MESSAGE


class DenyReasonTests(PermissionCase):
    def test_request_on_a_closed_turn_is_denied_with_turn_already_ended(self) -> None:
        """The turn ended before the request arrived: DENY carries the live 'turn already ended' reason."""
        self.service.void_all(self.binding)
        self.emit(bash_request("ls", approval_id="r9"))
        self.assertEqual(self.answers(), [("answer", "r9", Decision.DENY, None, None)])
        self.assertEqual(self.session.deny_messages, [("r9", "turn already ended")])
        self.assertEqual(CLOSED_TURN_MESSAGE, "turn already ended")
        self.assertEqual(self.posted(), [])

    def test_question_without_questions_is_denied_with_no_questions_to_answer(self) -> None:
        """An AskUserQuestion with no questions is refused with the live 'no questions to answer' reason."""
        self.emit(question_request(questions=(), approval_id="q9"))
        self.assertEqual(self.answers(), [("answer", "q9", Decision.DENY, None, None)])
        self.assertEqual(self.session.deny_messages, [("q9", "no questions to answer")])
        self.assertEqual(NO_QUESTIONS_MESSAGE, "no questions to answer")
        self.assertEqual(self.posted(), [])

    def test_card_that_cannot_be_posted_is_denied_with_could_not_reach_telegram(self) -> None:
        """Telegram refuses the card: DENY carries the live 'could not reach Telegram to ask' reason, not the owner's."""
        self.tg.fail_next("sendMessage", 400, description="chat not found")
        self.emit(bash_request())
        self.assertEqual(self.answers(), [("answer", "r1", Decision.DENY, None, None)])
        self.assertEqual(self.session.deny_messages, [("r1", "could not reach Telegram to ask")])
        self.assertEqual(NOT_POSTED_MESSAGE, "could not reach Telegram to ask")
        self.assertEqual(self.row("00000001"), "deny")

    def test_owner_deny_tap_sends_no_reason(self) -> None:
        """A real Deny tap is the owner's refusal: the session gets no reason, so DENY_MESSAGE is what the CLI sees."""
        self.emit(bash_request())
        self.assertTrue(self.service.answer_ask("00000001", False))
        self.assertEqual(self.answers(), [("answer", "r1", Decision.DENY, None, None)])
        self.assertEqual(self.session.deny_messages, [])
