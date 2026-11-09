"""Plan approval (core/plan_approval): the plan card of ExitPlanMode, its four buttons, and the follow-up turn.

The interactive TUI does not take an allow from the PermissionRequest hook for ExitPlanMode (it still draws its own
dialog), so an approval is answered as a deny that carries the approval text, and the follow-up stores the mode and
queues the implementing turn. A plan is never answered by bypass or by a saved rule.
"""

import unittest

from tests.test_permissions import CHAT, THREAD, PermissionCase
from tg_bridge.agents.claude.translate import ClaudeTranslator
from tg_bridge.agents.events import PLAN_CHOICES, PLAN_KIND, ApprovalRequest, Decision
from tg_bridge.bot.callbacks.approvals import ApprovalTaps
from tg_bridge.core.cards import PLAN_ABOVE_TEXT, PLAN_CARD_CHARS, PLAN_HEAD
from tg_bridge.core.permissions import BYPASS_MODE, PermissionService
from tg_bridge.core.plan_approval import (EDITS_MODE, PLAN_APPROVED_MESSAGE, PLAN_CHANGE_MESSAGE, PLAN_GO_TEXT,
                                          PLAN_SENDER, PlanFollowUp)

PLAN = "# Plan\n\n1. add the fee column\n2. write the tests"


def plan_request(plan: str = PLAN, approval_id: str = "p1") -> ApprovalRequest:
    return ApprovalRequest(approval_id=approval_id, kind=PLAN_KIND, tool="ExitPlanMode",
                           input={"plan": plan, "planFilePath": "/tmp/plan.md"}, summary="ExitPlanMode",
                           choices=PLAN_CHOICES)


class PlanCase(PermissionCase):
    def build_service(self) -> PermissionService:
        self.approved: list[tuple[int, int, str]] = []
        return PermissionService(asks=self.asks, rules=self.rules, cards=self.cards, messenger=self.messenger,
                                 workers=self.workers, waiting=self.waits, logger=self.logger,
                                 nonce_factory=lambda: next(self.nonces),
                                 plan_approved=lambda thread, chat, mode: self.approved.append((thread, chat, mode)))

    def deny_messages(self) -> list[tuple[str, str]]:
        return list(self.session.deny_messages)


class PlanCardTests(PlanCase):
    def test_the_card_shows_the_plan_and_four_buttons(self) -> None:
        self.emit(plan_request())
        self.assertEqual(len(self.posted()), 1)
        self.assertEqual(self.posted()[0]["text"], f"{PLAN_HEAD}\n\n{PLAN}")
        self.assertEqual(self.callbacks(), ["pP:00000001:b", "pP:00000001:e", "pc:00000001", "pd:00000001"])
        self.assertEqual(self.answers(), [])

    def test_a_long_plan_is_sent_in_messages_before_a_card_with_only_the_buttons(self) -> None:
        long_plan = "step\n" * (PLAN_CARD_CHARS // 4)
        self.emit(plan_request(long_plan))
        texts = [params["text"] for params in self.posted()]
        self.assertTrue(texts[0].startswith(PLAN_HEAD))
        self.assertEqual(texts[-1], PLAN_ABOVE_TEXT)
        self.assertEqual("".join(texts[:-1]).count("step"), long_plan.count("step"))

    def test_bypass_does_not_approve_a_plan(self) -> None:
        self.binding.mode = BYPASS_MODE
        self.emit(plan_request())
        self.assertEqual(self.answers(), [])
        self.assertEqual(len(self.posted()), 1)

    def test_a_saved_rule_does_not_approve_a_plan(self) -> None:
        self.rules.add(THREAD, "ExitPlanMode")
        self.emit(plan_request())
        self.assertEqual(self.answers(), [])
        self.assertEqual(len(self.posted()), 1)

    def test_allow_once_and_bypass_this_turn_cannot_approve_a_plan(self) -> None:
        self.emit(plan_request())
        self.assertFalse(self.service.answer_ask("00000001", True))
        self.assertFalse(self.service.answer_bypass("00000001"))
        self.assertEqual(self.answers(), [])


class PlanAnswerTests(PlanCase):
    def test_approve_denies_with_the_approval_text_and_queues_the_follow_up(self) -> None:
        self.emit(plan_request())
        self.assertTrue(self.service.answer_plan("00000001", BYPASS_MODE))
        self.assertEqual(self.answers(), [("answer", "p1", Decision.DENY, None, None)])
        self.assertEqual(self.deny_messages(), [("p1", PLAN_APPROVED_MESSAGE.format(mode=BYPASS_MODE))])
        self.assertEqual(self.approved, [(THREAD, CHAT, BYPASS_MODE)])
        self.assertEqual(self.row("00000001"), "approved")
        self.assertFalse(self.service.answer_plan("00000001", BYPASS_MODE), "a second tap finds nothing")
        self.assertEqual(len(self.approved), 1)

    def test_approve_refuses_an_ask_that_is_not_a_plan(self) -> None:
        self.emit(ApprovalRequest(approval_id="r1", kind="command", tool="Bash", input={"command": "ls"},
                                  summary="Bash", choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY})))
        self.assertFalse(self.service.answer_plan("00000001", BYPASS_MODE))
        self.assertEqual(self.approved, [])

    def test_change_waits_for_the_next_text_and_denies_with_it(self) -> None:
        self.emit(plan_request())
        self.assertTrue(self.service.answer_plan_change("00000001"))
        self.assertEqual(self.answers(), [])

        self.assertTrue(self.service.record_text(THREAD, "keep the amount in rials too"))

        self.assertEqual(self.deny_messages(),
                         [("p1", PLAN_CHANGE_MESSAGE.format(text="keep the amount in rials too"))])
        self.assertEqual(self.row("00000001"), "change")
        self.assertEqual(self.approved, [])
        self.assertFalse(self.service.record_text(THREAD, "a normal message"), "the wait ended with the feedback")

    def test_reject_is_a_plain_deny(self) -> None:
        self.emit(plan_request())
        self.assertTrue(self.service.answer_ask("00000001", False))
        self.assertEqual(self.answers(), [("answer", "p1", Decision.DENY, None, None)])
        self.assertEqual(self.approved, [])

    def test_a_failing_follow_up_still_answers_the_plan(self) -> None:
        def boom(*_: object) -> None:
            raise RuntimeError("no dispatcher")

        self.service._plan_approved = boom
        self.emit(plan_request())
        self.assertTrue(self.service.answer_plan("00000001", EDITS_MODE))
        self.assertEqual(self.deny_messages(), [("p1", PLAN_APPROVED_MESSAGE.format(mode=EDITS_MODE))])


class FakeTap:
    def __init__(self, *args: str) -> None:
        self.args = args
        self.message_text = "card"
        self.toasts: list[str] = []

    def toast(self, text: str) -> None:
        self.toasts.append(text)

    def edit(self, text: str) -> None:
        pass


class PlanTapTests(PlanCase):
    def test_the_two_approve_buttons_pick_their_modes_and_an_unknown_key_does_nothing(self) -> None:
        taps = ApprovalTaps(permissions=self.service)
        self.emit(plan_request())
        taps._plan(FakeTap("00000001", "x"))
        self.assertEqual(self.approved, [])
        taps._plan(FakeTap("00000001", "e"))
        self.assertEqual(self.approved, [(THREAD, CHAT, EDITS_MODE)])

    def test_change_button_opens_the_wait(self) -> None:
        taps = ApprovalTaps(permissions=self.service)
        self.emit(plan_request())
        tap = FakeTap("00000001")
        taps._plan_change(tap)
        self.assertEqual(self.cards.awaiting(THREAD), ("00000001", -1))
        self.assertTrue(tap.toasts)


class RecordingSessions:
    def __init__(self) -> None:
        self.modes: list[tuple[int, str]] = []

    def set_mode(self, thread_id: int, mode: str) -> None:
        self.modes.append((thread_id, mode))

    def get(self, thread_id: int) -> object:
        return type("Row", (), {"name": "accounting"})()


class RecordingDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def deliver(self, thread_id, topic, chat_id, sender, update_id, text) -> None:
        self.calls.append((thread_id, topic, chat_id, sender, update_id, text))


class FollowUpTests(unittest.TestCase):
    def test_the_follow_up_stores_the_mode_then_queues_the_go_message(self) -> None:
        sessions, dispatcher = RecordingSessions(), RecordingDispatcher()
        PlanFollowUp(sessions, dispatcher)(7561, CHAT, BYPASS_MODE)
        self.assertEqual(sessions.modes, [(7561, BYPASS_MODE)])
        self.assertEqual(dispatcher.calls,
                         [(7561, "accounting", CHAT, PLAN_SENDER, None, PLAN_GO_TEXT.format(mode=BYPASS_MODE))])


class TranslateTests(unittest.TestCase):
    def test_exit_plan_mode_is_a_plan_ask_with_only_approve_and_deny(self) -> None:
        event = {"type": "control_request", "request_id": "r9",
                 "request": {"subtype": "can_use_tool", "tool_name": "ExitPlanMode", "input": {"plan": PLAN}}}
        [request] = ClaudeTranslator().translate(event)
        self.assertEqual((request.kind, request.choices), (PLAN_KIND, PLAN_CHOICES))


if __name__ == "__main__":
    unittest.main()
