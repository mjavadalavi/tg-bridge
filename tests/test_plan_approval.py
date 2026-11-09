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
from tg_bridge.core.cards import PLAN_ABOVE_TEXT, PLAN_CARD_CHARS, PLAN_CHUNK_CHARS, PLAN_HEAD
from tg_bridge.core.permissions import BYPASS_MODE, PermissionService
from tg_bridge.core.plan_approval import (ASK_MODE, PLAN_APPROVED_MESSAGE, PLAN_CHANGE_MESSAGE, PLAN_GO_TEXT,
                                          PLAN_NOT_QUEUED_TEXT, PLAN_SENDER, PlanFollowUp)
from tg_bridge.core.topic_queues import Admission, Delivery
from tg_bridge.telegram.messenger import Messenger

PLAN = "# Plan\n\n1. add the fee column\n2. write the tests"


def plan_request(plan: str = PLAN, approval_id: str = "p1") -> ApprovalRequest:
    return ApprovalRequest(approval_id=approval_id, kind=PLAN_KIND, tool="ExitPlanMode",
                           input={"plan": plan, "planFilePath": "/tmp/plan.md"}, summary="ExitPlanMode",
                           choices=PLAN_CHOICES)


class PlanCase(PermissionCase):
    def build_service(self) -> PermissionService:
        self.approved: list[tuple[int, int, str]] = []
        def approved(thread: int, chat: int, mode: str) -> None:
            self.approved.append((thread, chat, mode))
            self.answered_before_follow_up = bool(self.answers())

        return PermissionService(asks=self.asks, rules=self.rules, cards=self.cards, messenger=self.messenger,
                                 workers=self.workers, waiting=self.waits, logger=self.logger,
                                 nonce_factory=lambda: next(self.nonces), plan_approved=approved)

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

    def test_a_very_long_plan_reaches_the_topic_whole(self) -> None:
        long_plan = "".join(f"step {n}\n" for n in range(6 * PLAN_CHUNK_CHARS // 7)) + "THE LAST STEP"
        self.emit(plan_request(long_plan))
        texts = [params["text"] for params in self.posted()]
        self.assertGreater(len(texts), 5)
        self.assertTrue(all(len(text) <= PLAN_CHUNK_CHARS for text in texts))
        sent = "\n".join(texts[:-1]).split()  # the sender trims the line end at each cut; no word is lost
        self.assertEqual(sent, f"{PLAN_HEAD}\n\n{long_plan}".split())

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


class FailingSecondText(Messenger):
    """A Messenger whose second plain-text send fails, as a send that failed after its retries."""

    texts_sent = 0

    def send_text(self, chat_id, thread_id, text) -> bool:
        self.texts_sent += 1
        return self.texts_sent != 2 and super().send_text(chat_id, thread_id, text)


class PlanPartLostTests(PlanCase):
    messenger_cls = FailingSecondText

    def test_no_card_when_a_part_of_the_plan_was_not_sent(self) -> None:
        self.emit(plan_request("step\n" * (3 * PLAN_CHUNK_CHARS // 5)))
        self.assertEqual(self.tg.buttons(THREAD), [])
        self.assertEqual(self.answers(), [("answer", "p1", Decision.DENY, None, None)])


class PlanAnswerTests(PlanCase):
    def test_approve_denies_with_the_approval_text_and_queues_the_follow_up(self) -> None:
        self.emit(plan_request())
        self.assertTrue(self.service.answer_plan("00000001", BYPASS_MODE))
        self.assertEqual(self.answers(), [("answer", "p1", Decision.DENY, None, None)])
        self.assertEqual(self.deny_messages(), [("p1", PLAN_APPROVED_MESSAGE.format(mode=BYPASS_MODE))])
        self.assertEqual(self.approved, [(THREAD, CHAT, BYPASS_MODE)])
        self.assertFalse(self.answered_before_follow_up, "the mode is stored and the turn queued before the answer")
        self.assertTrue(self.binding.plan_approved, "linger must not keep this session")
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

    def test_the_ask_before_edits_button_is_the_default_mode(self) -> None:
        self.assertEqual(ASK_MODE, "default")


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
        self.assertEqual(self.approved, [(THREAD, CHAT, ASK_MODE)])

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
    def __init__(self, admission: Admission = Admission.QUEUED, error: Exception | None = None) -> None:
        self.calls: list[tuple] = []
        self.admission, self.error = admission, error

    def deliver(self, thread_id, topic, chat_id, sender, update_id, text) -> Delivery:
        self.calls.append((thread_id, topic, chat_id, sender, update_id, text))
        if self.error is not None:
            raise self.error
        return Delivery(self.admission)


class RecordingNotices:
    def __init__(self) -> None:
        self.texts: list[tuple] = []
        self.lines: list[str] = []

    def send_text(self, chat_id, thread_id, text) -> None:
        self.texts.append((chat_id, thread_id, text))

    def log(self, line: str) -> None:
        self.lines.append(line)


class FollowUpTests(unittest.TestCase):
    def follow_up(self, dispatcher: RecordingDispatcher) -> tuple[RecordingSessions, RecordingNotices]:
        sessions, notices = RecordingSessions(), RecordingNotices()
        PlanFollowUp(sessions, dispatcher, notices, notices)(7561, CHAT, BYPASS_MODE)
        return sessions, notices

    def test_the_follow_up_stores_the_mode_then_queues_the_go_message(self) -> None:
        dispatcher = RecordingDispatcher()
        sessions, notices = self.follow_up(dispatcher)
        self.assertEqual(sessions.modes, [(7561, BYPASS_MODE)])
        self.assertEqual(dispatcher.calls,
                         [(7561, "accounting", CHAT, PLAN_SENDER, None, PLAN_GO_TEXT.format(mode=BYPASS_MODE))])
        self.assertEqual(notices.texts, [])

    def test_a_full_queue_is_told_to_the_topic(self) -> None:
        _, notices = self.follow_up(RecordingDispatcher(Admission.FULL))
        self.assertEqual(notices.texts, [(CHAT, 7561, PLAN_NOT_QUEUED_TEXT.format(mode=BYPASS_MODE))])

    def test_a_failure_is_logged_and_told_never_raised(self) -> None:
        _, notices = self.follow_up(RecordingDispatcher(error=RuntimeError("down")))
        self.assertEqual(len(notices.texts), 1)
        self.assertIn("plan follow-up failed: down", notices.lines[0])


class TranslateTests(unittest.TestCase):
    def test_exit_plan_mode_is_a_plan_ask_with_only_approve_and_deny(self) -> None:
        event = {"type": "control_request", "request_id": "r9",
                 "request": {"subtype": "can_use_tool", "tool_name": "ExitPlanMode", "input": {"plan": PLAN}}}
        [request] = ClaudeTranslator().translate(event)
        self.assertEqual((request.kind, request.choices), (PLAN_KIND, PLAN_CHOICES))


if __name__ == "__main__":
    unittest.main()
