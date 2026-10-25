"""cards: port of the markup, edit and nudge scenarios of fakecli/test_p2b_cards.py (labels A, C, D, F, G, I, K kept in
docstrings), plus the Codex-only deny warning and the in-flight cases that v3 handles explicitly.

Everything runs on FakeTelegram and FakeClock inside a TempHome. The nudge timer is a recording fake: the test fires it
by hand, after moving the clock, so no real sleep happens.
"""

import json
import os
import unittest

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FIRST_MESSAGE_ID, FakeTelegram
from tests.support import TempHome
from tg_bridge.agents.events import ApprovalRequest, Decision, Question
from tg_bridge.core.cards import (
    ASK_NUDGE_SECONDS,
    DENY_WARNING,
    NUDGE_TEXT,
    OTHER_WAIT_TEXT,
    ApprovalCards,
)
from tg_bridge.logging_ import Logger
from tg_bridge.telegram.keyboards import EMPTY_KEYBOARD, CallbackData, CallbackDataError
from tg_bridge.telegram.messenger import Messenger

THREAD = 9301
CHAT = -1001
ALL = frozenset({Decision.ALLOW_ONCE, Decision.ALLOW_ALWAYS, Decision.ALLOW_TURN, Decision.DENY})
NONCE = "a1b2c3d4"


class RecordingTimer:
    """A nudge timer that only records start/cancel and fires when a test calls fire()."""

    def __init__(self, seconds: float, fire) -> None:
        self.seconds = seconds
        self.fire_fn = fire
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        self.fire_fn()


class HookedMessenger(Messenger):
    """A Messenger whose send_buttons first runs a test hook, so a test can act while a card is still in flight."""

    hook = None

    def send_buttons(self, chat_id, thread_id, text, rows):
        if self.hook is not None:
            hook, self.hook = self.hook, None
            hook()
        return super().send_buttons(chat_id, thread_id, text, rows)


class Harness:
    """ApprovalCards wired to a fake Telegram, a fake clock, a log file and recording timers."""

    def __init__(self, home: TempHome, nudge_seconds: float = ASK_NUDGE_SECONDS) -> None:
        self.clock = FakeClock(1000.0)
        self.tg = FakeTelegram(clock=self.clock)
        self.log_path = os.path.join(home.state, "bridge.log")
        self.logger = Logger(self.log_path)
        self.messenger = HookedMessenger(self.tg, self.logger, self.clock)
        self.timers: list[RecordingTimer] = []
        self.cards = ApprovalCards(self.messenger, self.logger, self.clock, nudge_seconds=nudge_seconds,
                                   timer_factory=self._timer)

    def _timer(self, seconds, fire) -> RecordingTimer:
        timer = RecordingTimer(seconds, fire)
        self.timers.append(timer)
        return timer

    def post(self, request, nonce: str = NONCE, always_rule: str | None = None) -> bool:
        return self.cards.post(nonce, request, chat_id=CHAT, thread_id=THREAD, always_rule=always_rule)

    def sent_keyboard(self) -> list[list[dict]]:
        """The rows of the last card sent to the topic."""
        return self.tg.buttons(THREAD)[-1]

    def card_mid(self) -> int:
        """The message id of the last card: FakeTelegram numbers every sent message in order."""
        return FIRST_MESSAGE_ID + len([c for c in self.tg.calls if c.method == "sendMessage"])

    def edits(self) -> list[dict]:
        """Every editMessageText params dict, oldest first."""
        return [c.params for c in self.tg.calls if c.method == "editMessageText"]

    def last_edit(self) -> dict:
        return self.edits()[-1]

    def log_text(self) -> str:
        with open(self.log_path, encoding="utf-8") as handle:
            return handle.read()


def bash_request(command: str = "git status --short", choices=ALL, **extra) -> ApprovalRequest:
    return ApprovalRequest(approval_id="r1", kind="command", tool="Bash", input={"command": command},
                           summary="Bash", choices=choices, **extra)


def question_request(questions, choices=frozenset({Decision.DENY})) -> ApprovalRequest:
    return ApprovalRequest(approval_id="q1", kind="question", tool="AskUserQuestion",
                           input={"questions": []}, summary="AskUserQuestion", choices=choices,
                           questions=tuple(questions))


def labels(rows) -> list[list[str]]:
    return [[button["text"] for button in row] for row in rows]


def callbacks(rows) -> list[str]:
    return [button["callback_data"] for row in rows for button in row]


def markup_callbacks(markup: str) -> list[str]:
    return callbacks(json.loads(markup)["inline_keyboard"])


class PermissionCardTests(unittest.TestCase):
    """Permission cards: the four buttons, labels, callback budget and the Codex warning."""

    def setUp(self) -> None:
        self.home = TempHome().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.h = Harness(self.home)

    def test_four_buttons_in_order(self) -> None:
        """A: four buttons in order (pa, pA, pd, pt) with the Always label showing the rule."""
        self.assertTrue(self.h.post(bash_request(), always_rule="Bash(git status *)"))
        rows = self.h.sent_keyboard()
        self.assertEqual(callbacks(rows), [f"pa:{NONCE}", f"pA:{NONCE}", f"pd:{NONCE}", f"pt:{NONCE}"])
        self.assertEqual(labels(rows), [["✅ Allow once", "♾ Always: Bash(git status *)"],
                                        ["⛔ Deny", "🤖 Bypass this turn"]])

    def test_callback_data_within_64_bytes(self) -> None:
        """A: every callback_data is <= 64 bytes."""
        self.h.post(bash_request(), always_rule="Bash(git status *)")
        for data in callbacks(self.h.sent_keyboard()):
            self.assertLessEqual(len(data.encode()), 64, data)

    def test_card_text_shows_title_and_command(self) -> None:
        self.h.post(bash_request("npm test"), always_rule="Bash(npm test)")
        text = self.h.tg.texts(THREAD)[-1]
        self.assertEqual(text, "🔐 Bash\nnpm test")

    def test_always_label_is_cut_to_thirty_characters(self) -> None:
        """A: the Always label is shortened; the rule text itself is kept in the saved rule, not in the label."""
        rule = "Bash(" + "x" * 40 + ")"
        self.h.post(bash_request(), always_rule=rule)
        label = labels(self.h.sent_keyboard())[0][1]
        self.assertEqual(label, "♾ Always: " + rule[:29] + "…")

    def test_always_hidden_without_a_rule(self) -> None:
        """Always is offered by the backend but there is no rule text, so the button is hidden."""
        self.h.post(bash_request())
        self.assertEqual(callbacks(self.h.sent_keyboard()), [f"pa:{NONCE}", f"pd:{NONCE}", f"pt:{NONCE}"])

    def test_missing_choices_are_hidden(self) -> None:
        """A backend that offers only allow once and deny gets exactly those two buttons, one per row."""
        self.h.post(bash_request(choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY})))
        self.assertEqual(labels(self.h.sent_keyboard()), [["✅ Allow once"], ["⛔ Deny"]])

    def test_codex_cancel_only_deny_shows_warning(self) -> None:
        """A Codex request offering only accept/cancel (deny_aborts_turn) shows the Deny warning line."""
        self.h.post(bash_request(choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY}), deny_aborts_turn=True))
        self.assertIn(DENY_WARNING, self.h.tg.texts(THREAD)[-1])
        self.assertTrue(self.h.tg.texts(THREAD)[-1].endswith("Deny stops this turn"))

    def test_no_warning_when_deny_is_a_plain_decline(self) -> None:
        self.h.post(bash_request(choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY})))
        self.assertNotIn("Deny stops this turn", self.h.tg.texts(THREAD)[-1])

    def test_oversized_callback_raises_and_leaves_no_card(self) -> None:
        """An oversized nonce is refused before anything is sent, and no card is kept."""
        long_nonce = "n" * 62
        with self.assertRaises(CallbackDataError):
            self.h.post(bash_request(), nonce=long_nonce)
        self.assertFalse(self.h.cards.is_open(long_nonce))
        self.assertEqual(self.h.tg.texts(THREAD), [])

    def test_refused_send_returns_false_and_logs(self) -> None:
        """A Telegram refusal to send the card: post() returns False, the card is not open, the failure is logged."""
        self.h.tg.fail_next("sendMessage", 400, description="bad")
        self.assertFalse(self.h.post(bash_request()))
        self.assertFalse(self.h.cards.is_open(NONCE))
        self.assertIn("sendButtons failed", self.h.log_text())

    def test_duplicate_nonce_is_refused(self) -> None:
        self.h.post(bash_request())
        with self.assertRaises(ValueError):
            self.h.post(bash_request())


class CloseAndEditTests(unittest.TestCase):
    """Edit to outcome: allow, deny, void, withdrawn, a second close, and the send that is still in flight."""

    def setUp(self) -> None:
        self.home = TempHome().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.h = Harness(self.home)
        self.h.post(bash_request(), always_rule="Bash(git status *)")

    def test_allow_edits_card_to_outcome_and_drops_buttons(self) -> None:
        """D/I: a closed card keeps its text, gets the outcome line and loses its keyboard."""
        self.assertTrue(self.h.cards.close(NONCE, "allow"))
        edit = self.h.last_edit()
        self.assertEqual(edit["text"], "🔐 Bash\ngit status --short\n\n✅ allowed")
        self.assertEqual(edit["reply_markup"], EMPTY_KEYBOARD)
        self.assertFalse(self.h.cards.is_open(NONCE))

    def test_outcome_texts(self) -> None:
        for verdict, text in (("deny", "⛔ denied"), ("void", "⌛ void — turn ended"), ("withdrawn", "↩️ withdrawn")):
            with self.subTest(verdict=verdict):
                h = Harness(self.home)
                h.post(bash_request(), nonce="zz" + verdict[:2])
                self.assertTrue(h.cards.close("zz" + verdict[:2], verdict))
                self.assertTrue(h.last_edit()["text"].endswith("\n\n" + text))

    def test_second_close_and_unknown_nonce_return_false(self) -> None:
        self.assertTrue(self.h.cards.close(NONCE, "allow"))
        self.assertFalse(self.h.cards.close(NONCE, "deny"))
        self.assertFalse(self.h.cards.close("nope0000", "deny"))
        self.assertEqual(len(self.h.edits()), 1)

    def test_close_cancels_the_nudge_timer(self) -> None:
        self.h.cards.close(NONCE, "allow")
        self.assertTrue(self.h.timers[0].cancelled)

    def test_close_while_still_in_flight_paints_outcome_after_post(self) -> None:
        """A card closed before its message id is known: post() paints the outcome once it has the id, and no
        nudge timer is armed for it."""
        h = Harness(self.home)
        # the hook runs inside send_buttons: the card is registered, Telegram has not yet returned its message id
        h.messenger.hook = lambda: self.assertTrue(h.cards.close("fl000001", "void"))
        self.assertTrue(h.post(bash_request(), nonce="fl000001"))
        self.assertEqual(len(h.edits()), 1)
        self.assertEqual(h.edits()[0]["message_id"], h.card_mid())
        self.assertEqual(h.edits()[0]["text"], "🔐 Bash\ngit status --short\n\n⌛ void — turn ended")
        self.assertEqual(h.edits()[0]["reply_markup"], EMPTY_KEYBOARD)
        self.assertEqual(h.timers, [])
        self.assertFalse(h.cards.is_open("fl000001"))

    def test_refused_edit_is_logged_not_raised(self) -> None:
        self.h.tg.fail_next("editMessageText", 400, description="message is not modified in a way")
        self.assertTrue(self.h.cards.close(NONCE, "allow"))
        self.assertIn("editMessageText card " + NONCE, self.h.log_text())


class NudgeTests(unittest.TestCase):
    """The 10-minute nudge: Clock-driven, cancelled on close, never early."""

    def setUp(self) -> None:
        self.home = TempHome().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.h = Harness(self.home)
        self.h.post(bash_request(), always_rule="Bash(git status *)")

    def test_timer_is_armed_for_ten_minutes(self) -> None:
        """Default nudge: ASK_NUDGE_SECONDS (600) and the timer is started."""
        self.assertEqual(ASK_NUDGE_SECONDS, 600.0)
        self.assertEqual(self.h.timers[0].seconds, 600.0)
        self.assertTrue(self.h.timers[0].started)

    def test_early_fire_does_not_nudge(self) -> None:
        self.h.clock.advance(599)
        before = len(self.h.edits())
        self.h.timers[0].fire()
        self.assertEqual(len(self.h.edits()), before)

    def test_fire_after_ten_minutes_nudges_with_buttons(self) -> None:
        """A: the nudge adds the waiting line and keeps the buttons.

        Ported labels:
        A: nudge edits text and keeps the buttons
        """
        self.h.clock.advance(600)
        self.h.timers[0].fire()
        edit = self.h.last_edit()
        self.assertEqual(edit["text"], "🔐 Bash\ngit status --short" + NUDGE_TEXT)
        self.assertEqual(markup_callbacks(edit["reply_markup"]),
                         [f"pa:{NONCE}", f"pA:{NONCE}", f"pd:{NONCE}", f"pt:{NONCE}"])

    def test_nudge_after_close_does_nothing(self) -> None:
        self.h.cards.close(NONCE, "allow")
        self.h.clock.advance(600)
        before = len(self.h.edits())
        self.h.timers[0].fire()
        self.assertEqual(len(self.h.edits()), before)


class QuestionCardTests(unittest.TestCase):
    """AskUserQuestion cards: option buttons, Other, Skip, answers and the 'Other' wait (F, G, K)."""

    def setUp(self) -> None:
        self.home = TempHome().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.h = Harness(self.home)
        self.questions = [Question(text="Which db?", options=("pg", "sqlite")),
                          Question(text="Which env?", options=("dev", "prod"))]

    def test_both_questions_and_their_buttons(self) -> None:
        """F: both questions in the card text, options and Other per question, then Skip.

        Ported labels:
        F: labels carry the question number
        K: Other button on the question card
        """
        self.h.post(question_request(self.questions))
        text = self.h.tg.texts(THREAD)[-1]
        self.assertIn("❓ Which db?", text)
        self.assertIn("❓ Which env?", text)
        self.assertEqual(callbacks(self.h.sent_keyboard()),
                         [f"q:{NONCE}:0:0", f"q:{NONCE}:0:1", f"o:{NONCE}:0",
                          f"q:{NONCE}:1:0", f"q:{NONCE}:1:1", f"o:{NONCE}:1", f"pd:{NONCE}"])
        self.assertEqual(labels(self.h.sent_keyboard())[:3], [["1. pg"], ["1. sqlite"], ["1. ✏️ Other"]])

    def test_single_question_has_plain_labels_and_no_skip_without_deny(self) -> None:
        self.h.post(question_request(self.questions[:1], choices=frozenset()))
        self.assertEqual(labels(self.h.sent_keyboard()), [["pg"], ["sqlite"], ["✏️ Other"]])

    def test_a_question_without_options_gets_other_and_skip(self) -> None:
        """Ported labels:
        K: a question with no options gets Other and Skip
        """
        self.h.post(question_request((Question(text="Free?", options=()),)))
        self.assertEqual(callbacks(self.h.sent_keyboard()), [f"o:{NONCE}:0", f"pd:{NONCE}"])

    def test_answering_the_first_question_keeps_the_turn_waiting(self) -> None:
        """F: one answer is shown with a check mark and its buttons are removed; the progress is not complete.

        Ported labels:
        F: first answer shown, its buttons removed
        """
        self.h.post(question_request(self.questions))
        progress = self.h.cards.record_answer(NONCE, 0, "sqlite")
        self.assertIsNotNone(progress)
        self.assertFalse(progress.complete)
        self.assertEqual(progress.answers, {"Which db?": "sqlite"})
        edit = self.h.last_edit()
        self.assertIn("✔ sqlite", edit["text"])
        self.assertNotIn(f"q:{NONCE}:0:0", edit["reply_markup"])
        self.assertIn(f"q:{NONCE}:1:0", edit["reply_markup"])

    def test_second_answer_completes_and_close_allow_shows_both(self) -> None:
        self.h.post(question_request(self.questions))
        self.h.cards.record_answer(NONCE, 0, "sqlite")
        progress = self.h.cards.record_answer(NONCE, 1, "dev")
        self.assertTrue(progress.complete)
        self.assertEqual(progress.answers, {"Which db?": "sqlite", "Which env?": "dev"})
        self.assertTrue(self.h.cards.close(NONCE, "allow"))
        self.assertTrue(self.h.last_edit()["text"].endswith("✔ sqlite\n\n❓ Which env?\n✔ dev\n\n✅ allowed"))

    def test_tap_after_the_answer_is_refused(self) -> None:
        """F: a tap after the card is closed finds nothing."""
        self.h.post(question_request(self.questions[:1]))
        self.h.cards.record_answer(NONCE, 0, "pg")
        self.h.cards.close(NONCE, "allow")
        self.assertIsNone(self.h.cards.record_answer(NONCE, 0, "sqlite"))
        self.assertIsNone(self.h.cards.option_label(NONCE, 0, 0))

    def test_option_label_lookup(self) -> None:
        self.h.post(question_request(self.questions))
        self.assertEqual(self.h.cards.option_label(NONCE, 1, 1), "prod")
        self.assertIsNone(self.h.cards.option_label(NONCE, 1, 2))
        self.assertIsNone(self.h.cards.option_label(NONCE, 5, 0))

    def test_skip_is_absent_when_deny_is_not_offered(self) -> None:
        self.h.post(question_request(self.questions, choices=frozenset()))
        self.assertNotIn(f"pd:{NONCE}", callbacks(self.h.sent_keyboard()))

    def test_permission_card_rejects_question_calls(self) -> None:
        self.h.post(bash_request(), always_rule="Bash(git status *)")
        self.assertIsNone(self.h.cards.record_answer(NONCE, 0, "x"))
        self.assertFalse(self.h.cards.open_other(NONCE, 0))

    def test_other_wait_then_typed_answer(self) -> None:
        """K: Other sets the wait for the topic, the card asks for the typed answer, and the typed text answers

        Ported labels:
        K: Other tap waits for a typed answer (('98fce511', 0))
        the question and clears the wait."""
        self.h.post(question_request(self.questions[:1]))
        self.assertTrue(self.h.cards.open_other(NONCE, 0))
        self.assertEqual(self.h.cards.awaiting(THREAD), (NONCE, 0))
        self.assertIn("type your answer", self.h.last_edit()["text"])
        self.assertTrue(self.h.last_edit()["text"].endswith(OTHER_WAIT_TEXT.strip()))
        progress = self.h.cards.record_answer(NONCE, 0, "release-2.3")
        self.assertTrue(progress.complete)
        self.assertEqual(progress.answers, {"Which db?": "release-2.3"})
        self.assertIsNone(self.h.cards.awaiting(THREAD))

    def test_open_other_rejects_bad_index(self) -> None:
        self.h.post(question_request(self.questions))
        self.assertFalse(self.h.cards.open_other(NONCE, 2))
        self.assertIsNone(self.h.cards.awaiting(THREAD))

    def test_closing_the_card_drops_its_wait(self) -> None:
        self.h.post(question_request(self.questions))
        self.h.cards.open_other(NONCE, 0)
        self.h.cards.close(NONCE, "void")
        self.assertIsNone(self.h.cards.awaiting(THREAD))

    def test_wait_of_another_card_is_kept(self) -> None:
        self.h.post(question_request(self.questions), nonce="other000")
        self.h.cards.open_other("other000", 0)
        self.h.post(question_request(self.questions), nonce=NONCE)
        self.h.cards.close(NONCE, "void")
        self.assertEqual(self.h.cards.awaiting(THREAD), ("other000", 0))

    def test_answer_during_send_is_repainted_after_the_post(self) -> None:
        """An answer recorded while the card is still being sent is shown once the message id is known."""
        self.h.messenger.hook = lambda: self.h.cards.record_answer(NONCE, 0, "pg")
        self.h.post(question_request(self.questions[:1]))
        self.assertEqual(len(self.h.edits()), 1)
        self.assertIn("✔ pg", self.h.last_edit()["text"])


class CallbackBudgetTests(unittest.TestCase):
    def test_all_card_callbacks_decode(self) -> None:
        for data in (f"pa:{NONCE}", f"q:{NONCE}:1:0", f"o:{NONCE}:0", f"pd:{NONCE}"):
            decoded = CallbackData.decode(data)
            self.assertEqual(decoded.prefix, data.split(":")[0])
            self.assertEqual(decoded.encode(), data)


if __name__ == "__main__":
    unittest.main()
