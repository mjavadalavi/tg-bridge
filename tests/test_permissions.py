"""PermissionService (STEP 35): port of the non-subprocess scenarios of fakecli/test_permissions.py (labels A-J) and
fakecli/test_p2b_cards.py (labels A, D, E, F, G, I, K), plus the Codex always-rule and the two-tap idempotence test.

Everything runs on FakeBackend sessions (no subprocess), FakeTelegram and FakeClock inside a TempHome. Workers are inline
unless a test says otherwise, so every job finishes before the call returns.
"""

import dataclasses
import json
import os
import threading
import time
import unittest

from tests.fakes.backend import FAKE_CAPABILITIES, FakeBackend, RecordingSink, fake_spec
from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.agents.claude.backend import ClaudeRules
from tg_bridge.agents.codex.rules import CodexRuleMapper
from tg_bridge.agents.events import ApprovalRequest, ApprovalWithdrawn, Decision, Question, TurnResult, TurnStatus
from tg_bridge.core.cards import ApprovalCards
from tg_bridge.core.permissions import BYPASS_MODE, PermissionService, TurnBinding
from tg_bridge.core.workers import WorkerPool
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.asks import AskRepo
from tg_bridge.store.db import Database
from tg_bridge.store.rules import RuleRepo
from tg_bridge.telegram.keyboards import EMPTY_KEYBOARD
from tg_bridge.telegram.messenger import Messenger

THREAD = 9101
CHAT = -100
CWD = "/work"
ALL = frozenset({Decision.ALLOW_ONCE, Decision.ALLOW_ALWAYS, Decision.ALLOW_TURN, Decision.DENY})
QUESTIONS = (Question(text="Which db?", options=("pg", "sqlite")),
             Question(text="Which env?", options=("dev", "prod")))


class RecordingTimer:
    """A nudge timer that never fires by itself; the tests do not need the nudge here."""

    def __init__(self, seconds, fire) -> None:
        self.seconds = seconds

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        pass


class WaitRecorder:
    """The WaitObserver of the tests: every (thread, waiting) change in order."""

    def __init__(self) -> None:
        self.changes: list[tuple[int, bool]] = []

    def waiting_changed(self, thread_id: int, waiting: bool) -> None:
        self.changes.append((thread_id, waiting))


class InlineWorkers:
    """Runs each submitted job at once, on the calling thread."""

    def submit(self, fn, *args) -> None:
        fn(*args)


class BlockingMessenger(Messenger):
    """A Messenger whose card send waits on self.gate (when set) after announcing itself on self.sending."""

    gate = None
    sending = None
    returned = None

    def send_buttons(self, chat_id, thread_id, text, rows):
        if self.gate is not None:
            self.sending.set()
            self.gate.wait(15.0)
        result = super().send_buttons(chat_id, thread_id, text, rows)
        if self.returned is not None:
            self.returned.set()
        return result


class DeferredWorkers:
    """Holds submitted jobs until run() is called, to test a tap that arrives before the card is posted."""

    def __init__(self) -> None:
        self.jobs: list = []

    def submit(self, fn, *args) -> None:
        self.jobs.append((fn, args))

    def run(self) -> None:
        jobs, self.jobs = self.jobs, []
        for fn, args in jobs:
            fn(*args)


class TapInFlightMessenger(Messenger):
    """A Messenger whose first send_buttons runs a hook first, so a tap can land while the card is being sent."""

    hook = None

    def send_buttons(self, chat_id, thread_id, text, rows):
        if self.hook is not None:
            hook, self.hook = self.hook, None
            hook()
        return super().send_buttons(chat_id, thread_id, text, rows)


def bash_request(command="git status --short", approval_id="r1", choices=ALL, **extra) -> ApprovalRequest:
    return ApprovalRequest(approval_id=approval_id, kind="command", tool="Bash", input={"command": command},
                           summary="Bash", choices=choices, **extra)


def question_request(questions=QUESTIONS, approval_id="q1") -> ApprovalRequest:
    return ApprovalRequest(approval_id=approval_id, kind="question", tool="AskUserQuestion",
                           input={"questions": [q.text for q in questions]}, summary="question",
                           choices=frozenset({Decision.DENY}), questions=tuple(questions))


class PermissionCase(unittest.TestCase):
    """A PermissionService on real repos (migrated sqlite under TempHome), with FakeTelegram and one FakeBackend session."""

    messenger_cls = Messenger

    def setUp(self) -> None:
        self._home = TempHome()
        home = self._home.__enter__().home
        self.clock = FakeClock()
        self.tg = FakeTelegram(clock=self.clock)
        self.logger = Logger(os.path.join(home, "bridge.log"))
        self.messenger = self.messenger_cls(self.tg, self.logger, self.clock)
        self.db = Database(Paths.from_home(home))
        self.asks = AskRepo(self.db, self.clock)
        self.rules = RuleRepo(self.db, self.clock)
        self.cards = ApprovalCards(self.messenger, self.logger, self.clock, timer_factory=RecordingTimer)
        self.waits = WaitRecorder()
        self.workers = InlineWorkers()
        self.nonces = iter(f"{n:08x}" for n in range(1, 10_000))
        self.service = self.build_service()
        self.backend = FakeBackend(capabilities=self.capabilities())
        self.session = self.backend.open_session(fake_spec(thread_id=THREAD, chat_id=CHAT), RecordingSink())
        self.binding = TurnBinding(thread_id=THREAD, chat_id=CHAT, cwd=CWD, rules=self.rules_matcher(), mode="default")
        self.listener = self.service.bind_turn(self.binding)
        self.service.attach(self.binding, self.session)

    def tearDown(self) -> None:
        self.db.close()
        self._home.__exit__(None, None, None)

    # hooks for subclasses
    def capabilities(self):
        return FAKE_CAPABILITIES

    def rules_matcher(self):
        return ClaudeRules()

    def build_service(self) -> PermissionService:
        return PermissionService(asks=self.asks, rules=self.rules, cards=self.cards, messenger=self.messenger,
                                 workers=self.workers, waiting=self.waits, logger=self.logger,
                                 nonce_factory=lambda: next(self.nonces))

    # helpers
    def emit(self, event) -> None:
        self.listener.on_event(event)

    def posted(self) -> list[dict]:
        return [c.params for c in self.tg.calls if c.method == "sendMessage"]

    def card_edits(self) -> list[dict]:
        return [c.params for c in self.tg.calls if c.method == "editMessageText"]

    def callbacks(self) -> list[str]:
        """Every callback_data on the posted cards, in order."""
        out = []
        for keyboard in self.tg.buttons(THREAD):
            out += [button["callback_data"] for row in keyboard for button in row]
        return out

    def answers(self) -> list[tuple]:
        return [c for c in self.session.calls if c[0] == "answer"]

    def row(self, nonce: str) -> str:
        record = self.asks.by_nonce(nonce)
        return record.answer if record else "missing"

    def rule_texts(self) -> list[str]:
        return self.rules.texts(THREAD)

    def button_texts(self) -> list[str]:
        return [button["text"] for keyboard in self.tg.buttons(THREAD) for row in keyboard for button in row]


class PermissionFlowTests(PermissionCase):
    def test_A_card_is_posted_with_allow_always_deny_and_bypass_buttons(self) -> None:
        """A: one card with pa / pA / pd / pt buttons carrying the nonce; the ask is pending in the database.

        Ported labels:
        A: buttons are pa / pA / pd / pt with the nonce
        A: card text '🔐 Bash\\necho hi'
        A: one card posted (got 1)
        """
        self.emit(bash_request())
        self.assertEqual(len(self.posted()), 1)
        self.assertEqual(self.posted()[0]["text"], "🔐 Bash\ngit status --short")
        self.assertEqual(self.callbacks(), ["pa:00000001", "pA:00000001", "pd:00000001", "pt:00000001"])
        self.assertTrue(all(len(c.encode()) <= 64 for c in self.callbacks()))
        self.assertEqual(self.row("00000001"), "")
        self.assertEqual(self.waits.changes, [(THREAD, True)])

    def test_A_tap_allow_answers_the_session_once_and_closes_the_card(self) -> None:
        """A: tap Allow -> ok:allow; asks row = allow; card edited to allowed with an empty keyboard.

        Ported labels:
        A: asks row answer = allow
        A: card edited to allowed with empty keyboard
        A: tap Allow -> ok:allow (got {'ans': 'ok:allow', 'err': None})
        A: registries empty
        """
        self.emit(bash_request())
        self.assertTrue(self.service.answer_ask("00000001", True))
        self.assertEqual(self.answers(), [("answer", "r1", Decision.ALLOW_ONCE, None, None)])
        self.assertEqual(self.row("00000001"), "allow")
        self.assertEqual(self.asks.pending(THREAD), [])
        last = self.card_edits()[-1]
        self.assertTrue(last["text"].endswith("✅ allowed"))
        self.assertEqual(last["reply_markup"], EMPTY_KEYBOARD)
        self.assertEqual(self.waits.changes, [(THREAD, True), (THREAD, False)])

    def test_B_deny_answers_deny_and_marks_the_card_denied(self) -> None:
        """B: tap Deny -> ok:deny; row deny; card edited to denied.

        Ported labels:
        B: tap Deny -> ok:deny (got {'ans': 'ok:deny', 'err': None})
        """
        self.emit(bash_request())
        self.assertTrue(self.service.answer_ask("00000001", False))
        self.assertEqual(self.answers(), [("answer", "r1", Decision.DENY, None, None)])
        self.assertEqual(self.row("00000001"), "deny")
        self.assertTrue(self.card_edits()[-1]["text"].endswith("⛔ denied"))

    def test_C_saved_rule_allows_without_a_card_and_counts_a_hit(self) -> None:
        """C: a saved Bash(touch *) rule answers the request, no card is posted, hits is 1.

        Ported labels:
        C: hits incremented (1)
        C: no card posted
        C: rule allows (got 'ok:allow', None)
        """
        self.rules.add(THREAD, "Bash(touch *)")
        self.emit(bash_request("touch probe_ok.txt"))
        self.assertEqual(self.answers(), [("answer", "r1", Decision.ALLOW_ONCE, None, None)])
        self.assertEqual(self.posted(), [])
        self.assertEqual(self.rules.for_thread(THREAD)[0].hits, 1)

    def test_C_compound_command_does_not_match_a_pattern_rule(self) -> None:
        """H (compound refusal): 'touch a; rm -rf x' is not covered by Bash(touch *), so a card is posted."""
        self.rules.add(THREAD, "Bash(touch *)")
        self.emit(bash_request("touch a; rm -rf x"))
        self.assertEqual(len(self.posted()), 1)
        self.assertEqual(self.answers(), [])

    def test_always_saves_the_rule_and_the_next_identical_request_is_allowed(self) -> None:
        """A/C: Always saves 'Bash(git status *)'; the next 'git status' is allowed with no new card.

        Ported labels:
        A: Always -> allow ({'ans': 'ok:allow {"updatedInput": {"command": "git status --short"}, "updatedPermissions": [{"behavior": "allow", "destination": "session", "rules": [{"ruleContent": "git status *", "toolName": "Bash"}], "type": "addRules"}]}', 'err': None})
        A: perm_rules row stored (['Bash(git status *)'])
        A: Always label shows the synthesized rule ('♾ Always: Bash(git status *)')
        """
        self.emit(bash_request())
        self.assertIn("♾ Always: Bash(git status *)", self.button_texts())
        self.assertTrue(self.service.answer_always("00000001"))
        self.assertEqual(self.rule_texts(), ["Bash(git status *)"])
        self.assertEqual(self.answers(), [("answer", "r1", Decision.ALLOW_ALWAYS, None, None)])
        self.emit(bash_request("git status", approval_id="r2"))
        self.assertEqual(self.answers()[-1], ("answer", "r2", Decision.ALLOW_ONCE, None, None))
        self.assertEqual(len(self.posted()), 1)

    def test_always_is_refused_when_the_request_does_not_offer_it(self) -> None:
        """Without ALLOW_ALWAYS in the choices no Always button is shown and answer_always returns False."""
        self.emit(bash_request(choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY})))
        self.assertFalse(self.service.answer_always("00000001"))
        self.assertEqual(self.rule_texts(), [])
        self.assertEqual(self.answers(), [])

    def test_codex_always_rule_is_saved_as_bash_and_auto_accepts_next_time(self) -> None:
        """A Codex always-rule is saved as Bash(<prefix> *) in the shared grammar and auto-accepts the next command."""
        self.binding.rules = CodexRuleMapper()
        codex = ApprovalRequest(approval_id="rpc:1", kind="command", tool="commandExecution",
                                input={"command": "touch a.txt", "cwd": CWD, "prefix": "touch"},
                                summary="$ touch a.txt", choices=frozenset({Decision.ALLOW_ONCE, Decision.ALLOW_ALWAYS,
                                                                             Decision.ALLOW_TURN, Decision.DENY}))
        self.emit(codex)
        self.assertTrue(self.service.answer_always("00000001"))
        self.assertEqual(self.rule_texts(), ["Bash(touch *)"])
        second = ApprovalRequest(approval_id="rpc:2", kind="command", tool="commandExecution",
                                 input={"command": "touch b.txt", "cwd": CWD, "prefix": "touch"},
                                 summary="$ touch b.txt", choices=codex.choices)
        self.emit(second)
        self.assertEqual(self.answers()[-1], ("answer", "rpc:2", Decision.ALLOW_ONCE, None, None))
        self.assertEqual(len(self.posted()), 1)

    def test_two_taps_on_one_card_answer_the_session_exactly_once(self) -> None:
        """Two concurrent taps (double tap) on one card: exactly one call answers the session."""
        for round_no in range(5):
            approval_id = f"race{round_no}"
            self.emit(bash_request(approval_id=approval_id))
            wanted = self._nonce_for(approval_id)
            barrier = threading.Barrier(2)
            results: list[bool] = []

            def tap() -> None:
                barrier.wait()
                results.append(self.service.answer_ask(wanted, True))

            threads = [threading.Thread(target=tap) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(5)
            self.assertEqual(sorted(results), [False, True], f"round {round_no}")
            answered = [a for a in self.answers() if a[1] == approval_id]
            self.assertEqual(len(answered), 1, f"round {round_no} answered {len(answered)} times")

    def _nonce_for(self, approval_id: str) -> str:
        return next(r.nonce for r in self.asks.pending(THREAD) if r.request_id == approval_id)

    def test_replayed_tap_after_the_answer_finds_nothing(self) -> None:
        """A replayed callback after the answer: False, and the session is not answered again."""
        self.emit(bash_request())
        self.assertTrue(self.service.answer_ask("00000001", True))
        self.assertFalse(self.service.answer_ask("00000001", True))
        self.assertFalse(self.service.answer_ask("00000001", False))
        self.assertEqual(len(self.answers()), 1)

    def test_bypass_this_turn_switches_the_turn_and_allows_the_card(self) -> None:
        """D: Bypass this turn -> live switch to bypassPermissions, the card allowed, later asks allowed without a card,

        Ported labels:
        D: live switch to bypass sent ([('initialize', {'hooks': None}), ('set_permission_mode', {'mode': 'bypassPermissions'}), ('set_permission_mode', {'mode': 'default'})])
        D: no perm_rules row written by Bypass
        and no perm_rules row is written."""
        self.emit(bash_request())
        self.assertTrue(self.service.answer_bypass("00000001"))
        self.assertIn(("set_mode", BYPASS_MODE), self.session.calls)
        self.assertEqual(self.answers(), [("answer", "r1", Decision.ALLOW_ONCE, None, None)])
        self.assertEqual(self.row("00000001"), "allow")
        self.assertTrue(self.binding.turn_bypass)
        self.assertEqual(self.binding.mode, BYPASS_MODE)
        self.emit(bash_request("ls -la", approval_id="r2"))
        self.assertEqual(self.answers()[-1], ("answer", "r2", Decision.ALLOW_ONCE, None, None))
        self.assertEqual(len(self.posted()), 1)
        self.assertEqual(self.rule_texts(), [])

    def test_always_on_a_suggested_rule_and_on_write_saves_the_rule_text(self) -> None:
        """Ported labels:
        B: perm_rules row from the suggestion
        C: edit tools label Always: Edit
        C: Edit row in perm_rules
        """
        suggested = {"type": "control_request", "request_id": "r2", "request": {
            "subtype": "can_use_tool", "tool_name": "Bash", "input": {"command": "npm test"},
            "permission_suggestions": [{"type": "addRules", "rules": [{"toolName": "Bash", "ruleContent": "npm test"}],
                                        "behavior": "allow", "destination": "localSettings"}]}}
        self.emit(bash_request("npm test", approval_id="r2", raw=suggested))
        self.assertIn("♾ Always: Bash(npm test)", self.button_texts())
        self.assertTrue(self.service.answer_always("00000001"))
        self.assertIn("Bash(npm test)", self.rule_texts())
        write = ApprovalRequest(approval_id="r3", kind="tool", tool="Write",
                                input={"file_path": "/tmp/p2b/a.txt", "content": "x"}, summary="Write",
                                choices=ALL)
        self.emit(write)
        self.assertIn("♾ Always: Edit", self.button_texts())
        self.assertTrue(self.service.answer_always("00000002"))
        self.assertIn("Edit", self.rule_texts())

    def test_bypass_this_turn_ends_with_its_turn_and_the_next_turn_asks_again(self) -> None:
        """Ported labels:
        D: topic mode restored on the result
        D: session mode back to default, flag cleared
        D: the next turn posts a fresh card
        D: next turn can still be denied
        """
        self.emit(bash_request())
        self.assertTrue(self.service.answer_bypass("00000001"))
        self.assertIn(("set_mode", BYPASS_MODE), self.session.calls)
        self.emit(TurnResult(status=TurnStatus.OK, answer="done"))
        self.session.stop()
        restore = self.session.calls.index(("set_mode", "default"))
        self.assertLess(restore, self.session.calls.index(("stop", 5.0)), "the CLI must hear the restore while alive")
        self.assertEqual((self.binding.mode, self.binding.turn_bypass), ("default", False))
        self.assertEqual(self.service.void_all(self.binding), 0)
        self.assertEqual(self.session.calls.count(("set_mode", "default")), 1)
        turn = TurnBinding(thread_id=THREAD, chat_id=CHAT, cwd=CWD, rules=ClaudeRules(), mode="default")
        self.service.attach(turn, self.session)
        self.service.bind_turn(turn).on_event(bash_request("ls -la", approval_id="r2"))
        self.assertEqual(len(self.posted()), 2)
        self.assertTrue(self.service.answer_ask("00000002", False))
        self.assertEqual(self.answers()[-1], ("answer", "r2", Decision.DENY, None, None))

    def test_void_all_ends_a_bypass_turn_without_a_result_and_without_calling_the_cli(self) -> None:
        """The process ended before any result: void_all clears the turn's bypass flag and mode, and sends nothing,
        since the session is already stopped."""
        self.emit(bash_request())
        self.assertTrue(self.service.answer_bypass("00000001"))
        self.service.void_all(self.binding)
        self.assertEqual((self.binding.mode, self.binding.turn_bypass), ("default", False))
        self.assertEqual([c for c in self.session.calls if c[0] == "set_mode"], [("set_mode", BYPASS_MODE)])

    def test_bypass_mode_allows_new_asks_without_a_card(self) -> None:
        """E (mode part): a turn in bypassPermissions answers every permission request itself."""
        self.binding.mode = BYPASS_MODE
        self.emit(bash_request())
        self.assertEqual(self.answers(), [("answer", "r1", Decision.ALLOW_ONCE, None, None)])
        self.assertEqual(self.posted(), [])
        self.assertEqual(self.asks.pending(THREAD), [])

    def test_set_live_mode_bypass_allows_open_permission_cards_but_never_a_question(self) -> None:
        """E: bypass allows the open permission card, the question card still waits for its answers.

        Ported labels:
        E: live mode switched
        E: the permission card was allowed ([('r1', True)])
        E: question card still waiting for answers
        """
        self.emit(bash_request())
        self.emit(question_request())
        self.assertTrue(self.service.set_live_mode(THREAD, BYPASS_MODE))
        self.assertIn(("set_mode", BYPASS_MODE), self.session.calls)
        self.assertEqual(self.answers(), [("answer", "r1", Decision.ALLOW_ONCE, None, None)])
        self.assertEqual(self.row("00000001"), "allow")
        self.assertEqual(self.row("00000002"), "")

    def test_auto_on_during_a_live_turn_allows_the_open_card_and_switches_the_session(self) -> None:
        """What 'auto on' does to a running turn: the open card is allowed, no ask stays open, the session switches.

        Ported labels:
        E: one permission card open on the live session
        E: open card auto-allowed by auto on
        E: no open ask is left after auto on
        E: set_permission_mode bypass sent
        E: card closed as allowed
        E: live session mode updated
        """
        self.emit(bash_request())
        self.assertEqual([ask.nonce for ask in self.asks.pending(THREAD)], ["00000001"])
        self.assertTrue(self.service.set_live_mode(THREAD, BYPASS_MODE, by="auto on"))
        self.assertEqual(self.answers(), [("answer", "r1", Decision.ALLOW_ONCE, None, None)])
        self.assertIn(("set_mode", BYPASS_MODE), self.session.calls)
        self.assertEqual(self.binding.mode, BYPASS_MODE)
        self.assertEqual(self.asks.pending(THREAD), [])
        self.assertFalse(self.service.answer_ask("00000001", True))
        self.assertEqual(self.row("00000001"), "allow")
        self.assertTrue(self.card_edits()[-1]["text"].endswith("✅ allowed"))

    def test_auto_off_during_a_live_turn_leaves_the_open_card_waiting(self) -> None:
        """'auto off' returns the running turn to default: the open card waits for a tap, and a tap can still deny.

        Ported labels:
        F: auto off leaves the open card waiting
        F: set_permission_mode default sent
        F: the owner can still deny after auto off
        """
        self.emit(bash_request())
        self.assertTrue(self.service.set_live_mode(THREAD, "default", by="auto off"))
        self.assertEqual(self.answers(), [])
        self.assertEqual([ask.nonce for ask in self.asks.pending(THREAD)], ["00000001"])
        self.assertIn(("set_mode", "default"), self.session.calls)
        self.assertTrue(self.service.answer_ask("00000001", False))
        self.assertEqual(self.answers(), [("answer", "r1", Decision.DENY, None, None)])
        self.assertEqual(self.row("00000001"), "deny")

    def test_set_live_mode_reports_false_when_the_backend_cannot_switch_live(self) -> None:
        """Codex-style backend: set_mode is not live, the call says so, and the mode is still recorded for the turn."""
        backend = FakeBackend(capabilities=dataclasses.replace(FAKE_CAPABILITIES, live_mode=False))
        session = backend.open_session(fake_spec(thread_id=THREAD, chat_id=CHAT), RecordingSink())
        self.service.attach(self.binding, session)
        self.assertFalse(self.service.set_live_mode(THREAD, "plan"))
        self.assertIn(("set_mode", "plan"), session.calls)
        self.assertEqual(self.binding.mode, "plan")

    def test_set_live_mode_without_a_running_turn_is_false(self) -> None:
        self.service.void_all(self.binding)
        self.assertFalse(self.service.set_live_mode(THREAD, BYPASS_MODE))
        self.assertFalse(self.service.set_live_mode(THREAD + 1, BYPASS_MODE))

    def test_withdrawn_ask_closes_the_card_and_is_not_answered(self) -> None:
        """F: the CLI withdraws the ask -> row withdrawn, card edited withdrawn, the session gets no answer.

        Ported labels:
        F: asks row answer = withdrawn
        """
        self.emit(bash_request())
        self.emit(ApprovalWithdrawn(approval_id="r1"))
        self.assertEqual(self.row("00000001"), "withdrawn")
        self.assertTrue(self.card_edits()[-1]["text"].endswith("↩️ withdrawn"))
        self.assertEqual(self.answers(), [])
        self.assertEqual(self.waits.changes, [(THREAD, True), (THREAD, False)])

    def test_turn_end_voids_open_asks_and_denies_later_requests(self) -> None:
        """E: the turn ends with an ask pending -> row void, card closed as void; a late tap finds nothing; a request

        Ported labels:
        E: answer_ask returns False for a stale ask
        E: asks row answer = void
        E: turn ended with the ask pending (got 'ok:unanswered')
        I: answer_bypass / answer_always return False when gone
        I: turn ended with the ask pending ('ok:unanswered')
        after the end is denied at once."""
        self.emit(bash_request())
        self.assertEqual(self.service.void_all(self.binding), 1)
        self.assertEqual(self.row("00000001"), "void")
        self.assertTrue(self.card_edits()[-1]["text"].endswith("⌛ void — turn ended"))
        self.assertFalse(self.service.answer_ask("00000001", True))
        self.assertFalse(self.service.answer_bypass("00000001"))
        self.assertFalse(self.service.answer_always("00000001"))
        self.assertEqual(self.answers(), [])
        self.emit(bash_request("ls", approval_id="r9"))
        self.assertEqual(self.answers(), [("answer", "r9", Decision.DENY, None, None)])
        self.assertEqual(self.waits.changes[-1], (THREAD, False))

    def test_card_that_cannot_be_posted_denies_the_ask(self) -> None:
        """G: Telegram refuses the card -> the ask is denied, the row says deny, and waiting ends.

        Ported labels:
        G: unposted card denied ('ok:deny', <sqlite3.Row object at 0x7f4219987550>)
        G: waiting_asks restored
        """
        self.tg.fail_next("sendMessage", 400, description="chat not found")
        self.emit(bash_request())
        self.assertEqual(self.answers(), [("answer", "r1", Decision.DENY, None, None)])
        self.assertEqual(self.row("00000001"), "deny")
        self.assertEqual(self.waits.changes, [(THREAD, True), (THREAD, False)])

    def test_tap_while_the_card_is_in_flight_closes_it_once(self) -> None:
        """A tap that lands during the card send: the session is answered and the card ends with its outcome."""
        self.messenger = TapInFlightMessenger(self.tg, self.logger, self.clock)
        self.cards = ApprovalCards(self.messenger, self.logger, self.clock, timer_factory=RecordingTimer)
        self.service = self.build_service()
        self.listener = self.service.bind_turn(self.binding)
        self.service.attach(self.binding, self.session)
        self.messenger.hook = lambda: self.service.answer_ask("00000001", True)
        self.emit(bash_request())
        self.assertEqual(self.answers(), [("answer", "r1", Decision.ALLOW_ONCE, None, None)])
        self.assertEqual(len(self.posted()), 1)
        self.assertTrue(self.card_edits()[-1]["text"].endswith("✅ allowed"))
        self.assertEqual(self.card_edits()[-1]["reply_markup"], EMPTY_KEYBOARD)

    def test_tap_before_the_card_job_runs_posts_no_card(self) -> None:
        """An ask answered before its post job runs: the job sees the claim and sends nothing."""
        deferred = DeferredWorkers()
        self.service = PermissionService(asks=self.asks, rules=self.rules, cards=self.cards, messenger=self.messenger,
                                         workers=deferred, waiting=self.waits, logger=self.logger,
                                         nonce_factory=lambda: next(self.nonces))
        self.listener = self.service.bind_turn(self.binding)
        self.service.attach(self.binding, self.session)
        self.emit(bash_request())
        self.assertTrue(self.service.answer_ask("00000001", False))
        deferred.run()
        self.assertEqual(self.posted(), [])
        self.assertEqual(self.answers(), [("answer", "r1", Decision.DENY, None, None)])

    def test_waiting_is_reported_once_per_turn_not_per_card(self) -> None:
        """Two open asks: the observer hears 'waiting' once, and 'not waiting' only after the last one is answered.

        Ported labels:
        D: took 4.0s, waiting_asks back to 0
        D: waiting_asks == 1 while asked (1)
        """
        self.emit(bash_request(approval_id="a"))
        self.emit(bash_request("ls", approval_id="b"))
        self.assertEqual(self.waits.changes, [(THREAD, True)])
        self.service.answer_ask("00000001", True)
        self.assertEqual(self.waits.changes, [(THREAD, True)])
        self.service.answer_ask("00000002", True)
        self.assertEqual(self.waits.changes, [(THREAD, True), (THREAD, False)])

    def test_real_worker_pool_posts_the_card_off_the_calling_thread(self) -> None:
        """The card goes out on a WorkerPool thread, so the reader thread that delivered the request never waits for it."""
        pool = WorkerPool("perm-test", 2, self.logger)
        self.service = PermissionService(asks=self.asks, rules=self.rules, cards=self.cards, messenger=self.messenger,
                                         workers=pool, waiting=self.waits, logger=self.logger,
                                         nonce_factory=lambda: next(self.nonces))
        self.listener = self.service.bind_turn(self.binding)
        self.service.attach(self.binding, self.session)
        self.emit(bash_request())
        self.assertTrue(pool.join(5))
        self.assertEqual(len(self.posted()), 1)


class QuestionTests(PermissionCase):
    def test_F_question_card_collects_answers_then_allows_with_them(self) -> None:
        """F: option taps and a typed 'Other' answer: the session is answered only when both questions have one,

        Ported labels:
        F: a tap after the answer is expired (['expired — that turn already ended'])
        F: all answered -> allow ({'ans': 'ok:allow {"updatedInput": {"answers": {"Which db?": "sqlite", "Which env?": "dev"}, "questions": [{"header": "DB", "multiSelect": false, "options": [{"label": "pg"}, {"label": "sqlite"}], "question": "Which db?"}, {"header": "Env", "multiSelect": false, "options": [{"label": "dev"}, {"label": "prod"}], "question": "Which env?"}]}}', 'err': None})
        F: one answer is not enough, the turn waits
        K: the typed text is the answer to the question
        K: typed text -> allow ({'ans': 'ok:allow {"updatedInput": {"answers": {"Which branch should I deploy?": "release-2.3"}, "questions": [{"header": "Deploy", "multiSelect": false, "options": [{"label": "main"}, {"label": "staging"}], "question": "Which branch should I deploy?"}]}}', 'err': None})
        with answers keyed by question text; a later tap is refused."""
        self.emit(question_request())
        nonce = "00000001"
        self.assertTrue(self.service.answer_question(nonce, 0, 1))
        self.assertEqual(self.answers(), [])
        self.assertTrue(self.service.answer_other(nonce, 1))
        self.assertTrue(self.service.record_text(THREAD, "stage"))
        self.assertEqual(self.answers(), [("answer", "q1", Decision.ALLOW_ONCE, None,
                                           {"Which db?": "sqlite", "Which env?": "stage"})])
        self.assertEqual(self.row(nonce), "allow")
        self.assertFalse(self.service.answer_question(nonce, 0, 0))
        self.assertEqual(len(self.answers()), 1)

    def test_K_other_waits_for_the_typed_text_and_clears_after_it(self) -> None:
        """K: 'Other' sets the wait for the topic; the typed text answers it; afterwards nothing is waiting.

        Ported labels:
        K: the wait is cleared
        """
        self.emit(question_request())
        self.assertFalse(self.service.record_text(THREAD, "too early"))
        self.assertTrue(self.service.answer_other("00000001", 0))
        self.assertTrue(self.service.record_text(THREAD, "release-2.3"))
        self.assertFalse(self.service.record_text(THREAD, "again"))
        self.assertEqual(self.cards.awaiting(THREAD), None)

    def test_question_cannot_be_allowed_without_answers_and_skip_denies(self) -> None:
        """G/F: Allow on a question is refused; Skip (deny) answers the session with DENY.

        Ported labels:
        G: Skip denies the question ({'ans': 'ok:deny {"message": "denied by owner from Telegram"}', 'err': None})
        """
        self.emit(question_request())
        self.assertFalse(self.service.answer_ask("00000001", True))
        self.assertTrue(self.service.answer_ask("00000001", False))
        self.assertEqual(self.answers(), [("answer", "q1", Decision.DENY, None, None)])
        self.assertEqual(self.row("00000001"), "deny")

    def test_H_question_with_no_questions_is_denied_without_a_card(self) -> None:
        """H: an AskUserQuestion with no questions is denied at once.

        Ported labels:
        H: empty question list denied ('ok:deny {"message": "no questions to answer"}')
        """
        self.emit(ApprovalRequest(approval_id="q2", kind="question", tool="AskUserQuestion", input={},
                                  summary="q", choices=frozenset({Decision.DENY}), questions=()))
        self.assertEqual(self.answers(), [("answer", "q2", Decision.DENY, None, None)])
        self.assertEqual(self.posted(), [])

    def test_bypass_never_auto_answers_a_question(self) -> None:
        """E: in bypass the question still gets its card, and answer_bypass refuses a question."""
        self.binding.mode = BYPASS_MODE
        self.emit(question_request())
        self.assertEqual(len(self.posted()), 1)
        self.assertFalse(self.service.answer_bypass("00000001"))
        self.assertEqual(self.answers(), [])


class RecoveryTests(PermissionCase):
    def test_recover_cards_voids_asks_left_by_a_previous_run(self) -> None:
        """Start-up: a pending ask from a dead process is voided and its card is closed with the restart text."""
        self.asks.insert(nonce="deadbeef", thread_id=THREAD, chat_id=CHAT, request_id="old1", tool_name="Bash",
                         input={"command": "rm -rf build"})
        self.asks.set_message_id("deadbeef", 77)
        self.assertEqual(self.service.recover_cards(), 1)
        self.assertEqual(self.row("deadbeef"), "void")
        edit = [c for c in self.tg.calls if c.method == "editMessageText"][-1]
        self.assertEqual(edit.params["message_id"], 77)
        self.assertEqual(edit.params["reply_markup"], EMPTY_KEYBOARD)
        self.assertIn("bridge restarted", edit.params["text"])
        self.assertEqual(self.service.recover_cards(), 0)


class CardOrderTests(PermissionCase):
    def test_request_with_no_session_is_dropped_and_logged(self) -> None:
        self.binding.session = None
        self.emit(bash_request())
        self.assertEqual(self.posted(), [])
        with open(self.logger.path, encoding="utf-8") as handle:
            self.assertIn("has no session", handle.read())

    def test_card_payload_never_carries_the_raw_backend_request(self) -> None:
        """The card is built from the summary and the input only; the raw backend payload never reaches Telegram."""
        self.emit(bash_request(raw={"secret": "token-123"}))
        self.assertNotIn("token-123", json.dumps(self.posted()))


class BlockedCardTests(PermissionCase):
    """The reader hands a withdrawal to the service while the card of the ask is still being sent on a worker."""

    messenger_cls = BlockingMessenger

    def setUp(self) -> None:
        super().setUp()
        self.messenger.gate = threading.Event()
        self.messenger.sending = threading.Event()
        self.messenger.returned = threading.Event()
        self.addCleanup(self.messenger.gate.set)
        self.workers = WorkerPool("cards", 1, self.logger)
        self.addCleanup(self.workers.join, 5)
        self.service = self.build_service()
        self.listener = self.service.bind_turn(self.binding)
        self.service.attach(self.binding, self.session)

    def test_a_withdrawal_is_processed_while_the_card_send_is_blocked(self) -> None:
        """Ported labels:
        3: cancel processed while the card send is blocked
        3: cancel ran before the blocked send returned
        3: turn completed without waiting for the send

        The withdrawal (the CLI's cancel of the ask) returns at once and closes the ask, while the card send is still
        blocked on the worker; the send finishes afterwards.
        """
        self.emit(bash_request())
        self.assertTrue(self.messenger.sending.wait(15.0), "the card send started")
        started = time.monotonic()
        self.listener.on_event(ApprovalWithdrawn(approval_id="r1"))
        took = time.monotonic() - started
        self.assertFalse(self.messenger.returned.is_set(), "the withdrawal ran before the blocked send returned")
        self.assertLess(took, 2.0, f"the withdrawal waited for the send ({took:.2f}s)")
        self.assertEqual(self.row("00000001"), "withdrawn")
        self.messenger.gate.set()
        self.assertTrue(self.workers.join(5))
        self.assertTrue(self.messenger.returned.is_set())


if __name__ == "__main__":
    unittest.main()
