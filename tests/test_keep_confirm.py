"""KeepGuard and its confirm card: a change that would close a session kept open for background work waits for a tap
(section 5 D1, card C9d).

Scenarios: with nothing kept, with nothing pending (the quiet window) and during a user turn on the kept process, a
change applies at once as before. With background work pending on an idle or self-turn kept session, /mode, /model,
/launcher, /cwd, /new, /resume, the mode and folder picker taps and 'auto on' / 'auto off' post the card and change
nothing; 'change and close' applies the change and asks the session to close, 'keep the session' drops it, a card
answers once, a new card replaces the earlier one, and a token of another topic is expired. A change that stores the
value the topic already has, a search-card folder tap, and the quiet or nothing-kept typed commands are covered too.
Every test runs against a migrated database in a TempHome with FakeTelegram, a stub kept session (StubKeptLinger) and a
FakeClock; no subprocess starts.
"""

import json
import os
import unittest
from collections.abc import Callable
from typing import Any

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.inbound_support import ALLOWED_UID, CHAT, CLAUDE_CMD, THREAD, TOPIC, InboundCase
from tests.linger_stub import StubKeptLinger
from tests.support import TempHome
from tg_bridge.agents.claude.history import encoded_cwd
from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.bot.callbacks.keep_taps import KeepTaps
from tg_bridge.bot.callbacks.picker_taps import MODE_SET_TEXT, PickerTaps
from tg_bridge.bot.callbacks.router import Tap
from tg_bridge.bot.commands.base import CommandDeps, CommandRegistry
from tg_bridge.bot.commands.session import SessionCommands
from tg_bridge.bot.commands.settings import SettingsCommands
from tg_bridge.bot.commands.session import BOUND_ELSEWHERE_TEXT
from tg_bridge.bot.inbound import AUTO_OFF_TEXT, AUTO_ON_TEXT, MessageHandler
from tg_bridge.bot.keep_guard import (
    KEEP_CARD_TEXT,
    KEEP_NOT_APPLIED_TEXT,
    KEEP_CHANGED_LATER_TEXT,
    KEEP_CHANGED_TEXT,
    KEEP_EXPIRED_TEXT,
    KEEP_KEPT_TEXT,
    KEEP_NO_KIND,
    KEEP_TOAST_TEXT,
    KEEP_YES_KIND,
    SAME_SESSION_TEXT,
    WHAT_FOLDER,
    WHAT_MODE,
    ApplyResult,
    KeepGuard,
    applies,
)
from tg_bridge.config import Config, tilde
from tg_bridge.core.cards import ApprovalCards
from tg_bridge.core.cwd import SEARCH_KIND, CwdService, cwd_reply
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.launcher_switch import SAME_TEXT, LauncherSwitch
from tg_bridge.core.linger import LingerStatus
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.permissions import PermissionService
from tg_bridge.core.topic_backend import topic_backend
from tg_bridge.core.topic_queues import TopicQueues
from tg_bridge.core.workers import WorkerPool
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.asks import AskRepo
from tg_bridge.store.db import Database
from tg_bridge.store.inbox import CwdRepo
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.rules import RuleRepo
from tg_bridge.store.sessions import HistoryRepo, SessionRepo
from tg_bridge.store.turns import TurnRepo
from tg_bridge.telegram.keyboards import PICKER_PREFIX
from tg_bridge.telegram.messenger import Messenger

MID = 555
SID = "1f2e3d4c-aaaa-4bbb-8ccc-000000000091"
MODELS = ["haiku", "opus"]
PLAN_INDEX = 3  # the claude backend lists its modes in one fixed order; the card test picks one that is not the default


class StubLive:
    """LiveRegistry.running() only: a live turn for the topic when running is set."""

    def __init__(self, running: object | None = None) -> None:
        self.value = running

    def running(self, thread: int) -> object | None:
        return self.value


class FakeReplies:
    """The Telegram side of a tap, recorded: the toast, the edits (text and keyboard) and the new cards of the tap."""

    def __init__(self) -> None:
        self.toasts: list[str] = []
        self.edits: list[str] = []
        self.edit_rows: list[list[list[Any]]] = []
        self.cards: list[str] = []

    def answer(self, callback_id: str, text: str) -> None:
        self.toasts.append(text)

    def edit(self, chat_id: Any, message_id: int, text: str, rows: list[list[Any]]) -> None:
        self.edits.append(text)
        self.edit_rows.append(rows)

    def send_buttons(self, chat_id: Any, thread: int, text: str, rows: list[list[Any]]) -> None:
        self.cards.append(text)


def make_tap(replies: FakeReplies, args: tuple[str, ...], prefix: str = PICKER_PREFIX) -> Tap:
    return Tap(callback_id="cb", prefix=prefix, args=args, chat_id=CHAT, message_id=MID, message_text="",
               replies=replies)  # type: ignore[arg-type]


def card_tokens(callbacks: list[str], kind: str) -> list[str]:
    """The tokens of the buttons of one kind (cm:<thread>:<kind>:<token>)."""
    return [c.split(":")[3] for c in callbacks if c.split(":")[2] == kind]


class KeepGuardUnitTests(unittest.TestCase):
    """The guard alone: when it holds a change, what its buttons do, and that each card answers once."""

    def setUp(self) -> None:
        self.linger = StubKeptLinger(status=LingerStatus(pending=2, seconds_left=60.0, self_turn=False))
        self.live = StubLive()
        self.guard = KeepGuard(linger=self.linger, live=self.live)  # type: ignore[arg-type]
        self.applied: list[str] = []

    def hold(self, thread: int = THREAD, name: str = "one") -> Any:
        return self.guard.hold(thread, WHAT_MODE, applies(lambda: self.applied.append(name)))

    def test_nothing_kept_applies_at_once(self) -> None:
        self.linger.status_value = None
        self.assertIsNone(self.hold())
        self.assertEqual(self.applied, [])

    def test_the_quiet_window_applies_at_once(self) -> None:
        self.linger.status_value = LingerStatus(pending=0, seconds_left=60.0, self_turn=False)
        self.assertIsNone(self.hold())

    def test_a_user_turn_on_the_kept_process_applies_at_once(self) -> None:
        self.live.value = object()
        self.assertIsNone(self.hold())

    def test_idle_background_work_holds_the_change_behind_a_card(self) -> None:
        card = self.hold()
        self.assertIsNotNone(card)
        self.assertEqual(card.text, KEEP_CARD_TEXT.format(pending=2, what=WHAT_MODE))
        self.assertEqual(self.applied, [], "the change must wait for the card")
        callbacks = self.callbacks(card)
        self.assertTrue(callbacks[0].startswith(f"{PICKER_PREFIX}:{THREAD}:"))
        self.assertEqual(len(card_tokens(callbacks, KEEP_YES_KIND)), 1)
        self.assertEqual(len(card_tokens(callbacks, KEEP_NO_KIND)), 1)

    def test_a_self_turn_holds_the_change_too(self) -> None:
        self.live.value = object()
        self.linger.status_value = LingerStatus(pending=1, seconds_left=60.0, self_turn=True)
        self.assertIsNotNone(self.hold())

    def test_change_and_close_applies_once_and_asks_the_session_to_close(self) -> None:
        card = self.hold()
        token = card_tokens(self.callbacks(card), KEEP_YES_KIND)[0]
        self.linger.release_answer = True
        self.assertEqual(self.guard.decide(THREAD, token, yes=True), KEEP_CHANGED_TEXT)
        self.assertEqual(self.applied, ["one"])
        self.assertEqual(self.linger.requests, [("release", THREAD)])

    def test_change_and_close_during_a_self_turn_says_it_closes_later(self) -> None:
        self.live.value = object()
        self.linger.status_value = LingerStatus(pending=1, seconds_left=60.0, self_turn=True)
        token = card_tokens(self.callbacks(self.hold()), KEEP_YES_KIND)[0]
        self.linger.release_answer = False
        self.assertEqual(self.guard.decide(THREAD, token, yes=True), KEEP_CHANGED_LATER_TEXT)
        self.assertEqual(self.applied, ["one"])

    def test_keep_the_session_drops_the_change(self) -> None:
        token = card_tokens(self.callbacks(self.hold()), KEEP_NO_KIND)[0]
        self.assertEqual(self.guard.decide(THREAD, token, yes=False), KEEP_KEPT_TEXT)
        self.assertEqual(self.applied, [])
        self.assertEqual(self.linger.requests, [], "keeping the session asks for no release")

    def test_a_card_answers_once(self) -> None:
        token = card_tokens(self.callbacks(self.hold()), KEEP_YES_KIND)[0]
        self.guard.decide(THREAD, token, yes=True)
        self.assertEqual(self.guard.decide(THREAD, token, yes=True), KEEP_EXPIRED_TEXT)
        self.assertEqual(self.applied, ["one"], "a used card applies nothing again")

    def test_a_new_card_for_the_topic_replaces_the_earlier_one(self) -> None:
        first = card_tokens(self.callbacks(self.hold(name="first")), KEEP_YES_KIND)[0]
        second = card_tokens(self.callbacks(self.hold(name="second")), KEEP_YES_KIND)[0]
        self.assertEqual(self.guard.decide(THREAD, first, yes=True), KEEP_EXPIRED_TEXT)
        self.guard.decide(THREAD, second, yes=True)
        self.assertEqual(self.applied, ["second"])

    def test_a_token_answered_for_another_topic_is_expired_and_stays_usable(self) -> None:
        token = card_tokens(self.callbacks(self.hold()), KEEP_YES_KIND)[0]
        self.assertEqual(self.guard.decide(THREAD + 1, token, yes=True), KEEP_EXPIRED_TEXT)
        self.assertEqual(self.applied, [])
        self.guard.decide(THREAD, token, yes=True)
        self.assertEqual(self.applied, ["one"], "the card still answers for its own topic")

    def test_an_unknown_token_is_expired(self) -> None:
        self.assertEqual(self.guard.decide(THREAD, "deadbeef", yes=True), KEEP_EXPIRED_TEXT)
        self.assertEqual(self.applied, [])

    def test_a_yes_after_the_session_closed_on_its_own_applies_nothing(self) -> None:
        token = card_tokens(self.callbacks(self.hold()), KEEP_YES_KIND)[0]
        self.linger.status_value = None  # the kept session closed by itself while the card was open
        self.assertEqual(self.guard.decide(THREAD, token, yes=True), KEEP_EXPIRED_TEXT)
        self.assertEqual(self.applied, [])
        self.assertEqual(self.linger.requests, [])

    def test_a_card_left_after_its_kept_session_closed_applies_nothing(self) -> None:
        token = card_tokens(self.callbacks(self.hold()), KEEP_YES_KIND)[0]
        self.linger.status_value = None  # the kept session ended
        self.assertIsNone(self.hold(name="direct"), "with nothing kept the change applies at once, no card")
        self.assertEqual(self.guard.decide(THREAD, token, yes=True), KEEP_EXPIRED_TEXT)
        self.assertEqual(self.applied, [], "the old card must not apply its change again")

    def test_a_card_of_an_earlier_hold_applies_nothing_in_a_later_hold(self) -> None:
        token = card_tokens(self.callbacks(self.hold(name="first")), KEEP_YES_KIND)[0]
        self.linger.status_value = LingerStatus(pending=1, seconds_left=60.0, self_turn=False, hold_id=2)
        self.assertEqual(self.guard.decide(THREAD, token, yes=True), KEEP_EXPIRED_TEXT)
        self.assertEqual(self.applied, [])
        self.assertEqual(self.linger.requests, [], "the later hold is not released by the earlier card")

    def test_a_card_of_the_current_hold_still_applies(self) -> None:
        self.linger.status_value = LingerStatus(pending=1, seconds_left=60.0, self_turn=False, hold_id=2)
        token = card_tokens(self.callbacks(self.hold(name="later")), KEEP_YES_KIND)[0]
        self.linger.release_answer = True
        self.assertEqual(self.guard.decide(THREAD, token, yes=True), KEEP_CHANGED_TEXT)
        self.assertEqual(self.applied, ["later"])

    def test_the_cards_of_a_topic_whose_hold_ended_are_dropped_on_access(self) -> None:
        self.hold(name="gone")
        self.linger.status_value = None
        self.assertIsNone(self.guard.hold(THREAD, WHAT_MODE, applies(lambda: None)))
        self.assertEqual(self.guard._pending, {}, "the ended hold's card is dropped when its topic is looked at")

    def test_a_refused_change_releases_nothing_and_shows_its_refusal(self) -> None:
        card = self.guard.hold(THREAD, WHAT_MODE, lambda: ApplyResult(False, "refused here"))
        token = card_tokens(self.callbacks(card), KEEP_YES_KIND)[0]
        self.assertEqual(self.guard.decide(THREAD, token, yes=True), "refused here")
        self.assertEqual(self.linger.requests, [], "a refused change closes nothing")

    def test_a_refusal_without_text_says_nothing_changed(self) -> None:
        card = self.guard.hold(THREAD, WHAT_MODE, lambda: ApplyResult(False))
        token = card_tokens(self.callbacks(card), KEEP_YES_KIND)[0]
        self.assertEqual(self.guard.decide(THREAD, token, yes=True), KEEP_NOT_APPLIED_TEXT)
        self.assertEqual(self.linger.requests, [])

    def test_a_change_whose_kept_session_closes_while_it_applies_releases_nothing(self) -> None:
        def apply_while_closing() -> ApplyResult:
            self.linger.status_value = None  # the kept session closes while the change applies
            self.applied.append("closing")
            return ApplyResult(True)

        token = card_tokens(self.callbacks(self.guard.hold(THREAD, WHAT_MODE, apply_while_closing)), KEEP_YES_KIND)[0]
        self.assertEqual(self.guard.decide(THREAD, token, yes=True), KEEP_CHANGED_TEXT)
        self.assertEqual(self.applied, ["closing"])
        self.assertEqual(self.linger.requests, [], "nothing is left to close")

    def test_the_text_an_apply_returns_is_added_under_the_result(self) -> None:
        card = self.guard.hold(THREAD, WHAT_MODE, lambda: ApplyResult(True, "launcher reply"))
        token = card_tokens(self.callbacks(card), KEEP_YES_KIND)[0]
        self.linger.release_answer = True
        self.assertEqual(self.guard.decide(THREAD, token, yes=True), f"{KEEP_CHANGED_TEXT}\nlauncher reply")

    def test_a_same_value_applies_at_once_with_no_card(self) -> None:
        applied: list[str] = []
        self.assertIsNone(self.guard.hold(THREAD, WHAT_MODE, applies(lambda: applied.append("same")), same=True))
        self.assertEqual(applied, [], "the caller applies it, the guard holds nothing")

    @staticmethod
    def callbacks(card: Any) -> list[str]:
        return [button[1].encode() for row in card.rows for button in row]


class KeepCase(unittest.TestCase):
    """The settings, session and picker commands and the confirm taps wired to real repositories under a TempHome."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.base = os.path.realpath(self.home.home)
        self.project = os.path.join(self.base, "project")
        self.other = os.path.join(self.base, "other")
        os.makedirs(self.project, exist_ok=True)
        os.makedirs(self.other, exist_ok=True)
        self.clock = FakeClock()
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.config = Config(config_data(self.project, self.base), home=self.base, env={})
        self.backends = build_default_registry(self.config, self.logger, self.clock, title_launcher="")
        self.db = Database(Paths.from_home(self.base))
        self.addCleanup(self.db.close)
        self.sessions = SessionRepo(self.db, self.clock)
        self.history = HistoryRepo(self.db, self.clock)
        self.kv = KvRepo(self.db)
        self.live = LiveRegistry()
        self.telegram = FakeTelegram(self.clock)
        self.messenger = Messenger(self.telegram, self.logger, self.clock)
        self.out = WorkerPool("out", 1, self.logger)
        self.search = WorkerPool("search", 1, self.logger)
        self.permissions = PermissionService(
            asks=AskRepo(self.db, self.clock), rules=RuleRepo(self.db, self.clock),
            cards=ApprovalCards(self.messenger, self.logger, self.clock), messenger=self.messenger,
            workers=self.out, waiting=WaitRecorder(), logger=self.logger)
        handoff = HandoffService(self.kv, self.backends, self.logger)
        self.cwd = CwdService(sessions=self.sessions, kv=self.kv, paths=CwdRepo(self.db), messenger=self.messenger,
                              live=self.live, search_pool=self.search, out_pool=self.out, logger=self.logger,
                              clock=self.clock, handoff=handoff, finder=lambda term, roots, deadline: [])
        self.launchers = LauncherSwitch(self.config, self.sessions, self.history, handoff, self.backends)
        self.linger = StubKeptLinger(release=True)  # an idle kept session closes when asked
        self.keep = KeepGuard(linger=self.linger, live=self.live)  # type: ignore[arg-type]
        self.stops = SessionCommands(config=self.config, sessions=self.sessions, history=self.history,
                                     live=self.live, backends=self.backends, messenger=self.messenger,
                                     clock=self.clock, logger=self.logger, handoff=handoff,
                                     linger=self.linger, keep=self.keep)  # type: ignore[arg-type]
        self.settings = SettingsCommands(config=self.config, sessions=self.sessions, live=self.live,
                                         backends=self.backends, permissions=self.permissions,
                                         launchers=self.launchers, cwd=self.cwd, logger=self.logger, keep=self.keep)
        self.registry = CommandRegistry()
        self.settings.register(self.registry)
        self.stops.register(self.registry)
        self.deps = CommandDeps(
            config=self.config, sessions=self.sessions, turns=TurnRepo(self.db), rules=RuleRepo(self.db, self.clock),
            kv=self.kv, live=self.live, queues=TopicQueues(), backends=self.backends, messenger=self.messenger,
            out=self.out, linger=self.linger)  # type: ignore[arg-type]
        self.pickers = PickerTaps(config=self.config, sessions=self.sessions, live=self.live, backends=self.backends,
                                  permissions=self.permissions, launchers=self.launchers, stops=self.stops,
                                  rules=RuleRepo(self.db, self.clock), folders=self.cwd, logger=self.logger,
                                  keep=self.keep)
        self.confirm = KeepTaps(keep=self.keep)
        self.sessions.ensure(THREAD, TOPIC, CLAUDE_CMD, self.project, "default")
        self.replies = FakeReplies()

    def run_cmd(self, text: str) -> bool:
        handled = self.registry.dispatch(self.deps, CHAT, THREAD, TOPIC, text)
        self.assertTrue(self.out.join(5), "the replies did not go out")
        return handled

    def row(self) -> Any:
        return self.sessions.get(THREAD)

    def card_callbacks(self) -> list[str]:
        return [b["callback_data"] for keyboard in self.telegram.buttons(THREAD) for row in keyboard for b in row]

    def card_token(self, kind: str) -> str:
        return card_tokens(self.card_callbacks(), kind)[-1]

    def hold_with(self, pending: int) -> None:
        self.linger.status_value = LingerStatus(pending=pending, seconds_left=60.0, self_turn=False)

    def assert_card(self) -> None:
        self.assertIn("background task(s)", self.telegram.texts(THREAD)[-1])
        self.assertTrue(card_tokens(self.card_callbacks(), KEEP_YES_KIND), "the card has no change-and-close button")
        self.assertTrue(card_tokens(self.card_callbacks(), KEEP_NO_KIND), "the card has no keep button")


class KeptTypedCommandTests(KeepCase):
    """Typed commands with a kept session that has background work pending."""

    def setUp(self) -> None:
        super().setUp()
        self.hold_with(1)

    def test_mode_waits_for_the_card_and_change_and_close_applies_it(self) -> None:
        self.assertTrue(self.run_cmd("/mode plan"))
        self.assert_card()
        self.assertEqual(self.row().mode, "default", "the mode must not change before the card")
        self.assertEqual(self.linger.requests, [])
        self.tap_decide(KEEP_YES_KIND)
        self.assertEqual(self.row().mode, "plan")
        self.assertEqual(self.linger.requests, [("release", THREAD)])
        self.assertEqual(self.replies.edits, [KEEP_CHANGED_TEXT])
        self.assertEqual(self.replies.toasts, [KEEP_CHANGED_TEXT])

    def test_mode_keep_the_session_leaves_the_mode(self) -> None:
        self.run_cmd("/mode plan")
        self.tap_decide(KEEP_NO_KIND)
        self.assertEqual(self.row().mode, "default")
        self.assertEqual(self.linger.requests, [])
        self.assertEqual(self.replies.edits, [KEEP_KEPT_TEXT], "the tapped card is edited to the result, no buttons")

    def test_model_waits_for_the_card_then_applies_on_yes(self) -> None:
        self.run_cmd("/model opus")
        self.assert_card()
        self.assertEqual(self.row().model, "")
        self.tap_decide(KEEP_YES_KIND)
        self.assertEqual(self.row().model, "opus")

    def test_launcher_waits_for_the_card_and_keeps_the_launcher_on_no(self) -> None:
        before = self.row().launcher
        self.run_cmd("/launcher plain")
        self.assert_card()
        self.assertEqual(self.row().launcher, before)
        self.tap_decide(KEEP_NO_KIND)
        self.assertEqual(self.row().launcher, before)

    def test_cwd_waits_for_the_card_and_moves_the_folder_on_yes(self) -> None:
        self.run_cmd(f"/cwd {self.other}")
        self.assert_card()
        self.assertEqual(os.path.realpath(self.row().cwd), os.path.realpath(self.project))
        self.tap_decide(KEEP_YES_KIND)
        self.assertEqual(os.path.realpath(self.row().cwd), os.path.realpath(self.other))

    def test_a_refused_folder_change_says_nothing_changed_and_releases_nothing(self) -> None:
        self.cwd.apply = lambda thread, path: False  # the folder apply refuses the change
        self.run_cmd(f"/cwd {self.other}")
        self.assert_card()
        self.tap_decide(KEEP_YES_KIND)
        self.assertEqual(self.replies.edits, [KEEP_NOT_APPLIED_TEXT])
        self.assertEqual(self.linger.requests, [], "a refused folder change closes nothing")
        self.assertEqual(os.path.realpath(self.row().cwd), os.path.realpath(self.project))

    def test_new_waits_for_the_card_and_forgets_the_session_on_yes(self) -> None:
        self.sessions.set_uuid(THREAD, SID)
        self.run_cmd("/new")
        self.assert_card()
        self.assertEqual(self.row().uuid, SID)
        self.tap_decide(KEEP_YES_KIND)
        self.assertNotEqual(self.row().uuid, SID)

    def test_resume_waits_for_the_card_and_keeps_the_session_on_no(self) -> None:
        self.sessions.set_uuid(THREAD, SID)
        self.run_cmd("/resume 1f2e3d4c-aaaa-4bbb-8ccc-000000000092")
        self.assert_card()
        self.tap_decide(KEEP_NO_KIND)
        self.assertEqual(self.row().uuid, SID)

    def test_a_change_and_close_that_cannot_close_now_says_it_closes_later(self) -> None:
        self.run_cmd("/mode plan")
        token = self.card_token(KEEP_YES_KIND)
        self.linger.release_answer = False
        self.assertEqual(self.keep.decide(THREAD, token, yes=True), KEEP_CHANGED_LATER_TEXT)
        self.assertEqual(self.row().mode, "plan")

    def tap_decide(self, kind: str) -> None:
        tap = make_tap(self.replies, (str(THREAD), kind, self.card_token(kind)))
        if kind == KEEP_YES_KIND:
            self.confirm._yes(tap)
        else:
            self.confirm._no(tap)


class KeptStaleCardTests(KeepCase):
    """A confirm card that outlives its kept session or its hold: it expires and applies nothing (section 5 D1)."""

    def setUp(self) -> None:
        super().setUp()
        self.hold_with(1)

    def test_a_card_left_after_the_session_closed_does_not_apply_its_change(self) -> None:
        self.run_cmd("/mode plan")
        self.assert_card()
        token = self.card_token(KEEP_YES_KIND)
        self.linger.status_value = None  # the kept session ended
        self.run_cmd("/model opus")  # set directly: nothing is kept, so it applies at once
        self.assertEqual(self.row().model, "opus")
        self.assertEqual(self.keep.decide(THREAD, token, yes=True), KEEP_EXPIRED_TEXT)
        self.assertEqual(self.row().mode, "default", "the old card's mode must not apply")
        self.assertEqual(self.linger.requests, [])

    def test_a_card_from_an_earlier_hold_does_not_release_a_later_one(self) -> None:
        self.run_cmd("/mode plan")
        token = self.card_token(KEEP_YES_KIND)
        self.linger.status_value = LingerStatus(pending=1, seconds_left=60.0, self_turn=False, hold_id=2)
        self.assertEqual(self.keep.decide(THREAD, token, yes=True), KEEP_EXPIRED_TEXT)
        self.assertEqual(self.row().mode, "default")
        self.assertEqual(self.linger.requests, [], "the later kept session is not released by the old card")

    def test_a_resume_refused_for_a_session_bound_elsewhere_says_why_and_releases_nothing(self) -> None:
        self.sessions.ensure(THREAD + 1, "other topic", CLAUDE_CMD, self.other, "default")
        self.sessions.set_uuid(THREAD + 1, SID)
        self.run_cmd(f"/resume {SID}")
        self.assert_card()
        self.assertEqual(self.keep.decide(THREAD, self.card_token(KEEP_YES_KIND), yes=True),
                         BOUND_ELSEWHERE_TEXT.format(sid=SID))
        self.assertEqual(self.row().uuid, "", "the refused resume binds nothing")
        self.assertEqual(self.linger.requests, [], "a resume that did not happen closes nothing")

    def test_picking_the_current_launcher_again_releases_nothing(self) -> None:
        self.run_cmd("/launcher wrap")
        self.assert_card()
        self.assertEqual(self.keep.decide(THREAD, self.card_token(KEEP_YES_KIND), yes=True),
                         SAME_TEXT.format(name="wrap"))
        self.assertEqual(self.linger.requests, [], "the launcher did not change, so no session closes")


class KeptCurrentResumeTests(KeepCase):
    """A /resume of the session the topic already runs, with background work pending in its kept session: the change is
    unchanged, so no card is asked and nothing is released. A resume of another session still asks for the card."""

    def setUp(self) -> None:
        super().setUp()
        self.hold_with(1)
        self.sessions.set_uuid(THREAD, SID)

    def test_resume_of_the_current_session_asks_nothing_and_releases_nothing(self) -> None:
        self.run_cmd(f"/resume {SID}")
        self.assertEqual(self.card_callbacks(), [], "no confirm card for the session in use")
        self.assertEqual(self.linger.requests, [], "the kept session is not released")
        self.assertEqual(self.row().uuid, SID)
        self.assertEqual(self.telegram.texts(THREAD)[-1], SAME_SESSION_TEXT.format(sid=SID))
        self.assertIn("nothing changed", self.telegram.texts(THREAD)[-1])

    def test_resume_of_another_session_still_asks_for_the_card(self) -> None:
        self.run_cmd("/resume 1f2e3d4c-aaaa-4bbb-8ccc-000000000092")
        self.assert_card()
        self.assertEqual(self.row().uuid, SID, "the other session is not bound before the card")
        self.assertEqual(self.linger.requests, [])

    def test_resume_of_the_current_session_with_its_transcript_elsewhere_asks_for_the_card(self) -> None:
        folder = os.path.join(self.project, "cfg-b", "projects", encoded_cwd(os.path.realpath(self.project)))
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, SID + ".jsonl"), "w", encoding="utf-8") as out:
            out.write(json.dumps({"type": "ai-title", "aiTitle": "earlier question"}) + "\n")
        self.run_cmd(f"/resume {SID}")
        self.assert_card()
        self.assertEqual(self.linger.requests, [], "the card comes first: nothing is released yet")

    def test_resume_of_the_current_session_from_another_folder_still_asks_for_the_card(self) -> None:
        self.history.note_session(THREAD, SID, "earlier question", self.other, CLAUDE_CMD, "claude")
        self.run_cmd(f"/resume {SID}")
        self.assert_card()
        self.assertEqual(self.linger.requests, [])

    def test_the_picker_tap_of_the_current_session_asks_nothing_and_releases_nothing(self) -> None:
        self.history.note_session(THREAD, SID, "earlier question", self.project, CLAUDE_CMD, "claude")
        record = self.history.find(THREAD, SID)
        self.pickers._resume(make_tap(self.replies, (str(THREAD), "rs", str(record.id))))
        self.assertEqual(self.replies.toasts, [SAME_SESSION_TEXT.format(sid=SID)])
        self.assertEqual(self.replies.edits, [], "the picker is left as it is")
        self.assertEqual(self.card_callbacks(), [])
        self.assertEqual(self.linger.requests, [])


class KeptPickerTapTests(KeepCase):
    """The mode and launcher picker taps with a kept session that has background work pending."""

    def setUp(self) -> None:
        super().setUp()
        self.hold_with(1)

    def test_a_mode_tap_replaces_the_picker_with_the_card(self) -> None:
        modes = topic_backend(self.config, self.backends, self.row(), TOPIC).modes()
        tap = make_tap(self.replies, (str(THREAD), "md", str(PLAN_INDEX)))
        self.pickers._mode(tap)
        self.assertEqual(self.row().mode, "default")
        self.assertEqual(self.replies.edits, [KEEP_CARD_TEXT.format(pending=1, what=WHAT_MODE)])
        self.assertEqual(self.replies.toasts[-1], KEEP_TOAST_TEXT)
        self.assertNotEqual(modes[PLAN_INDEX].key, "default")

    def test_a_mode_tap_applies_at_once_when_nothing_is_kept(self) -> None:
        self.hold_with(0)
        tap = make_tap(self.replies, (str(THREAD), "md", str(PLAN_INDEX)))
        self.pickers._mode(tap)
        self.assertNotEqual(self.row().mode, "default")
        self.assertIn(MODE_SET_TEXT.format(mode=self.row().mode), self.replies.toasts)
        self.assertEqual(self.replies.cards, [])

    def test_a_resume_tap_replaces_the_picker_with_the_card(self) -> None:
        self.history.note_session(THREAD, SID, "earlier question", self.project, CLAUDE_CMD, "claude")
        record = self.history.find(THREAD, SID)
        self.pickers._resume(make_tap(self.replies, (str(THREAD), "rs", str(record.id))))
        self.assertEqual(self.row().uuid, "", "the resume must wait for the card")
        self.assertEqual(self.replies.toasts, [KEEP_TOAST_TEXT])
        self.assertIn("background task(s)", self.replies.edits[0])

    def test_a_launcher_tap_waits_for_the_card(self) -> None:
        before = self.row().launcher
        tap = make_tap(self.replies, (str(THREAD), "la", "1"))
        self.pickers._launcher(tap)
        self.assertEqual(self.row().launcher, before)
        self.assertEqual(len(self.replies.edits), 1)
        self.assertIn("background task(s)", self.replies.edits[0])


class QuietTypedCommandTests(KeepCase):
    """/mode, /model, /cwd and /new apply at once, with no card, when nothing is kept or the kept session is quiet."""

    def cases(self) -> list[tuple[str, Callable[[], bool]]]:
        other = os.path.realpath(self.other)
        return [
            ("/mode plan", lambda: self.row().mode == "plan"),
            ("/model opus", lambda: self.row().model == "opus"),
            (f"/cwd {self.other}", lambda: os.path.realpath(self.row().cwd) == other),
            ("/new", lambda: self.row().uuid != SID),
        ]

    def test_nothing_kept_and_the_quiet_window_apply_every_typed_change_at_once(self) -> None:
        quiet = LingerStatus(pending=0, seconds_left=60.0, self_turn=False)
        for label, status in (("nothing kept", None), ("quiet window", quiet)):
            for command, applied in self.cases():
                with self.subTest(state=label, command=command):
                    self.linger.status_value = status
                    self.sessions.set_uuid(THREAD, SID)
                    self.run_cmd(command)
                    self.assertTrue(applied(), "the change must apply at once")
                    self.assertEqual(self.card_callbacks(), [], "no confirm card for a quiet or unkept session")
                    self.assertEqual(self.linger.requests, [])


class KeptNoOpTypedTests(KeepCase):
    """A change that stores what the topic already has would close nothing, so it applies with no card even while
    background work is pending."""

    def setUp(self) -> None:
        super().setUp()
        self.hold_with(1)

    def test_the_mode_model_and_folder_already_set_apply_at_once(self) -> None:
        for command in ("/mode default", "/model default", f"/cwd {self.project}"):
            with self.subTest(command=command):
                self.run_cmd(command)
                self.assertEqual(self.card_callbacks(), [], "a no-op change must not post a card")
        self.assertEqual(self.row().mode, "default")
        self.assertEqual(self.row().model, "")
        self.assertEqual(os.path.realpath(self.row().cwd), os.path.realpath(self.project))
        self.assertEqual(self.linger.requests, [])

    def test_a_launcher_reply_is_shown_under_the_result_on_yes(self) -> None:
        self.run_cmd("/launcher plain")
        text = self.keep.decide(THREAD, self.card_token(KEEP_YES_KIND), yes=True)
        self.assertEqual(self.row().launcher_pick, "plain")
        self.assertTrue(text.startswith(KEEP_CHANGED_TEXT + "\n"), text)
        self.assertTrue(text.split("\n", 1)[1].strip(), "the launcher's reply must not be empty")


class KeepTapExpiryTests(KeepCase):
    """A confirm tap with an id that no card holds: a toast and the card edited to the expired text, nothing applied."""

    def test_an_unknown_token_toasts_and_edits_to_expired_for_both_buttons(self) -> None:
        self.hold_with(1)
        for kind in (KEEP_YES_KIND, KEEP_NO_KIND):
            with self.subTest(kind=kind):
                self.replies.toasts.clear()
                self.replies.edits.clear()
                tap = make_tap(self.replies, (str(THREAD), kind, "deadbeef"))
                if kind == KEEP_YES_KIND:
                    self.confirm._yes(tap)
                else:
                    self.confirm._no(tap)
                self.assertEqual(self.replies.toasts, [KEEP_EXPIRED_TEXT])
                self.assertEqual(self.replies.edits, [KEEP_EXPIRED_TEXT])
        self.assertEqual(self.row().mode, "default")
        self.assertEqual(self.linger.requests, [])


class KeptFolderSearchTests(KeepCase):
    """A search-card tap on a folder with a kept session that has background work pending (section 5 D1)."""

    def setUp(self) -> None:
        super().setUp()
        self.hold_with(1)
        self.cwd.remember_found(THREAD, MID, [self.other])

    def search_tap(self) -> None:
        self.pickers._folder_search(make_tap(self.replies, (str(THREAD), SEARCH_KIND, "0")))

    def edited_token(self, kind: str) -> str:
        callbacks = [button[1].encode() for rows in self.replies.edit_rows for row in rows for button in row]
        return card_tokens(callbacks, kind)[-1]

    def test_the_tap_posts_the_card_and_leaves_the_folder_alone(self) -> None:
        self.search_tap()
        self.assertEqual(os.path.realpath(self.row().cwd), os.path.realpath(self.project))
        self.assertEqual(self.replies.toasts, [KEEP_TOAST_TEXT])
        self.assertEqual(self.replies.edits, [KEEP_CARD_TEXT.format(pending=1, what=WHAT_FOLDER)])
        self.assertEqual(self.linger.requests, [])

    def test_change_and_close_moves_the_folder_and_asks_the_session_to_close(self) -> None:
        self.search_tap()
        token = self.edited_token(KEEP_YES_KIND)
        text = self.keep.decide(THREAD, token, yes=True)
        self.assertTrue(text.startswith(KEEP_CHANGED_TEXT + "\n"), text)  # the folder's own reply is shown under it
        self.assertEqual(os.path.realpath(self.row().cwd), os.path.realpath(self.other))
        self.assertEqual(self.linger.requests, [("release", THREAD)])

    def test_a_refused_folder_change_from_the_card_says_nothing_changed(self) -> None:
        self.cwd.apply = lambda thread, path: False  # the folder apply refuses the change
        self.search_tap()
        text = self.keep.decide(THREAD, self.edited_token(KEEP_YES_KIND), yes=True)
        self.assertEqual(text, cwd_reply(tilde(self.other, self.config.home), False),
                         "the search button's own refusal text is shown, not the change text")
        self.assertNotIn(KEEP_CHANGED_TEXT, text)
        self.assertEqual(self.linger.requests, [], "a refused folder change closes nothing")

    def test_keep_the_session_leaves_the_folder(self) -> None:
        self.search_tap()
        self.assertEqual(self.keep.decide(THREAD, self.edited_token(KEEP_NO_KIND), yes=False), KEEP_KEPT_TEXT)
        self.assertEqual(os.path.realpath(self.row().cwd), os.path.realpath(self.project))
        self.assertEqual(self.linger.requests, [])

    def test_the_quiet_window_moves_the_folder_at_once_with_no_card(self) -> None:
        self.hold_with(0)
        self.search_tap()
        self.assertEqual(os.path.realpath(self.row().cwd), os.path.realpath(self.other))
        self.assertFalse(any(self.replies.edit_rows), "the card's edit must carry no buttons")
        self.assertNotIn(KEEP_TOAST_TEXT, self.replies.toasts)


class KeptAutoTests(InboundCase):
    """'auto on' and 'auto off' with a kept session that has background work pending (the inbound handler)."""

    def setUp(self) -> None:
        super().setUp()
        self.linger = StubKeptLinger(status=LingerStatus(pending=1, seconds_left=60.0, self_turn=False), release=True)
        self.keep = KeepGuard(linger=self.linger, live=self.live)  # type: ignore[arg-type]
        self.handler = MessageHandler(
            config=self.config, topics=self.topics, sessions=self.sessions, kv=self.kv, inbox=self.inbox,
            dispatcher=self.dispatcher, permissions=self.permissions, live=self.live, cwd=self.folders,
            commands=self.commands, command_deps=self.deps, backends=self.backends, messenger=self.messenger,
            out=self.out, media=self.media, media_pool=self.media_pool, seats=self.seats, logger=self.logger,
            keep=self.keep)
        self.sessions.ensure(THREAD, TOPIC, CLAUDE_CMD, self.project, "default")

    def test_auto_on_waits_for_the_card(self) -> None:
        self.handle(self.message("auto on"))
        self.assertEqual(self.row().mode, "default")
        self.assertNotIn(AUTO_ON_TEXT, self.sent())
        self.assertTrue(any("background task(s)" in text for text in self.sent()))
        self.assertEqual(self.permissions.live_modes, [])

    def test_auto_off_yes_applies_the_mode_and_closes_the_session(self) -> None:
        self.sessions.set_mode(THREAD, "bypassPermissions")
        self.handle(self.message("auto off"))
        self.assertEqual(self.row().mode, "bypassPermissions")
        token = card_tokens(self.card_callbacks(), KEEP_YES_KIND)[-1]
        self.assertEqual(self.keep.decide(THREAD, token, yes=True), KEEP_CHANGED_TEXT)
        self.assertEqual(self.row().mode, "default")
        self.assertEqual(self.linger.requests, [("release", THREAD)])

    def test_auto_with_nothing_kept_applies_at_once(self) -> None:
        self.linger.status_value = None
        self.handle(self.message("auto on"))
        self.assertEqual(self.row().mode, "bypassPermissions")
        self.assertEqual(self.sent(), [AUTO_ON_TEXT])
        self.assertNotIn(AUTO_OFF_TEXT, self.sent())

    def card_callbacks(self) -> list[str]:
        return [b["callback_data"] for keyboard in self.telegram.buttons(THREAD) for row in keyboard for b in row]


class WaitRecorder:
    def waiting_changed(self, thread_id: int, waiting: bool) -> None:
        pass


def config_data(project: str, base: str) -> dict:
    """Two Claude launchers (wrap and plain) and one topic on wrap, the same shape the settings tests use."""
    return {
        "launchers": {
            "wrap": {"cmd": CLAUDE_CMD, "config_dir": os.path.join(project, "cfg-a")},
            "plain": {"cmd": "plain run", "config_dir": os.path.join(project, "cfg-b")},
        },
        "topics": {TOPIC: {"launcher": CLAUDE_CMD, "cwd": project}},
        "models": MODELS,
        "allow_user_ids": [ALLOWED_UID],
    }


if __name__ == "__main__":
    unittest.main()
