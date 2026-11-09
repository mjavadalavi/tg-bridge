"""ApprovalCards: the Telegram cards of permission asks and AskUserQuestion questions (the card half of C13).

Port of tg-bridge.py _card_markup, _question_markup, _question_body, _edit_card, _post_card, _nudge, _stop_timer,
answer_other, _record_answer, _drop_await, OUTCOME and the 10-minute nudge (ASK_NUDGE_SECONDS). The cards know nothing
about permissions: PermissionService (core/permissions.py) decides what a tap means, claims the ask in AskRepo, and
then calls this class to post, record, close or nudge the card. Card buttons come from ApprovalRequest.choices, so a
backend that offers fewer decisions gets a shorter card; a request that offers Deny as a cancel (deny_aborts_turn)
gets a warning line.

Callback data: pa/pA/pd/pt (permission buttons) and q/o (question options and 'Other') carry the nonce, checked by
CallbackData against the 64-byte limit. The Always button needs a rule text (always_rule); without one it is hidden.
A plan card (PLAN_KIND) shows the plan and pP:<nonce>:b / pP:<nonce>:e (run it), pc (change) and pd (reject); its
change waits for typed feedback like 'Other' does, under the question index PLAN_CHANGE_QI (core/plan_approval).

Lock map: self._lock guards _cards (nonce -> _Card) and _await_text (thread id -> (nonce, question index)). Telegram
calls and timer starts never run under it: each method copies what it needs, releases the lock, then acts.
A card is open until close(); a card closed while its sendMessage is still in flight is finished by post() once the
message id arrives, so the outcome is always painted.
"""

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from tg_bridge.agents.events import PLAN_KIND, ApprovalRequest, Decision, Question
from tg_bridge.clock import Clock
from tg_bridge.core.plan_approval import PLAN_CHANGE_QI
from tg_bridge.feed.format import short
from tg_bridge.logging_ import Logger
from tg_bridge.telegram.keyboards import EMPTY_KEYBOARD, CallbackData, keyboard_json
from tg_bridge.telegram.messenger import EDIT_TEXT_LIMIT, Messenger

ASK_NUDGE_SECONDS = 600.0
NUDGE_TEXT = "\n\n⏰ still waiting for your tap"
OTHER_WAIT_TEXT = "\n\n✏️ type your answer in this topic"
PLAN_HEAD = "📋 Plan — read it, then run it, ask for changes or reject it"
PLAN_ABOVE_TEXT = "📋 The plan is in the messages above — run it, ask for changes or reject it"
PLAN_CARD_CHARS = 3500  # a longer plan is sent as plain messages before its card, which then only holds the buttons
PLAN_CHUNK_CHARS = 3500  # one message of a long plan; every part is sent (Messenger.send_text would cut a long text)
PLAN_CHANGE_WAIT_TEXT = "\n\n✏️ type what to change in this topic — your next message goes to Claude"
DENY_WARNING = "\n⚠️ Deny stops this turn"
OUTCOME = MappingProxyType({
    "allow": "✅ allowed",
    "deny": "⛔ denied",
    "void": "⌛ void — turn ended",
    "withdrawn": "↩️ withdrawn",
    "approved": "✅ approved — the next turn implements it",
    "change": "✏️ sent back with your changes",
})
DETAIL_CHARS = 600
DETAIL_KEYS = ("command", "path", "file_path", "notebook_path")
ALWAYS_LABEL_CHARS = 30
OPTION_LABEL_CHARS = 40
ERROR_TEXT_LIMIT = 150


class NudgeTimer(Protocol):
    """A one-shot timer the nudge runs on. threading.Timer satisfies it; tests inject a recording fake."""

    def start(self) -> None: ...

    def cancel(self) -> None: ...


TimerFactory = Callable[[float, Callable[[], None]], NudgeTimer]


def threading_timer(seconds: float, fire: Callable[[], None]) -> NudgeTimer:
    """The production timer: a daemon threading.Timer that calls fire() once after seconds."""
    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    return timer


@dataclass(frozen=True)
class AnswerProgress:
    """What recording one question answer left behind: whether every question now has one, and all answers so far."""

    complete: bool
    answers: dict[str, str]


@dataclass
class _Card:
    """One posted (or being posted) card. Mutated only under ApprovalCards._lock."""

    nonce: str
    chat_id: int | str
    thread_id: int
    request: ApprovalRequest
    always_rule: str | None
    text: str
    posted_at: float
    posted_text: str = ""
    mid: int | None = None
    verdict: str | None = None
    timer: NudgeTimer | None = None
    answers: dict[str, str] = field(default_factory=dict)

    @property
    def questions(self) -> tuple[Question, ...]:
        return self.request.questions


@dataclass(frozen=True)
class _View:
    """Everything one Telegram edit needs, copied out of a card under the lock."""

    nonce: str
    chat_id: int | str
    mid: int
    text: str
    markup: str


def _detail(request: ApprovalRequest) -> str:
    """The line under the card title: the command or the path, else the input as JSON. Cut to DETAIL_CHARS."""
    for key in DETAIL_KEYS:
        value = request.input.get(key)
        if isinstance(value, str) and value:
            return value[:DETAIL_CHARS]
    return json.dumps(request.input, ensure_ascii=False)[:DETAIL_CHARS]


def question_body(questions: tuple[Question, ...], answers: dict[str, str]) -> str:
    """Text of a question card: every question, with its chosen answer once given."""
    parts = []
    for question in questions:
        text = question.text or "Question"
        part = f"❓ {text}"
        if question.text in answers:
            part += f"\n✔ {answers[question.text]}"
        parts.append(part)
    return "\n\n".join(parts)


def _plan_text(request: ApprovalRequest) -> str:
    plan = request.input.get("plan")
    return plan if isinstance(plan, str) else ""


def plan_chunks(text: str) -> list[str]:
    """The messages of a long plan, at most PLAN_CHUNK_CHARS each, split at a line end when one is near; none is lost."""
    chunks = []
    while len(text) > PLAN_CHUNK_CHARS:
        cut = text.rfind("\n", PLAN_CHUNK_CHARS // 2, PLAN_CHUNK_CHARS)
        cut = PLAN_CHUNK_CHARS if cut < 0 else cut + 1
        chunks.append(text[:cut])
        text = text[cut:]
    return [*chunks, text] if text else chunks


def _card_text(request: ApprovalRequest) -> str:
    if request.questions:
        return question_body(request.questions, {})
    if request.kind == PLAN_KIND:
        plan = _plan_text(request)
        return f"{PLAN_HEAD}\n\n{plan}" if len(plan) <= PLAN_CARD_CHARS else PLAN_ABOVE_TEXT
    text = f"🔐 {request.summary}\n{_detail(request)}"
    if request.deny_aborts_turn and Decision.DENY in request.choices:
        text += DENY_WARNING
    return text


def _permission_rows(card: _Card) -> list[list[tuple[str, CallbackData]]]:
    """Allow once and Always on the first row; Deny and Bypass this turn on the second. Missing choices are dropped."""
    nonce, choices = card.nonce, card.request.choices
    first: list[tuple[str, CallbackData]] = []
    second: list[tuple[str, CallbackData]] = []
    if Decision.ALLOW_ONCE in choices:
        first.append(("✅ Allow once", CallbackData("pa", (nonce,))))
    if Decision.ALLOW_ALWAYS in choices and card.always_rule:
        label = f"♾ Always: {short(card.always_rule, ALWAYS_LABEL_CHARS)}"
        first.append((label, CallbackData("pA", (nonce,))))
    if Decision.DENY in choices:
        second.append(("⛔ Deny", CallbackData("pd", (nonce,))))
    if Decision.ALLOW_TURN in choices:
        second.append(("🤖 Bypass this turn", CallbackData("pt", (nonce,))))
    return [row for row in (first, second) if row]


def _question_rows(card: _Card) -> list[list[tuple[str, CallbackData]]]:
    """Each unanswered question: one row per option, then 'Other'. Skip (a deny) closes the list when offered."""
    rows: list[list[tuple[str, CallbackData]]] = []
    many = len(card.questions) > 1
    for qi, question in enumerate(card.questions):
        if question.text in card.answers:
            continue
        for oi, label in enumerate(question.options):
            text = short(f"{qi + 1}. {label}" if many else label, OPTION_LABEL_CHARS)
            rows.append([(text, CallbackData("q", (card.nonce, qi, oi)))])
        other = f"{qi + 1}. ✏️ Other" if many else "✏️ Other"
        rows.append([(other, CallbackData("o", (card.nonce, qi)))])
    if Decision.DENY in card.request.choices:
        rows.append([("⛔ Skip", CallbackData("pd", (card.nonce,)))])
    return rows


def _plan_rows(card: _Card) -> list[list[tuple[str, CallbackData]]]:
    """Run it (bypass) and Run it asking before edits, one row each; Change and Reject on the last row."""
    nonce = card.nonce
    return [[("✅ Run it", CallbackData("pP", (nonce, "b")))],
            [("✅ Run it, ask before edits", CallbackData("pP", (nonce, "e")))],
            [("✏️ Change", CallbackData("pc", (nonce,))), ("⛔ Reject", CallbackData("pd", (nonce,)))]]


def _rows(card: _Card, with_buttons: bool) -> list[list[tuple[str, CallbackData]]]:
    if not with_buttons:
        return []
    if card.request.kind == PLAN_KIND:
        return _plan_rows(card)
    return _question_rows(card) if card.questions else _permission_rows(card)


class ApprovalCards:
    """Posts, edits, closes and nudges the Telegram cards of asks; holds the 'Other' wait per topic."""

    def __init__(
        self,
        messenger: Messenger,
        logger: Logger,
        clock: Clock,
        nudge_seconds: float = ASK_NUDGE_SECONDS,
        timer_factory: TimerFactory = threading_timer,
    ) -> None:
        self._messenger = messenger
        self._logger = logger
        self._clock = clock
        self._nudge_seconds = nudge_seconds
        self._timer_factory = timer_factory
        self._lock = threading.Lock()
        self._cards: dict[str, _Card] = {}
        self._await_text: dict[int, tuple[str, int]] = {}

    def post(
        self,
        nonce: str,
        request: ApprovalRequest,
        *,
        chat_id: int | str,
        thread_id: int,
        always_rule: str | None = None,
    ) -> bool:
        """Sends the card for an ask and arms its nudge. False when Telegram refused the send: the card is then
        forgotten, and the caller decides what the refused ask means (the permission service denies it)."""
        card = _Card(nonce=nonce, chat_id=chat_id, thread_id=thread_id, request=request, always_rule=always_rule,
                     text=_card_text(request), posted_at=self._clock.monotonic())
        rows = _rows(card, with_buttons=True)  # built before registering: an oversized callback raises here
        card.posted_text = card.text
        with self._lock:
            if nonce in self._cards:
                raise ValueError(f"card {nonce} is already posted")
            self._cards[nonce] = card
        try:
            if card.text == PLAN_ABOVE_TEXT:  # a long plan: its whole text goes first, in plain messages
                for chunk in plan_chunks(f"{PLAN_HEAD}\n\n{_plan_text(request)}"):
                    self._messenger.send_text(chat_id, thread_id, chunk)
            sent = self._messenger.send_buttons(chat_id, thread_id, card.text[:EDIT_TEXT_LIMIT], rows)
        except Exception:
            self._forget(card)
            raise
        if sent is None:
            self._forget(card)
            return False
        view, timer = None, None
        with self._lock:
            card.mid = int(sent["message_id"])
            if card.verdict is not None:  # closed while the send was in flight: paint the outcome now
                self._cards.pop(nonce, None)
                view = self._view(card, with_buttons=False, suffix=self._outcome_suffix(card.verdict))
            else:
                if card.text != card.posted_text:  # the question card changed while it was being sent
                    view = self._view(card, with_buttons=True)
                timer = self._timer_factory(self._nudge_seconds, lambda: self._nudge(nonce))
                card.timer = timer
        if view is not None:
            self._paint(view)
        if timer is not None:
            timer.start()
        return True

    def is_open(self, nonce: str) -> bool:
        """True while the card of this ask is posted and not yet closed."""
        with self._lock:
            return self._open_card(nonce) is not None

    def close(self, nonce: str, verdict: str) -> bool:
        """Ends the card with an outcome line and no buttons: 'allow', 'deny', 'void' or 'withdrawn'. False when the
        card is not open (already closed, or never posted). Also drops the 'Other' wait of this card."""
        with self._lock:
            card = self._open_card(nonce)
            if card is None:
                return False
            card.verdict = verdict
            self._drop_await(card)
            timer, card.timer = card.timer, None
            if card.mid is None:  # still being sent: post() paints the outcome when the message id arrives
                return True
            self._cards.pop(nonce, None)
            view = self._view(card, with_buttons=False, suffix=self._outcome_suffix(verdict))
        if timer is not None:
            timer.cancel()
        self._paint(view)
        return True

    def option_label(self, nonce: str, qi: int, oi: int) -> str | None:
        """The label of option oi of question qi on an open question card, or None when there is no such option."""
        with self._lock:
            card = self._open_card(nonce)
            if card is None or not 0 <= qi < len(card.questions):
                return None
            options = card.questions[qi].options
            return options[oi] if 0 <= oi < len(options) else None

    def open_other(self, nonce: str, qi: int) -> bool:
        """'Other' was tapped on question qi: the next text in the card's topic is that question's answer."""
        with self._lock:
            card = self._open_card(nonce)
            if card is None or not 0 <= qi < len(card.questions):
                return False
            self._await_text[card.thread_id] = (nonce, qi)
            card.text = question_body(card.questions, card.answers) + OTHER_WAIT_TEXT
            view = self._view(card, with_buttons=True)
        self._paint(view)
        return True

    def open_plan_change(self, nonce: str) -> bool:
        """'Change' was tapped on a plan card: the next text in the card's topic is the owner's feedback on the plan."""
        with self._lock:
            card = self._open_card(nonce)
            if card is None or card.request.kind != PLAN_KIND:
                return False
            self._await_text[card.thread_id] = (nonce, PLAN_CHANGE_QI)
            if not card.text.endswith(PLAN_CHANGE_WAIT_TEXT):
                card.text += PLAN_CHANGE_WAIT_TEXT
            view = self._view(card, with_buttons=True)
        self._paint(view)
        return True

    def awaiting(self, thread_id: int) -> tuple[str, int] | None:
        """(nonce, question index) of the typed answer the topic is waiting for, or None."""
        with self._lock:
            return self._await_text.get(thread_id)

    def record_answer(self, nonce: str, qi: int, value: str) -> AnswerProgress | None:
        """Stores the answer to question qi. The card shows the answers so far; the caller resolves the ask once
        AnswerProgress.complete is true. None when the card is not an open question card or qi is out of range."""
        with self._lock:
            card = self._open_card(nonce)
            if card is None or not 0 <= qi < len(card.questions):
                return None
            question = card.questions[qi]
            card.answers[question.text] = value
            if self._await_text.get(card.thread_id) == (nonce, qi):
                del self._await_text[card.thread_id]
            complete = all(q.text in card.answers for q in card.questions)
            card.text = question_body(card.questions, card.answers)
            progress = AnswerProgress(complete=complete, answers=dict(card.answers))
            view = None if complete else self._view(card, with_buttons=True)
        if view is not None:
            self._paint(view)
        return progress

    def _nudge(self, nonce: str) -> None:
        """Timer callback: after ASK_NUDGE_SECONDS on the clock, a still-open card says it waits for a tap."""
        with self._lock:
            card = self._open_card(nonce)
            if card is None or card.mid is None:
                return
            if self._clock.monotonic() - card.posted_at < self._nudge_seconds:
                return  # a timer that fired early (a late clock tick) must not nudge
            view = self._view(card, with_buttons=True, suffix=NUDGE_TEXT)
        self._paint(view)

    def _open_card(self, nonce: str) -> _Card | None:
        """The card if it is still open (caller holds the lock)."""
        card = self._cards.get(nonce)
        return card if card is not None and card.verdict is None else None

    def _drop_await(self, card: _Card) -> None:
        """Forget the 'Other' wait that belongs to this card (caller holds the lock)."""
        if self._await_text.get(card.thread_id, (None, None))[0] == card.nonce:
            del self._await_text[card.thread_id]

    def _forget(self, card: _Card) -> None:
        """Drops a card that never reached Telegram (the send failed or raised)."""
        with self._lock:
            if self._cards.get(card.nonce) is card:
                del self._cards[card.nonce]
            self._drop_await(card)

    def _view(self, card: _Card, with_buttons: bool, suffix: str = "") -> _View | None:
        """Snapshot of a card for one edit (caller holds the lock). None while the message id is still unknown:
        post() paints the current state once the id arrives."""
        if card.mid is None:
            return None
        markup = keyboard_json(_rows(card, with_buttons)) if with_buttons else EMPTY_KEYBOARD
        return _View(nonce=card.nonce, chat_id=card.chat_id, mid=card.mid, text=card.text + suffix, markup=markup)

    @staticmethod
    def _outcome_suffix(verdict: str) -> str:
        return "\n\n" + OUTCOME.get(verdict, verdict)

    def _paint(self, view: _View | None) -> None:
        """One editMessageText of a card. A refused edit is logged and never raised (the card is best effort)."""
        if view is None:
            return
        try:
            self._messenger.edit_text(view.chat_id, view.mid, view.text, reply_markup=view.markup)
        except Exception as exc:
            self._logger.log(f"editMessageText card {view.nonce}: {str(exc)[:ERROR_TEXT_LIMIT]}")
