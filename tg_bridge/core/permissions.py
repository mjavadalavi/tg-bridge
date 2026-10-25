"""PermissionService: approval listener, rule short-circuit, bypass and the taps that resolve asks (C13, permissions part).

Port of handle_permission, _post_card, answer_ask/always/bypass/other/question, _record_answer, handle_cancel, void_asks,
_apply_live_mode and the card part of recover(). The cards themselves are core/cards.py ApprovalCards; this class decides
what a tap means. bind_turn() returns the EventListener the turn bus delivers approval events to; the taps, the /mode
picker and the turn runner (void_all, recover_cards) call the other methods.

Ask lifecycle: pending (in self._asks) -> claimed (popped under the lock; only the claimer writes the AskRepo verdict,
answers the session and closes the card) -> closed. The claim is the idempotence point, so a double tap or a replayed
callback finds nothing and the session is answered exactly once. A card goes out on a worker: the post job checks the
claim before it sends, and closes the card if the ask was claimed while the send was in flight.

Lock map: self._lock guards _asks, _turns and every TurnBinding field. Repo, session, card, worker and observer calls
all run after the lock is released (copy, release, act).
"""

import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from tg_bridge.agents.base import AgentSession, RuleMatcher
from tg_bridge.agents.events import AgentEvent, ApprovalRequest, ApprovalWithdrawn, Decision, TurnResult
from tg_bridge.core.cards import ApprovalCards
from tg_bridge.logging_ import Logger
from tg_bridge.store.asks import VOID, AskRepo
from tg_bridge.store.rules import RuleRepo
from tg_bridge.telegram.keyboards import EMPTY_KEYBOARD
from tg_bridge.telegram.messenger import Messenger

BYPASS_MODE = "bypassPermissions"
QUESTION_KIND = "question"
VOID_RESTART_TEXT = "\n\n⌛ void — bridge restarted"
DETAIL_KEYS = ("command", "file_path", "notebook_path", "path")
DETAIL_CHARS = 600
ERROR_TEXT_LIMIT = 150
NONCE_BYTES = 4
# Reasons the bridge (not the owner) denies with. The model reads them, so the live texts are kept word for word.
CLOSED_TURN_MESSAGE = "turn already ended"
NO_QUESTIONS_MESSAGE = "no questions to answer"
NOT_POSTED_MESSAGE = "could not reach Telegram to ask"


def new_nonce() -> str:
    """A random card nonce: 8 hex characters, short enough for every callback prefix."""
    return secrets.token_hex(NONCE_BYTES)


class WaitObserver(Protocol):
    """Told when a turn starts or stops waiting for a person (the turn state uses it to give its slot back)."""

    def waiting_changed(self, thread_id: int, waiting: bool) -> None: ...


class Submitter(Protocol):
    """Runs a job on a worker thread. WorkerPool satisfies it; tests pass an inline runner."""

    def submit(self, fn: Callable[..., object], *args: object) -> None: ...


@dataclass(eq=False)
class TurnBinding:
    """What the permission service knows about one turn. Its mutable fields are guarded by PermissionService._lock."""

    thread_id: int
    chat_id: int
    cwd: str
    rules: RuleMatcher
    mode: str
    session: AgentSession | None = None
    turn_bypass: bool = False
    restore_mode: str = ""  # the topic's mode before 'Bypass this turn', put back when the turn ends
    waiting: int = 0
    closed: bool = False


@dataclass(eq=False)
class _Ask:
    """One permission question put to the user. Removed from PermissionService._asks by its claimer."""

    nonce: str
    binding: TurnBinding
    request: ApprovalRequest
    session: AgentSession
    always_rule: str | None
    verdict: str | None = None


class _TurnListener:
    """The EventListener of one turn: routes the approval events of that turn to the service."""

    def __init__(self, service: "PermissionService", binding: TurnBinding) -> None:
        self._service = service
        self._binding = binding

    def on_event(self, event: AgentEvent) -> None:
        if isinstance(event, ApprovalRequest):
            self._service.handle_request(self._binding, event)
        elif isinstance(event, ApprovalWithdrawn):
            self._service.handle_withdrawn(self._binding, event.approval_id)
        elif isinstance(event, TurnResult):
            self._service.end_turn_bypass(self._binding)


def _detail(inp: dict[str, Any]) -> str:
    """The text a recovered card shows under its title: the command or path, else the input as JSON."""
    for key in DETAIL_KEYS:
        value = inp.get(key)
        if isinstance(value, str) and value:
            return value[:DETAIL_CHARS]
    return str(inp)[:DETAIL_CHARS]


class PermissionService:
    """One per App. Turns approval events into card asks, saved-rule answers or bypass answers, and resolves the asks."""

    def __init__(self, *, asks: AskRepo, rules: RuleRepo, cards: ApprovalCards, messenger: Messenger,
                 workers: Submitter, waiting: WaitObserver, logger: Logger,
                 nonce_factory: Callable[[], str] = new_nonce) -> None:
        self._asks_repo = asks
        self._rules_repo = rules
        self._cards = cards
        self._messenger = messenger
        self._workers = workers
        self._waiting = waiting
        self._logger = logger
        self._new_nonce = nonce_factory
        self._lock = threading.Lock()
        self._asks: dict[str, _Ask] = {}
        self._turns: dict[int, TurnBinding] = {}

    # turn wiring

    def bind_turn(self, binding: TurnBinding) -> _TurnListener:
        """Registers a turn and returns the listener the turn's event bus delivers approval events to."""
        with self._lock:
            self._turns[binding.thread_id] = binding
        return _TurnListener(self, binding)

    def attach(self, binding: TurnBinding, session: AgentSession) -> None:
        """Gives the turn its session. Must happen before the session starts, so no approval arrives without one."""
        with self._lock:
            binding.session = session

    # approval events (called by the turn listener, on the backend reader thread: never blocks on a backend reply)

    def handle_request(self, binding: TurnBinding, req: ApprovalRequest) -> None:
        with self._lock:
            closed, mode, session = binding.closed, binding.mode, binding.session
        if session is None:
            self._logger.log(f"approval {req.approval_id} dropped: the turn has no session")
            return
        if closed:
            self._answer(session, req, Decision.DENY, deny_message=CLOSED_TURN_MESSAGE)
            return
        if req.kind == QUESTION_KIND:
            if not req.questions:
                self._answer(session, req, Decision.DENY, deny_message=NO_QUESTIONS_MESSAGE)
                return
        elif mode == BYPASS_MODE:
            self._answer(session, req, Decision.ALLOW_ONCE)
            return
        elif self._saved_rule(binding, req) is not None:
            self._answer(session, req, Decision.ALLOW_ONCE)
            return
        self._register(binding, session, req)

    def handle_withdrawn(self, binding: TurnBinding, approval_id: str) -> None:
        """The backend withdrew an ask (the CLI cancelled it, or Codex resolved it elsewhere): closed, not answered."""
        with self._lock:
            nonce = next((a.nonce for a in self._asks.values()
                          if a.binding is binding and a.request.approval_id == approval_id), None)
        if nonce is None:
            return
        ask = self._claim(nonce, "withdrawn")
        if ask is not None:
            self._finish(ask, None)

    # taps and commands (called from callbacks, commands and the typed 'Other' answer)

    def answer_ask(self, nonce: str, allow: bool, by: str = "owner") -> bool:
        """Allow once or Deny a pending ask. False when the ask is gone (already answered, withdrawn or voided).
        A question cannot be allowed without its answers, so allow=True on one returns False."""
        verdict = "allow" if allow else "deny"
        ask = self._claim(nonce, verdict, accept=lambda a: not (allow and a.request.kind == QUESTION_KIND))
        if ask is None:
            return False
        self._logger.log(f"ask {nonce}: {verdict} by {by}")
        self._finish(ask, Decision.ALLOW_ONCE if allow else Decision.DENY)
        return True

    def answer_always(self, nonce: str, by: str = "owner") -> bool:
        """Allow once and save the rule the backend proposed, so the same kind of request is allowed from now on."""
        ask = self._claim(nonce, "allow", accept=lambda a: a.always_rule is not None)
        if ask is None:
            return False
        if ask.always_rule is not None:
            self._rules_repo.add(ask.binding.thread_id, ask.always_rule)
        self._logger.log(f"ask {nonce}: always by {by}")
        self._finish(ask, Decision.ALLOW_ALWAYS)
        return True

    def answer_bypass(self, nonce: str, by: str = "owner") -> bool:
        """'Bypass this turn': the turn switches to bypass, and every open permission card is allowed (this one too).
        The switch runs on a worker first, as the live file does, so the allow reaches the CLI under bypass."""
        with self._lock:
            ask = self._asks.get(nonce)
            if ask is None or ask.request.kind == QUESTION_KIND or Decision.ALLOW_TURN not in ask.request.choices:
                return False
            if not ask.binding.turn_bypass:
                ask.binding.restore_mode = ask.binding.mode
            ask.binding.turn_bypass = True
            ask.binding.mode = BYPASS_MODE
            thread_id = ask.binding.thread_id
        self._workers.submit(self._bypass_job, thread_id, by)
        return True

    def answer_other(self, nonce: str, qi: int) -> bool:
        """'Other' on question qi: the next text in the topic is that question's answer."""
        with self._lock:
            ask = self._asks.get(nonce)
            if ask is None or ask.request.kind != QUESTION_KIND:
                return False
        return self._cards.open_other(nonce, qi)

    def answer_question(self, nonce: str, qi: int, oi: int) -> bool:
        """One option tap on question qi of a question card."""
        label = self._cards.option_label(nonce, qi, oi)
        return label is not None and self._record(nonce, qi, label)

    def record_text(self, thread_id: int, text: str) -> bool:
        """A typed message in a topic that waits for an 'Other' answer. True when it became that answer."""
        waiting = self._cards.awaiting(thread_id)
        return waiting is not None and self._record(waiting[0], waiting[1], text)

    def set_live_mode(self, thread_id: int, mode: str, by: str = "owner") -> bool:
        """Switches the running turn of a topic to mode. Returns whether the session applied it live (False: the mode
        applies from the next message, or no turn runs). Bypass also allows every open permission card of the turn."""
        with self._lock:
            binding = self._turns.get(thread_id)
            if binding is None or binding.closed:
                return False
            binding.mode = mode
            session = binding.session
        live = False
        if session is not None:
            try:
                live = bool(session.set_mode(mode))
            except Exception as exc:
                self._logger.log(f"session {thread_id}: set_mode {mode} failed: {str(exc)[:ERROR_TEXT_LIMIT]}")
        if mode == BYPASS_MODE:
            self._allow_open(binding, by)
        return live

    # end of turn and start-up

    def end_turn_bypass(self, binding: TurnBinding) -> None:
        """The turn's result arrived while its session is still alive: 'Bypass this turn' ends and the topic's mode goes
        back to the CLI now, so the CLI hears it before the stop (void_all runs after the stop, too late for that)."""
        with self._lock:
            if not binding.turn_bypass:
                return
            binding.turn_bypass = False
            binding.mode = binding.restore_mode
            session, mode, thread_id = binding.session, binding.mode, binding.thread_id
        if session is not None:
            self._workers.submit(self._restore_mode, thread_id, session, mode)

    def void_all(self, binding: TurnBinding) -> int:
        """The turn ended: every ask still open on it is void and its card closed. Returns how many were voided."""
        with self._lock:
            binding.closed = True
            if self._turns.get(binding.thread_id) is binding:
                del self._turns[binding.thread_id]
            mine = [a for a in self._asks.values() if a.binding is binding]
            for ask in mine:
                del self._asks[ask.nonce]
                ask.verdict = VOID
            was_waiting = binding.waiting > 0
            binding.waiting = 0
            if binding.turn_bypass:  # no result came (the process ended first): the flag ends here, the CLI is gone
                binding.turn_bypass = False
                binding.mode = binding.restore_mode
        for ask in mine:
            self._asks_repo.answer(ask.nonce, VOID)
            self._cards.close(ask.nonce, VOID)
        if was_waiting:
            self._notify(binding.thread_id, False)
        return len(mine)

    def recover_cards(self) -> int:
        """Start-up: every ask still pending in the database was never answered and its turn is gone. Each is voided
        and its card, when it has one, is closed. Returns how many asks were voided."""
        count = 0
        for record in self._asks_repo.pending():
            if not self._asks_repo.answer(record.nonce, VOID):
                continue
            count += 1
            if record.message_id is None:
                continue
            text = f"🔐 {record.tool_name}\n{_detail(record.tool_input())}{VOID_RESTART_TEXT}"
            try:
                self._messenger.edit_text(record.chat_id, record.message_id, text, reply_markup=EMPTY_KEYBOARD)
            except Exception as exc:
                self._logger.log(f"recover card {record.nonce}: {str(exc)[:ERROR_TEXT_LIMIT]}")
        return count

    # internals

    def _saved_rule(self, binding: TurnBinding, req: ApprovalRequest) -> str | None:
        """The first saved rule of the topic that matches the request, counted as a hit, or None."""
        for rule in self._rules_repo.texts(binding.thread_id):
            if binding.rules.matches(rule, req, binding.cwd):
                self._rules_repo.hit(binding.thread_id, rule)
                return rule
        return None

    def _register(self, binding: TurnBinding, session: AgentSession, req: ApprovalRequest) -> None:
        """Records the ask (database first, then memory, so an early tap always finds it) and posts its card."""
        nonce = self._new_nonce()
        always = binding.rules.rule_for(req) if Decision.ALLOW_ALWAYS in req.choices else None
        self._asks_repo.insert(nonce=nonce, thread_id=binding.thread_id, chat_id=binding.chat_id,
                               request_id=req.approval_id, tool_name=req.tool, input=req.input)
        ask = _Ask(nonce=nonce, binding=binding, request=req, session=session, always_rule=always)
        with self._lock:
            self._asks[nonce] = ask
            binding.waiting += 1
            became_waiting = binding.waiting == 1
        if became_waiting:
            self._notify(binding.thread_id, True)
        self._workers.submit(self._post, ask)

    def _post(self, ask: _Ask) -> None:
        """Worker job: sends the card of a registered ask. An ask answered before this runs is not posted; a card that
        cannot be sent denies the ask, since nobody could tap it."""
        with self._lock:
            if ask.verdict is not None:
                return
        try:
            posted = self._cards.post(ask.nonce, ask.request, chat_id=ask.binding.chat_id,
                                      thread_id=ask.binding.thread_id, always_rule=ask.always_rule)
        except Exception as exc:
            self._logger.log(f"ask {ask.nonce}: card not posted: {str(exc)[:ERROR_TEXT_LIMIT]}")
            posted = False
        if not posted:
            claimed = self._claim(ask.nonce, "deny")
            if claimed is not None:
                self._finish(claimed, Decision.DENY, deny_message=NOT_POSTED_MESSAGE)
            return
        self._logger.log(f"ask {ask.nonce}: {ask.request.tool} posted")
        with self._lock:
            verdict = ask.verdict
        if verdict is not None:  # claimed while the card was being sent: close the card now
            self._cards.close(ask.nonce, verdict)

    def _restore_mode(self, thread_id: int, session: AgentSession, mode: str) -> None:
        """Worker job: sends the topic's mode back to a session that is still running."""
        try:
            session.set_mode(mode)
        except Exception as exc:
            self._logger.log(f"session {thread_id}: restore mode {mode} failed: {str(exc)[:ERROR_TEXT_LIMIT]}")

    def _bypass_job(self, thread_id: int, by: str) -> None:
        self.set_live_mode(thread_id, BYPASS_MODE, by)

    def _allow_open(self, binding: TurnBinding, by: str) -> None:
        with self._lock:
            nonces = [a.nonce for a in self._asks.values() if a.binding is binding
                      and a.request.kind != QUESTION_KIND and Decision.ALLOW_ONCE in a.request.choices]
        for nonce in nonces:
            self.answer_ask(nonce, True, by=by)

    def _record(self, nonce: str, qi: int, value: str) -> bool:
        """Stores one answer on a question card. When it is the last one, the ask is claimed and allowed with the answers."""
        with self._lock:
            if nonce not in self._asks:
                return False
        progress = self._cards.record_answer(nonce, qi, value)
        if progress is None:
            return False
        if not progress.complete:
            return True
        ask = self._claim(nonce, "allow", accept=lambda a: a.request.kind == QUESTION_KIND)
        if ask is None:
            return False
        self._finish(ask, Decision.ALLOW_ONCE, answers=progress.answers)
        return True

    def _claim(self, nonce: str, verdict: str, accept: Callable[[_Ask], bool] | None = None) -> _Ask | None:
        """Removes a pending ask and records its verdict. Only the caller that gets the ask may answer it; None when the
        ask is gone or accept() refuses it. Waiting ends (observer told) when the turn's last open ask is claimed."""
        with self._lock:
            ask = self._asks.get(nonce)
            if ask is None or (accept is not None and not accept(ask)):
                return None
            del self._asks[nonce]
            ask.verdict = verdict
            binding = ask.binding
            binding.waiting = max(0, binding.waiting - 1)
            idle = binding.waiting == 0
        if idle:
            self._notify(binding.thread_id, False)
        return ask

    def _answer(self, session: AgentSession, req: ApprovalRequest, decision: Decision,
                answers: dict[str, str] | None = None, deny_message: str | None = None) -> None:
        try:
            session.answer(req, decision, answers=answers, deny_message=deny_message)
        except Exception as exc:
            self._logger.log(f"answer {req.approval_id} failed: {str(exc)[:ERROR_TEXT_LIMIT]}")

    def _finish(self, ask: _Ask, decision: Decision | None, answers: dict[str, str] | None = None,
                deny_message: str | None = None) -> None:
        """The second half of a claim: the verdict goes to the database, the session is answered (unless the backend
        withdrew the ask, decision None), and the card shows the outcome. deny_message is the bridge's own reason for a
        Deny; None is the owner's Deny."""
        verdict = ask.verdict or VOID
        self._asks_repo.answer(ask.nonce, verdict)
        if decision is not None:
            self._answer(ask.session, ask.request, decision, answers, deny_message=deny_message)
        self._cards.close(ask.nonce, verdict)

    def _notify(self, thread_id: int, waiting: bool) -> None:
        try:
            self._waiting.waiting_changed(thread_id, waiting)
        except Exception as exc:
            self._logger.log(f"waiting observer failed: {str(exc)[:ERROR_TEXT_LIMIT]}")
