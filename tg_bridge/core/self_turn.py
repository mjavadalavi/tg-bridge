"""SelfTurns: one self turn of a kept session, from its start to the 🔔 reply (linger).

A self turn is a turn the CLI starts by itself while the session is kept: a background task reports back, a wakeup
fires, or a background subagent finishes. It runs like a user turn: the same per-launch kit (feed, cards, view, answer
and usage collectors, the session's permission binding), the same wait loop (wait_turn), and the same turns row and
feed summary (TurnCloser). Its reply is posted with BACKGROUND_PREFIX. The session's input is never closed after a self
turn, so the session can take the next message.

A request to interrupt this one update (ctx.interrupt, set by LingerService) is sent to the session once, from the wait
loop, and the turn ends INTERRUPTED with the session kept. A stop of the whole session (/stop, the abort flag) ends the
turn through the same wait loop as a user turn; the caller decides what that means for the session.

Threads: run() runs on the topic's runner thread, and so does every call it makes to the session (the interrupt, the
wait). The session's events reach the feed and the listeners through the SessionRouter on the reader thread, as for
user turns; the router buffers them until run() attaches the gate.

Lock map: SelfTurns holds no lock of its own. LiveRegistry, WaitRouter and the per-turn objects lock their own state.
Nothing here holds a lock while it calls another service.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass

from tg_bridge.agents.base import AgentSession
from tg_bridge.clock import Clock
from tg_bridge.config import Config
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.permissions import PermissionService, TurnBinding
from tg_bridge.core.session_router import SessionRouter
from tg_bridge.core.turn_kit import (
    FeedFactory,
    TurnCloser,
    TurnPlan,
    TurnReply,
    build_kit,
    finish_attempt,
    open_view,
    sync_open_asks,
    wait_turn,
    wire_gate,
)
from tg_bridge.core.turn_listeners import SessionRecorder, WaitRouter
from tg_bridge.core.turn_state import TurnOutcome
from tg_bridge.core.turn_text import BACKGROUND_PREFIX, LAUNCH_ERROR_CHARS, NO_OUTPUT, RESTART_TEXT
from tg_bridge.core.view_recorder import ViewFactory
from tg_bridge.logging_ import Logger
from tg_bridge.store.turns import TurnRepo
from tg_bridge.telegram.answers import AnswerSender
from tg_bridge.telegram.messenger import Messenger

POLL_SECONDS = 1.0  # the wait loop's tick, as the user turn's (turn_runner.POLL_SECONDS; not imported, to avoid a cycle)
SEND_ERROR_CHARS = 120  # the send error text kept in the log (dispatcher.SEND_ERROR_CHARS; not imported, to avoid a cycle)


@dataclass(frozen=True)
class SelfTurnContext:
    """What one self turn runs on: the kept session's plan (text ''), the session, its router and permission binding,
    its session recorder, and the event that asks for an interrupt of this update only."""

    plan: TurnPlan
    session: AgentSession
    router: SessionRouter
    binding: TurnBinding
    recorder: SessionRecorder
    interrupt: threading.Event


@dataclass(frozen=True)
class SelfTurnEnd:
    """ran is False when the router was closed or a shutdown refused the turn: nothing ran and nothing was posted.
    outcome is the turn's outcome (None when nothing ran); alive says whether the session still runs after the turn."""

    ran: bool
    outcome: TurnOutcome | None
    alive: bool


class SelfTurns:
    """Runs self turns. One instance per App; each run() is one self turn, called from the topic's runner thread."""

    def __init__(
        self, *, config: Config, clock: Clock, logger: Logger, feeds: FeedFactory, views: ViewFactory | None,
        permissions: PermissionService, handoff: HandoffService, waits: WaitRouter, live: LiveRegistry,
        turns: TurnRepo, answers: AnswerSender, messenger: Messenger, poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self._config = config
        self._clock = clock
        self._logger = logger
        self._feeds = feeds
        self._views = views
        self._permissions = permissions
        self._handoff = handoff
        self._waits = waits
        self._live = live
        self._answers = answers
        self._messenger = messenger
        self._poll_seconds = poll_seconds
        self._closer = TurnCloser(turns=turns, clock=clock)

    def run(self, ctx: SelfTurnContext) -> SelfTurnEnd:
        """One self turn: a normal kit on the session's binding, the buffered events attached in order, the wait with
        the input left open, then the turns row, the feed summary and the 🔔 reply. Nothing is voided: the binding
        belongs to the session."""
        plan = ctx.plan
        thread = plan.thread_id
        kit = build_kit(plan, ctx.recorder, None, clock=self._clock, config=self._config, feeds=self._feeds,
                        permissions=self._permissions, handoff=self._handoff, pending_digest=lambda: "",
                        open_view=lambda: open_view(self._views, plan, self._logger), binding=ctx.binding)
        self._waits.register(thread, kit.turn)
        registered = False
        attached = False
        launch_err: str | None = None
        try:  # the view is finished by the finally, even when the wiring below raises
            gate = wire_gate(kit, self._logger)
            registered = self._live.begin_turn(thread, ctx.session, kit.feed)
            if registered:
                # the feed and the turn start before the gate is attached: the router drains the buffered events into
                # the gate at once, so an approval that was buffered must find a started turn (its timer pauses)
                kit.feed.start()
                kit.turn.start()
                sync_open_asks(self._waits, self._permissions, kit.binding)
                kit.turn.mark_running()
                attached = bool(ctx.router.attach(gate))
            if attached:
                wait_turn(kit.turn, ctx.session, kit.done, thread, self._live, self._poll_seconds,
                          after_result=lambda: True, on_tick=self._interrupt_once(ctx))
        except Exception as exc:  # a failure during the self turn ends it as a launch error, as for a user turn
            launch_err = f"launcher failed: {str(exc)[:LAUNCH_ERROR_CHARS]}"
        finally:
            if registered:
                self._live.end_turn(thread, ctx.session, kit.feed)
            self._waits.unregister(thread, kit.turn)
            if kit.view is not None:  # a self turn that failed emits no end event: the view must not stay running
                kit.view.finish()
            kit.feed.close()
        alive = ctx.session.alive()
        if not attached:
            if launch_err is not None:
                self._logger.log(f"self turn: thread {thread} did not run: {launch_err}")
            return SelfTurnEnd(ran=False, outcome=None, alive=alive)
        attempt = finish_attempt(kit, ctx.session, self._live, refused=False, launch_err=launch_err, stop_code=None,
                                 readable_tail=plan.backend.capabilities_for(plan.launcher).readable_tail,
                                 logger=self._logger)
        # a stop or a cap of a self turn closes the session (D2: only an interrupt keeps it), so its reply says so
        kept = attempt.outcome not in (TurnOutcome.STOPPED, TurnOutcome.CAPPED)
        reply = self._closer.close(plan, attempt, "", session_kept=kept)
        self._post(plan, reply)
        return SelfTurnEnd(ran=True, outcome=attempt.outcome, alive=ctx.session.alive())

    def _interrupt_once(self, ctx: SelfTurnContext) -> Callable[[], None]:
        """The wait loop's tick hook: sends the session one interrupt after ctx.interrupt was set, and no more. When the
        session does not take it (no control reply, or the TUI is gone), the update is stopped instead, as ⏭ Now and
        the stop taps stop a turn: the wait ends STOPPED, the hold is released, and a queued message is not blocked
        until the turn cap."""
        sent = False

        def tick() -> None:
            nonlocal sent
            if sent or not ctx.interrupt.is_set():
                return
            sent = True
            try:
                taken = ctx.session.interrupt()
            except Exception as exc:  # a raise is a refusal too: it must not fail the wait loop as a launch error
                self._logger.log(f"self turn: thread {ctx.plan.thread_id} interrupt raised: {str(exc)[:SEND_ERROR_CHARS]}")
                taken = False
            if taken:
                return
            self._logger.log(f"self turn: thread {ctx.plan.thread_id} had nothing to interrupt; stopping it")
            self._live.request_abort(ctx.plan.thread_id, stop_tap=True)
            threading.Thread(target=ctx.session.stop, daemon=True).start()

        return tick

    def _post(self, plan: TurnPlan, reply: TurnReply) -> None:
        """The 🔔 reply: the answer, or the error text. Nothing when a bridge restart stopped the turn. A send that fails
        is logged and dropped: the turn has already run, so the failure must not end the session."""
        try:
            if reply.answer is not None:
                text = f"{BACKGROUND_PREFIX}\n\n{reply.answer or NO_OUTPUT}"
                self._answers.send_answer(plan.chat_id, plan.thread_id, text, self._config.answer_doc_threshold())
            elif reply.err is not None and reply.err != RESTART_TEXT:
                self._messenger.send_text(plan.chat_id, plan.thread_id, f"{BACKGROUND_PREFIX}\n{reply.err}")
        except Exception as exc:
            self._logger.log(f"self turn: thread {plan.thread_id} reply not sent: {str(exc)[:SEND_ERROR_CHARS]}")
