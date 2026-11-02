"""TurnKit: the objects one launch of a turn shares, and the pieces that work on them.

build_kit() makes the per-launch objects that TurnRunner used to build inline: the turn state, the feed, the answer and
usage collectors, the permission binding, the listeners in their fixed order and the topic's view. open_view() makes that
view. wire_gate() puts the event bus and the end gate in front of those listeners, inside the launch's try. The session
sink is a SessionRouter (core/session_router.py) that holds the gate. wait_turn() is the wait loop of one launch,
finish_attempt() gathers the facts of a launch once its teardown is done, and TurnCloser writes the reply, the feed
summary and the turns row once the turn has ended. TurnPlan, TurnProgress, TurnReply and _Attempt are the types these
pieces pass around. They live here so that a long-lived session can build the per-turn parts again for each turn
without the whole TurnRunner.

Threads: the listeners of a kit are called on the backend's reader thread (see core/turn_listeners.py). build_kit,
wire_gate and wait_turn run on the thread that runs the turn, the topic's dispatcher. TurnCloser holds no state apart
from its repository and its clock.

Lock map: TurnKit holds no lock of its own. TurnState, AnswerCollector, UsageRecorder and the TurnEventBus each lock
their own state. Nothing here holds a lock while it calls another service.
"""

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from tg_bridge.agents.base import AgentBackend, AgentSession, AttachSpec
from tg_bridge.agents.events import AgentEvent, TurnResult, UsageEvent
from tg_bridge.clock import Clock, utc_stamp
from tg_bridge.config import Config, LauncherSpec
from tg_bridge.core.events_bus import EventListener, TurnEventBus
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.live import AbortFlags, LiveRegistry
from tg_bridge.core.permissions import PermissionService, TurnBinding
from tg_bridge.core.turn_listeners import (
    AnswerCollector,
    EndGate,
    HandoffClearer,
    ResultFreezer,
    SessionRecorder,
    UsageRecorder,
    WaitRouter,
)
from tg_bridge.core.turn_state import SlotLike, TurnOutcome, TurnPhase, TurnState
from tg_bridge.core.turn_text import NO_CONVERSATION, compose_reply, whole_seconds
from tg_bridge.core.view_recorder import ViewFactory, ViewRecorder
from tg_bridge.feed.format import final_header
from tg_bridge.logging_ import Logger
from tg_bridge.store.turns import TurnRepo

GRACEFUL_SECONDS = 10.0  # after a normal end the CLI gets this long to exit on its own (input closed, no interrupt)
STDERR_LINES = 20  # the stderr lines searched for the resume error
TAIL_LINES = 10  # the stderr lines shown in a process error
LOG_TAIL_CHARS = 2000  # the end of a raw tail written to the bridge log instead of the reply
_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")  # ANSI CSI and two-byte escapes
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # control characters, the newline and tab kept


def log_text(raw: str) -> str:
    """The tail as log text: the escape sequences and control characters of a terminal screen removed, the last
    LOG_TAIL_CHARS kept."""
    return _CONTROL.sub("", _ESCAPE.sub("", raw)).strip()[-LOG_TAIL_CHARS:]


class TurnProgress(Protocol):
    """The feed of one attempt (TurnFeed satisfies it). Called by the turn runner only."""

    def start(self) -> None: ...

    def on_event(self, event: AgentEvent) -> None: ...

    def close(self) -> None: ...

    def finish(self, header: str) -> None: ...


FeedFactory = Callable[[int | str, int, Callable[[], bool]], TurnProgress]


@dataclass(frozen=True)
class TurnReply:
    """What the turn produced. answer and err are both None only when a shutdown refused the spawn: nothing ran, and
    the caller re-marks the inbox row for a later try. terminal_busy is set (with both None) when the terminal session
    of an attached turn was not ready: nothing was pasted, and the message waits for the terminal again."""

    answer: str | None
    err: str | None
    outcome: TurnOutcome
    terminal_busy: bool = False


@dataclass(frozen=True)
class TurnPlan:
    """Everything decided before the first launch of a turn."""

    thread_id: int
    chat_id: int
    topic: str
    text: str
    images: tuple[str, ...]
    launcher: LauncherSpec
    backend: AgentBackend
    cwd: str
    mode: str
    model: str | None
    allow_rules: tuple[str, ...]
    deny_tools: tuple[str, ...]
    system_rules: str
    stored_uuid: str
    attach: AttachSpec | None = None  # set: the turn runs in that terminal session (nothing is spawned or kept)


@dataclass(frozen=True)
class _Attempt:
    """The facts of one launch, gathered for the reply, the turns row and the feed summary."""

    refused: bool
    outcome: TurnOutcome
    result: TurnResult | None
    usage: UsageEvent | None
    texts: tuple[str, ...]
    tools: int
    limit: float | None
    stop_tap: bool
    aborted: bool
    launch_err: str | None
    exit_code: int | None
    tail: str
    resume_failed: bool
    feed: TurnProgress
    started_wall: float
    started_mono: float
    terminal_busy: bool = False  # the attached terminal was not ready: refused, and the reply says so


@dataclass(frozen=True)
class TurnKit:
    """The objects of one launch. build_kit makes them; the turn runner and the wait loop read them."""

    turn: TurnState
    feed: TurnProgress
    done: threading.Event
    collector: AnswerCollector
    usage: UsageRecorder
    binding: TurnBinding
    started_wall: float
    started_mono: float
    digest: str
    listeners: list[EventListener]
    view: ViewRecorder | None


def build_kit(plan: TurnPlan, recorder: SessionRecorder, slot: SlotLike | None, *, clock: Clock, config: Config,
              feeds: FeedFactory, permissions: PermissionService, handoff: HandoffService,
              pending_digest: Callable[[], str], open_view: Callable[[], ViewRecorder | None],
              binding: TurnBinding | None = None) -> TurnKit:
    """The per-launch objects of one turn, built in the order the live file sets them up. pending_digest and open_view
    are called at the points where they always ran (the digest after the start times, the view after the listeners),
    so a launch that fails here still leaves the view to be finished by the caller. A given binding (the one a kept
    session was first started with) is used as it is; otherwise a new one is made for this launch."""
    thread = plan.thread_id
    turn = TurnState(thread, clock, slot, max_active_seconds=config.max_turn_seconds(),
                     hard_cap_seconds=config.hard_cap_seconds())
    feed = feeds(plan.chat_id, thread, lambda: turn.phase is TurnPhase.WAITING_APPROVAL)
    done = threading.Event()
    collector = AnswerCollector(done)
    usage = UsageRecorder()
    if binding is None:
        binding = TurnBinding(thread_id=thread, chat_id=plan.chat_id, cwd=plan.cwd, rules=plan.backend.rules,
                              mode=plan.mode)
    started_wall, started_mono = clock.now(), clock.monotonic()
    digest = pending_digest()
    # the freezer leads: the active timer stops at the result before any listener (or the wait loop) can act on it
    listeners = [ResultFreezer(turn.mark_result_received), feed, permissions.bind_turn(binding), usage,
                 recorder, collector]
    view = open_view()  # the readable mirror of the topic; None when there is none or it failed
    if view is not None:  # before the collector, which wakes the wait loop
        listeners.insert(-1, view)
    if digest:
        listeners.append(HandoffClearer(handoff, thread))
    return TurnKit(turn=turn, feed=feed, done=done, collector=collector, usage=usage, binding=binding,
                   started_wall=started_wall, started_mono=started_mono, digest=digest, listeners=listeners,
                   view=view)


def open_view(views: ViewFactory | None, plan: TurnPlan, logger: Logger) -> ViewRecorder | None:
    """The topic's view recorder for one attempt. A factory that raises costs the view only: it is logged and the
    turn runs without a recorder."""
    if views is None:
        return None
    try:
        return views(plan.thread_id, plan.topic, plan.launcher.transport, plan.launcher.name)
    except Exception as exc:  # the view is optional: a broken recorder never stops the turn
        logger.log(f"view: recorder not built, the turn runs without it: {exc!r}")
        return None


def wire_gate(kit: TurnKit, logger: Logger) -> EndGate:
    """The end gate in front of the turn's bus: whatever the process sends after its end never reaches the listeners.
    The caller builds it inside the launch's try, so a failure here ends the attempt like any other launch error."""
    bus = TurnEventBus(kit.listeners, logger)
    return EndGate(bus.publish, logger)


def sync_open_asks(waits: WaitRouter, permissions: PermissionService, binding: TurnBinding) -> None:
    """Tells a turn that starts while one of its session's cards is still open (a background agent asked while the
    session was idle) what an ApprovalRequest during the turn would have told it: it waits, and its active timer stops
    until the card is answered. Called right after turn.start(), through the same WaitRouter the permissions use. The
    count is read again after the notice: an answer that lands between the read and the notice may have sent its
    waiting=False before the notice reached the turn, so the turn is then told the waiting is over."""
    if permissions.open_asks(binding) == 0:
        return
    waits.waiting_changed(binding.thread_id, True)
    if permissions.open_asks(binding) == 0:
        waits.waiting_changed(binding.thread_id, False)


def wait_turn(turn: TurnState, session: AgentSession, done: threading.Event, thread_id: int, live: LiveRegistry,
              poll_seconds: float, *, after_result: Callable[[], bool] | None = None,
              on_tick: Callable[[], None] | None = None) -> None:
    """Waits until the turn has a result or its process is gone, checking the caps, the stop taps and a shutdown
    on every tick. A cap stops the session here; a dead process ends the wait after the dead-process delay.
    A normal end (a result arrived, no cap) closes the input and gives the CLI GRACEFUL_SECONDS to exit by itself:
    no interrupt is sent, so the CLI has nothing to answer. The stop in _attempt is only the fallback.
    after_result, asked after a normal end, returns True when the session must stay open: the input is then left
    open and nothing is closed (a kept session, or a self turn). on_tick runs at the start of every tick, on the
    thread that waits; the self turn uses it to send its one interrupt."""
    while not done.wait(poll_seconds):
        turn.retry_slot()  # a turn that gave its permit back for a card takes one again as soon as one is free
        if on_tick is not None:
            on_tick()
        if live.is_aborted(thread_id):
            turn.request_stop()
        if live.is_shutting_down():
            turn.begin_stopping()
        if turn.check_caps() is not None:
            turn.begin_stopping()
            session.halt_turn()  # the bridge's process is stopped; a terminal session gets an Escape only
            return
        if turn.observe_alive(session.alive()):
            return
    turn.mark_result_received()  # already done by the bus's ResultFreezer; kept so a result never leaves it running
    if turn.limit_hit is None and (after_result is None or not after_result()):
        session.end_gracefully(GRACEFUL_SECONDS)


def finish_attempt(kit: TurnKit, session: AgentSession | None, live: LiveRegistry, *, refused: bool,
                   launch_err: str | None, stop_code: int | None, readable_tail: bool, logger: Logger,
                   terminal_busy: bool = False) -> _Attempt:
    """The facts of one launch, read after its teardown: the stop flags (consumed, so the next turn starts unmarked),
    the outcome, and the stderr tails. The caller has already stopped the session and closed its feed. A refused
    launch reads no flags. stop_code is the exit code the stop reaped, the fallback when the reader saw none.
    readable_tail is the launcher's Capabilities flag: when False the tail is not for the reply; on an error outcome
    (the only one whose reply shows the tail) it goes to the bridge log with the thread id."""
    flags = AbortFlags(False, False) if refused else live.consume_abort(kit.turn.thread_id)
    if flags.aborted:  # ⏭ Now or a ⏹ tap: the stop was requested on purpose
        kit.turn.request_stop()
    # recorded inside the wait loop only: the teardown never reclassifies a finished turn
    limit = kit.turn.limit_hit
    result = kit.collector.result
    outcome = kit.turn.finish(result, launch_failed=launch_err is not None,
                              shutting_down=live.is_shutting_down())
    stderr = session.stderr_tail(STDERR_LINES) if session is not None else ""
    tail = session.stderr_tail(TAIL_LINES).strip() if session is not None else ""
    if tail and not readable_tail:
        if outcome is TurnOutcome.ERROR:
            logger.log(f"thread {kit.turn.thread_id}: process output, kept out of the reply (not readable):\n"
                       f"{log_text(tail)}")
        tail = ""
    return _Attempt(
        refused=refused, outcome=outcome, result=result, usage=kit.usage.usage, texts=kit.collector.texts,
        tools=kit.collector.tools, limit=limit, stop_tap=flags.stop_tap, aborted=flags.aborted,
        launch_err=launch_err,
        # the reader reports the exit only after the stdout EOF, which a child can delay past the stop: the code
        # the stop reaped is the fallback, as the live file reads the process's own return code
        exit_code=kit.collector.exit_code if kit.collector.exit_code is not None else stop_code,
        tail=tail,
        resume_failed=bool(result is not None and result.resume_failed) or NO_CONVERSATION in stderr,
        feed=kit.feed, started_wall=kit.started_wall, started_mono=kit.started_mono, terminal_busy=terminal_busy,
    )


class TurnCloser:
    """Closes a turn: the reply, the feed summary and the turns row. One instance per TurnRunner."""

    def __init__(self, *, turns: TurnRepo, clock: Clock) -> None:
        self._turns = turns
        self._clock = clock

    def close(self, plan: TurnPlan, attempt: _Attempt, fresh: str, *, session_kept: bool = True) -> TurnReply:
        if attempt.refused:
            return TurnReply(None, None, attempt.outcome, terminal_busy=attempt.terminal_busy)
        answer, err = compose_reply(
            attempt.outcome, result=attempt.result, texts=attempt.texts, limit=attempt.limit,
            stop_tap=attempt.stop_tap, tools=attempt.tools, launch_err=attempt.launch_err,
            exit_code=attempt.exit_code, tail=attempt.tail, session_kept=session_kept)
        wall = self._clock.monotonic() - attempt.started_mono
        attempt.feed.finish(final_header(attempt.outcome.stored, result=attempt.result, usage=attempt.usage,
                                         tools=attempt.tools, wall=wall, limit_hit=whole_seconds(attempt.limit)))
        result, usage = attempt.result, attempt.usage
        self._turns.insert(
            plan.thread_id, started=utc_stamp(attempt.started_wall), ended=utc_stamp(self._clock.now()),
            status=attempt.outcome.stored, subtype=result.subtype if result is not None else "",
            cost=usage.cost_usd if usage is not None else None,
            cost_known=usage.cost_known if usage is not None else True,
            in_tok=usage.input_tokens if usage is not None else None,
            out_tok=usage.output_tokens if usage is not None else None,
            num_turns=result.num_turns if result is not None else None,
            duration_ms=result.duration_ms if result is not None else None,
            error=err, backend=plan.backend.name)
        if fresh:  # the user sees why this reply has no earlier context
            if err:
                err = f"{fresh}\n{err}"
            else:
                answer = f"{fresh}\n\n{answer or ''}"
        return TurnReply(answer, err, attempt.outcome)
