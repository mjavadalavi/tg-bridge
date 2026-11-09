"""LingerService: keeps a topic's session open after a turn while background work is pending (linger).

A turn that starts background work (a Monitor, a background command or agent, a wakeup) leaves its process running
after the result. The service decides when that process is kept (offer), hands it to the next user message (claim),
runs the self turns the CLI starts by itself (wait_once) and closes it on the first end condition (release). The
kept topic's runner thread holds the session: the dispatcher calls wait_once while holding() is True.

Record states (HeldSession.state): TURN (a user turn runs), IDLE (no turn; background work may be pending), SELF (a self
turn runs), CLOSING (a release is in progress; the record is already out of the map). A shutdown leaves the kv marker and
does not stop the process: the lifecycle snapshot does that and the restart notice reads the marker.

End conditions are the HoldEnd values: SHUTDOWN, STOPPED, EXITED, CHANGED, CAP, DONE. The check order is not the enum
order. Each tick (wait_once, and the checks before each self turn in claim and resume) takes the early ends first:
STOPPED (an answered /stop) and CAP (the deadline passed). They win even over a pending self turn, so a self turn that
was already answered as closed never runs, and a stream of self turns cannot hold the session past its cap. Then a
pending self turn runs. Only with no self turn pending are the other ends checked, in enum order. So a tick applies
STOPPED, CAP, EXITED, CHANGED, DONE, with SHUTDOWN ahead of EXITED when no self turn is pending (a shutdown skips the
early check). A session that has also exited or changed topic while a self turn is pending runs that self turn first: it
ends as SELF_ENDED when the turn does not keep it, or as STOPPED or CAP when one of those early ends holds, not as
EXITED or CHANGED.

Threads: offer, claim, resume, wait_once, forget and every release run on the topic's runner thread. holding, idle_alive,
wake, request_release, request_interrupt, status and holds run on the bot and callback threads; the router calls the
wake hook on the reader thread. The record's services (session, router, permissions, kv, view, SelfTurns) are always
called with no lock held.

Lock map: _lock guards the _held map and the fields other threads read (state, deadline, release, wake, interrupt,
router, hold_id, self_seq) and the hold counter. The runner thread alone writes the other record fields, so it reads
them without the lock.
"""

import json
import threading
from dataclasses import replace

from tg_bridge.agents.base import AgentSession
from tg_bridge.agents.events import TurnResult, TurnStatus
from tg_bridge.clock import Clock, utc_stamp
from tg_bridge.config import Config, LauncherError
from tg_bridge.core.events_bus import TurnEventBus
from tg_bridge.core.linger_types import (
    KEEP_OUTCOMES,
    LINGER_CAP_TEXT,
    LINGER_EXIT_TEXT,
    LINGER_PREFIX,
    LINGER_QUIET_SECONDS,
    POLL_SECONDS,
    RELEASE_GRACE_SECONDS,
    HeldSession,
    HoldEnd,
    HoldResult,
    HoldState,
    LingerControl,
    LingerOffer,
    LingerStatus,
)
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.permissions import PermissionService
from tg_bridge.core.self_turn import SelfTurnContext, SelfTurns
from tg_bridge.core.topic_key import TopicKey, topic_key
from tg_bridge.core.view_recorder import STATE_IDLE, STATE_LINGERING
from tg_bridge.logging_ import Logger
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.sessions import SessionRepo

# the names below are defined in linger_types and re-exported here, so importers keep using tg_bridge.core.linger
__all__ = (
    "KEEP_OUTCOMES", "LINGER_CAP_TEXT", "LINGER_EXIT_TEXT", "LINGER_PREFIX", "LINGER_QUIET_SECONDS", "POLL_SECONDS",
    "RELEASE_GRACE_SECONDS", "HeldSession", "HoldEnd", "HoldResult", "HoldState", "LingerControl", "LingerOffer",
    "LingerStatus", "LingerService", "linger_key",
)


def linger_key(thread_id: int) -> str:
    """The kv key of a kept session of one topic."""
    return f"{LINGER_PREFIX}{thread_id}"


class LingerService:
    """Holds the kept sessions, one per topic. One instance per App, shared by its services."""

    def __init__(self, *, config: Config, sessions: SessionRepo, kv: KvRepo, live: LiveRegistry,
                 permissions: PermissionService, self_turns: SelfTurns, clock: Clock, logger: Logger,
                 poll_seconds: float = POLL_SECONDS) -> None:
        self._config = config
        self._sessions = sessions
        self._kv = kv
        self._live = live
        self._permissions = permissions
        self._self_turns = self_turns
        self._clock = clock
        self._logger = logger
        self._poll_seconds = poll_seconds
        self._lock = threading.Lock()
        self._held: dict[int, HeldSession] = {}
        self._holds_offered = 0  # the accepted offers so far: the source of each hold_id (guarded by _lock)

    # runner thread

    def offer(self, o: LingerOffer) -> bool:
        """Keeps the session of a user turn that just ended, when every condition holds. False (with the first failing
        reason logged) when it is not kept; the caller then stops the session as before."""
        thread = o.plan.thread_id
        reason = self._refusal(o)
        if reason is not None:
            self._logger.log(f"linger: thread {thread} not kept: {reason}")
            return False
        if not self._live.begin_linger(thread, o.session, self._config.linger_max_sessions()):
            self._logger.log(f"linger: thread {thread} not kept: the session cap is full")
            return False
        seconds = self._config.linger_seconds()
        now = self._clock.monotonic()
        plan = replace(o.plan, text="")
        key = TopicKey(launcher_name=plan.launcher.name, launcher_command=plan.launcher.command, cwd=plan.cwd,
                       mode=plan.mode, model=plan.model)
        pending = o.router.pending
        with self._lock:
            self._holds_offered += 1
            hold_id = self._holds_offered
            held = self._held.get(thread)
            if held is None:
                held = HeldSession(thread_id=thread, chat_id=plan.chat_id, topic=plan.topic, plan=plan, key=key,
                                   session=o.session, router=o.router, binding=o.binding, recorder=o.recorder,
                                   view=o.kit.view, deadline=now + seconds, state=HoldState.IDLE, quiet_since=None,
                                   release=False, wake=threading.Event(), interrupt=None, hold_id=hold_id, self_seq=0)
                self._held[thread] = held
            else:  # a later user turn on the kept session: the deadline restarts, the session and binding stay
                held.chat_id, held.topic, held.plan, held.key = plan.chat_id, plan.topic, plan, key
                held.recorder, held.view = o.recorder, o.kit.view
                held.deadline, held.state, held.quiet_since, held.release = now + seconds, HoldState.IDLE, None, False
                held.hold_id, held.self_seq = hold_id, 0
        o.router.hold(TurnEventBus([self._permissions.bind_turn(o.binding), o.recorder], self._logger).publish,
                      on_activity=held.wake.set)
        self._kv.set(linger_key(thread), json.dumps({"chat_id": plan.chat_id, "name": plan.topic,
                                                     "since": utc_stamp(self._clock.now())}))
        if o.kit.view is not None:
            o.kit.view.set_state(STATE_LINGERING)
        self._logger.log(f"linger: thread {thread} kept ({pending} pending, closes in {seconds:.0f} s)")
        return True

    def holds(self, thread_id: int, session: AgentSession) -> bool:
        """True when this topic's kept record holds this very session."""
        with self._lock:
            held = self._held.get(thread_id)
            return held is not None and held.session is session

    def claim(self, thread_id: int, key: TopicKey) -> HeldSession | None:
        """Hands the kept session to a user turn with this key (a pending self turn runs first); None when released."""
        held = self._get(thread_id)
        if held is None:
            return None
        if self._run_waiting(held) is not None:
            return None
        reason = self._end_reason(held, key, dead_counts=True)
        if reason is not None:
            self._release(held, reason)
            return None
        with self._lock:  # the release flag is read in the same block that sets TURN, so a request that landed since
            requested = held.release  # the end check above is not lost: a request_release sees IDLE only until this block
            if not requested:
                held.state = HoldState.TURN
        if requested:
            self._release(held, HoldEnd.STOPPED)
            return None
        return held

    def resume(self, thread_id: int) -> bool:
        """Runs the self turns that began after the claim, before the user turn's gate could attach (the turn's start
        and the attach are not one step), then makes the record a TURN again. False when the record was released."""
        held = self._get(thread_id)
        if held is None or self._run_waiting(held) is not None:
            return False
        with self._lock:  # the same release check as claim: a request that landed since the self turns is not lost
            requested = held.release
            if not requested:
                held.state = HoldState.TURN
        if requested:
            self._release(held, HoldEnd.STOPPED)
            return False
        return True

    def wait_once(self, thread_id: int) -> HoldResult:
        """One hold tick. The early ends come first: STOPPED (answered /stop) and CAP (the passed deadline), before any
        self turn, because they win even over a pending one. Then a pending self turn runs. Only with no self turn
        pending are SHUTDOWN, EXITED, CHANGED and DONE checked, in that order; else the tick waits for a wake-up. So a
        tick applies STOPPED, CAP, EXITED, CHANGED, DONE, with SHUTDOWN ahead of EXITED when no self turn is pending
        (during a shutdown the early check is skipped)."""
        held = self._get(thread_id)
        if held is None:
            return HoldResult(HoldEnd.DONE, 0, "")
        early = self._early_end(held)
        if early is not None:
            pending = held.router.pending
            self._release(held, early)
            return HoldResult(early, pending, held.chat_id)
        if held.router.self_pending:
            end = self._run_self(held)
            return HoldResult(end, held.router.pending, held.chat_id)
        self._track_quiet(held)
        reason = self._end_reason(held, None, dead_counts=False)
        if reason is not None:
            pending = held.router.pending
            self._release(held, reason)
            return HoldResult(reason, pending, held.chat_id)
        held.wake.wait(self._poll_seconds)
        held.wake.clear()  # a wake-up set after the wait is not lost: the next take_head sees the message anyway
        return HoldResult(None, held.router.pending, held.chat_id)

    def forget(self, thread_id: int, session: AgentSession) -> None:
        """Drops the record when it holds this session; the caller stops the session and voids the binding."""
        with self._lock:
            held = self._held.get(thread_id)
            if held is not None and held.session is session:
                del self._held[thread_id]
                held.state = HoldState.CLOSING
            else:
                held = None
        if held is not None:
            held.router.close()
        self._live.end_linger(thread_id, session)
        if not self._live.is_shutting_down():
            self._kv.delete(linger_key(thread_id))

    # any thread

    def holding(self, thread_id: int) -> bool:
        """True while a record exists for the topic (any state but CLOSING, which is never in the map)."""
        with self._lock:
            return thread_id in self._held

    def idle_alive(self, thread_id: int) -> bool:
        """True when the kept session is idle with no self turn pending: a message can be handed over quietly."""
        with self._lock:
            held = self._held.get(thread_id)
            if held is None or held.state is not HoldState.IDLE:
                return False
            router = held.router
        return not router.self_pending

    def wake(self, thread_id: int) -> None:
        with self._lock:
            held = self._held.get(thread_id)
            event = held.wake if held is not None else None
        if event is not None:
            event.set()

    def request_release(self, thread_id: int) -> bool:
        """Asks an idle kept session to close (/stop, the status ⏹); False when nothing is kept or idle."""
        with self._lock:
            held = self._held.get(thread_id)
            if held is None or held.state is not HoldState.IDLE:
                return False
            held.release = True
            event = held.wake
        event.set()
        return True

    def request_interrupt(self, thread_id: int) -> bool:
        """Interrupts the running self turn only (it ends INTERRUPTED and the session stays kept); False if none runs."""
        with self._lock:
            held = self._held.get(thread_id)
            if held is None or held.state is not HoldState.SELF or held.interrupt is None:
                return False
            event = held.interrupt
        event.set()
        return True

    def status(self, thread_id: int) -> LingerStatus | None:
        """What /status shows for a kept topic; None when nothing is kept."""
        with self._lock:
            held = self._held.get(thread_id)
            if held is None:
                return None
            deadline, in_self, router = held.deadline, held.state is HoldState.SELF, held.router
            hold_id, self_seq = held.hold_id, held.self_seq
        return LingerStatus(pending=router.pending, seconds_left=max(0.0, deadline - self._clock.monotonic()),
                            self_turn=in_self or router.self_pending, hold_id=hold_id, self_seq=self_seq)

    def in_self_turn(self, thread_id: int, hold_id: int, self_seq: int) -> bool:
        """True while this hold's self turn number self_seq runs (state SELF). A stop choice checks it before it acts."""
        with self._lock:
            held = self._held.get(thread_id)
            return (held is not None and held.state is HoldState.SELF
                    and held.hold_id == hold_id and held.self_seq == self_seq)

    # internals (runner thread unless stated)

    def _get(self, thread_id: int) -> HeldSession | None:
        with self._lock:
            return self._held.get(thread_id)

    def _refusal(self, o: LingerOffer) -> str | None:
        """The first condition that refuses the offer (its words for the log); None when it may be kept."""
        thread = o.plan.thread_id
        kit = o.kit
        if not o.plan.backend.capabilities_for(o.plan.launcher).linger:
            return "the backend cannot keep a session open"
        if self._config.linger_seconds() <= 0 or self._config.linger_max_sessions() <= 0:
            return "linger is off in the config"
        if self._live.is_shutting_down():
            return "the bridge is shutting down"
        if kit.turn.limit_hit is not None or kit.turn.stop_requested or self._live.is_aborted(thread):
            return "the turn hit a cap or was stopped"
        result = kit.collector.result
        if not isinstance(result, TurnResult) or result.status is not TurnStatus.OK:
            return "the turn did not end OK"
        if not o.session.alive() or o.router.exited:
            return "the process is gone"
        if o.binding.plan_approved:  # also an end of an existing hold (_end_reason), for a plan approved in a self turn
            return "a plan was approved: its CLI is still in plan mode, the next turn starts a fresh one"
        if o.router.pending <= 0 and not o.router.activity and not o.router.self_pending:
            return "no background work is pending"
        with self._lock:
            held = self._held.get(thread)
            if held is not None and held.router is not o.router:
                return "the kept session has another router"
        return None

    def _early_end(self, held: HeldSession) -> HoldEnd | None:
        """The end that comes before any self turn: a release already requested (STOPPED) or the deadline passed (CAP).
        None during a shutdown, which keeps its order (SHUTDOWN is decided by _end_reason, without a stop)."""
        if self._live.is_shutting_down():
            return None
        with self._lock:
            requested, deadline = held.release, held.deadline
        if requested:
            return HoldEnd.STOPPED
        if self._clock.monotonic() >= deadline:
            return HoldEnd.CAP
        return None

    def _run_waiting(self, held: HeldSession) -> HoldEnd | None:
        """Runs every self turn that waits for its gate, checking the early end before each one. None when the record is
        kept and idle, else the end it was released with."""
        while True:  # the end check comes before each self turn: a buffered self turn never runs after its end
            early = self._early_end(held)
            if early is not None:
                self._release(held, early)
                return early
            if not held.router.self_pending:
                return None
            end = self._run_self(held)  # each _run_self consumes one turn mark
            if end is not None:
                return end

    def _run_self(self, held: HeldSession) -> HoldEnd | None:
        """Runs the pending self turn here; None when kept (IDLE again), else released and the end returned. Once the
        router has seen the ProcessExited, the drain before it is done, so an exited process runs only a complete self
        turn (its TurnResult is buffered): an incomplete one is not waited for. Before that the process may still be
        draining its last lines (the process is reaped before the drain), so the self turn runs and its wait ends with the
        ProcessExited that follows the drain. A complete self turn that ends with the process gone keeps the hold for the
        next complete one, so two buffered updates both run."""
        if held.router.exited and not held.router.self_turn_complete:
            end_kind = HoldEnd.SHUTDOWN if self._live.is_shutting_down() else HoldEnd.EXITED
            self._release(held, end_kind)
            return end_kind
        interrupt = threading.Event()
        with self._lock:
            held.state = HoldState.SELF
            held.interrupt = interrupt
            held.self_seq += 1
        ctx = SelfTurnContext(plan=held.plan, session=held.session, router=held.router, binding=held.binding,
                              recorder=held.recorder, interrupt=interrupt)
        try:
            end = self._self_turns.run(ctx)
        except Exception as exc:
            self._logger.log(f"linger: thread {held.thread_id} self turn raised {type(exc).__name__}: {exc}")
            end = None
        more = held.router.self_turn_complete and not self._live.is_shutting_down()  # a complete update still buffered
        if end is not None and end.ran and end.outcome in KEEP_OUTCOMES and (end.alive or more):
            now = self._clock.monotonic()
            pending = held.router.pending
            with self._lock:  # self turns never move the deadline: only the user turn's offer does
                held.state = HoldState.IDLE
                held.interrupt = None
                held.quiet_since = None if pending > 0 else now
            if held.view is not None:
                held.view.set_state(STATE_LINGERING)
            return None
        if self._live.is_shutting_down():
            end_kind = HoldEnd.SHUTDOWN
        elif end is not None and end.ran and end.outcome in KEEP_OUTCOMES:  # the reply was posted; the process then left
            end_kind = HoldEnd.EXITED
        else:
            end_kind = HoldEnd.SELF_ENDED
        self._release(held, end_kind)
        return end_kind

    def _track_quiet(self, held: HeldSession) -> None:
        """Starts the quiet window at the first tick that sees no pending work, and clears it when work is pending."""
        pending = held.router.pending
        now = self._clock.monotonic()
        with self._lock:
            if pending > 0:
                held.quiet_since = None
            elif held.quiet_since is None:
                held.quiet_since = now

    def _end_reason(self, held: HeldSession, key: TopicKey | None, *, dead_counts: bool) -> HoldEnd | None:
        """The first end condition that holds, in HoldEnd order. key is a claiming turn's key; None reads it now.
        The process is gone once its ProcessExited was seen. The reader reaps the process before it drains the last
        lines and emits that event, so a dead process that has not sent it yet is still draining: a tick waits for it
        (dead_counts False), while a claiming turn cannot use a dead process and ends the hold at once (dead_counts)."""
        if self._live.is_shutting_down():
            return HoldEnd.SHUTDOWN
        with self._lock:
            requested, deadline, quiet_since = held.release, held.deadline, held.quiet_since
        if requested:
            return HoldEnd.STOPPED
        if held.router.exited or (dead_counts and not held.session.alive()):
            return HoldEnd.EXITED
        row = self._sessions.get(held.thread_id)
        if row is None or row.uuid != held.recorder.known:
            return HoldEnd.CHANGED
        try:
            current = key if key is not None else topic_key(self._config, row, held.topic)
        except LauncherError:
            return HoldEnd.CHANGED
        if current != held.key or held.binding.plan_approved:  # an approved plan's CLI is still in plan mode
            return HoldEnd.CHANGED
        now = self._clock.monotonic()
        if now >= deadline:
            return HoldEnd.CAP
        if held.router.pending == 0 and quiet_since is not None and now - quiet_since >= LINGER_QUIET_SECONDS:
            return HoldEnd.DONE
        return None

    def _release(self, held: HeldSession, end: HoldEnd) -> None:
        """Closes a kept session: the record leaves the map, the router closes, then the process stops (not on a
        shutdown, the lifecycle snapshot stops it), the cards are voided and the marker removed."""
        thread = held.thread_id
        with self._lock:
            held.state = HoldState.CLOSING
            if self._held.get(thread) is held:
                del self._held[thread]
        held.router.close()
        try:
            if end is not HoldEnd.SHUTDOWN:
                try:
                    held.session.stop(grace=RELEASE_GRACE_SECONDS)
                finally:  # a stop that raises still voids the cards and removes the marker
                    self._permissions.void_all(held.binding)
                    self._kv.delete(linger_key(thread))
        finally:
            self._live.end_linger(thread, held.session)
            if held.view is not None:
                held.view.set_state(STATE_IDLE)
            self._logger.log(f"linger: thread {thread} released ({end.value})")
