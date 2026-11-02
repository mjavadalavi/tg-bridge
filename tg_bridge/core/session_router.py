"""SessionRouter: the one sink of a Claude session for its whole life (DESIGN section 3, linger).

Until now a session's sink was the EndGate of the turn in progress, so everything after the turn's end was dropped. A
kept session (linger) also talks between turns: a background subagent asks for approval, the CLI starts a self turn by
itself (a task notification or a wakeup). The router is the sink for the whole life of the session and decides, per
event, where it goes:

- BackgroundWork sets the pending count and TurnBegan opens a self turn; neither one is delivered to a listener. A
  TurnBegan is kept in the queue as a mark: the events after it belong to that self turn, and one attach() takes exactly
  one self turn, so a second self turn that started meanwhile waits behind the first.
- While a turn's gate is attached, events go to that gate, as they did through EndGate: the first TurnResult or
  ProcessExited detaches the gate, and the events after it are dropped and logged once per idle stretch.
- While the session is held or armed and idle, the session events (approvals, session start) go to the session sink.
- While a self turn is pending and no gate is attached, events wait in the queue, in order, until the runner thread
  attaches its gate, which then receives them first. A TurnBegan opens a self turn when the session is held, or armed
  (arm(): the session may be kept after its turn, so a self turn that starts before hold() is not lost either). A
  TurnBegan also opens one while the attached gate is ending (its TurnResult or ProcessExited is queued but not yet
  delivered by the thread that drains): the self turn waits behind that end.
- A user turn's attach (user=True) is refused with SELF_FIRST while a self turn waits, so the user's gate never takes a
  self turn's events. The caller runs the waiting self turn first and attaches again.

Threads: the backend reader, tail and hook threads call __call__. The queue is drained by one thread at a time, and
every delivery runs after the lock is released, so the listeners see one strict order and never run under the lock.
attach() drains the queue on the calling thread unless another thread is draining it already.

Lock map: _lock guards every field of the router and the queue. Nothing is called while it is held: logging, the
listeners (gate, session sink, activity hook) all run after the release.
"""

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from tg_bridge.agents.events import (
    AgentEvent,
    ApprovalRequest,
    ApprovalWithdrawn,
    BackgroundWork,
    ProcessExited,
    SessionStarted,
    TurnBegan,
    TurnResult,
)
from tg_bridge.logging_ import Logger

SESSION_EVENTS = (ApprovalRequest, ApprovalWithdrawn, SessionStarted)   # what a held, idle session sends on
MAX_BUFFERED_EVENTS = 5000    # events kept while a self turn is pending (the turn marks do not count); beyond this the
                              # oldest event is dropped

Listener = Callable[[AgentEvent], None]


class Attach(Enum):
    """The result of attach(). Only ATTACHED is true."""

    ATTACHED = "attached"
    REFUSED = "refused"  # the router is closed, or a gate is attached already
    SELF_FIRST = "self-first"  # a user turn asked, and a self turn waits for its gate: run that one first

    def __bool__(self) -> bool:
        return self is Attach.ATTACHED


@dataclass(frozen=True)
class _TurnMark:
    """A TurnBegan in the queue: the events queued after it are one self turn's, up to that turn's result."""

    origin: str


class SessionRouter:
    """The sink of one Claude session. One instance is created when the session starts and lives as long as it does."""

    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._lock = threading.Lock()
        self._queue: deque[AgentEvent | _TurnMark] = deque()
        self._draining = False
        self._gate: Listener | None = None
        self._gate_ending = False  # the attached gate's TurnResult or ProcessExited is queued, not yet delivered
        self._closed = False
        self._held = False
        self._armed = False
        self._session_sink: Listener | None = None
        self._on_activity: Callable[[], None] | None = None
        self._turns_waiting = 0  # the turn marks in the queue: self turns that wait for their gate
        self._pending = 0
        self._activity = False
        self._exited = False
        self._drop_logged = False
        self._close_logged = False
        self._overflow_logged = False

    def attach(self, gate: Listener, user: bool = False) -> Attach:
        """A turn's sink (its EndGate). The events a pending self turn buffered are delivered to the gate, in order,
        before anything newer. user=True is a user turn: while a self turn waits for its gate, nothing is attached and
        SELF_FIRST is returned, so the self turn runs first. A self turn's own attach passes user=False."""
        with self._lock:
            if self._closed or self._gate is not None:
                return Attach.REFUSED
            if user and self._turns_waiting:
                return Attach.SELF_FIRST
            self._gate = gate
            self._gate_ending = False
            self._activity = False
            self._drop_logged = False
            self._overflow_logged = False
            start = not self._draining
            if start:
                self._draining = True
        if start:
            self._drain()
        return Attach.ATTACHED

    def arm(self, session_sink: Listener) -> None:
        """The session may be kept after its turn: from now on a TurnBegan opens a self turn, as when held, and an idle
        approval or session start goes to session_sink, as when held. session_sink is the sink hold() installs. Called
        before the turn's gate attaches, so a self turn that starts between the turn's result and hold() is buffered,
        not lost, and an approval asked in that window gets its card (voided when the session is not kept)."""
        with self._lock:
            if not self._closed:
                self._armed = True
                self._session_sink = session_sink

    def hold(self, session_sink: Listener, on_activity: Callable[[], None]) -> None:
        """The session is kept: idle session events go to session_sink, and TurnBegan opens a pending self turn.
        on_activity is called (on the thread that saw the event) whenever a background or self-turn event arrives."""
        with self._lock:
            if self._closed:
                return
            self._held = True
            self._session_sink = session_sink
            self._on_activity = on_activity

    def close(self) -> None:
        """Ends the router. Everything after this call is dropped; the first drop is logged."""
        with self._lock:
            self._closed = True
            self._held = False
            self._gate = None
            self._session_sink = None
            self._on_activity = None
            self._armed = False
            self._gate_ending = False
            self._turns_waiting = 0
            self._queue.clear()

    @property
    def pending(self) -> int:
        """The latest BackgroundWork count: the background tasks and wakeups the session is waiting for."""
        with self._lock:
            return self._pending

    @property
    def activity(self) -> bool:
        """True when a BackgroundWork arrived since the last attach()."""
        with self._lock:
            return self._activity

    @property
    def exited(self) -> bool:
        """True once a ProcessExited was seen."""
        with self._lock:
            return self._exited

    @property
    def self_pending(self) -> bool:
        """True while a self turn waits for its gate: one has begun and no gate has taken it yet (a second one may wait
        behind the first)."""
        with self._lock:
            return self._turns_waiting > 0

    @property
    def self_turn_complete(self) -> bool:
        """True when the first waiting self turn has its TurnResult buffered, so it can run to its reply with no more
        events from the process. False when no self turn waits or its result has not arrived. When the buffer overflows
        the oldest event may be a dropped TurnResult: the property then reads False, never True for a later turn, since
        a turn mark is never dropped and the scan stops at the next mark."""
        with self._lock:
            marked = False
            for item in self._queue:
                if isinstance(item, _TurnMark):
                    if marked:
                        return False
                    marked = True
                elif marked and isinstance(item, TurnResult):
                    return True
            return False

    def __call__(self, event: AgentEvent) -> None:
        notes: list[str] = []
        hook: Callable[[], None] | None = None
        start = False
        with self._lock:
            if self._closed:
                if not self._close_logged:
                    self._close_logged = True
                    notes.append(f"session router closed: dropping a later {type(event).__name__}")
            elif isinstance(event, BackgroundWork):
                self._pending = event.pending
                self._activity = True
                hook = self._on_activity
            elif isinstance(event, TurnBegan):
                if (self._held or self._armed) and (self._gate is None or self._gate_ending):
                    self._queue.append(_TurnMark(event.origin))
                    self._turns_waiting += 1
                    hook = self._on_activity
                    notes.append(f"self turn began ({event.origin or 'no origin'}): its events wait for a gate")
            else:
                if isinstance(event, (TurnResult, ProcessExited)) and self._gate is not None:
                    self._gate_ending = True  # the gate's end is queued: a self turn that begins now waits behind it
                if isinstance(event, ProcessExited):
                    self._exited = True
                    hook = self._on_activity
                self._queue.append(event)
                if self._turns_waiting and len(self._queue) - self._turns_waiting > MAX_BUFFERED_EVENTS:
                    self._drop_oldest_event()
                    if not self._overflow_logged:
                        self._overflow_logged = True
                        notes.append("session router buffer full: dropping the oldest buffered event")
                start = not self._draining
                if start:
                    self._draining = True
        for line in notes:
            self._logger.log(line)
        if hook is not None:
            hook()
        if start:
            self._drain()

    def _drain(self) -> None:
        """Delivers the queue in order until it is empty or a self turn waits for its gate. Runs on the calling thread,
        and only one thread drains at a time (the _draining flag, cleared under the lock on the way out). A turn mark
        with no gate attached stops the drain; with a gate attached the mark is consumed: the gate takes that self turn."""
        while True:
            with self._lock:
                if not self._queue or (self._gate is None and isinstance(self._queue[0], _TurnMark)):
                    self._draining = False
                    return
                item = self._queue.popleft()
                if isinstance(item, _TurnMark):
                    self._turns_waiting -= 1
                    continue
                target, notes = self._route(item)
            for line in notes:
                self._logger.log(line)
            if target is None:
                continue
            try:
                target(item)
            except Exception as exc:
                self._logger.log(f"session router: a listener raised {type(exc).__name__}: {exc}")

    def _drop_oldest_event(self) -> None:
        """Drops the oldest buffered event. A turn mark stays in the queue, so two self turns are never merged. Called
        under the lock."""
        for index, item in enumerate(self._queue):
            if not isinstance(item, _TurnMark):
                del self._queue[index]
                return

    def _route(self, event: AgentEvent) -> tuple[Listener | None, list[str]]:
        """Where one queued event goes (None: dropped). Called under the lock; the caller logs and delivers after it."""
        if self._gate is not None:
            gate = self._gate
            if isinstance(event, (TurnResult, ProcessExited)):
                self._gate = None
                self._drop_logged = False
            return gate, []
        if (self._held or self._armed) and isinstance(event, SESSION_EVENTS):
            return self._session_sink, []
        if self._drop_logged:
            return None, []
        self._drop_logged = True
        return None, [f"turn ended: ignoring a later {type(event).__name__} from the process"]
