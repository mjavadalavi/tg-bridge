"""TurnPhase enum + transition table, ActiveTimer, the turn caps, classify() -> TurnOutcome (C14, state part).

Port of the state the live file's _run_turn_body keeps in local variables and in its wait loop: the active-time cap
(max_turn_seconds, which does not count time parked on a permission card), the hard cap (hard_cap_seconds, wall time),
the slot give-back while a card is open (slot.give_back() on entering WAITING_APPROVAL, slot.retake() on leaving), the
dead-process rule (the wrapper exited but a child keeps stdout open: 3 s without a result, then the turn is reported),
and the final classification into the outcome stored in turns.status.

The phases are a plain enum plus one transition table (TRANSITIONS); an illegal move raises IllegalTransition. The
state is driven by the turn runner (STEP 38): start() before the process is spawned, mark_running() after send_user(),
waiting_changed() from PermissionService (TurnState is a WaitObserver for its own thread id), begin_stopping() on a
cap or shutdown, request_stop() on a ⏹ tap, observe_alive() in the wait loop and finish() once the turn has ended.
Nothing here sleeps or reads the wall clock: every time read goes through the injected Clock.

Lock map: self._lock guards the phase, the timers and every flag. The slot calls run under that same lock because
Slot.give_back()/retake() are non-blocking and never call back into TurnState, so no lock order can form. Clock and
slot calls are the only outside calls made while holding it.
"""

import enum
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from tg_bridge.agents.events import TurnResult, TurnStatus
from tg_bridge.clock import Clock

DEFAULT_MAX_ACTIVE_SECONDS = 1800.0  # config max_turn_seconds: work time, parked-on-a-card time excluded
DEFAULT_HARD_CAP_SECONDS = 21600.0  # config hard_cap_seconds: wall time from the spawn
DEAD_PROCESS_SECONDS = 3.0  # a process gone this long without a TurnResult ends the turn


class TurnPhase(enum.Enum):
    STARTING = "starting"  # spawned, nothing sent yet
    RUNNING = "running"  # the model works and the active timer runs
    WAITING_APPROVAL = "waiting_approval"  # at least one card is open: the timer is paused, the slot is given back
    STOPPING = "stopping"  # a stop, a cap or a shutdown was requested; the process is being torn down
    FINISHED = "finished"  # terminal: the outcome is known


class TurnOutcome(enum.Enum):
    OK = "ok"
    ERROR = "error"
    STOPPED = "stopped"  # a person pressed stop
    INTERRUPTED = "interrupted"  # the backend ended the turn as interrupted without a stop request
    CAPPED = "capped"  # max_turn_seconds or hard_cap_seconds reached
    RESTART = "restart"  # the bridge shut down while the turn ran

    @property
    def stored(self) -> str:
        """The value written to turns.status. RESTART is stored as 'interrupted', as the live file does, so the
        turns table, /cost and the rollback path read the same words as before."""
        return "interrupted" if self is TurnOutcome.RESTART else self.value


TRANSITIONS = MappingProxyType({
    TurnPhase.STARTING: frozenset({TurnPhase.RUNNING, TurnPhase.WAITING_APPROVAL, TurnPhase.STOPPING,
                                   TurnPhase.FINISHED}),
    TurnPhase.RUNNING: frozenset({TurnPhase.WAITING_APPROVAL, TurnPhase.STOPPING, TurnPhase.FINISHED}),
    TurnPhase.WAITING_APPROVAL: frozenset({TurnPhase.RUNNING, TurnPhase.STOPPING, TurnPhase.FINISHED}),
    TurnPhase.STOPPING: frozenset({TurnPhase.FINISHED}),
    TurnPhase.FINISHED: frozenset(),
})


class IllegalTransition(RuntimeError):
    """A phase change the transition table does not allow."""


class SlotLike(Protocol):
    """The part of core/slots.py Slot that a turn uses (duck-typed so this module does not import the slot pool)."""

    def give_back(self) -> None: ...

    def retake(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class TurnFacts:
    """Everything classify() needs, gathered by TurnState.finish() or by a test."""

    launch_failed: bool = False
    stop_requested: bool = False
    shutting_down: bool = False
    limit_hit: bool = False
    result: TurnResult | None = None


def classify(facts: TurnFacts) -> TurnOutcome:
    """The outcome of a turn, in the order the live file checks it: launch failure, stop tap, shutdown, cap, then the
    result. A shutdown wins over a missing or failed result (RESTART); a successful result that arrives during a
    shutdown is still OK."""
    if facts.launch_failed:
        return TurnOutcome.ERROR
    if facts.stop_requested:
        return TurnOutcome.STOPPED
    result = facts.result
    if facts.shutting_down and (result is None or result.status is not TurnStatus.OK):
        return TurnOutcome.RESTART
    if facts.limit_hit:
        return TurnOutcome.CAPPED
    if result is None:
        return TurnOutcome.ERROR
    if result.status is TurnStatus.OK:
        return TurnOutcome.OK
    if result.status is TurnStatus.INTERRUPTED:
        return TurnOutcome.INTERRUPTED
    return TurnOutcome.ERROR


class ActiveTimer:
    """Seconds of work. The clock runs between start() and finish, and stops while pause() holds it.

    pause() and resume() are idempotent, and both do nothing before start(), so the caller can call them on every
    change of state without checking first. seconds() includes the stretch that is running now."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._total = 0.0
        self._since: float | None = None  # monotonic time the running stretch began; None while paused or unstarted
        self._started = False

    @property
    def running(self) -> bool:
        return self._since is not None

    def start(self) -> None:
        if self._started:
            raise RuntimeError("the active timer was started twice")
        self._started = True
        self._since = self._clock.monotonic()

    def pause(self) -> None:
        if self._since is None:
            return
        self._total += self._clock.monotonic() - self._since
        self._since = None

    def resume(self) -> None:
        if not self._started or self._since is not None:
            return
        self._since = self._clock.monotonic()

    def seconds(self) -> float:
        if self._since is None:
            return self._total
        return self._total + (self._clock.monotonic() - self._since)


class TurnState:
    """The state of one user turn on one topic. Thread-safe; see the module docstring for the lock map."""

    def __init__(
        self,
        thread_id: int,
        clock: Clock,
        slot: SlotLike | None = None,
        max_active_seconds: float = DEFAULT_MAX_ACTIVE_SECONDS,
        hard_cap_seconds: float = DEFAULT_HARD_CAP_SECONDS,
        dead_after_seconds: float = DEAD_PROCESS_SECONDS,
    ) -> None:
        if max_active_seconds <= 0 or hard_cap_seconds <= 0 or dead_after_seconds < 0:
            raise ValueError("turn caps must be positive and the dead-process delay must not be negative")
        self.thread_id = thread_id
        self._clock = clock
        self._slot = slot
        self._max_active = float(max_active_seconds)
        self._hard_cap = float(hard_cap_seconds)
        self._dead_after = float(dead_after_seconds)
        self._lock = threading.Lock()
        self._phase = TurnPhase.STARTING
        self._active = ActiveTimer(clock)
        self._t0: float | None = None  # monotonic time of start(); the hard cap counts from here
        self._dead_since: float | None = None
        self._stop_requested = False
        self._limit_hit: float | None = None
        self._result_received = False
        self._unslotted = False
        self._outcome: TurnOutcome | None = None

    @property
    def phase(self) -> TurnPhase:
        with self._lock:
            return self._phase

    @property
    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

    @property
    def limit_hit(self) -> float | None:
        """The cap (in seconds) this turn reached, or None."""
        with self._lock:
            return self._limit_hit

    @property
    def unslotted(self) -> bool:
        """True while the turn runs without a permit because retake() found every permit busy."""
        with self._lock:
            return self._unslotted

    @property
    def outcome(self) -> TurnOutcome | None:
        with self._lock:
            return self._outcome

    def start(self) -> None:
        """Call just before the process is spawned: starts the hard-cap clock and the active timer."""
        with self._lock:
            if self._t0 is not None:
                raise RuntimeError("the turn was started twice")
            self._t0 = self._clock.monotonic()
            self._active.start()

    def mark_running(self) -> None:
        """After send_user(). Only moves STARTING to RUNNING: a card that arrived first has already moved the turn on."""
        with self._lock:
            if self._phase is TurnPhase.STARTING:
                self._move(TurnPhase.RUNNING)

    def waiting_changed(self, thread_id: int, waiting: bool) -> bool:
        """The WaitObserver hook used by PermissionService: True when the turn is waiting for a person, False when it
        no longer is. It is level-triggered and idempotent: a repeat, a change for another thread, or a change after
        the turn is stopping does nothing and returns False. Returns True when the phase changed."""
        if thread_id != self.thread_id:
            return False
        with self._lock:
            if waiting:
                if self._phase not in (TurnPhase.STARTING, TurnPhase.RUNNING):
                    return False
                self._move(TurnPhase.WAITING_APPROVAL)
                self._active.pause()
                if self._slot is not None:
                    self._slot.give_back()  # parked on a card: another topic may use the permit
                return True
            if self._phase is not TurnPhase.WAITING_APPROVAL:
                return False
            self._move(TurnPhase.RUNNING)
            self._active.resume()
            if self._slot is not None:
                self._unslotted = not self._slot.retake()  # non-blocking: the turn runs unslotted if none is free
            return True

    def retry_slot(self) -> bool:
        """Called on every tick of the wait loop. While the turn runs without a permit (retake() found every permit
        busy when the card was answered), it tries once more without waiting, so the turn takes a permit back as soon
        as one is free. True when the turn holds one again (this call or an earlier one)."""
        with self._lock:
            if self._slot is None:
                return False
            if self._phase is not TurnPhase.RUNNING or not self._unslotted:
                return not self._unslotted
            if self._slot.retake():
                self._unslotted = False
                return True
            return False

    def mark_result_received(self) -> None:
        """Called once the TurnResult has arrived. The active timer stops here, and no cap can be recorded from now on:
        the time that follows (closing the input, stopping the process, flushing the feed) is not the model's work, so a
        turn that finished OK is never reclassified as capped."""
        with self._lock:
            self._result_received = True
            self._active.pause()

    def check_caps(self) -> float | None:
        """The cap this turn has reached, or None. Active time is checked first, then the hard cap, as in the live file.
        The result is sticky: once a cap is hit it stays recorded and is returned by later calls. No new cap is recorded
        after mark_result_received(). It does not stop the turn by itself; the caller calls begin_stopping()."""
        with self._lock:
            if self._limit_hit is None and self._t0 is not None and not self._result_received:
                if self._active.seconds() >= self._max_active:
                    self._limit_hit = self._max_active
                elif self._clock.monotonic() - self._t0 >= self._hard_cap:
                    self._limit_hit = self._hard_cap
            return self._limit_hit

    def begin_stopping(self) -> bool:
        """A cap or a shutdown: move to STOPPING. Nothing is recorded as a stop request, so the outcome is CAPPED or
        RESTART. A turn already stopping or finished is left alone (returns False)."""
        with self._lock:
            if self._phase in (TurnPhase.STOPPING, TurnPhase.FINISHED):
                return False
            self._move(TurnPhase.STOPPING)
            return True

    def request_stop(self) -> bool:
        """A ⏹ tap or /stop: the outcome becomes STOPPED. False when the turn has already finished."""
        with self._lock:
            if self._phase is TurnPhase.FINISHED:
                return False
            self._stop_requested = True
            if self._phase is not TurnPhase.STOPPING:
                self._move(TurnPhase.STOPPING)
            return True

    def observe_alive(self, alive: bool) -> bool:
        """Called from the wait loop while no TurnResult has arrived. Returns True once the process has been gone for
        dead_after_seconds: the wrapper exited but a child kept stdout open, so no EOF will come."""
        with self._lock:
            if alive:
                self._dead_since = None
                return False
            now = self._clock.monotonic()
            if self._dead_since is None:
                self._dead_since = now
            return now - self._dead_since >= self._dead_after

    def finish(self, result: TurnResult | None, *, launch_failed: bool = False,
               shutting_down: bool = False) -> TurnOutcome:
        """The turn has ended: classify it, move to FINISHED and stop the active timer. Calling it twice raises
        IllegalTransition."""
        with self._lock:
            facts = TurnFacts(launch_failed=launch_failed, stop_requested=self._stop_requested,
                              shutting_down=shutting_down, limit_hit=self._limit_hit is not None, result=result)
            self._move(TurnPhase.FINISHED)
            self._active.pause()
            self._outcome = classify(facts)
            return self._outcome

    def active_seconds(self) -> float:
        """Work time so far, parked-on-a-card time excluded (for the /status and feed lines)."""
        with self._lock:
            return self._active.seconds()

    def _move(self, to: TurnPhase) -> None:
        if to not in TRANSITIONS[self._phase]:
            raise IllegalTransition(f"turn {self.thread_id}: {self._phase.value} -> {to.value} is not allowed")
        self._phase = to
