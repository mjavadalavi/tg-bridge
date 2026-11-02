"""FakeBackend / FakeSession: a scripted AgentBackend for turn-level tests. No process, no network.

A script is a list of turns; each turn is a list of items. The n-th send_user() (on any session of the backend)
plays the next turn on a worker thread and calls the EventSink for each item:

- an AgentEvent is emitted as it is;
- an ApprovalRequest is emitted, then the worker blocks until answer() is called with that approval_id;
- WaitForInterrupt blocks until interrupt() or stop() arrives.

When interrupt() ends a turn before its TurnResult, the fake emits TurnResult(INTERRUPTED). When stop() ends it,
nothing more is emitted, as a stopped process sends nothing. A blocked wait gives up after WAIT_SECONDS with a
TurnResult(ERROR), so a broken test fails instead of hanging. When no turn is left, the fake emits
TurnResult(ERROR, error='fake script exhausted').

A Later item (last in a turn) models a background wakeup: once the turn's items are played the session is idle, and
when the test sets the Later's gate the fake emits TurnBegan and plays the Later's items as a self turn. A Later may
be followed by another Later (chained wakeups).

Callers must call wait_idle() before the next send_user(): the worker clears its busy flag after the last item.
Each session records what was asked of it in `calls`; the backend records opened specs in `specs`.
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from tg_bridge.agents.base import (
    AgentBackend,
    AgentSession,
    Capabilities,
    EventSink,
    ModeChoice,
    SessionSpec,
)
from tg_bridge.agents.events import (
    AgentEvent,
    ApprovalRequest,
    Decision,
    SessionStarted,
    TurnBegan,
    TurnResult,
    TurnStatus,
)
from tg_bridge.config import Config, LauncherSpec

WAIT_SECONDS = 10.0
GATE_POLL_SECONDS = 0.05  # how often a waiting Later re-checks its gate (Event.set() does not wake the session's condition)
DEFAULT_MODELS = ("fake-model",)
FAKE_CAPABILITIES = Capabilities(
    live_mode=True,
    live_model=True,
    questions=True,
    cost_usd=True,
    images=True,
    preassigned_id=True,
)


@dataclass(frozen=True)
class WaitForInterrupt:
    """Script item: stop here until interrupt() or stop() arrives."""


@dataclass(frozen=True)
class Later:
    """Script item, last in a turn: after the turn's items, wait for gate (bounded by WAIT_SECONDS), then emit
    TurnBegan(origin) and play items as a self turn. Nothing is played when the session was stopped first."""

    gate: threading.Event
    items: tuple["ScriptItem", ...]
    origin: str = "task-notification"


ScriptItem = AgentEvent | WaitForInterrupt | Later


def _check_turn(turn: tuple[ScriptItem, ...]) -> None:
    """A Later ends its turn: an item after it would never play, so a script like that is refused at once."""
    for index, item in enumerate(turn):
        if isinstance(item, Later):
            if index != len(turn) - 1:
                raise ValueError("a Later must be the last item of its turn")
            _check_turn(item.items)


class FakeRules:
    """RuleMatcher where a rule matches a request for the same tool name, and 'always' saves the request's hint."""

    def matches(self, rule: str, req: ApprovalRequest, cwd: str) -> bool:
        return rule == req.tool

    def rule_for(self, req: ApprovalRequest) -> str | None:
        return req.always_hint


class FakeBackend(AgentBackend):
    """A backend whose sessions play the scripted turns it was built with."""

    name = "fake"

    def __init__(self, scripts: Iterable[Iterable[ScriptItem]] = (), *,
                 capabilities: Capabilities = FAKE_CAPABILITIES,
                 models: Sequence[str] = DEFAULT_MODELS, stderr: str = "") -> None:
        self.capabilities = capabilities
        self.rules = FakeRules()
        self.specs: list[SessionSpec] = []
        self.sessions: list["FakeSession"] = []
        self._models = tuple(models)
        self.stderr = stderr
        turns = tuple(tuple(turn) for turn in scripts)
        for turn in turns:
            _check_turn(turn)
        self._turns: deque[tuple[ScriptItem, ...]] = deque(turns)
        self._lock = threading.Lock()
        self._counter = 0

    def modes(self) -> tuple[ModeChoice, ...]:
        return (ModeChoice(key="default", label="Default"),
                ModeChoice(key="plan", label="Plan", note="applies from next message"))

    def models(self, cfg: Config) -> tuple[str, ...]:
        return self._models

    def open_session(self, spec: SessionSpec, sink: EventSink) -> "FakeSession":
        with self._lock:
            self.specs.append(spec)
            session = FakeSession(self, spec, sink)
            self.sessions.append(session)
        return session

    def turns_left(self) -> int:
        with self._lock:
            return len(self._turns)

    def take_turn(self) -> tuple[ScriptItem, ...]:
        """The next scripted turn, or a single ERROR result when the script has run out."""
        with self._lock:
            if self._turns:
                return self._turns.popleft()
        return (TurnResult(TurnStatus.ERROR, answer=None, error="fake script exhausted"),)

    def next_session_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"fake-session-{self._counter}"


class FakeSession(AgentSession):
    """One fake session. start() emits SessionStarted; send_user() plays a scripted turn on a worker thread."""

    exit_code: int | None = None  # what stop() returns, as a real process session returns its return code

    def __init__(self, backend: FakeBackend, spec: SessionSpec, sink: EventSink) -> None:
        self._backend = backend
        self._spec = spec
        self._sink = sink
        self.session_id: str | None = None
        self.calls: list[tuple[Any, ...]] = []
        self.deny_messages: list[tuple[str, str]] = []  # (approval_id, reason) of each Deny that carried one
        self._cond = threading.Condition()
        self._started = False
        self._pending_started = False
        self._closed = False
        self._busy = False
        self._interrupted = False
        self._answered: set[str] = set()
        self._thread: threading.Thread | None = None

    # AgentSession interface

    def start(self) -> None:
        with self._cond:
            if self._started:
                raise RuntimeError("fake session already started")
            self._started = True
            self.calls.append(("start",))
        spec = self._spec
        self.session_id = spec.resume_id or spec.new_id or self._backend.next_session_id()
        # like the Claude adapter (system/init after the first user message), SessionStarted waits for the first turn
        # and is emitted on the turn's worker thread, never from start()
        self._pending_started = True

    def _emit_started(self) -> None:
        if self._pending_started:
            self._pending_started = False
            spec = self._spec
            self._sink(SessionStarted(session_id=self.session_id, model=spec.model, mode=spec.mode,
                                      cwd=spec.cwd, resumed=spec.resume_id is not None))

    def send_user(self, text: str, images: tuple[str, ...] = ()) -> None:
        with self._cond:
            if not self._started or self._closed:
                raise RuntimeError("fake session is not running")
            if self._busy:
                raise RuntimeError("fake session already has a turn running")
            turn = self._backend.take_turn()
            self._busy = True
            self._interrupted = False
            self.calls.append(("send_user", text, tuple(images)))
            self._thread = threading.Thread(target=self._play, args=(turn,), name="fake-turn", daemon=True)
            self._thread.start()

    def answer(self, req: ApprovalRequest, decision: Decision, *,
               updated_input: dict | None = None, answers: dict | None = None,
               deny_message: str | None = None) -> None:
        with self._cond:
            self.calls.append(("answer", req.approval_id, decision, updated_input, answers))
            if deny_message is not None:
                self.deny_messages.append((req.approval_id, deny_message))
            self._answered.add(req.approval_id)
            self._cond.notify_all()

    def interrupt(self) -> bool:
        with self._cond:
            self.calls.append(("interrupt",))
            if not self._busy:
                return False
            self._interrupted = True
            self._cond.notify_all()
            return True

    def set_mode(self, mode: str) -> bool:
        with self._cond:
            self.calls.append(("set_mode", mode))
        return self._backend.capabilities.live_mode

    def set_model(self, model: str) -> bool:
        with self._cond:
            self.calls.append(("set_model", model))
        return self._backend.capabilities.live_model

    def end_gracefully(self, timeout: float) -> bool:
        """EOF-style end: no interrupt is recorded; the fake process exits at once."""
        with self._cond:
            self.calls.append(("end_gracefully", timeout))
            self._closed = True
            self._cond.notify_all()
        return True

    def stop(self, grace: float = 5.0) -> int | None:
        with self._cond:
            self.calls.append(("stop", grace))
            self._closed = True
            self._cond.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=grace)
        return self.exit_code

    def alive(self) -> bool:
        with self._cond:
            return self._started and not self._closed

    def stderr_tail(self, n: int = 20) -> str:
        return "\n".join(self._backend.stderr.splitlines()[-n:])

    # test helpers

    def wait_idle(self, timeout: float = WAIT_SECONDS) -> bool:
        """True once no turn is running, False if one still runs after timeout seconds."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    # worker thread

    def _play(self, items: tuple[ScriptItem, ...]) -> None:
        # owns_busy: this thread set the busy flag and must clear it. A waiting Later does not own it (a user turn may
        # run meanwhile), so its timeout or stop must not clear the flag of that other turn.
        owns_busy = True
        try:
            self._emit_started()
            while True:
                later = self._play_turn(items)
                self._mark_idle()
                owns_busy = False
                if later is None or not self._await_later(later):
                    return
                owns_busy = True
                items = later.items
        except TimeoutError as exc:
            if not self._closed:
                self._sink(TurnResult(TurnStatus.ERROR, answer=None, error=str(exc)))
        finally:
            if owns_busy:
                self._mark_idle()

    def _play_turn(self, items: tuple[ScriptItem, ...]) -> Later | None:
        """Plays one turn's items. Returns the trailing Later when the turn ended normally, else None."""
        result_seen = False
        later = None
        for item in items:
            if isinstance(item, Later):
                later = item
                break
            if isinstance(item, WaitForInterrupt):
                self._block(self._ended)
            else:
                self._sink(item)
                result_seen = result_seen or isinstance(item, TurnResult)
                if isinstance(item, ApprovalRequest):
                    approval_id = item.approval_id
                    self._block(lambda: approval_id in self._answered or self._ended())
            if self._ended():
                break
        if not result_seen and self._interrupted and not self._closed:
            self._sink(TurnResult(TurnStatus.INTERRUPTED, answer=None, subtype="interrupted"))
        return None if self._ended() else later

    def _await_later(self, later: Later) -> bool:
        """Waits until the gate is set and no user turn runs, then marks the session busy and emits TurnBegan.
        False when the session was stopped first (nothing plays). Raises TimeoutError after WAIT_SECONDS.

        A gate that is never set ends the wait with an ERROR TurnResult that has no TurnBegan before it, while the
        session is idle. The real session has no such result; a test that forgets to set its gate sees it instead of
        hanging. Only the thread that set busy clears it (see _play), so a timed-out or stopped Later leaves the busy
        flag of a user turn that started meanwhile alone."""
        deadline = time.monotonic() + WAIT_SECONDS
        with self._cond:
            while not self._closed and (not later.gate.is_set() or self._busy):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"fake session waited more than {WAIT_SECONDS:g}s for a later turn")
                self._cond.wait(min(remaining, GATE_POLL_SECONDS))
            if self._closed:
                return False
            self._busy = True
            self._interrupted = False
        self._sink(TurnBegan(origin=later.origin))
        return True

    def _mark_idle(self) -> None:
        with self._cond:
            self._busy = False
            self._cond.notify_all()

    def _ended(self) -> bool:
        return self._interrupted or self._closed

    def _block(self, ready: Callable[[], bool]) -> None:
        deadline = time.monotonic() + WAIT_SECONDS
        with self._cond:
            while not ready():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"fake session waited more than {WAIT_SECONDS:g}s")
                self._cond.wait(remaining)


class RecordingSink:
    """An EventSink that keeps every event in arrival order. wait_for() blocks until a matching event arrives."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self._cond = threading.Condition()

    def __call__(self, event: AgentEvent) -> None:
        with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    def wait_for(self, predicate: Callable[[AgentEvent], bool], timeout: float = WAIT_SECONDS) -> bool:
        """True once an event matching predicate has been recorded, False after timeout seconds."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while not any(predicate(event) for event in self.events):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True


def fake_spec(**overrides: Any) -> SessionSpec:
    """A SessionSpec with fake defaults, for tests. Keyword arguments replace individual fields."""
    launcher = LauncherSpec(name="fake", backend="fake", command="fake", argv=("fake",), config_dir="/nonexistent")
    values: dict[str, Any] = {
        "thread_id": 1,
        "chat_id": -100,
        "topic": "fake topic",
        "launcher": launcher,
        "cwd": "/nonexistent",
        "mode": "default",
        "model": None,
        "resume_id": None,
        "new_id": None,
        "allow_rules": (),
        "deny_tools": (),
        "system_rules": "",
        "env": {},
        "errfile_dir": "/nonexistent",
    }
    values.update(overrides)
    return SessionSpec(**values)
