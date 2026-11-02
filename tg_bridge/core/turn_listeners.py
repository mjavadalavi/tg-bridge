"""The listeners of one turn's event bus besides the feed and the permissions (DESIGN section 2, Observer).

UsageRecorder keeps the last usage of the turn (cumulative, so the last one wins). SessionRecorder stores the session id
the first time it is known (sessions.uuid and session_history) and the CLI's slash commands (kv slash:<thread>).
AnswerCollector gathers the top-level answer texts, counts tool calls, and signals the wait loop when the turn has ended.
WaitRouter is the WaitObserver the PermissionService is built with: it forwards a thread's waiting changes to the
TurnState of the turn running on that thread.

Threads: every listener's on_event runs on the backend reader thread. Each one guards its state with its own lock, and
none of them calls another service while holding it.
"""

import json
import threading
from collections.abc import Callable

from tg_bridge.agents.events import (
    AgentEvent,
    ProcessExited,
    SessionStarted,
    TextEvent,
    ToolStart,
    TurnResult,
    UsageEvent,
)
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.permissions import WaitObserver
from tg_bridge.logging_ import Logger
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.sessions import HistoryRepo, SessionRepo

SLASH_PREFIX = "slash:"


def slash_key(thread_id: int) -> str:
    """The kv key of a topic's CLI slash commands (read by /help)."""
    return f"{SLASH_PREFIX}{thread_id}"


class EndGate:
    """The sink of one attempt, in front of the bus: it passes events on until the attempt's first end (a TurnResult
    or a ProcessExited) and drops everything after it.

    A CLI can answer a stop with a second result (an interrupt of a finished turn gets error_during_execution, with no
    usage). Without this gate that answer would replace the real usage and clear the feed's held answer, so the turn
    would store zero tokens and post its answer twice. The drop is logged once per attempt.
    """

    def __init__(self, deliver: Callable[[AgentEvent], None], logger: Logger) -> None:
        self._deliver = deliver
        self._logger = logger
        self._lock = threading.Lock()
        self._ended = False
        self._dropped = 0

    def __call__(self, event: AgentEvent) -> None:
        with self._lock:
            if self._ended:
                self._dropped += 1
                if self._dropped == 1:
                    self._logger.log(f"turn ended: ignoring a later {type(event).__name__} from the process")
                return
            if isinstance(event, (TurnResult, ProcessExited)):
                self._ended = True
        self._deliver(event)

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped


class ResultFreezer:
    """Freezes the turn's active timer the moment its TurnResult is published. It is the first listener of the bus, so
    it runs before the feed, the permissions or the AnswerCollector (which wakes the wait loop) see the result: no cap
    can be recorded for a turn that has already finished, however long the listeners after it take."""

    def __init__(self, freeze: Callable[[], None]) -> None:
        self._freeze = freeze

    def on_event(self, event: AgentEvent) -> None:
        if isinstance(event, TurnResult):
            self._freeze()


class UsageRecorder:
    """Keeps the most recent UsageEvent of the turn; None when the backend sent none."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage: UsageEvent | None = None

    def on_event(self, event: AgentEvent) -> None:
        if isinstance(event, UsageEvent):
            with self._lock:
                self._usage = event

    @property
    def usage(self) -> UsageEvent | None:
        with self._lock:
            return self._usage


class AnswerCollector:
    """The answer texts, tool count and end of one turn.

    Texts with a parent_id (a subagent's words) are not the answer and are skipped. The first TurnResult or ProcessExited
    sets `done`, which ends the turn's wait loop, and later ones are ignored: a turn has one result. A stop that follows a
    finished turn can still make the CLI answer again (only when it did not exit by itself); EndGate drops that answer
    before it gets here, and this collector ignores it too.
    Tool calls are counted from every ToolStart, subagents included.
    """

    def __init__(self, done: threading.Event) -> None:
        self._done = done
        self._lock = threading.Lock()
        self._texts: list[str] = []
        self._tools = 0
        self._result: TurnResult | None = None
        self._exit_code: int | None = None

    def on_event(self, event: AgentEvent) -> None:
        if isinstance(event, TextEvent):
            if event.parent_id is None and event.text.strip():
                with self._lock:
                    self._texts.append(event.text)
        elif isinstance(event, ToolStart):
            with self._lock:
                self._tools += 1
        elif isinstance(event, TurnResult):
            with self._lock:
                if self._result is None and self._exit_code is None:
                    self._result = event
                    self._done.set()
        elif isinstance(event, ProcessExited):
            with self._lock:
                if self._result is None and self._exit_code is None:
                    self._exit_code = event.returncode
                    self._done.set()

    @property
    def result(self) -> TurnResult | None:
        with self._lock:
            return self._result

    @property
    def texts(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._texts)

    @property
    def tools(self) -> int:
        with self._lock:
            return self._tools

    @property
    def exit_code(self) -> int | None:
        with self._lock:
            return self._exit_code


class SessionRecorder:
    """Records the session id of a topic the first time it is known, and the CLI's slash commands.

    remember() is called before the process exists for a pre-assigned id (Claude --session-id) and on every
    SessionStarted (Codex learns its thread id there). A repeated id is not written again. Writes are repository calls
    and never take this recorder's lock for longer than the comparison.
    """

    def __init__(self, *, sessions: SessionRepo, history: HistoryRepo, kv: KvRepo, thread_id: int, text: str,
                 cwd: str, launcher: str, backend: str, known_uuid: str) -> None:
        self._sessions = sessions
        self._history = history
        self._kv = kv
        self._thread_id = thread_id
        self._text = text
        self._cwd = cwd
        self._launcher = launcher
        self._backend = backend
        self._lock = threading.Lock()
        self._known = known_uuid

    @property
    def known(self) -> str:
        with self._lock:
            return self._known

    def remember(self, uuid: str) -> None:
        with self._lock:
            if not uuid or uuid == self._known:
                return
            self._known = uuid
        self._sessions.set_uuid(self._thread_id, uuid)
        self._history.note_session(self._thread_id, uuid, self._text, self._cwd, self._launcher, self._backend)

    def on_event(self, event: AgentEvent) -> None:
        if isinstance(event, SessionStarted):
            self.remember(event.session_id)
            if event.commands:
                self._kv.set(slash_key(self._thread_id), json.dumps(list(event.commands)))


class WaitRouter:
    """The PermissionService's WaitObserver for the whole App: routes each thread's changes to its running turn.

    register() is called when a turn starts and unregister() when it ends; a change for a thread with no turn is ignored.
    The observer is called after the router's lock is released.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._observers: dict[int, WaitObserver] = {}

    def register(self, thread_id: int, observer: WaitObserver) -> None:
        with self._lock:
            self._observers[thread_id] = observer

    def unregister(self, thread_id: int, observer: WaitObserver) -> None:
        with self._lock:
            if self._observers.get(thread_id) is observer:
                del self._observers[thread_id]

    def waiting_changed(self, thread_id: int, waiting: bool) -> None:
        with self._lock:
            observer = self._observers.get(thread_id)
        if observer is not None:
            observer.waiting_changed(thread_id, waiting)


class HandoffClearer:
    """Discards the pending handoff digest once the fresh session that was given it has started. A turn that dies before
    SessionStarted never reaches this, so its digest stays in kv for the next try."""

    def __init__(self, handoff: HandoffService, thread_id: int) -> None:
        self._handoff = handoff
        self._thread_id = thread_id

    def on_event(self, event: AgentEvent) -> None:
        if isinstance(event, SessionStarted):
            self._handoff.discard(self._thread_id)
