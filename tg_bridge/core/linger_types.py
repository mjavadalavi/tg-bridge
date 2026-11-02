"""The value types and constants of LingerService (linger): the hold states and end conditions, what a tick and the
status report, the offer a user turn makes, the record of a kept topic and the LingerControl the dispatcher holds a
topic with. linger.py re-exports all of them, so every importer keeps using tg_bridge.core.linger.
"""

import enum
import threading
from dataclasses import dataclass
from typing import Protocol

from tg_bridge.agents.base import AgentSession
from tg_bridge.core.permissions import TurnBinding
from tg_bridge.core.session_router import SessionRouter
from tg_bridge.core.topic_key import TopicKey
from tg_bridge.core.turn_kit import TurnKit, TurnPlan
from tg_bridge.core.turn_listeners import SessionRecorder
from tg_bridge.core.turn_state import TurnOutcome
from tg_bridge.core.view_recorder import ViewRecorder

LINGER_PREFIX = "linger:"  # the kv key of a kept session is 'linger:<thread>'; a restart reads it for its notice
LINGER_QUIET_SECONDS = 20.0  # pending work at 0 for this long ends the hold (DONE): the late notification has come
RELEASE_GRACE_SECONDS = 10.0  # a released session gets this long to exit before stop() uses its signal ladder
POLL_SECONDS = 1.0  # the hold loop's tick: wait_once waits at most this long for a wake-up (the dispatcher's tick)
KEEP_OUTCOMES = (TurnOutcome.OK, TurnOutcome.INTERRUPTED)  # a self turn that ends like these keeps the session
# the topic's notices when a kept session closes with background work pending ({minutes}: ceil(linger_seconds / 60))
LINGER_CAP_TEXT = ("⌛ closed the session that was kept open for background work: {minutes} min passed since your last "
                   "message (linger_seconds). Ask again if you still need it.")
LINGER_EXIT_TEXT = "⚠️ the session kept open for background work exited before {pending} background task(s) finished."


class HoldState(enum.Enum):
    TURN = "turn"
    IDLE = "idle"
    SELF = "self"
    CLOSING = "closing"


class HoldEnd(enum.Enum):
    STOPPED = "stopped"  # /stop, the status ⏹ or a feed ⏹ asked for the release
    EXITED = "exited"  # the process is gone
    CHANGED = "changed"  # the topic's launcher, folder, mode, model or session changed
    CAP = "cap"  # linger_seconds passed since the last user turn
    DONE = "done"  # no background work is pending and the quiet window passed
    SELF_ENDED = "self-ended"  # a self turn ended badly (not OK or INTERRUPTED, or the process is gone)
    SHUTDOWN = "shutdown"  # the bridge is shutting down


@dataclass(frozen=True)
class HoldResult:
    """What one wait_once tick saw; end is None while the session is held."""

    end: HoldEnd | None
    pending: int
    chat_id: int | str


@dataclass(frozen=True)
class LingerStatus:
    """What /status shows for a kept topic. hold_id numbers the accepted offers of the service (1, 2, ...); self_seq
    counts the self turns that started in that hold (0 before the first one)."""

    pending: int
    seconds_left: float
    self_turn: bool
    hold_id: int = 0
    self_seq: int = 0


@dataclass(frozen=True)
class LingerOffer:
    """A user turn's end: its plan, session, router, binding, recorder and kit."""

    plan: TurnPlan
    session: AgentSession
    router: SessionRouter
    binding: TurnBinding
    recorder: SessionRecorder
    kit: TurnKit


class LingerControl(Protocol):
    """What the Dispatcher uses to hold a topic's runner thread."""

    def holding(self, thread_id: int) -> bool: ...

    def idle_alive(self, thread_id: int) -> bool: ...

    def wake(self, thread_id: int) -> None: ...

    def wait_once(self, thread_id: int) -> HoldResult: ...

    def request_release(self, thread_id: int) -> bool: ...


@dataclass(eq=False)
class HeldSession:
    """One kept topic; linger.py's docstring says which thread writes which field. hold_id is the accepted offer that
    made this hold, self_seq the self turns started in it."""

    thread_id: int
    chat_id: int | str
    topic: str
    plan: TurnPlan
    key: TopicKey
    session: AgentSession
    router: SessionRouter
    binding: TurnBinding
    recorder: SessionRecorder
    view: ViewRecorder | None
    deadline: float
    state: HoldState
    quiet_since: float | None
    release: bool
    wake: threading.Event
    interrupt: threading.Event | None
    hold_id: int
    self_seq: int
