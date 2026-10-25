"""TurnEventBus (Observer subject) and the EventListener Protocol.

One bus is built per turn by the turn runner, with the turn's listeners (feed, permissions, usage, session recorder,
answer collector) in a fixed order. The backend's reader thread calls publish() for every normalized AgentEvent, so
publish() runs on that thread and must not block on anything slow; each listener does its own queueing.

A listener that raises is logged and skipped for that one event: the other listeners still receive it, and the same
listener still receives the next event. The bus holds no state apart from its listener tuple and its logger, so one
turn's failure can never leak into another turn.
"""

from typing import Iterable, Protocol

from tg_bridge.agents.events import AgentEvent
from tg_bridge.logging_ import Logger


class EventListener(Protocol):
    """Anything that wants the normalized events of one turn."""

    def on_event(self, event: AgentEvent) -> None: ...


class TurnEventBus:
    """Delivers each event to every listener, in the order the listeners were given."""

    def __init__(self, listeners: Iterable[EventListener], logger: Logger) -> None:
        self._listeners: tuple[EventListener, ...] = tuple(listeners)
        self._logger = logger

    @property
    def listeners(self) -> tuple[EventListener, ...]:
        return self._listeners

    def publish(self, event: AgentEvent) -> None:
        for listener in self._listeners:
            try:
                listener.on_event(event)
            except Exception as exc:  # one broken listener must not stop the turn or the other listeners
                self._logger.log(
                    f"turn listener {type(listener).__name__} failed on {type(event).__name__}: {exc!r}"
                )
