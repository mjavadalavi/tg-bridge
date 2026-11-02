"""The result of a stop request (SessionCommands.stop_turn and stop_now), kept apart from session.py to stay under the
module size limit. CHOOSE carries the hold and self turn of the background update the choice card is posted for."""

import enum
from dataclasses import dataclass


class StopKind(enum.Enum):
    """What a stop request did: nothing was there to stop, a running turn was stopped, an idle kept session was asked
    to close, or a background update runs and the caller must ask which stop (section 5 D2)."""

    NOTHING = "nothing"
    STOPPED = "stopped"
    RELEASED = "released"
    CHOOSE = "choose"


@dataclass(frozen=True)
class StopResult:
    """The kind of stop; truthy when something was running or kept (CHOOSE counts: something is running). For CHOOSE,
    hold_id and self_seq are the LingerStatus of the update, so the card's buttons act on that update only (K9)."""

    kind: StopKind
    hold_id: int = 0
    self_seq: int = 0

    def __bool__(self) -> bool:
        return self.kind is not StopKind.NOTHING
