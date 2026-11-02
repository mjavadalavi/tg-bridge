"""The wait before a message is typed into the TUI (tui_session.py send_user). While a permission request is pending, no key
is typed: a key would answer the TUI's own dialog. The wait is bounded; past the bound the requests are denied, so the
dialog closes before the message is typed. Moved out of tui_session to keep it under the size ceiling.
"""

from collections.abc import Callable
from typing import Protocol

from tg_bridge.agents.claude.hookbridge import TURN_ENDED
from tg_bridge.clock import Clock

HOOK_WAIT_SECONDS = 60.0  # a pending permission request is waited for at most this long before a message
HOOK_POLL_SECONDS = 0.5


class PendingRequests(Protocol):
    def pending_count(self) -> int: ...


def wait_for_hooks(server: PendingRequests, clock: Clock, deny: Callable[[str], object],
                   log: Callable[[str], None]) -> None:
    """Waits, bounded, for the pending requests to clear. Past the bound they are denied through deny (all of them, a
    background subagent's included, since the message is about to be typed)."""
    waited = 0.0
    while server.pending_count() and waited < HOOK_WAIT_SECONDS:
        clock.sleep(HOOK_POLL_SECONDS)
        waited += HOOK_POLL_SECONDS
    if server.pending_count():
        log("permission requests still pending before the next message: denied")
        deny(TURN_ENDED)
