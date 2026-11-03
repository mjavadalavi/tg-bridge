"""TerminalTaps: the three buttons of the terminal confirm card that a change posts when the topic's session is open
in a terminal tab (card H2, after the keep taps of section 5 D1; card K-D).

The buttons are picker kinds tc (close the tab and run the change now), tl (run the change when the owner closes the
tab) and tn (cancel), cm:<thread>:<kind>:<token>. The token names the held change in TerminalGuard; each tap answers
through the guard and edits the card to the result. tn leaves the card with no buttons; tl keeps one cancel button
while the change waits; tc ends the tab's Claude Code on a spawned thread, because close_now blocks while the tab
closes. A close that cannot run (no child to end, or the tab did not close in time) leaves the change pending, so the
card keeps its later and cancel buttons. The card and its change were made by the commands, so this object holds no
state.

Threads: the callback thread of the tap (CallbackRouter.dispatch). tc finishes on the spawned thread, and the report of
a change runs on whichever thread applies it (the spawned thread, the guard's watcher thread, or the dispatcher's
runner). A deferred change's report waits for the tap's own deferred edit first, so the result never lands before it.
The object takes no lock of its own: TerminalGuard locks the pending map and the tap replies go out through the
router's queue.
"""

import threading
from collections.abc import Callable

from tg_bridge.bot import live_actions
from tg_bridge.bot.callbacks.router import CallbackRouter, Tap
from tg_bridge.bot.terminal_guard import (
    TERMINAL_CLOSE_KIND,
    TERMINAL_CLOSE_TIMEOUT_TEXT,
    TERMINAL_CLOSE_WAIT_SECONDS,
    TERMINAL_DEFERRED_TEXT,
    TERMINAL_LATER_BUTTON,
    TERMINAL_LATER_KIND,
    TERMINAL_NO_BUTTON,
    TERMINAL_NO_CHILD_TEXT,
    TERMINAL_NO_KIND,
    TerminalGuard,
)
from tg_bridge.telegram.keyboards import PICKER_PREFIX, CallbackData

TERMINAL_CLOSING_TOAST = "closing the tab…"
LATER_REPORT_WAIT_SECONDS = 5.0  # the longest a deferred change's result waits for the tap's own deferred edit


def _button_row(thread: int, token: str, label: str, kind: str) -> list[tuple[str, CallbackData]]:
    """One button of the card on its own row, with the callback data terminal_card gives it."""
    return [(label, CallbackData(PICKER_PREFIX, (str(thread), kind, token)))]


class TerminalTaps:
    """The tc, tl and tn picker kinds of the terminal confirm card. The terminal guard is the one shared by every change
    that asks; spawn runs the close on a thread of its own, so the tap's thread does not wait for the tab."""

    def __init__(self, *, terminal: TerminalGuard, spawn: Callable[..., object] = live_actions.spawn) -> None:
        self._terminal = terminal
        self._spawn = spawn

    def register(self, router: CallbackRouter) -> None:
        router.on_picker(TERMINAL_CLOSE_KIND, self._close)
        router.on_picker(TERMINAL_LATER_KIND, self._later)
        router.on_picker(TERMINAL_NO_KIND, self._cancel)

    def _cancel(self, tap: Tap) -> None:
        text = self._terminal.cancel(int(tap.args[0]), tap.args[2])  # the router checked the thread is a number
        tap.toast(text)
        tap.edit(text)

    def _later(self, tap: Tap) -> None:
        thread, token = int(tap.args[0]), tap.args[2]
        shown = threading.Event()  # set once this tap's own edit went out, so a faster result waits for it

        def report(result: str) -> None:
            shown.wait(LATER_REPORT_WAIT_SECONDS)
            tap.edit(result)

        try:
            text = self._terminal.defer(thread, token, report=report)
            tap.toast(text)
            if text == TERMINAL_DEFERRED_TEXT:
                tap.edit(text, [_button_row(thread, token, TERMINAL_NO_BUTTON, TERMINAL_NO_KIND)])
            else:
                tap.edit(text)
        finally:
            shown.set()

    def _close(self, tap: Tap) -> None:
        tap.toast(TERMINAL_CLOSING_TOAST)
        tap.edit(TERMINAL_CLOSING_TOAST)  # the card drops its buttons while the tab closes
        self._spawn(self._finish_close, tap, int(tap.args[0]), tap.args[2])

    def _finish_close(self, tap: Tap, thread: int, token: str) -> None:
        text = self._terminal.close_now(thread, token, report=tap.edit)
        if text is not None:
            self._show(tap, thread, token, text)

    def _show(self, tap: Tap, thread: int, token: str, text: str) -> None:
        """Edits the card to text. A change that still waits keeps its later and cancel buttons, other text has none."""
        timeout_text = TERMINAL_CLOSE_TIMEOUT_TEXT.format(seconds=TERMINAL_CLOSE_WAIT_SECONDS)
        if text in (TERMINAL_NO_CHILD_TEXT, timeout_text):
            tap.edit(text, [_button_row(thread, token, TERMINAL_LATER_BUTTON, TERMINAL_LATER_KIND),
                            _button_row(thread, token, TERMINAL_NO_BUTTON, TERMINAL_NO_KIND)])
        else:
            tap.edit(text)
