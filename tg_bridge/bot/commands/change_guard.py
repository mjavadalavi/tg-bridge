"""ask_first: the confirm card a /new or /resume change waits for when a guard holds it (card H1, section 5 D1).

A change that would move a topic off its session can be held by the TerminalGuard (the session is open in a
terminal tab) or by the KeepGuard (the session is kept open for background work). ask_first() asks the terminal guard
first, so its card wins, and asks the keep guard only when the terminal guard holds nothing. The caller posts the card
it gets back and answers a button tap with the toast it gets back.

The two cannot both apply at once: a tab cannot claim a topic whose session is kept open (terminal_claim.take_claim
refuses while the meta is running or lingering), and no bridge turn starts while a claim is live. The terminal guard is
still asked first, so the order holds either way.
"""

from collections.abc import Callable

from tg_bridge.bot.keep_guard import KEEP_TOAST_TEXT, ApplyResult, KeepGuard
from tg_bridge.bot.pickers import Picker
from tg_bridge.bot.terminal_guard import TerminalGuard

TERMINAL_TOAST_TEXT = "the session is open in a terminal tab — confirm first"


def ask_first(terminal: TerminalGuard | None, keep: KeepGuard | None, thread: int, what: str,
              terminal_apply: Callable[[], ApplyResult],
              keep_apply: Callable[[], ApplyResult]) -> tuple[Picker, str] | None:
    """The (card, toast) of the first guard that holds the change, or None when neither holds it (apply it now).

    Each guard gets its own apply: the terminal card shows the reply of the change, the keep card shows only the
    change's result. None for a guard means that guard is not wired in, so it holds nothing.
    """
    if terminal is not None:
        card = terminal.hold(thread, what, terminal_apply)
        if card is not None:
            return card, TERMINAL_TOAST_TEXT
    if keep is not None:
        card = keep.hold(thread, what, keep_apply)
        if card is not None:
            return card, KEEP_TOAST_TEXT
    return None
