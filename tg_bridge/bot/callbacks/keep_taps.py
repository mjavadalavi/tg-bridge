"""KeepTaps: the two buttons of the confirm card that a change posts when it would close a kept session (section 5 D1).

The buttons are picker kinds ky (change and close) and kn (keep the session), cm:<thread>:<kind>:<token>. The token
names the held change in KeepGuard; the tap applies or drops it through KeepGuard.decide and edits the card to the
result, with no buttons left on it. The card and its change were made by the commands, so this object holds no state.

Threads: the callback thread of the tap (CallbackRouter.dispatch). The object takes no lock of its own: KeepGuard.decide
locks the pending map and the tap replies go out through the router's queue.
"""

from tg_bridge.bot.callbacks.router import CallbackRouter, Tap
from tg_bridge.bot.keep_guard import KEEP_NO_KIND, KEEP_YES_KIND, KeepGuard


class KeepTaps:
    """The ky and kn picker kinds of the confirm card. The keep guard is the one shared by every change that asks."""

    def __init__(self, *, keep: KeepGuard) -> None:
        self._keep = keep

    def register(self, router: CallbackRouter) -> None:
        router.on_picker(KEEP_YES_KIND, self._yes)
        router.on_picker(KEEP_NO_KIND, self._no)

    def _yes(self, tap: Tap) -> None:
        self._decide(tap, yes=True)

    def _no(self, tap: Tap) -> None:
        self._decide(tap, yes=False)

    def _decide(self, tap: Tap, *, yes: bool) -> None:
        text = self._keep.decide(int(tap.args[0]), tap.args[2], yes)  # the router checked the thread is a number
        tap.toast(text)
        tap.edit(text)
