"""Approval taps: the buttons of a permission card and of an AskUserQuestion card (C13 taps).

- pa / pd: allow once / deny a permission ask. pA: allow and save the backend's proposed rule. pt: bypass for this
  turn. q:<nonce>:<question>:<option>: a picked option. o:<nonce>:<question>: 'Other', which waits for a typed answer.

Each tap is a call on PermissionService, which decides whether the ask is still open. A tap that finds nothing open
(a double tap, an old card, a turn that already ended) gets the 'expired' toast and the card's text is kept with its
buttons removed. The service claims an ask before it answers it, so a repeated tap never answers twice.
"""

from tg_bridge.bot.callbacks.router import CallbackRouter, Tap
from tg_bridge.core.permissions import PermissionService

EXPIRED_TEXT = "expired — that turn already ended"
ALLOWED_TEXT = "Allowed ✅"
DENIED_TEXT = "Denied ⛔"
ALWAYS_TEXT = "Always allowed ♾"
BYPASS_TEXT = "Bypass on for this turn 🤖"
RECORDED_TEXT = "Recorded ✅"
OTHER_TEXT = "Type your answer in this topic ✏️"


class ApprovalTaps:
    """The approval and question buttons, each one a single call on PermissionService."""

    def __init__(self, *, permissions: PermissionService) -> None:
        self._permissions = permissions

    def register(self, router: CallbackRouter) -> None:
        router.on("pa", self._allow)
        router.on("pd", self._deny)
        router.on("pA", self._always)
        router.on("pt", self._bypass)
        router.on("q", self._option)
        router.on("o", self._other)

    @staticmethod
    def _nonce(tap: Tap, fields: int) -> str | None:
        """The card's nonce when the tap carries exactly fields arguments, the first of them non-empty."""
        if len(tap.args) != fields or not tap.args[0]:
            return None
        return tap.args[0]

    @staticmethod
    def _settle(tap: Tap, ok: bool, toast: str) -> None:
        """Answers the tap: the toast when the ask was still open, else the expired toast and the card without buttons."""
        if ok:
            tap.toast(toast)
            return
        tap.toast(EXPIRED_TEXT)
        tap.edit(tap.message_text)

    def _allow(self, tap: Tap) -> None:
        nonce = self._nonce(tap, 1)
        if nonce is not None:
            self._settle(tap, self._permissions.answer_ask(nonce, True), ALLOWED_TEXT)

    def _deny(self, tap: Tap) -> None:
        nonce = self._nonce(tap, 1)
        if nonce is not None:
            self._settle(tap, self._permissions.answer_ask(nonce, False), DENIED_TEXT)

    def _always(self, tap: Tap) -> None:
        nonce = self._nonce(tap, 1)
        if nonce is not None:
            self._settle(tap, self._permissions.answer_always(nonce), ALWAYS_TEXT)

    def _bypass(self, tap: Tap) -> None:
        nonce = self._nonce(tap, 1)
        if nonce is not None:
            self._settle(tap, self._permissions.answer_bypass(nonce), BYPASS_TEXT)

    def _option(self, tap: Tap) -> None:
        nonce = self._nonce(tap, 3)
        if nonce is None or not tap.args[1].isdigit() or not tap.args[2].isdigit():
            return
        ok = self._permissions.answer_question(nonce, int(tap.args[1]), int(tap.args[2]))
        self._settle(tap, ok, RECORDED_TEXT)

    def _other(self, tap: Tap) -> None:
        nonce = self._nonce(tap, 2)
        if nonce is None or not tap.args[1].isdigit():
            return
        ok = self._permissions.answer_other(nonce, int(tap.args[1]))
        self._settle(tap, ok, OTHER_TEXT)
