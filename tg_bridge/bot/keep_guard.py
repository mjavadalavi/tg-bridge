"""KeepGuard: asks before a change would close a session kept open for background work (section 5 D1, card C9d).

A topic whose kept session has background work pending (a Monitor, a background command or agent, a wakeup) keeps its
process after the turn. A change that would close that process (mode, model, launcher, folder, session) must not
apply at once. hold() stores the change under a short token and returns the confirm card for the caller to post; the
card's two buttons reach decide(): yes applies the change and asks the kept session to close, no drops the change.

hold() returns None, and the caller applies the change as it always did, when nothing is kept, when nothing is pending
(the quiet window), when the change would store the value the topic already has (same=True: nothing would close, so no
card) or when a user turn runs on the kept process (that turn applies the change as it does today).

Each held change keeps the hold_id of the kept session it was offered for (LingerStatus.hold_id). A tap applies only
while the topic still has that same hold: once the hold ended (no kept session) or another hold took its place, the card
has expired and applies nothing. The entries of a topic whose hold has ended are dropped lazily, when the topic is
next looked at (hold or decide). LingerService numbers every accepted offer, so a new user turn on the same kept session
also takes a new hold_id: a card posted before that turn expires too (stricter than the spec, and safe).

An apply returns an ApplyResult: applied False (the handler's own refusal, for example a session in use elsewhere) is
shown as the result and releases nothing; applied True releases the kept session only when one is still kept.

Threads: hold() runs on the poll thread (typed commands and inbound messages) or on a callback thread (taps); decide()
runs on a callback thread. The change's apply callable runs with no lock held. The pending map is the only state and
is guarded by _lock; a new hold() for a topic replaces the topic's earlier unanswered card, and decide() takes the entry
out of the map, so each card is answered once.

Lock map: _lock guards _pending.
"""

import os
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass

from tg_bridge.bot.pickers import Picker
from tg_bridge.core.linger import LingerService
from tg_bridge.core.live import LiveRegistry
from tg_bridge.telegram.keyboards import PICKER_PREFIX, CallbackData

KEEP_YES_KIND = "ky"  # cm: kind of the card's change-and-close button; the arg is the card's token
KEEP_NO_KIND = "kn"   # cm: kind of the card's keep button; the arg is the card's token
KEEP_TOKEN_BYTES = 4  # 8 hex characters: short enough for the 64-byte callback limit, unique enough per topic
KEEP_CARD_TEXT = ("⚠️ this topic has a session kept open for {pending} background task(s). "
                  "Changing {what} closes it and stops that work.")
KEEP_YES_BUTTON = "✅ change and close"
KEEP_NO_BUTTON = "↩ keep the session"
KEEP_CHANGED_TEXT = "✅ changed; the kept session was closed."
KEEP_CHANGED_LATER_TEXT = "✅ changed; the kept session closes when its running background update ends."
KEEP_KEPT_TEXT = "↩ kept — nothing changed."
KEEP_EXPIRED_TEXT = "this choice has expired"
KEEP_NOT_APPLIED_TEXT = "nothing changed."
SAME_SESSION_TEXT = "↩️ session {sid} — already the current session, nothing changed"  # a /resume of it
KEEP_TOAST_TEXT = "a background task is running — confirm first"
WHAT_MODE = "mode"
WHAT_MODEL = "model"
WHAT_LAUNCHER = "launcher"
WHAT_FOLDER = "folder"
WHAT_SESSION = "session"


@dataclass(frozen=True)
class ApplyResult:
    """What a held change reports when its card is answered: applied False when the handler refused it (text is the
    refusal); applied True with an optional text (a launcher's reply) that is shown under the result."""

    applied: bool
    text: str | None = None


APPLIED = ApplyResult(True)


def applies(change: Callable[[], object]) -> Callable[[], ApplyResult]:
    """The apply of a change that cannot be refused: it runs change, and what change returns is not a refusal."""

    def apply() -> ApplyResult:
        change()
        return APPLIED

    return apply


@dataclass(frozen=True)
class _Pending:
    """One unanswered card: the topic it was posted for, the hold it was posted in and the change it holds."""

    thread: int
    hold_id: int
    apply: Callable[[], ApplyResult]


def keep_card(thread: int, token: str, pending: int, what: str) -> Picker:
    """The confirm card: its text and the two buttons (cm:<thread>:ky:<token> and cm:<thread>:kn:<token>)."""
    rows = [[(KEEP_YES_BUTTON, CallbackData(PICKER_PREFIX, (str(thread), KEEP_YES_KIND, token))),
              (KEEP_NO_BUTTON, CallbackData(PICKER_PREFIX, (str(thread), KEEP_NO_KIND, token)))]]
    return Picker(KEEP_CARD_TEXT.format(pending=pending, what=what), rows)


def same_folder(current: str, path: str) -> bool:
    """True when the topic's stored folder already is path (the realpath rule of CwdService.apply): nothing changes."""
    return os.path.realpath(os.path.expanduser(current or "")) == path


class KeepGuard:
    """Holds the changes that would close a kept session until its topic answers the card. One instance, shared by
    the commands, the taps and the inbound handler, so a card posted by one of them is answered by the taps."""

    def __init__(self, *, linger: LingerService, live: LiveRegistry) -> None:
        self._linger = linger
        self._live = live
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}

    def hold(self, thread: int, what: str, apply: Callable[[], ApplyResult], same: bool = False) -> Picker | None:
        """The card when the change must wait for the answer (apply is then stored, not run); None to apply it now.
        same: the change stores what the topic already has, so nothing would close and no card is needed."""
        if same:
            return None
        status = self._linger.status(thread)
        hold_id = status.hold_id if status is not None else None
        with self._lock:
            self._drop_ended(thread, hold_id)
        if status is None or status.pending <= 0:
            return None
        if self._live.running(thread) is not None and not status.self_turn:
            return None  # a user turn runs on the kept process: the change applies as it always did
        token = secrets.token_hex(KEEP_TOKEN_BYTES)
        with self._lock:
            for key in [key for key, held in self._pending.items() if held.thread == thread]:
                del self._pending[key]
            self._pending[token] = _Pending(thread, status.hold_id, apply)
        return keep_card(thread, token, status.pending, what)

    def decide(self, thread: int, token: str, yes: bool) -> str:
        """Answers a card: yes applies its change and asks the kept session to close, no drops the change. Returns the
        text the card is edited to. A card has expired (applies nothing) when the token is unknown or used, belongs to
        another topic, or its hold has ended or was replaced by another one. A change refused by its apply shows the
        refusal; a change applied with a text (the launcher's reply) adds that text under the result."""
        status = self._linger.status(thread)
        hold_id = status.hold_id if status is not None else None
        with self._lock:
            held = self._pending.get(token)
            if held is None or held.thread != thread or hold_id is None or held.hold_id != hold_id:
                self._drop_ended(thread, hold_id)
                return KEEP_EXPIRED_TEXT
            del self._pending[token]
        if not yes:
            return KEEP_KEPT_TEXT
        result = held.apply()
        if not result.applied:
            return result.text or KEEP_NOT_APPLIED_TEXT
        # the change is applied. Only a kept session of the same hold still needs a close: none is left when it has
        # closed on its own, and a replaced hold is not this card's to close. request_release is False when a self
        # turn runs (it closes when that turn ends).
        after = self._linger.status(thread)
        if after is None or after.hold_id != hold_id:
            text = KEEP_CHANGED_TEXT
        else:
            closed = self._linger.request_release(thread) or self._linger.status(thread) is None
            text = KEEP_CHANGED_TEXT if closed else KEEP_CHANGED_LATER_TEXT
        return f"{text}\n{result.text}" if result.text else text

    def _drop_ended(self, thread: int, hold_id: int | None) -> None:
        """Drops the topic's entries that belong to another hold than hold_id (None: no hold). Caller holds _lock."""
        for key in [key for key, held in self._pending.items() if held.thread == thread and held.hold_id != hold_id]:
            del self._pending[key]
