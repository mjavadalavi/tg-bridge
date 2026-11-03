"""TerminalGuard: asks before /new or /resume changes a topic whose session is open in a terminal tab (cards H1, K-C,
L1).

A topic's session can be open in a `tg-bridge view` tab, which holds a live claim on it
(core/terminal_claim.py). While that claim is live, a change that would move the topic to another session must not
apply at once. hold() stores the change under a short token and returns the card for the caller to post. The card's
three rows reach three answers, each once:
- close_now(): the tab's Claude Code is ended (SIGTERM to its CLI child), the change runs when the tab is free.
- defer(): the change waits until the owner closes the tab; a watcher thread runs it then and reports the result.
- cancel(): the change is dropped.

In close_now and defer the tab's session is not taken back: the tab's ended record is consumed (take_ended), and
terminal_handoff.adopt_ended reads that record. The change happens only on a tap. A tap after the tab already closed
still applies: the owner asked for the change while the tab was open.

Every apply (close_now's wait, settle, before_turn) takes the deferred entry out of the map, consumes the ended record
and runs apply(), all under _apply_lock. settle() is the poll and watcher path: it also waits while the topic's turn
mark exists, since a process of the topic runs then. before_turn() is the dispatcher's hook, run before a turn the
bridge spawns. The dispatcher holds the turn mark for that turn, so the hook does not check it, and it applies the
change before the turn's adopt_ended reads the ended record. A settle that comes during an apply waits for it and then
sees the final state.

hold() returns None, and the caller applies the change as it always did, when view_dir is '' (no terminal view), when
the topic has no live claim, or when reading the claim fails with an OSError.

Threads: hold(), cancel(), close_now() and defer() run on a callback thread or the poll thread. settle() runs on the
watcher thread of a deferred change and inside close_now's wait; before_turn() runs on the dispatcher's runner thread.
The apply callable and the report callable run under _apply_lock, never under _lock. The order is _apply_lock, then
_lock. A new hold() for a topic replaces the topic's earlier unanswered card, and every answer takes its entry out of
the map, so each card is answered once.

Lock map: _apply_lock serialises the applies (pop, take_ended, apply, report). _lock guards _pending and _settled.
_settled holds the tokens of close_now's changes that an apply ran, until close_now reads them.
"""

import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace

from tg_bridge.bot.keep_guard import KEEP_NOT_APPLIED_TEXT, ApplyResult
from tg_bridge.bot.pickers import Picker
from tg_bridge.clock import Clock, SystemClock
from tg_bridge.core import terminal_claim
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.terminal_claim import Claim
from tg_bridge.telegram.keyboards import PICKER_PREFIX, CallbackData

TERMINAL_CLOSE_KIND = "tc"  # cm: kind of the card's close-and-run button; the arg is the card's token
TERMINAL_LATER_KIND = "tl"  # cm: kind of the card's run-when-the-tab-closes button; the arg is the card's token
TERMINAL_NO_KIND = "tn"     # cm: kind of the card's cancel button; the arg is the card's token
TERMINAL_TOKEN_BYTES = 6  # 12 hex characters: short enough for the 64-byte callback limit, unique enough per topic
TERMINAL_POLL_SECONDS = 2.0
TERMINAL_CLOSE_WAIT_SECONDS = 20.0
TERMINAL_CARD_TEXT = ("🖥 this topic's session is open in the terminal (tg-bridge view). The {what} change needs the "
                      "tab's Claude Code closed first.")
TERMINAL_CLOSE_BUTTON = "⏹ close the tab and run it now"
TERMINAL_LATER_BUTTON = "⏳ run it when I close the tab"
TERMINAL_NO_BUTTON = "↩ cancel"
TERMINAL_KEPT_TEXT = "cancelled — the terminal session is unchanged."
TERMINAL_EXPIRED_TEXT = "this choice has expired"
TERMINAL_DONE_TEXT = "done."
TERMINAL_CLOSED_TEXT = "🖥 the tab's Claude Code was closed.\n{text}"
TERMINAL_CLOSE_TIMEOUT_TEXT = ("the tab did not close within {seconds:g} s — nothing changed. "
                               "Exit Claude Code in the tab and try again.")
TERMINAL_NO_CHILD_TEXT = "the tab cannot be closed from here (it runs an older tg-bridge view) — nothing changed."
TERMINAL_UNREADABLE_TEXT = "the tab's state could not be read — nothing changed."
TERMINAL_DEFERRED_TEXT = "⏳ ok — the change runs when you close the tab; the tab's session is not taken back."
TERMINAL_LATER_DONE_TEXT = "🖥 the tab closed.\n{text}"


@dataclass(frozen=True)
class _Pending:
    """One unanswered card: the topic it was posted for, the change it holds, and, once deferred, the report of its
    result. closing marks a change that close_now waits for: its result goes to report under the closed line."""

    thread: int
    apply: Callable[[], ApplyResult]
    deferred: bool = False
    report: Callable[[str], None] | None = None
    closing: bool = False


def terminal_card(thread: int, token: str, what: str) -> Picker:
    """The confirm card: its text and its three buttons, one per row (cm:<thread>:<kind>:<token>)."""
    def button(label: str, kind: str) -> tuple[str, CallbackData]:
        return label, CallbackData(PICKER_PREFIX, (str(thread), kind, token))

    rows = [[button(TERMINAL_CLOSE_BUTTON, TERMINAL_CLOSE_KIND)],
            [button(TERMINAL_LATER_BUTTON, TERMINAL_LATER_KIND)],
            [button(TERMINAL_NO_BUTTON, TERMINAL_NO_KIND)]]
    return Picker(TERMINAL_CARD_TEXT.format(what=what), rows)


def _run_change(held: _Pending, done_wrapper: str) -> str:
    """Runs the held change and returns its text: the refusal when it was refused, else done_wrapper filled with
    the change's text (TERMINAL_DONE_TEXT when it has none)."""
    result = held.apply()
    if not result.applied:
        return result.text or KEEP_NOT_APPLIED_TEXT
    return done_wrapper.format(text=result.text or TERMINAL_DONE_TEXT)


class TerminalGuard:
    """Holds the changes that would move a topic off the session its terminal tab has open, until the topic answers the
    card. One instance, shared by the commands and the taps, so a card posted by one is answered by the other."""

    def __init__(self, *, view_dir: str, live: LiveRegistry | None = None, clock: Clock = SystemClock(),
                 close: Callable[[Claim], bool] = terminal_claim.signal_child,
                 start_thread: Callable[..., threading.Thread] = threading.Thread) -> None:
        self._view_dir = view_dir
        self._registry = live
        self._clock = clock
        self._close = close
        self._start_thread = start_thread
        self._apply_lock = threading.Lock()
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}
        self._settled: set[str] = set()

    def hold(self, thread: int, what: str, apply: Callable[[], ApplyResult]) -> Picker | None:
        """The card when the change must wait for the answer (apply is then stored, not run); None to apply it now."""
        if not self._view_dir or not self._has_live_claim(thread):
            return None
        token = secrets.token_hex(TERMINAL_TOKEN_BYTES)
        with self._lock:
            for key in [key for key, held in self._pending.items() if held.thread == thread]:
                del self._pending[key]
            self._pending[token] = _Pending(thread, apply)
        return terminal_card(thread, token, what)

    def cancel(self, thread: int, token: str) -> str:
        """Drops the held change (a deferred one too). Returns the text the card is edited to."""
        return TERMINAL_KEPT_TEXT if self._take(thread, token) is not None else TERMINAL_EXPIRED_TEXT

    def close_now(self, thread: int, token: str, report: Callable[[str], None]) -> str | None:
        """Ends the tab's Claude Code and runs the change once the tab is free. A tab that is already closed runs it at
        once, and its text is returned. Otherwise the change waits for the tab, and whichever path finds the tab free
        first applies it and passes its text to report. Returns None once the change was applied (report carried its
        text), or the text the card is edited to: a refusal, no child to close, an unreadable claim, a timeout, or a
        cancel meanwhile. An unreadable claim leaves the entry as it was, so the card can still be answered."""
        with self._lock:
            held = self._pending.get(token)
            if held is None or held.thread != thread or held.deferred:
                return TERMINAL_EXPIRED_TEXT
        try:
            claim = terminal_claim.live_claim(self._view_dir, thread)
        except OSError:
            return TERMINAL_UNREADABLE_TEXT
        if claim is None:
            return self._apply_now(thread, token)
        if claim.child <= 0 or not self._close(claim):
            return TERMINAL_NO_CHILD_TEXT
        with self._lock:
            held = self._pending.get(token)
            if held is None or held.thread != thread:
                return TERMINAL_EXPIRED_TEXT
            self._pending[token] = replace(held, deferred=True, report=report, closing=True)
        return self._await_close(thread, token)

    def defer(self, thread: int, token: str, report: Callable[[str], None]) -> str:
        """Keeps the change until the owner closes the tab: a watcher thread runs it then and passes its result to
        report. Returns the text the card is edited to."""
        with self._lock:
            held = self._pending.get(token)
            if held is None or held.thread != thread or held.closing:
                return TERMINAL_EXPIRED_TEXT
            self._pending[token] = replace(held, deferred=True, report=report)
        self._start_thread(target=self._watch, args=(thread, token), daemon=True).start()
        return TERMINAL_DEFERRED_TEXT

    def settle(self, thread: int) -> str | None:
        """The poll and watcher path: runs the topic's deferred change when its tab is free, no turn runs and no turn
        mark exists. Returns the text it applied, or None when nothing is deferred or the topic is still busy."""
        return self._apply_deferred(thread, turn_mark=True)

    def before_turn(self, thread: int) -> None:
        """The dispatcher's hook, run after the terminal wait and before a turn the bridge spawns. The turn mark is that
        turn's own, so only the claim and the running turn are checked. The change applies here, before the turn's
        adopt_ended reads the ended record."""
        self._apply_deferred(thread, turn_mark=False)

    def _apply_deferred(self, thread: int, *, turn_mark: bool) -> str | None:
        """Applies the topic's deferred change once its tab is free (and, with turn_mark, no turn mark exists)."""
        with self._apply_lock:
            if not self._is_deferred(thread) or not self._released(thread, turn_mark=turn_mark):
                return None
            held = self._pop_deferred(thread)
            if held is None:
                return None
            terminal_claim.take_ended(self._view_dir, thread)  # the tab's session is not taken back
            text = _run_change(held, TERMINAL_CLOSED_TEXT if held.closing else TERMINAL_LATER_DONE_TEXT)
            if held.report is not None:
                try:
                    held.report(text)
                except Exception:  # a failed post must not lose the change's result or stop the watcher's bookkeeping
                    pass
            return text

    def _apply_now(self, thread: int, token: str) -> str:
        """The change of a card whose tab is already closed: applied at once, and its text returned."""
        with self._apply_lock:
            with self._lock:
                held = self._pending.get(token)
                if held is None or held.thread != thread or held.deferred:
                    return TERMINAL_EXPIRED_TEXT
                del self._pending[token]
            terminal_claim.take_ended(self._view_dir, thread)  # the tab's session is not taken back
            return _run_change(held, TERMINAL_CLOSED_TEXT)

    def _await_close(self, thread: int, token: str) -> str | None:
        """Polls until the closing change is applied (None) or cancelled (KEPT). When the close wait passes first, the
        entry goes back to pending, so a later defer or cancel can still answer the card (TIMEOUT)."""
        deadline = self._clock.monotonic() + TERMINAL_CLOSE_WAIT_SECONDS
        while True:
            self.settle(thread)
            with self._lock:
                if token in self._settled:
                    self._settled.discard(token)
                    return None
                held = self._pending.get(token)
                if held is None:
                    return TERMINAL_KEPT_TEXT
                if self._clock.monotonic() >= deadline:
                    self._pending[token] = replace(held, deferred=False, report=None, closing=False)
                    return TERMINAL_CLOSE_TIMEOUT_TEXT.format(seconds=TERMINAL_CLOSE_WAIT_SECONDS)
            self._clock.sleep(TERMINAL_POLL_SECONDS)

    def _watch(self, thread: int, token: str) -> None:
        """The watcher of a deferred change: sleeps, then settles, until the change is answered or replaced. The sleep
        comes first so the tap's edit of the card to the deferred text lands before any done text."""
        while self._still_deferred(token):
            self._clock.sleep(TERMINAL_POLL_SECONDS)
            self.settle(thread)

    def _is_deferred(self, thread: int) -> bool:
        with self._lock:
            return any(held.thread == thread and held.deferred for held in self._pending.values())

    def _pop_deferred(self, thread: int) -> _Pending | None:
        """Takes the topic's deferred change out of the map; a closing one is recorded for close_now to read."""
        with self._lock:
            token = next((key for key, held in self._pending.items()
                          if held.thread == thread and held.deferred), None)
            if token is None:
                return None
            held = self._pending.pop(token)
            if held.closing:
                self._settled.add(token)
            return held

    def _still_deferred(self, token: str) -> bool:
        with self._lock:
            held = self._pending.get(token)
            return held is not None and held.deferred

    def _take(self, thread: int, token: str) -> _Pending | None:
        """Removes the card's entry when it belongs to thread; None when it is unknown, used, or another topic's."""
        with self._lock:
            held = self._pending.get(token)
            if held is None or held.thread != thread:
                return None
            del self._pending[token]
            return held

    def _released(self, thread: int, *, turn_mark: bool) -> bool:
        """True when the topic has no live claim and no running turn; with turn_mark also no turn mark, since a process
        of the topic may still run. An unreadable claim counts as still held."""
        try:
            if terminal_claim.live_claim(self._view_dir, thread) is not None:
                return False
            if turn_mark and terminal_claim.turn_marked(self._view_dir, thread):
                return False
        except OSError:
            return False
        return self._registry is None or self._registry.running(thread) is None

    def _has_live_claim(self, thread: int) -> bool:
        """True when the topic has a live terminal claim. An unreadable claim counts as none."""
        try:
            return terminal_claim.live_claim(self._view_dir, thread) is not None
        except OSError:
            return False
