"""TerminalWait: how a topic's head message waits while a terminal claim (tg-bridge view) holds the topic (spec T3).

A claim names a live Claude Code that the owner runs in a view tab. A claim with a pane and a session uuid is attachable:
the message is then typed into that session, and no turn mark is taken (TerminalTurn.attach names the session). It waits
only while the pane is busy or has text typed in it; the topic is told once with WAITING_TEXT. The pane is read through
the injected pane_idle probe (agents/claude/attached_session.pane_idle in the app), so tests fake it.

An attachable pane that stays busy or typed in for STALL_SECONDS gets one more notice, STALL_TEXT, that names the reason
the injected pane_reason probe gives (the reason strings are the tmux_pane REASON_* values; this module keeps them as
plain strings, since core does not import agents/claude). Nothing is sent to the pane.

A claim without a pane or uuid (an old view process) keeps the old wait: the topic is told TERMINAL_NOTICE once and the
message waits until the claim goes. With no claim, the turn mark is taken (terminal_claim.begin_turn) and the bridge spawns
its own process; a claim that goes while the message waits gives that same normal path. Without a view folder, or
without a probe, nothing is checked and the claims are not attachable.

An attached turn carries a liveness check (still_attached): the claim it started with must still be the topic's claim
(same pid and pane), so a CLI that exits mid-turn ends the turn, since the view pane releases the claim then.

Threads: the dispatcher's runner thread of the topic calls wait(); the inbound handler calls waiting(). Lock map: _lock
guards the set of waiting topics only. The claim lock is taken inside terminal_claim, only for file I/O, and nothing is
sent under it.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

from tg_bridge.agents.base import AttachSpec
from tg_bridge.clock import Clock
from tg_bridge.core import terminal_claim
from tg_bridge.core.removals import Removals
from tg_bridge.logging_ import Logger
from tg_bridge.telegram.messenger import Messenger

WAITING_TEXT = ("🖥 waiting: the terminal session is busy or has text typed in it — "
                "your message goes in when it is free.")
SEND_ERROR_CHARS = 120
STALL_SECONDS = 120.0  # an attachable pane not idle for this long gets the STALL_TEXT notice, once per stall
SECONDS_PER_MINUTE = 60
STALL_TEXT = "🖥 still waiting after {minutes} min: {detail}"
STALL_DEFAULT = "the terminal session is not idle"
IDLE_REASON = "idle"  # the pane reason when the pane freed since the idle poll: no stall notice, the next poll decides
STALL_UNREADABLE = ("the terminal pane cannot be read — exit Claude Code in the tab or close it, "
                    "and the message runs here")
STALL_TEXTS = MappingProxyType({
    "busy": "Claude Code is working in the terminal session — your message goes in when it is idle "
            "(Esc in the tab stops that work)",
    "typed": "text is typed in the terminal's input box — the bridge never touches it; send or clear it in the tab, "
             "or exit Claude Code there",
    "dialog": "a dialog is open in the terminal session — answer it in the tab",
    "mode": "the terminal pane is in a tmux mode (copy mode) — press q in the tab",
    "gone": STALL_UNREADABLE,
    "no-cli": STALL_UNREADABLE,
})


@dataclass(frozen=True)
class TerminalTurn:
    """How the head message's turn may start. attach None: the bridge's own process, whose turn mark is taken (the caller
    clears it once no process is left). attach set: the message goes into that terminal session, and nothing is marked."""

    attach: AttachSpec | None = None


def attach_spec(claim: terminal_claim.Claim, still_attached: Callable[[], bool],
                claim_uuid: Callable[[], str]) -> AttachSpec:
    return AttachSpec(socket=claim.socket, pane=claim.pane, uuid=claim.uuid, config_dir=claim.config_dir, cwd=claim.cwd,
                      pid=claim.pid, still_attached=still_attached, hooked=claim.hooked, claim_uuid=claim_uuid)


class TerminalWait:
    """The claim wait of the dispatcher. One instance per Dispatcher; wait() runs on the topic's runner thread."""

    def __init__(self, *, view_dir: str, removals: Removals, shutting_down: Callable[[], bool], clock: Clock,
                 logger: Logger, messenger: Messenger, poll_seconds: float,
                 pane_idle: Callable[[terminal_claim.Claim], bool] | None,
                 holding: Callable[[int], bool], tick: Callable[[int, int], int],
                 pane_reason: Callable[[terminal_claim.Claim], str] | None = None,
                 stall_seconds: float = STALL_SECONDS) -> None:
        self._view_dir = view_dir
        self._removals = removals
        self._is_shutting_down = shutting_down
        self._clock = clock
        self._logger = logger
        self._messenger = messenger
        self._poll = poll_seconds
        self._pane_idle = pane_idle  # None: no claim is attachable, every claim keeps the old wait
        self._pane_reason = pane_reason  # None: the stall notice names the default reason
        self._stall_seconds = stall_seconds
        self._holding = holding      # True while the topic's kept session is held (its hold is ticked while waiting)
        self._tick = tick            # one hold tick: returns the count of consecutive failed ticks
        self._lock = threading.Lock()  # guards _waiting only
        self._waiting: set[int] = set()  # the topics whose runner is in wait() now

    def waiting(self, thread_id: int) -> bool:
        """True while the topic's runner is in wait(): its head message waits for the terminal."""
        with self._lock:
            return thread_id in self._waiting

    def wait(self, thread_id: int, chat_id: int | str) -> TerminalTurn | None:
        """Waits until the head message may run. A TerminalTurn when it may (its turn mark is taken unless attach is set),
        or None when a shutdown came first: nothing was marked and the message stays pending. A topic removed meanwhile
        gets a plain TerminalTurn, so the turn drops the message itself. Without a view folder it returns at once."""
        if not self._view_dir:
            return TerminalTurn()
        with self._lock:
            self._waiting.add(thread_id)
        try:
            return self._wait(thread_id, chat_id)
        finally:
            with self._lock:
                self._waiting.discard(thread_id)

    def _wait(self, thread_id: int, chat_id: int | str) -> TerminalTurn | None:
        notified = False
        failures = 0
        stalled_since: float | None = None  # the monotonic time the attachable pane was first seen not idle
        told = False  # the stall notice went out for the current stall
        while not self._removals.active(thread_id):
            claim = self._read(thread_id)
            if claim is None:
                stalled_since, told = None, False
                if self._begin(thread_id):
                    return TerminalTurn()
            elif self._attachable(claim) and self._idle(claim):
                return TerminalTurn(attach=attach_spec(claim, self._still_attached(thread_id, claim),
                                                       self._claim_session(thread_id, claim.pid)))
            else:
                if not notified:  # the topic is told once per wait, with the text that fits the claim
                    notified = True
                    self._notify(chat_id, thread_id,
                                 WAITING_TEXT if self._attachable(claim) else terminal_claim.TERMINAL_NOTICE)
                if not self._attachable(claim):
                    stalled_since, told = None, False
                elif stalled_since is None:
                    stalled_since = self._clock.monotonic()
                elif not told and self._clock.monotonic() - stalled_since >= self._stall_seconds:
                    detail = self._stall_detail(claim)
                    if detail is not None:  # None: idle now, no notice; told stays False, the next poll decides
                        told = True
                        minutes = max(1, round((self._clock.monotonic() - stalled_since) / SECONDS_PER_MINUTE))
                        self._notify(chat_id, thread_id, STALL_TEXT.format(minutes=minutes, detail=detail))
            if self._is_shutting_down():
                return None
            if self._holding(thread_id):
                failures = self._tick(thread_id, failures)
            time.sleep(self._poll)
        return TerminalTurn()

    def _still_attached(self, thread_id: int, claim: terminal_claim.Claim) -> Callable[[], bool]:
        """The liveness check of an attached turn: the topic's claim is still the one this turn started with (same pid and
        pane). The view pane releases its claim when its CLI exits, so a released or replaced claim reads as gone. An
        unreadable claim file is taken as still held: the views are best-effort and must not end a turn by themselves."""
        def still() -> bool:
            try:
                now = terminal_claim.read_claim(self._view_dir, thread_id)
            except OSError as exc:
                self._logger.log(f"terminal: thread {thread_id} claim check failed: {str(exc)[:SEND_ERROR_CHARS]}")
                return True
            return now is not None and now.pid == claim.pid and now.pane == claim.pane
        return still

    def _claim_session(self, thread_id: int, pid: int) -> Callable[[], str]:
        """The session the tab's claim names now, read on each call: a /clear or /resume that the tab's hook reports
        moves it on. '' when the claim is gone or does not name this pid, and when the claim file cannot be read."""
        def current() -> str:
            try:
                claim = terminal_claim.own_claim(self._view_dir, thread_id, pid)
            except OSError:
                return ""
            return claim.uuid if claim is not None else ""
        return current

    def _attachable(self, claim: terminal_claim.Claim) -> bool:
        return self._pane_idle is not None and bool(claim.pane and claim.uuid)

    def _idle(self, claim: terminal_claim.Claim) -> bool:
        """The pane probe. A probe that raises reads as busy: the message keeps waiting and is logged."""
        try:
            return bool(self._pane_idle(claim))
        except Exception as exc:
            self._logger.log(f"terminal: pane probe failed: {str(exc)[:SEND_ERROR_CHARS]}")
            return False

    def _stall_detail(self, claim: terminal_claim.Claim) -> str | None:
        """The reason the pane is not idle, as the stall notice names it; None when the probe says the pane is idle now
        (it freed since the idle poll), so there is nothing to explain. A reason probe that raises reads as the
        default."""
        if self._pane_reason is None:
            return STALL_DEFAULT
        try:
            reason = self._pane_reason(claim)
        except Exception as exc:
            self._logger.log(f"terminal: pane reason probe failed: {str(exc)[:SEND_ERROR_CHARS]}")
            return STALL_DEFAULT
        if reason == IDLE_REASON:
            return None
        return STALL_TEXTS.get(reason, STALL_DEFAULT)

    def _read(self, thread_id: int) -> terminal_claim.Claim | None:
        try:
            return terminal_claim.read_claim(self._view_dir, thread_id)
        except OSError as exc:  # the views are best-effort: the claim is taken as absent
            self._logger.log(f"terminal: thread {thread_id} claim read failed: {str(exc)[:SEND_ERROR_CHARS]}")
            return None

    def _begin(self, thread_id: int) -> bool:
        try:
            return terminal_claim.begin_turn(self._view_dir, thread_id, self._clock)
        except OSError as exc:  # the views are best-effort: the turn runs, its feed may be missing
            self._logger.log(f"terminal: thread {thread_id} claim check failed: {str(exc)[:SEND_ERROR_CHARS]}")
            return True

    def _notify(self, chat_id: int | str, thread_id: int, text: str) -> None:
        try:
            self._messenger.send_text(chat_id, thread_id, text)
        except Exception as exc:  # the notice is lost, the message still waits and runs
            self._logger.log(f"terminal: thread {thread_id} notice not sent: {str(exc)[:SEND_ERROR_CHARS]}")
