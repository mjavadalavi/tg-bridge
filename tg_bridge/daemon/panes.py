"""Panes: the tmux side of the bridge, panes_alive, type_to_pane and spawn_seat (C21).

Port of the live file's panes_alive, type_to_pane, spawn_seat and SPAWN_PREAMBLE (tg-bridge.py:1992-2032). Every tmux
call goes through the injected runner, so tests script tmux and no process starts. The waits (0.4 s between the text
and Enter, the 30 s prompt wait polled every 1.5 s) go through the injected Clock.

A tmux that cannot run (not installed, timed out) is an answer, not an exception: panes_alive returns no panes and the
send and spawn calls report a failure. The live file let the exception reach the poll thread instead.
"""

import subprocess
from collections.abc import Sequence
from typing import Protocol

from tg_bridge.clock import Clock

SPAWN_CWD = "<your-workspace>"  # the folder a new seat starts in; the live file's constant, set per install
TMUX_SESSION = "claude"  # new seats open as windows of this tmux session
SEAT_NAME_CHARS = 40
KEYS_GAP_SECONDS = 0.4  # between the typed text and Enter, as the live file waited
PROMPT_WAIT_SECONDS = 30
PROMPT_POLL_SECONDS = 1.5
PROMPT_GLYPH = "❯"
PROMPT_BUSY = "esc to interrupt"
PROMPT_SHELL_END = "$"
LIST_TIMEOUT = 10
SEND_TIMEOUT = 10
CAPTURE_TIMEOUT = 10
SPAWN_TIMEOUT = 20
EMPTY_TEXT = "empty text — nothing typed"
NOT_ALIVE_TEXT = (
    "pane {pane} is not alive — the target session was closed or moved to a new pane; update the topic config"
)
TYPED_TEXT = "typed into pane {pane}"
SPAWN_FAILED_TEXT = "spawn failed: {detail}"
TMUX_FAILED_TEXT = "tmux did not run: {detail}"
NO_PROMPT_TEXT = "window up (pane {pane}) but no prompt detected in 30s — send your message again"
ERROR_CHARS = 200
SPAWN_PREAMBLE = (
    "[tg-bridge · topic «{topic}»] {sender} is on Telegram — they CANNOT see this tmux pane. Rules: "
    '(1) Anything they must see goes through: ~/.local/bin/tg-bridge send "{topic}" "<reply>" — '
    "send a short note after every substantive step and always a final summary. "
    "(2) Never stay silent: if the request is vague, state your best interpretation and proceed "
    "(note your assumptions), or ask exactly ONE short clarifying question — via the send command. "
    "(3) Keep Telegram replies short plain text. "
    "--- {sender}'s message: "
)


class TmuxRunner(Protocol):
    """Runs `tmux <args>` and returns the finished process. Raises OSError or SubprocessError when it cannot run."""

    def __call__(self, args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]: ...


def run_tmux(args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """The real runner: one tmux process, output captured as text."""
    return subprocess.run(["tmux", *args], capture_output=True, text=True, timeout=timeout)


def seat_window_name(session_name: str) -> str:
    """The tmux window name of a seat: dots and spaces become dashes, at most SEAT_NAME_CHARS long."""
    return session_name.replace(".", "-").replace(" ", "-")[:SEAT_NAME_CHARS] or "seat"


def at_prompt(screen: str) -> bool:
    """Whether a captured pane shows the Claude TUI prompt (the glyph, the busy hint, or a shell prompt)."""
    return PROMPT_GLYPH in screen or PROMPT_BUSY in screen or screen.rstrip().endswith(PROMPT_SHELL_END)


class Panes:
    """Talks to tmux through the injected runner and waits through the injected clock. Holds no state of its own."""

    def __init__(self, *, clock: Clock, run: TmuxRunner = run_tmux, spawn_cwd: str = SPAWN_CWD) -> None:
        self._clock = clock
        self._run = run
        self._spawn_cwd = spawn_cwd

    @property
    def spawn_cwd(self) -> str:
        """The folder a spawned seat starts in, and the folder its topic's session row records."""
        return self._spawn_cwd

    def panes_alive(self) -> set[str]:
        """The ids of every live tmux pane (empty when tmux cannot run)."""
        result = self._tmux(["list-panes", "-a", "-F", "#{pane_id}"], LIST_TIMEOUT)
        return set(result.stdout.split()) if result is not None else set()

    def type_to_pane(self, pane: str, text: str) -> tuple[bool, str]:
        """Types the text into a live interactive session's pane, as if at the keyboard, then presses Enter.

        No AI turn starts here: the target session does its own work. Returns (ok, one-line info for the topic)."""
        if not (text or "").strip():
            return False, EMPTY_TEXT
        if pane not in self.panes_alive():
            return False, NOT_ALIVE_TEXT.format(pane=pane)
        flat = text.replace("\n", " ")
        if self._tmux(["send-keys", "-t", pane, "-l", "--", flat], SEND_TIMEOUT) is None:
            return False, TMUX_FAILED_TEXT.format(detail="send-keys did not run")
        self._clock.sleep(KEYS_GAP_SECONDS)
        self._tmux(["send-keys", "-t", pane, "Enter"], SEND_TIMEOUT)
        return True, TYPED_TEXT.format(pane=pane)

    def spawn_seat(self, command: str, session_name: str, topic: str, sender: str,
                   text: str) -> tuple[bool, str, str | None]:
        """Opens a live seat in a new window of the claude session, waits for its prompt, then types the first message
        behind SPAWN_PREAMBLE. Returns (ok, info, pane); pane is None when no window was made.

        When the window opens but no prompt shows within PROMPT_WAIT_SECONDS, the result is ok with the 'send your
        message again' info and the message is not typed."""
        return self.open_seat(command, session_name, SPAWN_PREAMBLE.format(topic=topic, sender=sender) + text)

    def open_seat(self, command: str, session_name: str, first_text: str) -> tuple[bool, str, str | None]:
        """The seat opener without the preamble: opens the window, waits for its prompt, then types first_text exactly as
        given. This is the live file's spawn_seat(sp_cmd, sess_name, first_text), which the `spawn` subcommand uses.
        Returns (ok, info, pane) like spawn_seat."""
        result = self._tmux(["new-window", "-d", "-P", "-F", "#{pane_id}", "-t", TMUX_SESSION,
                             "-n", seat_window_name(session_name), "-c", self._spawn_cwd, command], SPAWN_TIMEOUT)
        if result is None:
            return False, TMUX_FAILED_TEXT.format(detail="tmux could not start"), None
        pane = result.stdout.strip()
        if result.returncode != 0 or not pane.startswith("%"):
            detail = (result.stderr or result.stdout).strip()[:ERROR_CHARS]
            return False, SPAWN_FAILED_TEXT.format(detail=detail), None
        if not self._wait_for_prompt(pane):
            return True, NO_PROMPT_TEXT.format(pane=pane), pane
        ok, info = self.type_to_pane(pane, first_text)
        return ok, info, pane

    def _wait_for_prompt(self, pane: str) -> bool:
        deadline = self._clock.monotonic() + PROMPT_WAIT_SECONDS
        while self._clock.monotonic() < deadline:
            result = self._tmux(["capture-pane", "-p", "-t", pane], CAPTURE_TIMEOUT)
            if result is not None and at_prompt(result.stdout):
                return True
            self._clock.sleep(PROMPT_POLL_SECONDS)
        return False

    def _tmux(self, args: list[str], timeout: float) -> subprocess.CompletedProcess[str] | None:
        """One tmux call. None when tmux cannot run; a non-zero exit is still returned for the caller to read."""
        try:
            return self._run(args, timeout)
        except (OSError, subprocess.SubprocessError):
            return None
