"""TmuxPane: the keys, the screen and the existence of one tmux pane (the attached session, attached_session.py).

The pane is the one `tg-bridge view` runs its Claude Code in. socket is the tmux server's socket path ('' for tmux's default
server), pane the pane id ('%3'). Every call is one tmux process with its arguments as a list, no shell. A tmux that cannot
run, a call that times out, a non-zero exit and an empty pane id are answers, not exceptions: the methods return False or ''.

The pure functions at the end read a captured screen: input_empty (the input box holds no text), working (the CLI is busy,
or a dialog of its own waits for a key), shows_cli (the screen is the CLI's: its box, or one of the states above) and idle
(input box empty and not working: a message can be pasted now), and screen_reason (why it cannot be, as one REASON_ value).
A pane in a tmux mode (copy-mode) takes no keys for the app, so ready() is False there, and paste() leaves the mode first;
reason() gives the same answer for the pane itself, with REASON_GONE and REASON_MODE too.

Threads: none of its own; the owner calls it from its own thread. Lock map: none.
"""

import subprocess
from collections.abc import Sequence
from typing import Protocol

from tg_bridge.clock import Clock, SystemClock

TMUX_TIMEOUT_SECONDS = 10
PASTE_GAP_SECONDS = 0.4          # between the paste and Enter, so the TUI has read the pasted text
PASTE_GAP_PER_CHUNK = 0.1        # a large paste needs longer: this much more per PASTE_CHUNK_BYTES of text
PASTE_CHUNK_BYTES = 2048
PASTE_GAP_MAX_SECONDS = 2.0
RULE = "─"                        # the horizontal rules that frame the input box
RULE_MIN_RUN = 8                  # a rule line starts with at least this many RULE glyphs (its label may follow)
PROMPT = "❯"                      # the prompt glyph, also the trust dialog's selection marker
NBSP = " "
BUSY_HINT = "esc to interrupt"    # the spinner line while a turn runs
DIALOG_FOOTER = "Enter to confirm · Esc to cancel"  # a selection dialog of the CLI waits for a key
TAIL_LINES = 6                    # without an input box, the busy hint and the dialog footer are looked for in the last lines
SPINNER_ABOVE_BOX = 3             # with a box, the spinner line is looked for from this many lines above its top rule
REASON_IDLE = "idle"              # a message can be pasted now
REASON_BUSY = "busy"              # the CLI runs a turn (its spinner shows BUSY_HINT)
REASON_DIALOG = "dialog"          # a selection dialog of the CLI waits for a key (its footer is DIALOG_FOOTER)
REASON_TYPED = "typed"            # the input box holds text
REASON_NO_CLI = "no-cli"          # the screen has no input box of the CLI
REASON_MODE = "mode"              # the pane is in a tmux mode (copy-mode): keys do not reach the app
REASON_GONE = "gone"              # tmux does not show the pane


class TmuxRunner(Protocol):
    """Runs `tmux <args>` and returns the finished process; text, when given, is its stdin. Raises OSError or
    SubprocessError when it cannot run."""

    def __call__(self, args: Sequence[str], timeout: float,
                 text: str | None = None) -> subprocess.CompletedProcess[str]: ...


def run_tmux(args: Sequence[str], timeout: float, text: str | None = None) -> subprocess.CompletedProcess[str]:
    """The real runner: one tmux process, output captured as text."""
    return subprocess.run(["tmux", *args], input=text, capture_output=True, text=True, timeout=timeout, check=False)


class TmuxPane:
    """One tmux pane on one server. Holds no state beyond its arguments."""

    def __init__(self, socket: str, pane: str, *, run: TmuxRunner = run_tmux, clock: Clock = SystemClock()) -> None:
        self._base = ["-S", socket] if socket else []
        self._pane = pane
        self._buffer = "tgbridge-" + ("".join(ch for ch in pane if ch.isdigit()) or "0")
        self._run = run
        self._clock = clock

    def capture(self) -> str:
        """The pane's visible text, one line per row. '' when tmux cannot show it."""
        return self._call(["capture-pane", "-p", "-t", self._pane]) or ""

    def paste(self, text: str) -> bool:
        """Leaves a tmux mode first (copy-mode would take the keys), pastes text as a bracketed paste (the TUI takes it as
        typed text; -r keeps the newlines, which tmux would otherwise turn into carriage returns), waits paste_gap, then
        presses Enter. False when any step fails; Enter is not sent after a failed paste."""
        if self.in_mode() and self._call(["send-keys", "-t", self._pane, "-X", "cancel"]) is None:
            return False
        if self._call(["load-buffer", "-b", self._buffer, "-"], text) is None:
            return False
        if self._call(["paste-buffer", "-p", "-r", "-d", "-b", self._buffer, "-t", self._pane]) is None:
            return False
        self._clock.sleep(paste_gap(text))
        return self._call(["send-keys", "-t", self._pane, "Enter"]) is not None

    def in_mode(self) -> bool:
        """True when the pane is in a tmux mode of its own (copy-mode, a choose tree): keys typed there never reach the app."""
        return (self._call(["display-message", "-p", "-t", self._pane, "#{pane_in_mode}"]) or "").strip() == "1"

    def ready(self) -> bool:
        """True when a message can be pasted into the pane now: it exists, it is in no tmux mode, and its screen is idle."""
        return self.exists() and not self.in_mode() and idle(self.capture())

    def reason(self) -> str:
        """Why a message cannot be pasted into the pane now, one of the REASON_ values; REASON_IDLE when it can. The pane
        must exist and be out of any tmux mode first, then its screen decides (screen_reason)."""
        if not self.exists():
            return REASON_GONE
        if self.in_mode():
            return REASON_MODE
        return screen_reason(self.capture())

    def escape(self) -> bool:
        """Presses Escape in the pane (the CLI's interrupt key)."""
        return self._call(["send-keys", "-t", self._pane, "Escape"]) is not None

    def exists(self) -> bool:
        """True when tmux shows the pane under its id."""
        shown = self._call(["display-message", "-p", "-t", self._pane, "#{pane_id}"])
        return shown is not None and shown.strip() == self._pane

    def _call(self, args: Sequence[str], text: str | None = None) -> str | None:
        """The stdout of one tmux call, or None when it could not run or exited non-zero. Without a pane id nothing runs:
        tmux would otherwise pick its own current pane."""
        if not self._pane:
            return None
        try:
            done = self._run([*self._base, *args], TMUX_TIMEOUT_SECONDS, text)
        except (OSError, subprocess.SubprocessError):
            return None
        return (done.stdout or "") if done.returncode == 0 else None


def paste_gap(text: str) -> float:
    """The seconds between a paste and its Enter: PASTE_GAP_SECONDS, plus PASTE_GAP_PER_CHUNK per PASTE_CHUNK_BYTES of the
    text (UTF-8), at most PASTE_GAP_MAX_SECONDS."""
    chunks = len(text.encode("utf-8")) / PASTE_CHUNK_BYTES
    return min(PASTE_GAP_SECONDS + PASTE_GAP_PER_CHUNK * chunks, PASTE_GAP_MAX_SECONDS)


def _is_rule(line: str) -> bool:
    """A horizontal rule, alone or with a label after the run ('────── tg:master ─'). The prompt line never starts with
    a run of rule glyphs, so it is not a rule."""
    return line.strip().startswith(RULE * RULE_MIN_RUN)


def _blank(text: str) -> bool:
    return not text.replace(NBSP, " ").strip()


def _last_lines(screen: str, count: int) -> list[str]:
    """The last count lines that are not blank at the bottom: capture-pane pads the screen with empty rows."""
    lines = screen.splitlines()
    while lines and _blank(lines[-1]):
        lines.pop()
    return lines[-count:]


def _input_box(lines: list[str]) -> tuple[int, int] | None:
    """The (top, bottom) rule rows of the input box: the last two rules, with a prompt line between them. None when the
    screen has no such box."""
    rules = [index for index, line in enumerate(lines) if _is_rule(line)]
    if len(rules) < 2:
        return None
    top, bottom = rules[-2], rules[-1]
    if not any(lines[index].lstrip().startswith(PROMPT) for index in range(top + 1, bottom)):
        return None
    return top, bottom


def input_empty(screen: str) -> bool:
    """True when the input box of the CLI holds no text. The box is the text between the last two horizontal rules; its
    prompt line is the last line starting with the glyph, and any line below the prompt, up to the bottom rule, counts
    as text too. No box on the screen is not empty: the screen is not an idle prompt then."""
    lines = screen.splitlines()
    box = _input_box(lines)
    if box is None:
        return False
    top, bottom = box
    prompts = [index for index in range(top + 1, bottom) if lines[index].lstrip().startswith(PROMPT)]
    first = prompts[-1]
    typed = lines[first].lstrip()[len(PROMPT):]
    return _blank(typed) and all(_blank(line) for line in lines[first + 1:bottom])


def _scanned(screen: str) -> list[str]:
    """The lines that the busy hint and the dialog footer are looked for in: with an input box, from SPINNER_ABOVE_BOX lines
    above its top rule to the end of the screen (the statusline sits below the box); without one, the last TAIL_LINES."""
    lines = screen.splitlines()
    box = _input_box(lines)
    return _last_lines(screen, TAIL_LINES) if box is None else lines[max(0, box[0] - SPINNER_ABOVE_BOX):]


def working(screen: str) -> bool:
    """True when the CLI is busy (its spinner shows 'esc to interrupt') or a selection dialog of its own waits for a key
    (its footer 'Enter to confirm · Esc to cancel'). The lines looked at are those of _scanned."""
    return any(BUSY_HINT in line.lower() or DIALOG_FOOTER in line for line in _scanned(screen))


def shows_cli(screen: str) -> bool:
    """True when the screen is the CLI's own: its input box is on it, or it is working or waits at a dialog of its own."""
    return _input_box(screen.splitlines()) is not None or working(screen)


def idle(screen: str) -> bool:
    """True when a message can be pasted now: the input box is empty and the CLI is not working."""
    return input_empty(screen) and not working(screen)


def screen_reason(screen: str) -> str:
    """Why a message cannot be pasted into the screen now, one of the REASON_ values: a dialog or a busy hint in _scanned
    (dialog first), then no input box, then text in the box. REASON_IDLE when none holds: then idle(screen) is True."""
    scan = _scanned(screen)
    if any(DIALOG_FOOTER in line for line in scan):
        return REASON_DIALOG
    if any(BUSY_HINT in line.lower() for line in scan):
        return REASON_BUSY
    if _input_box(screen.splitlines()) is None:
        return REASON_NO_CLI
    if not input_empty(screen):
        return REASON_TYPED
    return REASON_IDLE
