"""The startup dialogs of the Claude TUI: the folder trust dialog, the external imports dialog, and any other selection dialog
that asks before the prompt. The screen tests that find each dialog and its prompt, and the key presses that answer them.
The screen is the whitespace-free text of PtyProcess.compact_screen(), as the TUI draws it.
"""

from collections.abc import Callable
from typing import Protocol

from tg_bridge.clock import Clock

TRUST_MARKERS = ("Yes,Itrustthisfolder", "Quicksafetycheck")  # the trust dialog, in the whitespace-free screen text
PROMPT_GLYPH = "❯"
NO_CHOICE = "No,exit"           # the dialog's first choice: a glyph right before it is the dialog's cursor, not the prompt
CONFIRM_FOOTER = "Entertoconfirm·Esctocancel"  # the footer of a selection dialog: a glyph before it is that dialog's cursor
EXTERNAL_TITLE = "Externalimports:"
EXTERNAL_CHOICES = ("No,disableexternalimports", "Yes,allowexternalimports")  # the external dialog's two choices
SELECTION_CHOICES = (NO_CHOICE, TRUST_MARKERS[0], "No,disableexternalimports", "Yes,allowexternalimports")
                                  # the choices a dialog's cursor sits right before
UNKNOWN_DIALOG_CHARS = 120      # how much of an unknown dialog's screen text goes to the log
TRUST_KEY_GAP_SECONDS = 0.8     # Down, a pause, then the cursor is read again: the dialog redraws after each key
TRUST_SETTLE_SECONDS = 1.0      # the dialog's screen must stay unchanged this long first: it may still be mounting
TRUST_POLL_SECONDS = 0.1        # how often the screen is read while it settles
TRUST_ATTEMPTS = 3              # Down presses at most before the cursor must be on Yes


class Screen(Protocol):
    """What the startup checks need from the PTY session."""

    def compact_screen(self) -> str: ...

    def press_down(self) -> bool: ...

    def press_enter(self) -> bool: ...

    def press_esc(self) -> bool: ...

    def alive(self) -> bool: ...

    def wait_for(self, predicate: Callable[[str], bool], timeout: float) -> bool: ...


def _trust_dialog_shown(compact: str) -> bool:
    return any(marker in compact for marker in TRUST_MARKERS)


def _is_selection_cursor(compact: str, glyph: int) -> bool:
    """True when the glyph at glyph is a dialog's cursor: it sits right before one of the dialogs' choices."""
    after = glyph + len(PROMPT_GLYPH)
    return any(compact.startswith(choice, after) for choice in SELECTION_CHOICES)


def _is_dialog_glyph(compact: str, glyph: int) -> bool:
    """True for a dialog's cursor: right before a known choice, or with a dialog footer after it (a fragment of the dialog
    can end at its cursor before the choices arrive)."""
    return _is_selection_cursor(compact, glyph) or compact.find(CONFIRM_FOOTER, glyph) >= 0


def _prompt_glyph(compact: str) -> int:
    """The index of the last prompt glyph that is not a dialog's cursor, or -1. A fragment of a dialog can end at its
    cursor before the other choice arrives: that glyph must not count as the prompt."""
    glyph = compact.rfind(PROMPT_GLYPH)
    while glyph >= 0 and _is_dialog_glyph(compact, glyph):
        glyph = compact.rfind(PROMPT_GLYPH, 0, glyph)
    return glyph


def _prompt_ready(compact: str) -> bool:
    """The prompt glyph is drawn after the last trust-dialog text. The screen keeps old text (it is not a terminal
    emulator), so the trust text stays visible after the dialog is accepted: a plain 'no trust text' check would never
    see the prompt again. A dialog's own cursor glyphs never count (see _prompt_glyph)."""
    glyph = _prompt_glyph(compact)
    return glyph >= 0 and glyph > max(compact.rfind(marker) for marker in TRUST_MARKERS)


def _dialog_up(compact: str) -> bool:
    """True when a dialog's footer is drawn after the last glyph, so the dialog is the screen's last thing. Old text stays
    on the screen, so this can also hold for a dialog that is already gone: the caller settles the screen first."""
    return compact.rfind(CONFIRM_FOOTER) > compact.rfind(PROMPT_GLYPH)


def _external_dialog(compact: str) -> bool:
    """True when the external imports dialog is the screen's current dialog (its title is on the screen, and its choices
    follow the last glyph)."""
    tail = compact[max(compact.rfind(PROMPT_GLYPH), 0):]
    return _dialog_up(compact) and EXTERNAL_TITLE in compact and any(choice in tail for choice in EXTERNAL_CHOICES)


def _trust_dialog_current(compact: str) -> bool:
    """True when the last glyph is the trust dialog's cursor: on 'No, exit' or on 'Yes, I trust this folder'."""
    glyph = compact.rfind(PROMPT_GLYPH)
    after = glyph + len(PROMPT_GLYPH)
    return glyph >= 0 and (compact.startswith(NO_CHOICE, after) or compact.startswith(TRUST_MARKERS[0], after))


def _trust_cursor_on_yes(compact: str) -> bool:
    """True when the dialog's selection marker is on 'Yes, I trust this folder': the last prompt glyph is directly before
    that choice. The screen keeps old text, so the last glyph is the current cursor."""
    glyph = compact.rfind(PROMPT_GLYPH)
    return glyph >= 0 and compact.startswith(TRUST_MARKERS[0], glyph + len(PROMPT_GLYPH))


def _trust_footer(compact: str) -> bool:
    """True when the footer after the last glyph belongs to the trust dialog: its text is on the screen and its cursor is
    the last glyph. The dialog's own footer ('Enter to confirm') stays on the screen after it is accepted, so it is not
    an unknown dialog; the prompt, or the next dialog, is drawn after it."""
    return _trust_dialog_shown(compact) and _trust_dialog_current(compact)


def startup_visible(compact: str) -> bool:
    """True once the screen shows something the startup checks decide on: the trust dialog, the prompt, or a dialog."""
    return _trust_dialog_shown(compact) or _prompt_ready(compact) or _dialog_up(compact)


def prompt_showing(screen: Screen) -> bool:
    """True when the screen shows the prompt now (a dialog answered by the user, for one, has gone)."""
    return _prompt_ready(screen.compact_screen())


def must_accept(screen: Screen, clock: Clock, prompt_timeout: float) -> bool:
    """True when the folder trust dialog is the current screen and must be accepted. A screen without trust text is decided
    at once. With trust text the screen settles first: a read can split the dialog right after its cursor glyph, and a
    fragment must not decide. A prompt, or another dialog drawn after the trust text, is not the trust dialog."""
    if not _trust_dialog_shown(screen.compact_screen()):
        return False
    _settle(screen, clock, prompt_timeout)
    return _trust_dialog_current(screen.compact_screen())


def accept_trust_dialog(screen: Screen, clock: Clock, prompt_timeout: float) -> bool:
    """Presses Down until the cursor is on 'Yes', then Enter. Enter is pressed only when the cursor is seen on Yes: on
    'No, exit' it ends the TUI. The dialog may still be mounting, so the screen first settles. False when the cursor
    never reaches Yes; no Enter is pressed then. After Enter the screen is waited on until it changes from the one Enter
    was pressed on (the dialog's text stays up until the next paint), bounded by prompt_timeout."""
    _settle(screen, clock, prompt_timeout)
    for _ in range(TRUST_ATTEMPTS):
        if _trust_cursor_on_yes(screen.compact_screen()):
            break
        screen.press_down()
        clock.sleep(TRUST_KEY_GAP_SECONDS)
    if not _trust_cursor_on_yes(screen.compact_screen()):
        return False
    before = screen.compact_screen()
    screen.press_enter()
    _wait_past(screen, clock, before, prompt_timeout)
    return True


def await_prompt(screen: Screen, clock: Clock, timeout: float, log: Callable[[str], None]) -> str | None:
    """Waits for the prompt. The external imports dialog is declined once with Esc, which is its own default (disable),
    never 'allow'. Any other dialog that is still up once the screen settles is unknown: nothing is typed into it. None
    when the prompt is ready, else the reason it is not."""
    deadline = clock.monotonic() + timeout
    declined = False
    while clock.monotonic() < deadline:
        if _prompt_ready(screen.compact_screen()):
            return None
        _settle(screen, clock, deadline - clock.monotonic())
        compact = screen.compact_screen()
        if _prompt_ready(compact):
            return None
        if not _dialog_up(compact):
            continue
        if not _external_dialog(compact) and _trust_footer(compact):
            continue  # the accepted trust dialog's own footer, left on the screen: the next paint decides
        if not _external_dialog(compact):
            log(f"unknown dialog on startup: {compact[-UNKNOWN_DIALOG_CHARS:]}")
            return "no prompt: an unknown dialog is open, nothing was typed"
        if not declined:
            declined = True
            screen.press_esc()
            log("declined the external imports dialog")
    return "no prompt after the startup dialogs"


def get_past_startup(screen: Screen, clock: Clock, timeout: float, log: Callable[[str], None]) -> str | None:
    """Accepts the folder trust dialog when it is the current screen, then waits for the prompt (await_prompt). None when
    the prompt is ready, else the reason the session cannot go on, for the caller to log."""
    if must_accept(screen, clock, timeout):
        if not screen.alive():
            return "the TUI exited during the trust dialog"
        if not accept_trust_dialog(screen, clock, timeout):
            return "the folder trust dialog was not accepted"
        log("accepted the folder trust dialog")
    return await_prompt(screen, clock, timeout, log)


def startup(screen: Screen, clock: Clock, prompt_timeout: float, log: Callable[[str], None]) -> str | None:
    """The whole startup of a spawned TUI: waits for a screen that decides, gets past the dialogs (get_past_startup). None
    when the prompt is ready, else the reason it is not. The reason is logged once, as is 'prompt ready'."""
    if not screen.wait_for(startup_visible, prompt_timeout):
        log("no prompt or trust dialog (the TUI may have exited)")
        return "no prompt or trust dialog"
    problem = get_past_startup(screen, clock, prompt_timeout, log)
    log(problem or "prompt ready")
    return problem


def _wait_past(screen: Screen, clock: Clock, before: str, timeout: float) -> None:
    """Returns once the compact screen differs from before, or at the timeout: the screen after Enter may be drawn late."""
    deadline = clock.monotonic() + timeout
    while screen.compact_screen() == before and clock.monotonic() < deadline:
        clock.sleep(TRUST_POLL_SECONDS)


def _settle(screen: Screen, clock: Clock, prompt_timeout: float) -> None:
    """Returns once the compact screen has stayed unchanged for TRUST_SETTLE_SECONDS, or at the prompt timeout."""
    deadline = clock.monotonic() + prompt_timeout
    last = screen.compact_screen()
    since = clock.monotonic()
    while clock.monotonic() - since < TRUST_SETTLE_SECONDS and clock.monotonic() < deadline:
        clock.sleep(TRUST_POLL_SECONDS)
        current = screen.compact_screen()
        if current != last:
            last, since = current, clock.monotonic()
