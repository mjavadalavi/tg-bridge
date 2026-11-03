"""ask_first (card H1 and section 5 D1): which guard's confirm card a /new or /resume change waits for.

Scenarios: the terminal guard holds the change, so its card and toast win even when the keep guard would hold it too;
when only the keep guard holds it, its card and toast come back; when neither holds it, or neither is wired in, the
result is None and the change applies at once. Each guard is asked with its own apply. The two toast texts are the
ones the taps answer with. The guards are stubs: the real guards have their own tests (test_terminal_guard.py and
test_keep_confirm.py).
"""

import unittest

from tg_bridge.bot.commands.change_guard import TERMINAL_TOAST_TEXT, ask_first
from tg_bridge.bot.keep_guard import KEEP_TOAST_TEXT, ApplyResult
from tg_bridge.bot.pickers import Picker

THREAD = 7
WHAT = "session"


class StubGuard:
    """A guard whose hold() returns the card it was given, or None, and records the call."""

    def __init__(self, card: Picker | None) -> None:
        self.card = card
        self.calls: list[tuple[int, str, object]] = []

    def hold(self, thread: int, what: str, apply: object) -> Picker | None:
        self.calls.append((thread, what, apply))
        return self.card


def apply_terminal() -> ApplyResult:
    return ApplyResult(True, "terminal")


def apply_keep() -> ApplyResult:
    return ApplyResult(True)


class TerminalWins(unittest.TestCase):
    def test_the_terminal_card_and_toast_win_when_both_guards_hold(self) -> None:
        terminal_card, keep_card = Picker("terminal", []), Picker("keep", [])
        terminal, keep = StubGuard(terminal_card), StubGuard(keep_card)
        asked = ask_first(terminal, keep, THREAD, WHAT, apply_terminal, apply_keep)
        self.assertEqual(asked, (terminal_card, TERMINAL_TOAST_TEXT))
        self.assertEqual(keep.calls, [], "the keep guard is not asked when the terminal guard holds the change")

    def test_the_terminal_guard_gets_its_own_apply_and_the_thread_and_what(self) -> None:
        terminal = StubGuard(Picker("terminal", []))
        ask_first(terminal, None, THREAD, WHAT, apply_terminal, apply_keep)
        self.assertEqual(terminal.calls, [(THREAD, WHAT, apply_terminal)])


class KeepHoldsAlone(unittest.TestCase):
    def test_the_keep_card_and_toast_come_back_when_only_the_keep_guard_holds(self) -> None:
        keep_card = Picker("keep", [])
        terminal, keep = StubGuard(None), StubGuard(keep_card)
        asked = ask_first(terminal, keep, THREAD, WHAT, apply_terminal, apply_keep)
        self.assertEqual(asked, (keep_card, KEEP_TOAST_TEXT))
        self.assertEqual(keep.calls, [(THREAD, WHAT, apply_keep)], "the keep guard gets its own apply")


class NeitherHolds(unittest.TestCase):
    def test_none_when_neither_guard_holds_the_change(self) -> None:
        terminal, keep = StubGuard(None), StubGuard(None)
        self.assertIsNone(ask_first(terminal, keep, THREAD, WHAT, apply_terminal, apply_keep))
        self.assertEqual(len(terminal.calls), 1)
        self.assertEqual(len(keep.calls), 1)

    def test_none_when_no_guard_is_wired_in(self) -> None:
        self.assertIsNone(ask_first(None, None, THREAD, WHAT, apply_terminal, apply_keep))


class ToastTexts(unittest.TestCase):
    def test_the_terminal_toast_says_the_session_is_in_a_terminal_tab(self) -> None:
        self.assertEqual(TERMINAL_TOAST_TEXT, "the session is open in a terminal tab — confirm first")

    def test_the_keep_toast_is_the_keep_guard_text(self) -> None:
        self.assertEqual(KEEP_TOAST_TEXT, "a background task is running — confirm first")


if __name__ == "__main__":
    unittest.main()
