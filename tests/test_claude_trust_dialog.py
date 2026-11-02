"""_prompt_ready and its helpers (trust_dialog.py): the prompt glyph counts as the prompt only when it is not the dialog's
own selection cursor. The screen is the whitespace-free text the TUI draws, so a fragment of the dialog can end at the
cursor on 'No, exit' before the 'Yes, I trust this folder' line arrives; that fragment is not a ready prompt. A glyph
after the dialog's text, once the dialog is accepted, is the prompt."""

import unittest

from tests.fakes.clock import FakeClock
from tg_bridge.agents.claude.trust_dialog import (_prompt_ready, _trust_dialog_shown, accept_trust_dialog, await_prompt,
                                                 get_past_startup, must_accept)

QUESTION = "Quicksafetycheck:Isthisaprojectyoucreatedoryoutrustit?"
YES = "Yes,Itrustthisfolder"
NO = "No,exit"
GLYPH = "❯"
PROMPT_TIMEOUT = 45.0
STALE_TRUST = QUESTION + NO + GLYPH + YES  # the trust text stays on the screen once the dialog is accepted
FOOTER = "Entertoconfirm·Esctocancel"  # the trust dialog's footer, compact: it stays on the screen after Enter too
TRUST_SCREEN = STALE_TRUST + FOOTER  # the trust dialog with the cursor on Yes, as it is shown before Enter
LATE_PAINT = 5.0  # the next paint after Enter comes this long after the screen was first read (3 s after Enter)
# the external imports dialog as the whitespace-free screen text shows it (the live screen's own '...' included)
EXTERNAL = ("Externalimports:/home/mjavadalavi/.claude/RTK.md Important:OnlyuseClaudeCodewithfilesyoutrust..."
            + GLYPH + "No,disableexternalimports Yes,allowexternalimports Entertoconfirm·Esctocancel")
UNKNOWN = "Pickamodelforthissession" + GLYPH + "SonnetOpus Entertoconfirm·Esctocancel"


class TimedScreen:
    """A screen whose compact text changes at set times of the fake clock: frames is (time, text) pairs in time order."""

    def __init__(self, clock: FakeClock, frames: list[tuple[float, str]]) -> None:
        self._clock = clock
        self._frames = frames
        self.presses: list[str] = []

    def compact_screen(self) -> str:
        now = self._clock.monotonic()
        text = ""
        for at, frame in self._frames:
            if at <= now:
                text = frame
        return text

    def press_down(self) -> bool:
        self.presses.append("down")
        return True

    def press_enter(self) -> bool:
        self.presses.append("enter")
        return True

    def press_esc(self) -> bool:
        self.presses.append("esc")
        return True

    def alive(self) -> bool:
        return True


class PromptReadyTests(unittest.TestCase):
    def test_a_fragment_ending_at_the_cursor_on_no_is_not_the_prompt(self) -> None:
        fragment = QUESTION + GLYPH + NO  # the Yes choice has not arrived yet
        self.assertTrue(_trust_dialog_shown(fragment))
        self.assertFalse(_prompt_ready(fragment))

    def test_a_full_dialog_with_the_cursor_on_no_is_not_the_prompt(self) -> None:
        self.assertFalse(_prompt_ready(QUESTION + GLYPH + NO + YES))

    def test_a_full_dialog_with_the_cursor_on_yes_is_not_the_prompt(self) -> None:
        self.assertFalse(_prompt_ready(QUESTION + NO + GLYPH + YES))

    def test_only_cursor_glyphs_on_the_dialog_are_not_the_prompt(self) -> None:
        self.assertFalse(_prompt_ready(QUESTION + GLYPH + NO + GLYPH + NO))

    def test_a_prompt_drawn_after_the_accepted_dialog_is_ready(self) -> None:
        accepted = QUESTION + NO + GLYPH + YES + "Entertoconfirm" + GLYPH
        self.assertTrue(_prompt_ready(accepted))

    def test_a_plain_prompt_with_no_dialog_is_ready(self) -> None:
        self.assertTrue(_prompt_ready("Welcomeback" + GLYPH))


class ReactiveScreen:
    """A screen whose text changes on a key press: after_esc replaces it on Esc, after_enter on Enter (None keeps it).
    Records every key press."""

    def __init__(self, text: str, after_esc: str | None = None, after_enter: str | None = None) -> None:
        self.text = text
        self.after_esc = after_esc
        self.after_enter = after_enter
        self.presses: list[str] = []

    def compact_screen(self) -> str:
        return self.text

    def press_down(self) -> bool:
        self.presses.append("down")
        return True

    def press_enter(self) -> bool:
        self.presses.append("enter")
        if self.after_enter is not None:
            self.text = self.after_enter
        return True

    def press_esc(self) -> bool:
        self.presses.append("esc")
        if self.after_esc is not None:
            self.text = self.after_esc
        return True

    def alive(self) -> bool:
        return True


class StartupDialogTests(unittest.TestCase):
    def test_the_external_imports_dialog_is_not_the_prompt_even_with_the_trust_text_above_it(self) -> None:
        self.assertFalse(_prompt_ready(EXTERNAL))
        self.assertFalse(_prompt_ready(STALE_TRUST + EXTERNAL))

    def test_the_external_dialog_is_declined_with_esc_then_the_prompt_after_it_is_ready(self) -> None:
        screen = ReactiveScreen(STALE_TRUST + EXTERNAL, after_esc=STALE_TRUST + EXTERNAL + GLYPH)
        log: list[str] = []
        self.assertIsNone(await_prompt(screen, FakeClock(), PROMPT_TIMEOUT, log.append))
        self.assertEqual(screen.presses, ["esc"])  # once, and never Enter or Down: 'allow' is not chosen
        self.assertEqual(log, ["declined the external imports dialog"])

    def test_the_external_dialog_after_the_trust_text_is_not_the_trust_dialog(self) -> None:
        screen = ReactiveScreen(STALE_TRUST + EXTERNAL)
        self.assertFalse(must_accept(screen, FakeClock(), PROMPT_TIMEOUT))
        self.assertEqual(screen.presses, [])

    def test_an_unknown_confirm_dialog_is_not_the_prompt_and_nothing_is_typed(self) -> None:
        self.assertFalse(_prompt_ready(UNKNOWN))
        screen = ReactiveScreen(UNKNOWN)
        log: list[str] = []
        reason = await_prompt(screen, FakeClock(), PROMPT_TIMEOUT, log.append)
        self.assertIsNotNone(reason)
        self.assertEqual(screen.presses, [])
        self.assertTrue(log and log[0].startswith("unknown dialog on startup: "))
        self.assertIn("Pickamodel", log[0])

    def test_get_past_startup_accepts_the_trust_dialog_then_declines_the_external_one(self) -> None:
        clock = FakeClock()
        screen = ReactiveScreen(QUESTION + NO + GLYPH + YES, after_enter=STALE_TRUST + EXTERNAL,
                                after_esc=STALE_TRUST + EXTERNAL + GLYPH)
        log: list[str] = []
        self.assertIsNone(get_past_startup(screen, clock, PROMPT_TIMEOUT, log.append))
        self.assertEqual(screen.presses, ["enter", "esc"])
        self.assertEqual(log, ["accepted the folder trust dialog", "declined the external imports dialog"])


class DelayedPaintTests(unittest.TestCase):
    """The next paint after Enter is late. The accepted trust dialog keeps its footer on the screen meanwhile: that footer
    is not an unknown dialog, and the prompt or the external dialog that comes later is what decides."""

    def test_the_trust_footer_left_on_the_screen_is_waited_on_and_a_late_prompt_is_ready(self) -> None:
        clock = FakeClock()
        start = clock.monotonic()
        screen = TimedScreen(clock, [(start, TRUST_SCREEN), (start + 3.0, TRUST_SCREEN + GLYPH)])
        log: list[str] = []
        self.assertIsNone(await_prompt(screen, clock, PROMPT_TIMEOUT, log.append))
        self.assertEqual(log, [])  # no unknown dialog, and nothing was typed or pressed
        self.assertEqual(screen.presses, [])

    def test_accept_waits_for_the_screen_to_change_after_enter(self) -> None:
        clock = FakeClock()
        start = clock.monotonic()
        screen = TimedScreen(clock, [(start, TRUST_SCREEN), (start + LATE_PAINT, TRUST_SCREEN + GLYPH)])
        self.assertTrue(accept_trust_dialog(screen, clock, PROMPT_TIMEOUT))
        self.assertEqual(screen.presses, ["enter"])
        self.assertGreaterEqual(clock.monotonic(), start + LATE_PAINT)  # Enter was pressed at ~2 s, the wait runs on

    def test_a_late_prompt_after_the_accepted_trust_dialog_is_ready(self) -> None:
        clock = FakeClock()
        start = clock.monotonic()
        screen = TimedScreen(clock, [(start, TRUST_SCREEN), (start + LATE_PAINT, TRUST_SCREEN + GLYPH)])
        log: list[str] = []
        self.assertIsNone(get_past_startup(screen, clock, PROMPT_TIMEOUT, log.append))
        self.assertEqual(screen.presses, ["enter"])
        self.assertEqual(log, ["accepted the folder trust dialog"])

    def test_a_late_external_dialog_after_the_accepted_trust_dialog_is_declined_not_reported_unknown(self) -> None:
        clock = FakeClock()
        start = clock.monotonic()
        external = STALE_TRUST + EXTERNAL
        screen = TimedScreen(clock, [(start, TRUST_SCREEN), (start + LATE_PAINT, external),
                                     (start + LATE_PAINT + 2.0, external + GLYPH)])
        log: list[str] = []
        self.assertIsNone(get_past_startup(screen, clock, PROMPT_TIMEOUT, log.append))
        self.assertEqual(screen.presses, ["enter", "esc"])  # Esc declines the external dialog; never Enter or Down
        self.assertEqual(log, ["accepted the folder trust dialog", "declined the external imports dialog"])


class MustAcceptTests(unittest.TestCase):
    def test_a_fragment_split_after_the_cursor_is_accepted_once_the_full_dialog_settles(self) -> None:
        fragment = QUESTION + GLYPH  # a read that ends right after the cursor glyph, before 'No, exit' arrives
        full = QUESTION + GLYPH + NO + YES
        self.assertTrue(_prompt_ready(fragment))  # the bare fragment alone would read as the prompt
        clock = FakeClock()
        screen = TimedScreen(clock, [(0.0, fragment), (0.3, full)])
        self.assertTrue(must_accept(screen, clock, PROMPT_TIMEOUT))
        self.assertTrue(clock.sleeps)  # the screen was given time to settle before the decision

    def test_a_settled_full_dialog_is_accepted(self) -> None:
        clock = FakeClock()
        screen = TimedScreen(clock, [(0.0, QUESTION + NO + GLYPH + YES)])
        self.assertTrue(must_accept(screen, clock, PROMPT_TIMEOUT))

    def test_a_screen_with_no_trust_text_is_decided_at_once(self) -> None:
        clock = FakeClock()
        screen = TimedScreen(clock, [(0.0, "Welcomeback" + GLYPH)])
        self.assertFalse(must_accept(screen, clock, PROMPT_TIMEOUT))
        self.assertEqual(clock.sleeps, [])  # no settle wait without trust text

    def test_a_prompt_drawn_after_the_dialog_text_is_not_accepted(self) -> None:
        accepted = QUESTION + NO + GLYPH + YES + "Entertoconfirm" + GLYPH
        clock = FakeClock()
        screen = TimedScreen(clock, [(0.0, QUESTION + NO + GLYPH + YES), (0.5, accepted)])
        self.assertFalse(must_accept(screen, clock, PROMPT_TIMEOUT))
        self.assertEqual(screen.presses, [])  # nothing is typed into a prompt that is already there


if __name__ == "__main__":
    unittest.main()
