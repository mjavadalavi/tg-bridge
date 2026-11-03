"""TerminalTaps and its confirm card (cards H2, K-D): the close, later and cancel buttons answer the held change.

Scenarios: 'cancel' toasts and edits the card to the kept text and applies nothing; 'later' toasts and edits the card to
the deferred text with one cancel button, and the change runs when the tab is free (settle) and the card is edited to
the done text through the report, after the deferred edit; 'close' toasts the closing text, ends the tab's Claude Code
(a fake close that removes the claim) and edits the card to the closed text once, applying the change once; a close
with no child to end, or a tab that does not close in time, leaves the change pending and the card with its later and
cancel buttons; a second tap on a card that was already answered toasts and edits to the expired text and applies
nothing more. The taps are called directly with a Tap built over a fake reply side, and spawn runs its call at once.
A claim is a real file that names this process (os.getpid()), so the claim is live. The guard's clock is a FakeClock,
so the close wait does not sleep. Unless a test says otherwise, the watcher is a NoWatcher and the test calls settle.
"""

import json
import os
import threading
import unittest
from collections.abc import Callable
from typing import Any

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.bot.callbacks.router import Tap
from tg_bridge.bot.callbacks.terminal_taps import TERMINAL_CLOSING_TOAST, TerminalTaps
from tg_bridge.bot.keep_guard import ApplyResult
from tg_bridge.bot.terminal_guard import (
    TERMINAL_CLOSE_KIND,
    TERMINAL_CLOSE_TIMEOUT_TEXT,
    TERMINAL_CLOSE_WAIT_SECONDS,
    TERMINAL_CLOSED_TEXT,
    TERMINAL_DEFERRED_TEXT,
    TERMINAL_DONE_TEXT,
    TERMINAL_EXPIRED_TEXT,
    TERMINAL_KEPT_TEXT,
    TERMINAL_LATER_DONE_TEXT,
    TERMINAL_LATER_KIND,
    TERMINAL_NO_BUTTON,
    TERMINAL_NO_CHILD_TEXT,
    TERMINAL_NO_KIND,
    TerminalGuard,
)
from tg_bridge.core import terminal_claim
from tg_bridge.core.terminal_claim import Claim
from tg_bridge.telegram.keyboards import PICKER_PREFIX, CallbackData

THREAD = 5150
WHAT = "session"
CHAT_ID = 1
MESSAGE_ID = 2
CHILD_PID = 4242
CLOSED_DONE = TERMINAL_CLOSED_TEXT.format(text=TERMINAL_DONE_TEXT)
LATER_DONE = TERMINAL_LATER_DONE_TEXT.format(text=TERMINAL_DONE_TEXT)
TIMEOUT_TEXT = TERMINAL_CLOSE_TIMEOUT_TEXT.format(seconds=TERMINAL_CLOSE_WAIT_SECONDS)


class Applied:
    """An apply callable that counts its runs and returns the result it was built with."""

    def __init__(self, result: ApplyResult = ApplyResult(True)) -> None:
        self.result = result
        self.runs = 0

    def __call__(self) -> ApplyResult:
        self.runs += 1
        return self.result


class FakeReplies:
    """The Telegram side of a tap: records the toasts and the card edits, in order."""

    def __init__(self) -> None:
        self.toasts: list[str] = []
        self.edits: list[str] = []
        self.edit_rows: list[list[list[Any]]] = []

    def answer(self, callback_id: str, text: str) -> None:
        self.toasts.append(text)

    def edit(self, chat_id: Any, message_id: int, text: str, rows: list[list[Any]]) -> None:
        self.edits.append(text)
        self.edit_rows.append(rows)


class NoWatcher:
    """Stands in for the watcher thread of a deferred change: the test settles the change itself."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def start(self) -> None:
        pass


class ThreadWatcher:
    """The watcher on a thread of its own, as in production, so a test can race it against the tap's own edit."""
    started: list[threading.Thread] = []

    def __init__(self, **kwargs: Any) -> None:
        self.thread = threading.Thread(**kwargs)

    def start(self) -> None:
        ThreadWatcher.started.append(self.thread)
        self.thread.start()


def run_now(target: Callable[..., object], *args: object, **kwargs: object) -> None:
    """Stands in for the background-thread helper: the call runs at once, on this thread."""
    target(*args, **kwargs)


def make_tap(replies: FakeReplies, args: tuple[str, ...], prefix: str = PICKER_PREFIX) -> Tap:
    return Tap(callback_id="cb", prefix=prefix, args=args, chat_id=CHAT_ID, message_id=MESSAGE_ID, message_text="",
               replies=replies)  # type: ignore[arg-type]


class TerminalTapsCase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.view = os.path.join(self.home.home, "view")
        os.makedirs(self.view, mode=0o700)
        self.clock = FakeClock()
        self.watcher_class: Callable[..., Any] = NoWatcher
        self.guard = TerminalGuard(view_dir=self.view, close=self.close_claim, clock=self.clock,
                                   start_thread=lambda **kwargs: self.watcher_class(**kwargs))  # type: ignore[arg-type]
        self.taps = TerminalTaps(terminal=self.guard, spawn=run_now)
        self.replies = FakeReplies()
        self.applied = Applied()
        self.closes = 0
        self.close_removes_claim = True
        with open(terminal_claim.claim_path(self.view, THREAD), "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started": 1.0}, handle)
        self.card = self.guard.hold(THREAD, WHAT, self.applied)
        self.token = self.card.rows[0][0][1].args[2]
        self.later_row, self.cancel_row = self.card.rows[1], self.card.rows[2]

    def close_claim(self, claim: Claim) -> bool:
        """The fake close of the tab's Claude Code: the tab's claim goes away, as it does when the CLI ends."""
        self.closes += 1
        if self.close_removes_claim:
            os.remove(terminal_claim.claim_path(self.view, THREAD))
        return True

    def remove_claim(self) -> None:
        """The owner closed the tab on its own."""
        os.remove(terminal_claim.claim_path(self.view, THREAD))

    def set_child(self) -> None:
        terminal_claim.set_child(self.view, THREAD, os.getpid(), CHILD_PID)

    def tap(self, kind: str, token: str) -> None:
        tap = make_tap(self.replies, (str(THREAD), kind, token))
        if kind == TERMINAL_CLOSE_KIND:
            self.taps._close(tap)
        elif kind == TERMINAL_LATER_KIND:
            self.taps._later(tap)
        else:
            self.taps._cancel(tap)

    def clear_replies(self) -> None:
        self.replies.toasts.clear()
        self.replies.edits.clear()
        self.replies.edit_rows.clear()


class CancelTapTests(TerminalTapsCase):
    def test_cancel_does_not_apply_and_edits_the_card_to_kept(self) -> None:
        self.tap(TERMINAL_NO_KIND, self.token)
        self.assertEqual(self.applied.runs, 0)
        self.assertEqual(self.replies.toasts, [TERMINAL_KEPT_TEXT])
        self.assertEqual(self.replies.edits, [TERMINAL_KEPT_TEXT])
        self.assertEqual(self.replies.edit_rows, [[]])

    def test_cancel_after_a_close_with_no_child_edits_the_card_to_kept(self) -> None:
        self.tap(TERMINAL_CLOSE_KIND, self.token)
        self.clear_replies()
        self.tap(TERMINAL_NO_KIND, self.token)
        self.assertEqual(self.applied.runs, 0)
        self.assertEqual(self.replies.edits, [TERMINAL_KEPT_TEXT])
        self.assertEqual(self.replies.edit_rows, [[]])


class LaterTapTests(TerminalTapsCase):
    def test_later_keeps_the_change_and_edits_the_card_to_deferred_with_one_cancel_button(self) -> None:
        self.tap(TERMINAL_LATER_KIND, self.token)
        self.assertEqual(self.applied.runs, 0)
        self.assertEqual(self.replies.toasts, [TERMINAL_DEFERRED_TEXT])
        self.assertEqual(self.replies.edits, [TERMINAL_DEFERRED_TEXT])
        self.assertEqual(self.replies.edit_rows, [[self.cancel_row]])

    def test_the_change_runs_when_the_tab_is_free_and_the_done_text_comes_after_the_deferred_edit(self) -> None:
        self.tap(TERMINAL_LATER_KIND, self.token)
        self.remove_claim()
        self.assertEqual(self.guard.settle(THREAD), LATER_DONE)
        self.assertEqual(self.applied.runs, 1)
        self.assertEqual(self.replies.edits, [TERMINAL_DEFERRED_TEXT, LATER_DONE])
        self.assertEqual(self.replies.edit_rows, [[self.cancel_row], []])

    def test_later_after_a_close_with_no_child_defers_and_the_change_runs_when_the_tab_is_free(self) -> None:
        self.tap(TERMINAL_CLOSE_KIND, self.token)
        self.clear_replies()
        self.tap(TERMINAL_LATER_KIND, self.token)
        self.assertEqual(self.replies.toasts, [TERMINAL_DEFERRED_TEXT])
        self.assertEqual(self.replies.edits, [TERMINAL_DEFERRED_TEXT])
        self.assertEqual(self.replies.edit_rows, [[self.cancel_row]])
        self.remove_claim()
        self.assertEqual(self.guard.settle(THREAD), LATER_DONE)
        self.assertEqual(self.replies.edits, [TERMINAL_DEFERRED_TEXT, LATER_DONE])
        self.assertEqual(self.applied.runs, 1)

    def test_the_result_waits_for_the_deferred_edit_when_the_watcher_runs_first(self) -> None:
        self.watcher_class = ThreadWatcher
        self.remove_claim()
        self.tap(TERMINAL_LATER_KIND, self.token)
        for thread in ThreadWatcher.started:
            thread.join(10)
        self.assertEqual(self.applied.runs, 1)
        self.assertEqual(self.replies.edits, [TERMINAL_DEFERRED_TEXT, LATER_DONE])


class CloseTapTests(TerminalTapsCase):
    def test_close_ends_the_tab_then_runs_the_change_and_edits_the_card_to_closed_once(self) -> None:
        self.set_child()
        self.tap(TERMINAL_CLOSE_KIND, self.token)
        self.assertEqual(self.closes, 1)
        self.assertEqual(self.replies.toasts, [TERMINAL_CLOSING_TOAST])
        self.assertEqual(self.applied.runs, 1)
        self.assertEqual(self.replies.edits, [TERMINAL_CLOSING_TOAST, CLOSED_DONE])
        self.assertEqual(self.replies.edit_rows, [[], []])

    def test_close_edits_the_card_to_closing_without_buttons_before_the_close_runs(self) -> None:
        calls: list[tuple[Callable[..., object], tuple[object, ...]]] = []

        def record(target: Callable[..., object], *args: object) -> None:
            calls.append((target, args))

        self.taps = TerminalTaps(terminal=self.guard, spawn=record)
        self.set_child()
        self.tap(TERMINAL_CLOSE_KIND, self.token)
        self.assertEqual(self.closes, 0)
        self.assertEqual(self.applied.runs, 0)
        self.assertEqual(self.replies.edits, [TERMINAL_CLOSING_TOAST])
        self.assertEqual(self.replies.edit_rows, [[]])
        self.assertEqual(len(calls), 1)
        target, args = calls[0]
        target(*args)
        self.assertEqual(self.closes, 1)
        self.assertEqual(self.applied.runs, 1)
        self.assertEqual(self.replies.edits, [TERMINAL_CLOSING_TOAST, CLOSED_DONE])
        self.assertEqual(self.replies.edit_rows, [[], []])

    def test_close_with_no_child_keeps_the_change_and_shows_the_later_and_cancel_buttons(self) -> None:
        self.tap(TERMINAL_CLOSE_KIND, self.token)
        self.assertEqual(self.closes, 0)
        self.assertEqual(self.applied.runs, 0)
        self.assertEqual(self.replies.toasts, [TERMINAL_CLOSING_TOAST])
        self.assertEqual(self.replies.edits, [TERMINAL_CLOSING_TOAST, TERMINAL_NO_CHILD_TEXT])
        self.assertEqual(self.replies.edit_rows, [[], [self.later_row, self.cancel_row]])

    def test_a_tab_that_does_not_close_in_time_keeps_the_change_with_the_later_and_cancel_buttons(self) -> None:
        self.set_child()
        self.close_removes_claim = False
        self.tap(TERMINAL_CLOSE_KIND, self.token)
        self.assertEqual(self.applied.runs, 0)
        self.assertEqual(self.replies.edits, [TERMINAL_CLOSING_TOAST, TIMEOUT_TEXT])
        self.assertEqual(self.replies.edit_rows, [[], [self.later_row, self.cancel_row]])

    def test_a_second_tap_on_an_answered_card_shows_expired_and_applies_nothing(self) -> None:
        self.set_child()
        self.tap(TERMINAL_CLOSE_KIND, self.token)
        self.clear_replies()
        self.tap(TERMINAL_CLOSE_KIND, self.token)
        self.assertEqual(self.closes, 1)
        self.assertEqual(self.applied.runs, 1)
        self.assertEqual(self.replies.toasts, [TERMINAL_CLOSING_TOAST])
        self.assertEqual(self.replies.edits, [TERMINAL_CLOSING_TOAST, TERMINAL_EXPIRED_TEXT])
        self.assertEqual(self.replies.edit_rows, [[], []])


if __name__ == "__main__":
    unittest.main()
