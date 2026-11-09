"""Plan approval: what the buttons of a plan card (ExitPlanMode) do, for every transport.

A plan ask is never answered by a saved rule or by bypass: the owner reads the plan and picks one of four buttons.
- ✅ Run it / ✅ Ask before edits: the plan is approved. The ask is answered with a deny that carries
  PLAN_APPROVED_MESSAGE: the interactive TUI does not take an allow from the PermissionRequest hook for ExitPlanMode
  (it still draws its own "Exit plan mode?" dialog, which nobody can answer from Telegram), but it does take a deny.
  The model ends the turn, the topic's mode is stored as the picked one, and PlanFollowUp queues PLAN_GO_TEXT as the
  topic's next message, so the next turn starts in that mode and implements the plan. Print takes the same path.
- ✏️ Change: the next text in the topic is the owner's feedback; the ask is denied with it (PLAN_CHANGE_MESSAGE),
  and the model revises the plan and calls ExitPlanMode again, which posts a new card.
- ⛔ Reject: a plain deny.

This module holds the texts, the mode keys of the buttons and PlanFollowUp. The cards (core/cards.py) render the plan,
PermissionService (core/permissions.py) claims the ask, and bot/callbacks/approvals.py routes the taps.
"""

from types import MappingProxyType
from typing import Protocol


PLAN_CHANGE_QI = -1  # the 'question index' of a plan card waiting for typed feedback (ApprovalCards._await_text)
BYPASS_MODE = "bypassPermissions"
EDITS_MODE = "acceptEdits"
PLAN_MODES = MappingProxyType({"b": BYPASS_MODE, "e": EDITS_MODE})  # the callback key of each approve button -> the topic mode
PLAN_SENDER = "tg-bridge"
PLAN_APPROVED_MESSAGE = (
    "PLAN APPROVED by the owner in Telegram. This is an approval, not a rejection. Do not call any more tools in this "
    "turn: reply with one short line that you are starting, and end the turn. The bridge starts your next turn in "
    "{mode} permission mode; implement the approved plan then.")
PLAN_GO_TEXT = "✅ The plan is approved (permission mode: {mode}). Implement it now, as planned."
PLAN_CHANGE_MESSAGE = (
    "The owner read the plan in Telegram and wants changes before approving it:\n\n{text}\n\n"
    "Revise the plan accordingly and call ExitPlanMode again with the whole new plan.")


class _Sessions(Protocol):
    def set_mode(self, thread_id: int, mode: str) -> None: ...

    def get(self, thread_id: int) -> object | None: ...


class _Deliver(Protocol):
    def deliver(self, thread_id: int, topic: str, chat_id: int | str, sender: str, update_id: int | None,
                text: str) -> object: ...


class PlanFollowUp:
    """After an approval: stores the topic's new mode and queues the message that starts the implementing turn. It runs
    while the planning turn is still open, so the message waits in the queue and runs once that turn has ended."""

    def __init__(self, sessions: _Sessions, dispatcher: _Deliver) -> None:
        self._sessions = sessions
        self._dispatcher = dispatcher

    def __call__(self, thread_id: int, chat_id: int | str, mode: str) -> None:
        self._sessions.set_mode(thread_id, mode)
        row = self._sessions.get(thread_id)
        topic = str(getattr(row, "name", "") or thread_id)
        self._dispatcher.deliver(thread_id, topic, chat_id, PLAN_SENDER, None, PLAN_GO_TEXT.format(mode=mode))
