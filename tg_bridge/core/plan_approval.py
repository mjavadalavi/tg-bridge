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

from tg_bridge.core.topic_queues import Admission, Delivery

PLAN_CHANGE_QI = -1  # the 'question index' of a plan card waiting for typed feedback (ApprovalCards._await_text)
BYPASS_MODE = "bypassPermissions"
ASK_MODE = "default"  # Claude asks before each edit and command (acceptEdits would allow every edit)
PLAN_MODES = MappingProxyType({"b": BYPASS_MODE, "e": ASK_MODE})  # approve button key -> the topic mode
PLAN_SENDER = "tg-bridge"
PLAN_APPROVED_MESSAGE = (
    "PLAN APPROVED by the owner in Telegram. This is an approval, not a rejection. Do not call any more tools in this "
    "turn: reply with one short line that you are starting, and end the turn. The bridge starts your next turn in "
    "{mode} permission mode; implement the approved plan then.")
PLAN_GO_TEXT = "✅ The plan is approved (permission mode: {mode}). Implement it now, as planned."
PLAN_NOT_QUEUED_TEXT = ("⚠️ the plan is approved and the mode is now {mode}, but its turn could not be queued: "
                        "send 'implement the plan' when the queue has room.")
ERROR_TEXT_LIMIT = 150
PLAN_CHANGE_MESSAGE = (
    "The owner read the plan in Telegram and wants changes before approving it:\n\n{text}\n\n"
    "Revise the plan accordingly and call ExitPlanMode again with the whole new plan.")


class _Sessions(Protocol):
    def set_mode(self, thread_id: int, mode: str) -> None: ...

    def get(self, thread_id: int) -> object | None: ...


class _Deliver(Protocol):
    def deliver(self, thread_id: int, topic: str, chat_id: int | str, sender: str, update_id: int | None,
                text: str) -> Delivery: ...


class _Notify(Protocol):
    def send_text(self, chat_id: int | str, thread_id: int | str, text: str) -> None: ...


class _Log(Protocol):
    def log(self, line: str) -> None: ...


class PlanFollowUp:
    """After an approval: stores the topic's new mode and queues the message that starts the implementing turn. It runs
    before the planning turn is answered, so the message waits in the queue and runs once that turn has ended, in the
    new mode. It never raises: a refused message (queue full) or a failure is told to the topic and logged."""

    def __init__(self, sessions: _Sessions, dispatcher: _Deliver, messenger: _Notify, logger: _Log) -> None:
        self._sessions = sessions
        self._dispatcher = dispatcher
        self._messenger = messenger
        self._logger = logger

    def __call__(self, thread_id: int, chat_id: int | str, mode: str) -> None:
        try:
            self._sessions.set_mode(thread_id, mode)
            row = self._sessions.get(thread_id)
            topic = str(getattr(row, "name", "") or thread_id)
            text = PLAN_GO_TEXT.format(mode=mode)
            queued = self._dispatcher.deliver(thread_id, topic, chat_id, PLAN_SENDER, None, text)
        except Exception as exc:
            self._logger.log(f"thread {thread_id}: plan follow-up failed: {str(exc)[:ERROR_TEXT_LIMIT]}")
            queued = None
        if queued is None or queued.admission in (Admission.FULL, Admission.DROPPED):
            self._messenger.send_text(chat_id, thread_id, PLAN_NOT_QUEUED_TEXT.format(mode=mode))
