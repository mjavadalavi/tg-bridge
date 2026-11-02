"""terminal_handoff: the two places where a terminal session and the topic's stored session meet (TurnRunner).

adopt_ended(): a topic that has no session yet takes the session its view tab ran and closed (the pane's .session record),
so the next normal turn resumes it instead of starting a second one. A stored uuid is never replaced here.

move_claim(): an attached turn that moved to another transcript (a /clear or /resume in the tab) moves the live claim to the
new uuid, so the next message follows that file at once. Only the claim this turn attached to (its pid) is changed.

Threads: the turn's runner thread. The file work runs under the topic's flock (terminal_claim); the repositories lock
themselves. Nothing here holds a lock of its own.
"""

from dataclasses import replace

from tg_bridge.core import terminal_claim
from tg_bridge.core.turn_kit import TurnPlan
from tg_bridge.logging_ import Logger
from tg_bridge.store.sessions import HistoryRepo, SessionRepo

SHORT_ERROR_CHARS = 120


def adopt_ended(view_dir: str, plan: TurnPlan, *, sessions: SessionRepo, history: HistoryRepo, logger: Logger) -> TurnPlan:
    """The plan with the view session's uuid as its stored uuid, when the topic had none. The record is read and removed
    whatever the outcome: a stored uuid that is already there is kept and the terminal's uuid is logged as not adopted."""
    if not view_dir:
        return plan
    try:
        ended = terminal_claim.take_ended(view_dir, plan.thread_id)
    except OSError as exc:  # best-effort: the turn runs with the topic's own session
        logger.log(f"terminal: thread {plan.thread_id} session record failed: {str(exc)[:SHORT_ERROR_CHARS]}")
        return plan
    if ended is None or not ended.uuid:
        return plan
    if plan.stored_uuid:
        logger.log(f"terminal session {ended.uuid} not adopted: topic already has {plan.stored_uuid}")
        return plan
    sessions.set_uuid(plan.thread_id, ended.uuid)
    history.adopt_session(plan.thread_id, ended.uuid, plan.launcher.command, plan.cwd, plan.backend.name)
    logger.log(f"terminal session {ended.uuid} adopted by thread {plan.thread_id}")
    return replace(plan, stored_uuid=ended.uuid)


def move_claim(view_dir: str, thread_id: int, pid: int, uuid: str, logger: Logger) -> None:
    """Points the live claim held by pid at uuid. A claim that is gone or names another pid is left alone."""
    if not view_dir:
        return
    try:
        moved = terminal_claim.update_uuid(view_dir, thread_id, pid, uuid)
    except OSError as exc:  # best-effort: the next message then follows the stale claim, as before
        logger.log(f"terminal: thread {thread_id} claim move failed: {str(exc)[:SHORT_ERROR_CHARS]}")
        return
    if moved:
        logger.log(f"terminal: thread {thread_id} claim now names session {uuid}")
