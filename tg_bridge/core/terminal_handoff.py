"""terminal_handoff: the two places where a terminal session and the topic's stored session meet (TurnRunner).

adopt_ended(): a topic that has no session yet takes the session its view tab ran and closed (the pane's .session
record), so the next normal turn resumes it instead of starting a second one. A stored uuid is replaced only when it is
one of the sessions this same tab ran (started with or moved off) and the tab ended on another (a /clear in the tab).
A session that Telegram bound meanwhile from elsewhere (/new, /resume <other>) is therefore never overwritten.

move_claim(): an attached turn that moved to another transcript (a /clear or /resume in the tab) moves the live claim to the
new uuid, so the next message follows that file at once. Only the claim this turn attached to (its pid) is changed, and
only while the claim still names the uuid the turn started from: the tab's SessionStart hook is authoritative, so the
daemon's move never overwrites a newer uuid the hook wrote.

A tab session is adopted only when it ran under the same Claude config folder as the topic's launcher: a transcript
lives under its folder's projects/, so a resume under another folder would start a fresh session.

Threads: the turn's runner thread. The file work runs under the topic's flock (terminal_claim); the repositories lock
themselves. Nothing here holds a lock of its own.
"""

import os
from dataclasses import replace

from tg_bridge.core import terminal_claim
from tg_bridge.core.turn_kit import TurnPlan
from tg_bridge.logging_ import Logger
from tg_bridge.store.sessions import HistoryRepo, SessionRepo

SHORT_ERROR_CHARS = 120


def adopt_ended(view_dir: str, plan: TurnPlan, *, sessions: SessionRepo, history: HistoryRepo, logger: Logger) -> TurnPlan:
    """The plan with the view session's uuid as its stored uuid, when the topic had none or the tab moved off the stored one.
    A stored uuid is replaced only when it is one of the sessions this same tab ran (it started from it or moved off
    it), so a session Telegram bound meanwhile from elsewhere is never overwritten. The session is also adopted only
    when it ran under the plan's launcher config folder (or the record names none). The record is read and removed
    whatever the outcome; a stored uuid outside the tab's lineage, or a session of another config folder, is kept and
    the terminal's uuid is logged as not adopted."""
    if not view_dir:
        return plan
    try:
        ended = terminal_claim.take_ended(view_dir, plan.thread_id)
    except OSError as exc:  # best-effort: the turn runs with the topic's own session
        logger.log(f"terminal: thread {plan.thread_id} session record failed: {str(exc)[:SHORT_ERROR_CHARS]}")
        return plan
    if ended is None or not ended.uuid:
        return plan
    lineage = {ended.started_uuid, *ended.seen} - {""}
    moved = bool(plan.stored_uuid) and plan.stored_uuid in lineage and ended.uuid != plan.stored_uuid
    if plan.stored_uuid and not moved:
        logger.log(f"terminal session {ended.uuid} not adopted: topic already has {plan.stored_uuid}")
        return plan
    if not _same_folder(ended.config_dir, plan.launcher.config_dir):
        logger.log(f"terminal session {ended.uuid} not adopted: it ran under {ended.config_dir}, "
                   f"the topic's launcher uses {plan.launcher.config_dir}")
        return plan
    sessions.set_uuid(plan.thread_id, ended.uuid)
    history.adopt_session(plan.thread_id, ended.uuid, plan.launcher.command, plan.cwd, plan.backend.name)
    if moved:
        logger.log(
            f"terminal session {ended.uuid} adopted by thread {plan.thread_id} (the tab moved off {plan.stored_uuid})"
        )
    else:
        logger.log(f"terminal session {ended.uuid} adopted by thread {plan.thread_id}")
    return replace(plan, stored_uuid=ended.uuid)


def _same_folder(ended_dir: str, plan_dir: str) -> bool:
    """True when the record names no config folder (an old record) or names the plan's launcher folder."""
    if not ended_dir:
        return True
    return os.path.realpath(ended_dir) == os.path.realpath(plan_dir)


def move_claim(view_dir: str, thread_id: int, pid: int, uuid: str, logger: Logger, *, expected: str) -> None:
    """Points the live claim held by pid at uuid, when the claim still names expected (the uuid the attached turn
    started from). A claim that is gone, names another pid, or already names another uuid (the SessionStart hook wrote a
    newer one meanwhile) is left alone: the hook's uuid is authoritative and the daemon's move never overwrites it.
    A hooked claim is left alone as well: the SessionStart hook is its only writer, and the bridge relocates a hooked
    tab only to the session the claim already names, so the daemon never needs to write it. The hooked check runs
    inside update_uuid under the claim's lock, so a hook that marks the claim meanwhile is never overwritten."""
    if not view_dir:
        return
    try:
        moved = terminal_claim.update_uuid(view_dir, thread_id, pid, uuid, expected=expected, only_unhooked=True)
    except OSError as exc:  # best-effort: the next message then follows the stale claim, as before
        logger.log(f"terminal: thread {thread_id} claim move failed: {str(exc)[:SHORT_ERROR_CHARS]}")
        return
    if moved:
        logger.log(f"terminal: thread {thread_id} claim now names session {uuid}")
