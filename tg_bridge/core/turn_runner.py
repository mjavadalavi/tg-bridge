"""TurnRunner: one user turn on an AgentSession (C14, neutral).

Port of the live _run_turn_body and run_turn_v2 (tg-bridge.py:1718-1985). The backend, the permission service, the feed
and every repository are injected, so the same runner serves Claude and Codex and is tested with FakeBackend.

One turn: plan (launcher precedence, allow and deny tools, RULES text) -> the active:<thread> marker -> an attempt (open
the session, attach the permissions, register it as live, start, send the user text, wait for the end) -> at most one
relaunch as a fresh session when the stored session is gone -> the turns row (its insert also stamps sessions.last_active)
-> the feed summary -> the reply (answer, err).

Lock map: TurnRunner holds no lock of its own. LiveRegistry, WaitRouter and Database each lock themselves, and the
per-turn objects (TurnState, AnswerCollector, SessionRecorder) lock their own state. Nothing here holds a lock while it
calls another service.
"""

import json
import threading
import uuid as uuidlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from tg_bridge.agents.base import AgentBackend, AgentSession, SessionSpec
from tg_bridge.agents.events import AgentEvent, SessionStarted, TurnResult, UsageEvent
from tg_bridge.agents.registry import BackendRegistry, UnknownBackend
from tg_bridge.clock import Clock, utc_stamp
from tg_bridge.config import Config, LauncherError, LauncherSpec
from tg_bridge.core.events_bus import TurnEventBus
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.live import AbortFlags, LiveRegistry
from tg_bridge.core.media import PHOTO_NOTE
from tg_bridge.core.permissions import BYPASS_MODE, PermissionService, TurnBinding
from tg_bridge.core.turn_listeners import (
    AnswerCollector,
    EndGate,
    ResultFreezer,
    SessionRecorder,
    UsageRecorder,
    WaitRouter,
)
from tg_bridge.core.turn_state import SlotLike, TurnOutcome, TurnPhase, TurnState
from tg_bridge.core.turn_text import (
    LAUNCH_ERROR_CHARS,
    NO_CONVERSATION,
    TIMEZONE,
    compose_reply,
    format_rules,
    fresh_note,
    whole_seconds,
)
from tg_bridge.feed.format import final_header
from tg_bridge.logging_ import Logger
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.rules import RuleRepo
from tg_bridge.store.sessions import HistoryRepo, SessionRepo
from tg_bridge.store.turns import TurnRepo

ACTIVE_PREFIX = "active:"
DEFAULT_ALLOWED_TOOLS = "Read,Glob,Grep"
KILL_SWITCH_TOOL = "KillShell"  # refused by the CLI even in bypass mode
POLL_SECONDS = 1.0  # the wait loop's tick: caps, stop taps and dead-process checks (the live file's done.wait(1))
STDERR_LINES = 20  # the stderr lines searched for the resume error
TAIL_LINES = 10  # the stderr lines shown in a process error
MAX_LAUNCHES = 2  # the first launch and the one fresh relaunch after a missing resume
GRACEFUL_SECONDS = 10.0  # after a normal end the CLI gets this long to exit on its own (input closed, no interrupt)


class TurnProgress(Protocol):
    """The feed of one attempt (TurnFeed satisfies it). Called by the turn runner only."""

    def start(self) -> None: ...

    def on_event(self, event: AgentEvent) -> None: ...

    def close(self) -> None: ...

    def finish(self, header: str) -> None: ...


FeedFactory = Callable[[int | str, int, Callable[[], bool]], TurnProgress]


@dataclass(frozen=True)
class TurnReply:
    """What the turn produced. answer and err are both None only when a shutdown refused the spawn: nothing ran, and
    the caller re-marks the inbox row for a later try."""

    answer: str | None
    err: str | None
    outcome: TurnOutcome


@dataclass(frozen=True)
class TurnPlan:
    """Everything decided before the first launch of a turn."""

    thread_id: int
    chat_id: int
    topic: str
    text: str
    images: tuple[str, ...]
    launcher: LauncherSpec
    backend: AgentBackend
    cwd: str
    mode: str
    model: str | None
    allow_rules: tuple[str, ...]
    deny_tools: tuple[str, ...]
    system_rules: str
    stored_uuid: str


@dataclass(frozen=True)
class _Attempt:
    """The facts of one launch, gathered for the reply, the turns row and the feed summary."""

    refused: bool
    outcome: TurnOutcome
    result: TurnResult | None
    usage: UsageEvent | None
    texts: tuple[str, ...]
    tools: int
    limit: float | None
    stop_tap: bool
    aborted: bool
    launch_err: str | None
    exit_code: int | None
    tail: str
    resume_failed: bool
    feed: TurnProgress
    started_wall: float
    started_mono: float


def active_key(thread_id: int) -> str:
    """The kv key of a turn in progress. It survives a restart on purpose (recover() reads it)."""
    return f"{ACTIVE_PREFIX}{thread_id}"


class _HandoffClearer:
    """Discards the pending handoff digest once the fresh session that was given it has started. A turn that dies before
    SessionStarted never reaches this, so its digest stays in kv for the next try."""

    def __init__(self, handoff: HandoffService, thread_id: int) -> None:
        self._handoff = handoff
        self._thread_id = thread_id

    def on_event(self, event: AgentEvent) -> None:
        if isinstance(event, SessionStarted):
            self._handoff.discard(self._thread_id)


class TurnRunner:
    """Runs user turns. One instance per App; each run() is one turn on one topic, called from that topic's dispatcher."""

    def __init__(
        self, *, config: Config, registry: BackendRegistry, live: LiveRegistry, permissions: PermissionService,
        waits: WaitRouter, sessions: SessionRepo, history: HistoryRepo, rules: RuleRepo, turns: TurnRepo,
        kv: KvRepo, handoff: HandoffService, feeds: FeedFactory, clock: Clock, logger: Logger,
        base_env: Mapping[str, str], errfile_dir: str, poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self._config = config
        self._registry = registry
        self._live = live
        self._permissions = permissions
        self._waits = waits
        self._sessions = sessions
        self._history = history
        self._rules = rules
        self._turns = turns
        self._kv = kv
        self._handoff = handoff
        self._feeds = feeds
        self._clock = clock
        self._logger = logger
        self._base_env = dict(base_env)
        self._errfile_dir = errfile_dir
        self._poll_seconds = poll_seconds

    def run(self, *, thread_id: int, chat_id: int, topic: str, text: str, sender: str,
            slot: SlotLike | None = None, images: tuple[str, ...] = ()) -> TurnReply:
        """One user turn. Sets the active marker, runs the turn, and removes the marker unless a shutdown began.
        images are local photo paths for native image input (they are dropped for a backend without it)."""
        key = active_key(thread_id)
        self._kv.set(key, json.dumps({"chat_id": chat_id, "name": topic,
                                      "started": utc_stamp(self._clock.now())}))
        try:
            reply = self._run_body(thread_id, chat_id, topic, text, sender, slot, images)
        finally:
            if not self._live.is_shutting_down():  # a shutdown keeps the marker on purpose
                self._kv.delete(key)
        if reply.answer is None and reply.err is None:  # refused by a shutdown before any input: nothing ran
            self._kv.delete(key)
        return reply

    # planning

    def _run_body(self, thread_id: int, chat_id: int, topic: str, text: str, sender: str,
                  slot: SlotLike | None, images: tuple[str, ...]) -> TurnReply:
        try:
            plan = self._plan(thread_id, chat_id, topic, text, sender, images)
        except (LauncherError, UnknownBackend) as exc:  # refused before any session id is stored
            return TurnReply(None, f"launcher failed: {exc}", TurnOutcome.ERROR)
        recorder = SessionRecorder(sessions=self._sessions, history=self._history, kv=self._kv,
                                   thread_id=thread_id, text=text, cwd=plan.cwd,
                                   launcher=plan.launcher.command, backend=plan.backend.name,
                                   known_uuid=plan.stored_uuid)
        preassigned = plan.backend.capabilities.preassigned_id
        resume_id = plan.stored_uuid or None
        new_id = None
        if resume_id is None and preassigned:  # the id is stored BEFORE the process exists
            new_id = str(uuidlib.uuid4())
            recorder.remember(new_id)
        attempt = self._attempt(plan, recorder, resume_id, new_id, slot)
        fresh = ""
        if resume_id is not None and self._relaunch_fresh(attempt):
            uid = resume_id
            fresh = fresh_note(uid=uid, cwd=plan.cwd, launcher=plan.launcher.command, keeps_id=preassigned,
                               home=self._config.home)
            self._logger.log(f"resume {uid} not found — restarting as new session "
                             + ("with same id" if preassigned else "as a new thread"))
            attempt = self._attempt(plan, recorder, None, uid if preassigned else None, slot)
        return self._finish_turn(plan, attempt, fresh)

    def _plan(self, thread_id: int, chat_id: int, topic: str, text: str, sender: str,
              images: tuple[str, ...]) -> TurnPlan:
        row = self._sessions.get(thread_id)
        if row is None:
            raise RuntimeError(f"topic {thread_id} has no sessions row")
        cfgt = self._config.topic(topic)
        launcher = self._config.resolve_launcher(row.launcher_pick, str(cfgt.get("launcher") or ""), row.launcher)
        backend = self._registry.for_launcher(launcher)
        mode = row.mode or self._config.default_mode()
        allow: tuple[str, ...] = ()
        if mode != BYPASS_MODE:  # bypass ignores the allow list, as the live file does
            allowed = cfgt.get("allowed_tools", DEFAULT_ALLOWED_TOOLS)
            allow = tuple(str(rule) for rule in [allowed, *self._rules.texts(thread_id)] if rule)
        denied = cfgt.get("deny_tools") or []
        if isinstance(denied, str):
            denied = [denied]
        if not backend.capabilities.images and images:
            # a backend without image input gets each saved photo as a text note, so a photo is never dropped silently;
            # a note the text already carries is not added twice
            notes = [PHOTO_NOTE.format(path=path) for path in images]
            missing = [note for note in notes if note not in text]
            text = "\n".join([text, *missing]).strip()
            images = ()
        return TurnPlan(
            thread_id=thread_id, chat_id=chat_id, topic=topic, text=text, images=tuple(images),
            launcher=launcher, backend=backend,
            cwd=row.cwd or str(cfgt.get("cwd") or "") or self._config.home, mode=mode, model=row.model or None,
            allow_rules=allow, deny_tools=(KILL_SWITCH_TOOL, *(str(tool) for tool in denied)),
            system_rules=format_rules(topic, sender), stored_uuid=row.uuid,
        )

    def _relaunch_fresh(self, attempt: _Attempt) -> bool:
        """True when the stored session was not found and nothing else ended the turn (cap, stop tap)."""
        return not attempt.refused and attempt.resume_failed and attempt.limit is None and not attempt.aborted

    # one launch

    def _attempt(self, plan: TurnPlan, recorder: SessionRecorder, resume_id: str | None, new_id: str | None,
                 slot: SlotLike | None) -> _Attempt:
        thread = plan.thread_id
        turn = TurnState(thread, self._clock, slot, max_active_seconds=self._config.max_turn_seconds(),
                         hard_cap_seconds=self._config.hard_cap_seconds())
        feed = self._feeds(plan.chat_id, thread, lambda: turn.phase is TurnPhase.WAITING_APPROVAL)
        done = threading.Event()
        collector = AnswerCollector(done)
        usage = UsageRecorder()
        binding = TurnBinding(thread_id=thread, chat_id=plan.chat_id, cwd=plan.cwd, rules=plan.backend.rules,
                              mode=plan.mode)
        started_wall, started_mono = self._clock.now(), self._clock.monotonic()
        digest = self._pending_digest(thread, resume_id)
        # the freezer leads: the active timer stops at the result before any listener (or the wait loop) can act on it
        listeners = [ResultFreezer(turn.mark_result_received), feed, self._permissions.bind_turn(binding), usage,
                     recorder, collector]
        if digest:
            listeners.append(_HandoffClearer(self._handoff, thread))
        bus = TurnEventBus(listeners, self._logger)
        gate = EndGate(bus.publish, self._logger)  # whatever the process sends after its end never reaches the listeners
        spec = self._spec(plan, resume_id, new_id, digest)
        session: AgentSession | None = None
        launch_err: str | None = None
        refused = False
        registered = False
        stop_code: int | None = None
        self._waits.register(thread, turn)
        try:
            session = plan.backend.open_session(spec, gate)
            self._permissions.attach(binding, session)
            registered = self._live.begin_turn(thread, session, feed)
            if registered:
                refused, launch_err = self._drive(plan, turn, session, feed, done)
            else:
                refused = True  # a shutdown began: the spawn is refused
        except Exception as exc:  # a launch that failed before or during start
            launch_err = f"launcher failed: {str(exc)[:LAUNCH_ERROR_CHARS]}"
        finally:
            if session is not None:
                stop_code = session.stop(grace=10)  # cheap when the process has already exited; its exit code is kept
            if registered:
                self._live.end_turn(thread, session, feed)
            self._permissions.void_all(binding)  # the turn is over: its open permission cards are closed
            self._waits.unregister(thread, turn)
            feed.close()  # the feed posts what is left before the answer goes out
        flags = AbortFlags(False, False) if refused else self._live.consume_abort(thread)
        if flags.aborted:  # ⏭ Now or a ⏹ tap: the stop was requested on purpose
            turn.request_stop()
        limit = turn.limit_hit  # recorded inside the wait loop only: the teardown above never reclassifies a finished turn
        result = collector.result
        outcome = turn.finish(result, launch_failed=launch_err is not None,
                              shutting_down=self._live.is_shutting_down())
        stderr = session.stderr_tail(STDERR_LINES) if session is not None else ""
        return _Attempt(
            refused=refused, outcome=outcome, result=result, usage=usage.usage, texts=collector.texts,
            tools=collector.tools, limit=limit, stop_tap=flags.stop_tap, aborted=flags.aborted,
            launch_err=launch_err,
            # the reader reports the exit only after the stdout EOF, which a child can delay past the stop: the code
            # the stop reaped is the fallback, as the live file reads the process's own return code
            exit_code=collector.exit_code if collector.exit_code is not None else stop_code,
            tail=(session.stderr_tail(TAIL_LINES).strip() if session is not None else ""),
            resume_failed=bool(result is not None and result.resume_failed) or NO_CONVERSATION in stderr,
            feed=feed, started_wall=started_wall, started_mono=started_mono,
        )

    def _drive(self, plan: TurnPlan, turn: TurnState, session: AgentSession, feed: TurnProgress,
               done: threading.Event) -> tuple[bool, str | None]:
        """Starts the session, sends the text and waits for the turn to end. Returns (refused, launch_err)."""
        try:
            feed.start()
            turn.start()
            session.start()
            if self._live.is_shutting_down():  # the shutdown snapshot may have missed this process: stop it here
                session.stop(grace=1)
                return True, None
            turn.mark_running()
            session.send_user(plan.text, plan.images)
            self._wait(turn, session, done, plan.thread_id)
            return False, None
        except Exception as exc:
            return False, f"launcher failed: {str(exc)[:LAUNCH_ERROR_CHARS]}"

    def _wait(self, turn: TurnState, session: AgentSession, done: threading.Event, thread_id: int) -> None:
        """Waits until the turn has a result or its process is gone, checking the caps, the stop taps and a shutdown
        on every tick. A cap stops the session here; a dead process ends the wait after the dead-process delay.
        A normal end (a result arrived, no cap) closes the input and gives the CLI GRACEFUL_SECONDS to exit by itself:
        no interrupt is sent, so the CLI has nothing to answer. The stop in _attempt is only the fallback."""
        while not done.wait(self._poll_seconds):
            turn.retry_slot()  # a turn that gave its permit back for a card takes one again as soon as one is free
            if self._live.is_aborted(thread_id):
                turn.request_stop()
            if self._live.is_shutting_down():
                turn.begin_stopping()
            if turn.check_caps() is not None:
                turn.begin_stopping()
                session.stop()
                return
            if turn.observe_alive(session.alive()):
                return
        turn.mark_result_received()  # already done by the bus's ResultFreezer; kept so a result never leaves it running
        if turn.limit_hit is None:
            session.end_gracefully(GRACEFUL_SECONDS)

    def _pending_digest(self, thread_id: int, resume_id: str | None) -> str:
        """The handoff digest a fresh session takes as context; '' for a resumed session or when none is pending."""
        if resume_id is not None:
            return ""
        digest = self._handoff.peek(thread_id) or ""
        if digest:
            self._logger.log(f"handoff: thread {thread_id} fresh session takes a digest of {len(digest)} chars")
        return digest

    def _spec(self, plan: TurnPlan, resume_id: str | None, new_id: str | None, digest: str) -> SessionSpec:
        system_rules = f"{plan.system_rules}\n\n{digest}" if digest else plan.system_rules
        return SessionSpec(
            thread_id=plan.thread_id, chat_id=plan.chat_id, topic=plan.topic, launcher=plan.launcher,
            cwd=plan.cwd, mode=plan.mode, model=plan.model, resume_id=resume_id, new_id=new_id,
            allow_rules=plan.allow_rules, deny_tools=plan.deny_tools, system_rules=system_rules,
            env=dict(self._base_env, TZ=TIMEZONE), errfile_dir=self._errfile_dir,
        )

    # the end of a turn

    def _finish_turn(self, plan: TurnPlan, attempt: _Attempt, fresh: str) -> TurnReply:
        if attempt.refused:
            return TurnReply(None, None, attempt.outcome)
        answer, err = compose_reply(
            attempt.outcome, result=attempt.result, texts=attempt.texts, limit=attempt.limit,
            stop_tap=attempt.stop_tap, tools=attempt.tools, launch_err=attempt.launch_err,
            exit_code=attempt.exit_code, tail=attempt.tail)
        wall = self._clock.monotonic() - attempt.started_mono
        attempt.feed.finish(final_header(attempt.outcome.stored, result=attempt.result, usage=attempt.usage,
                                         tools=attempt.tools, wall=wall, limit_hit=whole_seconds(attempt.limit)))
        result, usage = attempt.result, attempt.usage
        self._turns.insert(
            plan.thread_id, started=utc_stamp(attempt.started_wall), ended=utc_stamp(self._clock.now()),
            status=attempt.outcome.stored, subtype=result.subtype if result is not None else "",
            cost=usage.cost_usd if usage is not None else None,
            cost_known=usage.cost_known if usage is not None else True,
            in_tok=usage.input_tokens if usage is not None else None,
            out_tok=usage.output_tokens if usage is not None else None,
            num_turns=result.num_turns if result is not None else None,
            duration_ms=result.duration_ms if result is not None else None,
            error=err, backend=plan.backend.name)
        if fresh:  # the user sees why this reply has no earlier context
            if err:
                err = f"{fresh}\n{err}"
            else:
                answer = f"{fresh}\n\n{answer or ''}"
        return TurnReply(answer, err, attempt.outcome)
