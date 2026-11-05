"""TurnRunner: one user turn on an AgentSession (C14, neutral).

Port of the live _run_turn_body and run_turn_v2 (tg-bridge.py:1718-1985). The backend, the permission service, the feed
and every repository are injected, so the same runner serves Claude and Codex and is tested with FakeBackend.

One turn: plan (launcher precedence, allow and deny tools, RULES text) -> the active:<thread> marker -> an attempt (open
the session, attach the permissions, register it as live, start, send the user text, wait for the end) -> at most one
relaunch as a fresh session when the stored session is gone -> the turns row (its insert also stamps
sessions.last_active) -> the feed summary -> the reply (answer, err). The per-launch objects and the end of the turn are
in core/turn_kit.py. The session's sink is a SessionRouter (core/session_router.py) that holds the attempt's end gate.

Memory: a Claude topic's SessionSpec names its own auto-memory folder, <memory root>/<thread id>, created 0700 before
the launch (_topic_memory). Claude Code keeps one memory per cwd and topics share a cwd, so each topic gets its own.

Attached: a turn with an attach target (a claim of a terminal tab) runs in that terminal session: the resume id is the
claim's uuid, nothing is kept or offered for linger, no mark is taken here, and no relaunch is tried (_run_attached).

Linger: before a new launch is planned, the LingerService is asked for the topic's kept session (claim). When it has
one with the same key, the turn runs on that running process with no spawn: the attempt reuses its router and
permission binding and attaches a new gate to the router. When a turn ends normally, its session is offered to the
service (the wait's after_result). A kept session is not stopped; every other one is, as before.

Lock map: TurnRunner holds no lock of its own. LiveRegistry, WaitRouter and Database each lock themselves, and the
per-turn objects (TurnState, AnswerCollector, SessionRecorder) lock their own state. Nothing here holds a lock while it
calls another service.
"""

import json
import threading
import uuid as uuidlib
from collections.abc import Callable, Mapping
from dataclasses import replace

from tg_bridge.agents.base import AgentSession, AttachSpec, SessionSpec, TerminalNotReady
from tg_bridge.agents.registry import BackendRegistry, UnknownBackend
from tg_bridge.clock import Clock, utc_stamp
from tg_bridge.config import BACKEND_CLAUDE, Config, LauncherError
from tg_bridge.core import terminal_handoff
from tg_bridge.core.events_bus import TurnEventBus
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.linger import HeldSession, LingerOffer, LingerService
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.media import PHOTO_NOTE
from tg_bridge.core.permissions import BYPASS_MODE, PermissionService, TurnBinding
from tg_bridge.core.session_router import Attach, SessionRouter
from tg_bridge.core.topic_key import TopicKey, topic_cwd, topic_launcher, topic_mode, topic_model
from tg_bridge.core.turn_kit import (
    FeedFactory,
    TurnCloser,
    TurnKit,
    TurnPlan,
    TurnProgress,
    TurnReply,
    _Attempt,
    build_kit,
    finish_attempt,
    open_view,
    sync_open_asks,
    wait_turn,
    wire_gate,
)
from tg_bridge.core.turn_listeners import EndGate, SessionRecorder, WaitRouter
from tg_bridge.core.turn_state import SlotLike, TurnOutcome, TurnState
from tg_bridge.core.turn_text import (
    LAUNCH_ERROR_CHARS,
    TIMEZONE,
    format_rules,
    fresh_note,
)
from tg_bridge.core.view_recorder import ViewFactory
from tg_bridge.logging_ import Logger
from tg_bridge.paths import topic_memory
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.rules import RuleRepo
from tg_bridge.store.sessions import HistoryRepo, SessionRepo
from tg_bridge.store.turns import TurnRepo

ACTIVE_PREFIX = "active:"
DEFAULT_ALLOWED_TOOLS = "Read,Glob,Grep"
KILL_SWITCH_TOOL = "KillShell"  # refused by the CLI even in bypass mode
POLL_SECONDS = 1.0  # the wait loop's tick: caps, stop taps and dead-process checks (the live file's done.wait(1))
MAX_LAUNCHES = 2  # the first launch and the one fresh relaunch after a missing resume
SELF_FIRST_RETRIES = 3  # the attach of a claimed turn is retried this often after a self turn ran in front of it
BUSY_TEXT = "background turns kept starting before the message could run"


def active_key(thread_id: int) -> str:
    """The kv key of a turn in progress. It survives a restart on purpose (recover() reads it)."""
    return f"{ACTIVE_PREFIX}{thread_id}"


class TurnRunner:
    """Runs user turns. One instance per App; each run() is one turn on one topic, called from that topic's dispatcher."""

    def __init__(
        self, *, config: Config, registry: BackendRegistry, live: LiveRegistry, permissions: PermissionService,
        waits: WaitRouter, sessions: SessionRepo, history: HistoryRepo, rules: RuleRepo, turns: TurnRepo,
        kv: KvRepo, handoff: HandoffService, feeds: FeedFactory, clock: Clock, logger: Logger,
        base_env: Mapping[str, str], errfile_dir: str, poll_seconds: float = POLL_SECONDS,
        views: ViewFactory | None = None, view_dir: str = "", linger: LingerService | None = None,
        memory_dir: str = "",
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
        self._views = views
        self._view_dir = view_dir
        self._linger = linger  # None: no session is kept after a turn (linger off), as before
        self._memory_dir = memory_dir  # the root of the topics' auto-memory folders; '' = Claude Code's default
        self._closer = TurnCloser(turns=turns, clock=clock)

    def run(self, *, thread_id: int, chat_id: int, topic: str, text: str, sender: str,
            slot: SlotLike | None = None, images: tuple[str, ...] = (), attach: AttachSpec | None = None) -> TurnReply:
        """One user turn. Sets the active marker, runs the turn, and removes the marker unless a shutdown began.
        images are local photo paths for native image input (they are dropped for a backend without it). attach names the
        terminal session the turn runs in (None: the bridge spawns its own process)."""
        key = active_key(thread_id)
        self._kv.set(key, json.dumps({"chat_id": chat_id, "name": topic,
                                      "started": utc_stamp(self._clock.now())}))
        try:
            reply = self._run_body(thread_id, chat_id, topic, text, sender, slot, images, attach)
        finally:
            if not self._live.is_shutting_down():  # a shutdown keeps the marker on purpose
                self._kv.delete(key)
        if reply.answer is None and reply.err is None:  # refused by a shutdown before any input: nothing ran
            self._kv.delete(key)
        return reply

    # planning

    def _run_body(self, thread_id: int, chat_id: int, topic: str, text: str, sender: str,
                  slot: SlotLike | None, images: tuple[str, ...], attach: AttachSpec | None) -> TurnReply:
        try:
            plan = self._plan(thread_id, chat_id, topic, text, sender, images)
        except (LauncherError, UnknownBackend) as exc:  # refused before any session id is stored
            return TurnReply(None, f"launcher failed: {exc}", TurnOutcome.ERROR)
        if attach is not None:
            return self._run_attached(plan, attach, slot)
        plan = terminal_handoff.adopt_ended(self._view_dir, plan, sessions=self._sessions, history=self._history,
                                            logger=self._logger)
        held = self._claim(plan)
        if held is not None:  # the kept process of this topic runs the turn: no spawn, no resume, no relaunch
            try:
                attempt = self._attempt(plan, held.recorder, None, None, slot, held=held)
            except Exception:  # raised before _attempt's own try (the kit, the wait registration): release the claim
                self._release_claimed(held)
                raise
            return self._closer.close(plan, attempt, "")
        recorder = SessionRecorder(sessions=self._sessions, history=self._history, kv=self._kv,
                                   thread_id=thread_id, text=text, cwd=plan.cwd,
                                   launcher=plan.launcher.command, backend=plan.backend.name,
                                   known_uuid=plan.stored_uuid)
        preassigned = plan.backend.capabilities_for(plan.launcher).preassigned_id
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
        return self._closer.close(plan, attempt, fresh)

    def _plan(self, thread_id: int, chat_id: int, topic: str, text: str, sender: str,
              images: tuple[str, ...]) -> TurnPlan:
        row = self._sessions.get(thread_id)
        if row is None:
            raise RuntimeError(f"topic {thread_id} has no sessions row")
        cfgt = self._config.topic(topic)
        launcher = topic_launcher(self._config, row, topic)
        backend = self._registry.for_launcher(launcher)
        mode = topic_mode(self._config, row)
        allow: tuple[str, ...] = ()
        if mode != BYPASS_MODE:  # bypass ignores the allow list, as the live file does
            allowed = cfgt.get("allowed_tools", DEFAULT_ALLOWED_TOOLS)
            allow = tuple(str(rule) for rule in [allowed, *self._rules.texts(thread_id)] if rule)
        denied = cfgt.get("deny_tools") or []
        if isinstance(denied, str):
            denied = [denied]
        if not backend.capabilities_for(launcher).images and images:
            # a backend without image input gets each saved photo as a text note, so a photo is never dropped silently;
            # a note the text already carries is not added twice
            notes = [PHOTO_NOTE.format(path=path) for path in images]
            missing = [note for note in notes if note not in text]
            text = "\n".join([text, *missing]).strip()
            images = ()
        return TurnPlan(
            thread_id=thread_id, chat_id=chat_id, topic=topic, text=text, images=tuple(images),
            launcher=launcher, backend=backend,
            cwd=topic_cwd(self._config, row, topic), mode=mode, model=topic_model(row),
            allow_rules=allow, deny_tools=(KILL_SWITCH_TOOL, *(str(tool) for tool in denied)),
            system_rules=format_rules(topic, sender), stored_uuid=row.uuid,
        )

    def _run_attached(self, plan: TurnPlan, attach: AttachSpec, slot: SlotLike | None) -> TurnReply:
        """One turn in the terminal session of a view tab. The session resumes the claim's uuid, so a fresh one is stored by
        the recorder when it starts (SessionStarted). Nothing is kept: the terminal's process is not the bridge's, and the
        session is stopped (its transcript follow closed) when the turn ends: the terminal's session stays open, so the
        reply says it is kept. Only Claude sessions can be attached."""
        if plan.backend.name != BACKEND_CLAUDE:
            return TurnReply(None, f"the terminal session is a Claude session, this topic runs on {plan.backend.name}",
                             TurnOutcome.ERROR)
        plan = replace(plan, attach=attach)
        recorder = SessionRecorder(sessions=self._sessions, history=self._history, kv=self._kv,
                                   thread_id=plan.thread_id, text=plan.text, cwd=plan.cwd,
                                   launcher=plan.launcher.command, backend=plan.backend.name,
                                   known_uuid=plan.stored_uuid)
        attempt = self._attempt(plan, recorder, attach.uuid, None, slot)
        if not attempt.refused and recorder.known not in ("", attach.uuid):  # the turn moved to another transcript
            terminal_handoff.move_claim(self._view_dir, plan.thread_id, attach.pid, recorder.known, self._logger,
                                        expected=attach.uuid)
        return self._closer.close(plan, attempt, "")

    def _relaunch_fresh(self, attempt: _Attempt) -> bool:
        """True when the stored session was not found and nothing else ended the turn (cap, stop tap)."""
        return not attempt.refused and attempt.resume_failed and attempt.limit is None and not attempt.aborted

    def _may_linger(self, plan: TurnPlan) -> bool:
        """True when this turn's session may be kept after it: a linger service is wired, the turn is not an attached one
        (a terminal's process is never kept), the backend can keep a session open, and the config allows linger (both the
        seconds and the session cap are above zero)."""
        return (self._linger is not None and plan.attach is None and plan.backend.capabilities_for(plan.launcher).linger
                and self._config.linger_seconds() > 0 and self._config.linger_max_sessions() > 0)

    def _claim(self, plan: TurnPlan) -> HeldSession | None:
        """The kept record of this topic when it takes this turn (same launcher, folder, mode and model), else None."""
        if self._linger is None:
            return None
        key = TopicKey(launcher_name=plan.launcher.name, launcher_command=plan.launcher.command, cwd=plan.cwd,
                       mode=plan.mode, model=plan.model)
        return self._linger.claim(plan.thread_id, key)

    def _release_claimed(self, held: HeldSession) -> None:
        """Ends a claimed record whose turn could not start: it leaves the map, its process stops, its cards are voided.
        Safe after _attempt's own finally has already done the same (each step is idempotent)."""
        self._linger.forget(held.thread_id, held.session)
        try:
            held.session.stop(grace=10)
        finally:
            held.router.close()
            self._permissions.void_all(held.binding)

    # one launch

    def _attempt(self, plan: TurnPlan, recorder: SessionRecorder, resume_id: str | None, new_id: str | None,
                 slot: SlotLike | None, held: HeldSession | None = None) -> _Attempt:
        """One launch of the turn. A held record reuses its kept session and binding (no spawn). The session is stopped
        here unless the linger service keeps it after the turn (after_result offers it)."""
        thread = plan.thread_id
        kit = build_kit(plan, recorder, slot, clock=self._clock, config=self._config, feeds=self._feeds,
                        permissions=self._permissions, handoff=self._handoff,
                        pending_digest=lambda: self._pending_digest(thread, resume_id),
                        open_view=lambda: open_view(self._views, plan, self._logger),
                        binding=held.binding if held is not None else None)
        if held is None:  # a held turn registers after its self-first servicing, in _drive_kept
            self._waits.register(thread, kit.turn)
        router = held.router if held is not None else SessionRouter(self._logger)  # the sink, for the session's life
        session: AgentSession | None = held.session if held is not None else None
        launch_err: str | None = None
        refused = False
        registered = False
        stop_code: int | None = None
        kept = False  # set by after() when the linger service takes the session at this turn's end
        not_ready = False  # set when an attached terminal was not ready (TerminalNotReady)

        def after() -> bool:
            """The wait's after_result: at a normal end it offers the session to the linger service. True leaves the
            input open only when the session was kept. Any other end (linger off, refused, or background work still
            pending) gets the graceful end as before: the CLI has GRACEFUL_SECONDS to exit by itself, then the stop."""
            nonlocal kept
            if self._linger is None or session is None or plan.attach is not None:
                return False
            kept = self._linger.offer(LingerOffer(plan=plan, session=session, router=router, binding=kit.binding,
                                                  recorder=recorder, kit=kit))
            return kept

        try:  # from here on the view is finished by the finally, even when the wiring below raises
            gate = wire_gate(kit, self._logger)
            if held is None:
                if self._may_linger(plan):
                    # a self turn that starts after this turn's result waits for the offer, and an approval asked in
                    # that window gets its card; the sink is the one that linger's offer installs with hold()
                    router.arm(TurnEventBus([self._permissions.bind_turn(kit.binding), recorder], self._logger).publish)
                router.attach(gate, user=True)
                session = plan.backend.open_session(self._spec(plan, resume_id, new_id, kit.digest), router)
                self._permissions.attach(kit.binding, session)
            registered = self._live.begin_turn(thread, session, kit.feed)
            if not registered:
                refused = True  # a shutdown began: the spawn (or the held turn) is refused
            elif held is None:
                refused, launch_err = self._drive(plan, kit.turn, kit.binding, session, kit.feed, kit.done, after)
            else:
                refused, launch_err = self._drive_kept(plan, kit, router, session, gate, after)
        except TerminalNotReady:  # the terminal took no message: nothing ran, so the launch is refused (no reply)
            refused = True
            not_ready = True
        except Exception as exc:  # a launch that failed before or during start
            launch_err = f"launcher failed: {str(exc)[:LAUNCH_ERROR_CHARS]}"
        finally:
            # a kept session stays open with its router and binding; anything else is stopped and its cards voided
            if not kept:
                if session is not None:
                    # cheap when the process has already exited; its exit code is kept
                    stop_code = session.stop(grace=10)
                router.close()  # after the stop, so the events of the stop flow through the gate as before
                if self._linger is not None and session is not None:
                    self._linger.forget(thread, session)  # drops the held record of this session, if there is one
                self._permissions.void_all(kit.binding)  # the turn is over: its open permission cards are closed
            if registered:
                self._live.end_turn(thread, session, kit.feed)
            self._waits.unregister(thread, kit.turn)
            if kit.view is not None:  # a launch that failed emits no end event: the view must not stay running
                kit.view.finish()
            kit.feed.close()  # the feed posts what is left before the answer goes out
        return finish_attempt(kit, session, self._live, refused=refused, launch_err=launch_err,
                              stop_code=stop_code, readable_tail=plan.backend.capabilities_for(plan.launcher).readable_tail,
                              logger=self._logger, terminal_busy=not_ready)

    def _drive(self, plan: TurnPlan, turn: TurnState, binding: TurnBinding, session: AgentSession, feed: TurnProgress,
               done: threading.Event, after: Callable[[], bool]) -> tuple[bool, str | None]:
        """Starts the session, sends the text and waits for the turn to end. Returns (refused, launch_err)."""
        try:
            feed.start()
            turn.start()
            sync_open_asks(self._waits, self._permissions, binding)
            session.start()
            if self._live.is_shutting_down():  # the shutdown snapshot may have missed this process: stop it here
                session.stop(grace=1)
                return True, None
            turn.mark_running()
            session.send_user(plan.text, plan.images)
            wait_turn(turn, session, done, plan.thread_id, self._live, self._poll_seconds, after_result=after)
            return False, None
        except TerminalNotReady:  # raised to _attempt, which refuses the launch without a reply
            raise
        except Exception as exc:
            return False, f"launcher failed: {str(exc)[:LAUNCH_ERROR_CHARS]}"

    def _drive_kept(self, plan: TurnPlan, kit: TurnKit, router: SessionRouter, session: AgentSession, gate: EndGate,
                    after: Callable[[], bool]) -> tuple[bool, str | None]:
        """Runs a turn on a kept session, which is already running. The gate is attached first (a self turn that waits
        runs in front of it). Then the feed and the turn start, the turn is registered for the waits, and the text goes
        to the running process, so the self turns' time never counts against this turn's caps. A router that refuses
        the gate ends the launch."""
        try:
            failed = self._attach_user(plan, kit, router, session, gate)
            if failed is not None:
                return False, failed
            self._waits.register(plan.thread_id, kit.turn)
            kit.feed.start()
            kit.turn.start()
            sync_open_asks(self._waits, self._permissions, kit.binding)
            if self._live.is_shutting_down():  # the shutdown refuses the turn: stop the held process, as _drive does
                session.stop(grace=1)
                return True, None
            kit.turn.mark_running()
            session.send_user(plan.text, plan.images)
            wait_turn(kit.turn, session, kit.done, plan.thread_id, self._live, self._poll_seconds, after_result=after)
            return False, None
        except Exception as exc:
            return False, f"launcher failed: {str(exc)[:LAUNCH_ERROR_CHARS]}"

    def _attach_user(self, plan: TurnPlan, kit: TurnKit, router: SessionRouter, session: AgentSession,
                     gate: EndGate) -> str | None:
        """Attaches the user turn's gate. A self turn that began after the claim (its events wait in the router) runs
        first, through the linger service, and the attach is retried; the retries are bounded. None when attached, else
        the launch error. The self turn's run removes this turn's live entry (SelfTurns registers its own turn under the
        same thread and unregisters it at its end), so the live entry is registered again before each retry. Nothing of
        the user turn is started here: its feed, turn and wait registration start after this returns."""
        for _ in range(SELF_FIRST_RETRIES + 1):
            result = router.attach(gate, user=True)
            if result is Attach.ATTACHED:
                return None
            if result is not Attach.SELF_FIRST:
                return "session closed"
            if self._linger is None or not self._linger.resume(plan.thread_id):
                return "session closed"
            if not self._live.begin_turn(plan.thread_id, session, kit.feed):
                return "session closed"
        return BUSY_TEXT

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
            env=dict(self._base_env, TZ=TIMEZONE), errfile_dir=self._errfile_dir, view_dir=self._view_dir,
            tui_size=self._config.tui_size(), attach=plan.attach, memory_dir=self._topic_memory(plan),
        )

    def _topic_memory(self, plan: TurnPlan) -> str:
        """The topic's own auto-memory folder, <memory root>/<thread id>, created 0700 before the launch. '' for a
        backend other than Claude (it has no auto-memory setting) and when no memory root was given."""
        if plan.launcher.backend != BACKEND_CLAUDE:
            return ""
        return topic_memory(self._memory_dir, plan.thread_id)
