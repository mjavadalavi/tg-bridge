"""Session commands: /stop, /new (alias /clear) and /resume, plus the shared actions their buttons call (C19 session part).

Port of the live _stop_turn, _new_session, stop_and_new, _finish_new, _resume_session and resume_tap of tg-bridge.py.
The texts are the live ones. One stop path serves /stop, the status button, the feed's button and the stop-and-new
button: stop_now() marks the turn through LiveRegistry.request_abort and stops its session on a thread of its own,
because a stop can take ~10 s and the caller must not wait for it. A kept session with nothing running is asked to
close (LingerService.request_release), and a kept session in a background update is not stopped at once: stop_turn()
returns CHOOSE, and the caller posts the choice card (section 5 D2), whose buttons call interrupt_update() or stop_now().

/resume <uuid> and the /resume picker follow the live rules. The folder the session ran in becomes the topic's cwd when
it still exists. A session made by another launcher keeps the current launcher when both share a config folder or
its transcript can be copied (same backend only); otherwise the launcher pick switches to the one that made it, and
a launcher that is not in config refuses the resume. Sessions of every backend are listed, newest first, through
HistoryRepo.list_for once per backend. Claude sessions still on disk for the folder are added first
(HistoryRepo.remember_disk), so a session started outside the bridge can be resumed from the list too.

The launcher a session is resumed under is decided by where its transcript file is: the first other Claude launcher
whose config folder holds it is the source when the current launcher's folder does not, and the history row decides
only when no folder holds it. A session that another topic is bound to is refused.
"""

import os
import threading
import uuid as uuidlib
from collections.abc import Mapping
from dataclasses import dataclass

from tg_bridge.agents.registry import BackendRegistry
from tg_bridge.bot.commands.base import Alias, Command, CommandContext, CommandRegistry
from tg_bridge.bot.commands.change_guard import ask_first
from tg_bridge.bot.commands.disk_sessions import remember_disk_sessions
from tg_bridge.bot.commands.info import launcher_state
from tg_bridge.bot.commands.stop_types import StopKind, StopResult
from tg_bridge.bot.keep_guard import SAME_SESSION_TEXT, WHAT_SESSION, ApplyResult, KeepGuard, applies
from tg_bridge.bot.pickers import RESUME_LIMIT, Picker, resume_picker, stop_choice_picker
from tg_bridge.bot.terminal_guard import TerminalGuard
from tg_bridge.clock import Clock
from tg_bridge.config import BACKEND_CLAUDE, Config, LauncherError, LauncherSpec, tilde
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.linger import LingerService, LingerStatus
from tg_bridge.core.live import LiveRegistry
from tg_bridge.logging_ import Logger
from tg_bridge.store.sessions import KNOWN_BACKENDS, HistoryRecord, HistoryRepo, SessionRecord, SessionRepo
from tg_bridge.telegram.keyboards import EMPTY_KEYBOARD, PICKER_PREFIX, CallbackData
from tg_bridge.telegram.messenger import Messenger

STOP_WAIT_SECONDS = 30.0        # /new after a stop waits this long for the turn to end
STOP_POLL_SECONDS = 0.5
STOP_BUTTON_KIND = "nw"         # cm: kind of the stop-and-new button
NO_SESSION_TEXT = "no session in this topic"
NOTHING_RUNNING_TEXT = "nothing is running in this topic."
STOP_NEW_TOAST = "Stopping ⏹ — then a fresh start"
NOT_IDLE_TEXT = "a turn is running — send /stop first, then try again."
NOT_IDLE_TAP_TEXT = "a turn is running — send /stop first"
NEW_BUSY_TEXT = "a turn is running — send /stop first, then try again."
NEW_BUTTON_TEXT = "⏹ Stop and start fresh"
NEW_TEXT = "🆕 the next message starts a fresh session. /resume <uuid> brings the old one back."
NEW_DONE_TEXT = "🆕 stopped; the next message starts a fresh session"
STILL_STOPPING_TEXT = "⏹ still stopping after {wait:g} s — send /new again in a moment"
RESUME_USAGE_TEXT = "no earlier sessions in this topic yet. usage: /resume <session uuid>"
RESUME_ARG_USAGE_TEXT = "usage: /resume <session uuid>"
RESUMED_TEXT = "↩️ the next message resumes session {sid}{suffix}"
RESUMED_TAP_TEXT = "↩️ the next message resumes it{suffix}"
REFUSED_TEXT = "session {sid} belongs to launcher {launcher}, which is not in config — it cannot run here"
BOUND_ELSEWHERE_TEXT = "session {sid} is in use by another topic — /stop it there first"
FOLDER_GONE_TEXT = " — its folder {folder} no longer exists, so it runs in the current one"
FOLDER_MOVED_TEXT = " in {folder}"
SWITCHED_TEXT = " — launcher switched to {name}, which made this session"
COPIED_TEXT = " — history copied from launcher {source} into {name}"
SHARED_TEXT = " — {name} shares its history folder with {source}"
NOT_IN_LIST_TEXT = "not in the list anymore"
LINGER_STOPPED_TEXT = "⏹ closed the session that was kept open for background work."
UPDATE_INTERRUPTED_TEXT = "⏹ this background update was interrupted; the session stays open."
UPDATE_CLOSED_TEXT = "🛑 closing the session that was kept open for background work."
UPDATE_GONE_TEXT = "no background update is running any more."


@dataclass(frozen=True)
class ResumeTap:
    """The outcome of a /resume button tap: the toast to answer with, and the picker to edit the message into when the
    session was bound (None when nothing changed, so the message stays as it is)."""

    toast: str
    picker: Picker | None = None


def _same_dir(a: str, b: str) -> bool:
    """True when two folders are the same folder, after ~ expansion and symlink resolution."""
    return bool(a) and bool(b) and os.path.realpath(os.path.expanduser(a)) == os.path.realpath(os.path.expanduser(b))


class SessionCommands:
    """The /stop, /new and /resume commands and the shared actions of their buttons.

    Every collaborator is injected. The only state is the service references; a turn's state lives in LiveRegistry
    and a topic's session in SessionRepo.
    """

    def __init__(self, *, config: Config, sessions: SessionRepo, history: HistoryRepo, live: LiveRegistry,
                 backends: BackendRegistry, messenger: Messenger, clock: Clock, logger: Logger,
                 handoff: HandoffService, stop_wait: float = STOP_WAIT_SECONDS,
                 linger: LingerService | None = None, keep: KeepGuard | None = None,
                 terminal: TerminalGuard | None = None) -> None:
        self._config = config
        self._linger = linger  # None: no session is kept, so nothing can be asked to close or to be interrupted
        self._keep = keep      # None: no change to /new or /resume waits for a confirm card (section 5 D1)
        self._terminal = terminal  # None: no change waits for the card of an open terminal tab (card H1)
        self._sessions = sessions
        self._history = history
        self._handoff = handoff
        self._live = live
        self._backends = backends
        self._messenger = messenger
        self._clock = clock
        self._logger = logger
        self._stop_wait = stop_wait

    def register(self, registry: CommandRegistry) -> None:
        """Adds /stop, /new (with /clear as its alias) and /resume to the registry."""
        registry.register(Command("stop", "stop the running turn (the session is kept)", self._stop))
        registry.register(Command("new", "start a fresh session with the next message", self._new,
                                  aliases=(Alias("clear", "same as /new: forget this conversation, start fresh"),)))
        registry.register(Command("resume", "continue an earlier session: buttons, or /resume <uuid>", self._resume))

    # --- shared actions (commands and buttons call these)

    def stop_turn(self, thread: int) -> StopResult:
        """The stop of /stop, the status button and the feed's button. While a background update runs in a kept
        session, nothing is stopped yet: the result is CHOOSE and the caller posts stop_choice_picker (section 5 D2).
        Otherwise it is stop_now()."""
        update = self._running_update(thread)
        if update is not None:
            return StopResult(StopKind.CHOOSE, update.hold_id, update.self_seq)
        return self.stop_now(thread)

    def stop_now(self, thread: int) -> StopResult:
        """Stops the running turn of a topic and its session (background work included), or asks an idle kept session
        to close. The turn is marked as stopped on purpose (abort and the feed's stop mark) and its session is stopped on
        a daemon thread, because stop() can take ~10 s."""
        running = self._live.request_abort(thread, stop_tap=True)
        if running is not None:  # halt_turn: a terminal session gets an Escape, the bridge's process is stopped
            threading.Thread(target=running.session.halt_turn, daemon=True).start()
            return StopResult(StopKind.STOPPED)
        if self._linger is not None and self._linger.request_release(thread):
            return StopResult(StopKind.RELEASED)
        return StopResult(StopKind.NOTHING)

    def interrupt_update(self, thread: int) -> bool:
        """Interrupts the running background update only; the session stays kept. False when none runs."""
        return self._linger is not None and self._linger.request_interrupt(thread)

    def choice_current(self, thread: int, hold_id: int, self_seq: int) -> bool:
        """True while the self turn that a choice card was posted for still runs (K9): a tap after it ended, or after
        another one started, or after its hold ended, is expired and acts on nothing."""
        return self._linger is not None and self._linger.in_self_turn(thread, hold_id, self_seq)

    def _running_update(self, thread: int) -> LingerStatus | None:
        """The status of the background update that runs in a kept session (a self turn), or None."""
        if self._linger is None or self._live.running(thread) is None:
            return None
        status = self._linger.status(thread)
        return status if status is not None and status.self_turn else None

    def new_session(self, thread: int) -> None:
        """Forgets the topic's session id (it stays reachable through /resume, the id moves to prev_uuid) and any
        pending handoff digest: a fresh start must not carry the summary of the conversation it forgets."""
        self._sessions.new_session(thread)
        self._handoff.discard(thread)

    def stop_and_new(self, thread: int, chat_id: int | str, message_id: int | None) -> str:
        """The stop-and-new button: stops the turn as /stop does, then starts a background wait that runs /new and edits
        the same message. Returns the toast to answer the tap with."""
        self.stop_now(thread)  # a background update is stopped with the session here: starting fresh drops both
        threading.Thread(target=self.finish_new, args=(thread, chat_id, message_id), daemon=True).start()
        return STOP_NEW_TOAST

    def finish_new(self, thread: int, chat_id: int | str, message_id: int | None) -> str:
        """The background half of stop_and_new: waits up to the stop wait for the turn to end, then /new and the edit.

        Returns the text the message was edited to. The wait uses the injected clock, so tests run it without sleeping.
        """
        deadline = self._clock.monotonic() + self._stop_wait
        while self._live.running(thread) is not None and self._clock.monotonic() < deadline:
            self._clock.sleep(STOP_POLL_SECONDS)
        if self._live.running(thread) is not None:
            text = STILL_STOPPING_TEXT.format(wait=self._stop_wait)
        else:
            self.new_session(thread)
            text = NEW_DONE_TEXT
        if message_id:
            try:
                self._messenger.edit_text(chat_id, message_id, text, reply_markup=EMPTY_KEYBOARD)
            except Exception as exc:  # the new session already started; a refused edit is only logged
                self._logger.log(f"stop and new: edit failed: {str(exc)[:120]}")
        return text

    def resume_picker(self, thread: int) -> Picker:
        """The /resume picker of a topic: its newest sessions across every backend, the one in use marked.

        The Claude sessions on disk for the topic's folder are remembered first, so they list with the sessions the
        bridge ran itself. A session another topic is bound to is left out, except this topic's own current id. The
        list is ordered by the file's modification time for a session the disk listing found just now, else by its
        start time; the rows show their start time either way.
        """
        row = self._sessions.get(thread)
        current_uuid = row.uuid if row else ""
        _, current_command = launcher_state(self._config, row, row.name if row else "")
        names = {command: name for name, command in self._config.launcher_choices().items()}
        taken = self._sessions.bound_uuids(thread) - {current_uuid}
        found: dict[str, str] = {}
        if row is not None:
            found = remember_disk_sessions(self._config, self._backends, self._history, thread, row.cwd,
                                           current_command, taken)
        newest_first = sorted(self._picker_records(thread, taken, found),
                              key=lambda record: (found.get(record.uuid, record.started), record.id), reverse=True)
        return resume_picker(thread, newest_first[:RESUME_LIMIT], current_uuid, current_command, names)

    def _picker_records(self, thread: int, taken: set[str], found: dict[str, str]) -> list[HistoryRecord]:
        """The candidate rows of the picker: per backend, the newest history rows that no other topic is bound to, read
        with one bounded query each (the limit is RESUME_LIMIT plus the number of excluded ids, so that many rows remain
        after the exclusion). A session the disk listing found this time is added even when its row is older than the
        window."""
        limit = RESUME_LIMIT + len(taken)
        records: list[HistoryRecord] = []
        for backend in KNOWN_BACKENDS:
            records += [r for r in self._history.list_newest(thread, backend, limit) if r.uuid not in taken]
        listed = {record.uuid for record in records}
        for uuid in found:
            record = self._history.find(thread, uuid) if uuid not in listed else None
            if record is not None:
                records.append(record)
        return records

    def resume_tap(self, thread: int, history_id: int) -> ResumeTap:
        """A tap on a /resume picker button: the same change as /resume <uuid>, refused while a turn runs."""
        row = self._sessions.get(thread)
        if row is None:
            return ResumeTap(NO_SESSION_TEXT)
        record = self._history.find_by_id(thread, history_id)
        if record is None:
            return ResumeTap(NOT_IN_LIST_TEXT)
        if self._live.running(thread) is not None:
            return ResumeTap(NOT_IDLE_TAP_TEXT)
        if self._resume_unchanged(row, record.uuid):
            return ResumeTap(SAME_SESSION_TEXT.format(sid=record.uuid))
        asked = ask_first(self._terminal, self._keep, thread, WHAT_SESSION,
                          lambda: self._resume_reply(thread, record.uuid, RESUMED_TAP_TEXT),
                          lambda: self._resume_change(thread, record.uuid))
        if asked is not None:
            return ResumeTap(asked[1], asked[0])
        ok, suffix = self.resume_session(thread, record.uuid)
        if not ok:
            return ResumeTap(suffix)
        return ResumeTap(RESUMED_TAP_TEXT.format(suffix=suffix), self.resume_picker(thread))

    def _resume_unchanged(self, row: SessionRecord, sid: str) -> bool:
        """True when resuming sid changes nothing: the topic's current session, in its folder, under its launcher."""
        record = self._history.find(row.thread_id, sid)
        home, home_command = (record.cwd, record.launcher) if record else ("", "")
        cwd, spec = home or row.cwd or self._config.home, self._current_spec(row)
        cur_command = launcher_state(self._config, row, row.name)[1]
        if sid != row.uuid or self._folder_change(row, home)[1] or home_command not in ("", cur_command):
            return False
        return self._holds(spec, cwd, sid) or self._transcript_source(sid, cwd, spec, self._config.launcher_specs()) is None

    def _resume_change(self, thread: int, sid: str) -> ApplyResult:
        """The /resume change a card waits for: a refusal (a session in use elsewhere, another launcher's) is the result."""
        ok, text = self.resume_session(thread, sid)
        return ApplyResult(True) if ok else ApplyResult(False, text)

    def _resume_reply(self, thread: int, sid: str, reply: str) -> ApplyResult:
        """The /resume change a terminal card waits for: the same reply the direct change gives, formatted from reply
        (RESUMED_TEXT or RESUMED_TAP_TEXT), or the refusal. Refused while a turn runs (the card may outlive it)."""
        if self._live.running(thread) is not None:
            return ApplyResult(False, NOT_IDLE_TEXT)
        ok, suffix = self.resume_session(thread, sid)
        return ApplyResult(True, reply.format(sid=sid, suffix=suffix)) if ok else ApplyResult(False, suffix)

    def _new_change(self, thread: int) -> ApplyResult:
        """The /new change a terminal card waits for: the same reply the direct /new gives, or the refusal while a turn
        runs (the card may outlive it)."""
        if self._live.running(thread) is not None:
            return ApplyResult(False, NOT_IDLE_TEXT)
        self.new_session(thread)
        return ApplyResult(True, NEW_TEXT)

    def resume_session(self, thread: int, sid: str) -> tuple[bool, str]:
        """Binds the topic to an existing session id (the one in use moves to prev_uuid). Returns (ok, text).

        ok False: the text is the refusal. ok True: the text is the reply suffix, '' when nothing about the folder or the
        launcher changed (see the module docstring for the rules).
        """
        row = self._sessions.get(thread)
        if row is None:
            return False, NO_SESSION_TEXT
        if sid in self._sessions.bound_uuids(thread):
            return False, BOUND_ELSEWHERE_TEXT.format(sid=sid)
        record = self._history.find(thread, sid)
        home = record.cwd if record else ""
        home_command = record.launcher if record else ""
        cur_name, cur_command = launcher_state(self._config, row, row.name)
        specs = self._config.launcher_specs()
        cur_spec = self._current_spec(row)
        if cur_spec is not None:
            cur_name = cur_spec.name
        cwd = home or row.cwd or self._config.home
        source = None
        holds = self._holds(cur_spec, cwd, sid)
        if not holds:  # its own folder holds it: nothing to copy or switch
            source = self._transcript_source(sid, cwd, cur_spec, specs)
            if source is None:  # no folder holds the transcript: the history row decides
                foreign = bool(home_command) and home_command != cur_command  # made by another launcher
                source = next((spec for spec in specs.values() if spec.command == home_command), None) if foreign else None
                if foreign and source is None:
                    return False, REFUSED_TEXT.format(sid=sid, launcher=home_command)
        folder_text, newcwd = self._folder_change(row, home)
        switch = source
        copied = False
        if switch is not None and cur_spec is not None:
            if _same_dir(switch.config_dir, cur_spec.config_dir):
                switch = None
            elif switch.backend == cur_spec.backend and self._copy(switch, cur_spec, row, home, sid, newcwd):
                switch, copied = None, True
        launcher_text = self._launcher_text(source, switch, cur_name, copied)
        self._sessions.bind_session(thread, sid, cwd=newcwd, launcher_pick=switch.name if switch else None)
        self._handoff.discard(thread)
        if switch is not None:
            self._sessions.set_backend(thread, switch.backend)
        if source is not None and switch is None:
            self._history.adopt_session(thread, sid, cur_command, newcwd or row.cwd, source.backend)
        if holds and cur_spec is not None and record is not None and home_command != cur_command:
            # the current folder holds the transcript, so the row's old launcher label is stale: the picker follows the topic
            self._history.adopt_session(thread, sid, cur_command, newcwd or row.cwd, cur_spec.backend)
        return True, folder_text + launcher_text

    # --- helpers of resume_session

    def _current_spec(self, row: SessionRecord) -> LauncherSpec | None:
        """The launcher the topic's next turn runs under, resolved as the turn runner resolves it; None when none does.

        An unnamed launcher (a command that is not in config) counts too, so its own config folder is the one checked.
        """
        cfgt = self._config.topic(row.name)
        try:
            return self._config.resolve_launcher(row.launcher_pick, str(cfgt.get("launcher") or ""), row.launcher)
        except LauncherError:
            return None

    def _holds(self, spec: LauncherSpec | None, cwd: str, sid: str) -> bool:
        """True when the launcher's backend finds sid's transcript for cwd in the launcher's config folder."""
        return spec is not None and self._backends.get(spec.backend).has_transcript(spec, cwd, sid)

    def _transcript_source(self, sid: str, cwd: str, cur_spec: LauncherSpec | None,
                           specs: Mapping[str, LauncherSpec]) -> LauncherSpec | None:
        """The other Claude launcher whose config folder holds sid's transcript for cwd, or None.

        A launcher whose config folder is the current one is skipped (it is the current folder, checked by the caller).
        The first match in config order wins.
        """
        for spec in specs.values():
            if spec.backend != BACKEND_CLAUDE or not spec.config_dir:
                continue
            if cur_spec is not None and _same_dir(spec.config_dir, cur_spec.config_dir):
                continue
            if self._backends.get(spec.backend).has_transcript(spec, cwd, sid):
                return spec
        return None

    def _folder_change(self, row: SessionRecord, home: str) -> tuple[str, str | None]:
        """(reply suffix, new cwd or None): the session's folder becomes the cwd when it exists."""
        if not home or _same_dir(home, row.cwd):
            return "", None
        shown = tilde(home, self._config.home)
        if os.path.isdir(home):
            return FOLDER_MOVED_TEXT.format(folder=shown), home
        return FOLDER_GONE_TEXT.format(folder=shown), None

    def _copy(self, source: LauncherSpec, target: LauncherSpec, row: SessionRecord, home: str, sid: str,
              newcwd: str | None) -> bool:
        """Copies the session's transcript into the current launcher's folder. True when the copy is in place."""
        backend = self._backends.get(source.backend)
        return backend.carry_history(source, target, home or row.cwd or self._config.home, sid,
                                     dst_cwd=newcwd or row.cwd or self._config.home)

    @staticmethod
    def _launcher_text(source: LauncherSpec | None, switch: LauncherSpec | None, cur_name: str | None,
                       copied: bool) -> str:
        if switch is not None:
            return SWITCHED_TEXT.format(name=switch.name)
        if source is None:
            return ""
        if copied:
            return COPIED_TEXT.format(source=source.name, name=cur_name)
        return SHARED_TEXT.format(name=cur_name, source=source.name)

    # --- command handlers

    def _stop(self, ctx: CommandContext, args: str) -> None:
        result = self.stop_turn(ctx.thread_id)
        if result.kind is StopKind.CHOOSE:
            picker = stop_choice_picker(ctx.thread_id, result.hold_id, result.self_seq)
            ctx.say_buttons(picker.text, picker.rows)
        elif result.kind is StopKind.RELEASED:
            ctx.say(LINGER_STOPPED_TEXT)
        elif result.kind is StopKind.NOTHING:
            ctx.say(NOTHING_RUNNING_TEXT)

    def _new(self, ctx: CommandContext, args: str) -> None:
        if self._live.running(ctx.thread_id) is not None:  # the running process still holds the old session
            button = CallbackData(PICKER_PREFIX, (ctx.thread_id, STOP_BUTTON_KIND, 0))
            ctx.say_buttons(NEW_BUSY_TEXT, [[(NEW_BUTTON_TEXT, button)]])
            return
        asked = ask_first(self._terminal, self._keep, ctx.thread_id, WHAT_SESSION,
                          lambda: self._new_change(ctx.thread_id),
                          applies(lambda: self.new_session(ctx.thread_id)))
        if asked is not None:
            ctx.say_buttons(asked[0].text, asked[0].rows)
            return
        self.new_session(ctx.thread_id)
        ctx.say(NEW_TEXT)

    def _resume(self, ctx: CommandContext, args: str) -> None:
        if self._live.running(ctx.thread_id) is not None:
            ctx.say(NOT_IDLE_TEXT)
            return
        if not args:
            picker = self.resume_picker(ctx.thread_id)
            if picker.rows:
                ctx.say_buttons(picker.text, picker.rows)
            else:
                ctx.say(RESUME_USAGE_TEXT)
            return
        try:
            sid = str(uuidlib.UUID(args))
        except ValueError:
            ctx.say(RESUME_ARG_USAGE_TEXT)
            return
        row = self._sessions.get(ctx.thread_id)
        if row is not None and self._resume_unchanged(row, sid):
            ctx.say(SAME_SESSION_TEXT.format(sid=sid))
            return
        asked = ask_first(self._terminal, self._keep, ctx.thread_id, WHAT_SESSION,
                          lambda: self._resume_reply(ctx.thread_id, sid, RESUMED_TEXT),
                          lambda: self._resume_change(ctx.thread_id, sid))
        if asked is not None:
            ctx.say_buttons(asked[0].text, asked[0].rows)
            return
        ok, suffix = self.resume_session(ctx.thread_id, sid)
        ctx.say(RESUMED_TEXT.format(sid=sid, suffix=suffix) if ok else suffix)
