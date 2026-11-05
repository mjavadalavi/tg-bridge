"""AttachedSession: a Claude Code session that runs in a `tg-bridge view` tab (the attached transport, spec T2).

Nothing is spawned here. The CLI is the process of the terminal's own pane; this session types into that pane and reads
the session's transcript. start() checks that the pane exists and reports the session as resumed. send_user() pastes the
message into the pane (TmuxPane: bracketed paste, then Enter) and follows the transcript for the turn (TranscriptTurns:
the echo line, the answer, the turn_duration). interrupt() presses one Escape in the pane (_press_escape), by a rule
that reads no screen: no Escape while one already covers the bridge's open turn, and none within ESCAPE_REPEAT_SECONDS
of the last one (a new paste lifts that wait). The permissions are asked in the terminal itself, outside this process, so answer() is False. stop()
closes the transcript follow and nothing else: the terminal's CLI is never signalled or typed into.

The owner passes plain strings (the tmux socket, the pane id, the session uuid, the launcher's config folder, the topic's
folder), never a core type. idle() tells the owner whether a message can be pasted now (TmuxPane.ready).

A turn whose echo is not read in the followed transcript relocates (_relocate). The tab's claim decides where it
looks: a claim that names another session moves the turn to that session's transcript only. A hooked tab (its
SessionStart hook keeps the claim current) never scans the other transcripts of the folder. Only a tab started without
the hook scans the folder for the echo, as before the hook existed.

After the bridge's own Escape, the CLI may put the pasted message back into its input box, where it blocks every queued
message (TmuxPane.ready is False for typed text). When that Escape interrupted the bridge's open turn, _press_escape
starts a restore thread (_clear_restored). It empties the box, but only when the box holds that message (is_restored)
and no newer paste came since; each key of the clear checks the box again (_still_ours), so text the user typed in the
tab is never touched. The restore waits while the CLI works, up to RESTORE_CAP_SECONDS. stop() stops new restores, then
waits for the running ones (RESTORE_JOIN_SECONDS). A restore still running then is abandoned: it presses no further key,
so the next turn's paste cannot meet its keys.

An Escape is sent whatever the screen shows: a single Escape is harmless at an idle prompt, and it cancels a dialog of
the CLI as it always did. Two Escapes close together are not: at an idle prompt the second opens the CLI's Rewind
dialog. So the taps that follow are held back (_covered). One Escape covers the bridge's open paste turn: while that
turn is open, no further Escape is sent for it. Whatever the turn, no Escape is sent within ESCAPE_REPEAT_SECONDS of the
last one sent, unless a paste came after that Escape: the paste makes the CLI busy, so its turn is judged on its own.

Threads: the transcript tail thread delivers the lines (TranscriptTurns); the owner's thread calls the methods, and an
interrupt may come from another thread (the ⏹ tap). The restore thread (_clear_restored) starts after an Escape.
Lock map: _keys serialises the keys typed into the pane (the paste, the Escape and the restore's clear). It guards
_pasted, _paste_seq, _escaped_seq, _escaped_gen, _escaped_at and _stopped, and the restore target is decided under it,
before the Escape. _restores (the count of running restores) is guarded by _restores_idle, a Condition that stop() waits
on outside _keys; _keys is taken first when both are held. _abandoned is set by stop() without _keys: a one-way flag
that the restore reads before each key.
The turn state is TranscriptTurns' own. No lock is held while a turn is opened or followed.
"""

import re
import threading
from collections.abc import Callable

from tg_bridge.agents.base import AgentSession, Capabilities, EventSink, SessionSpec, TerminalNotReady, always_attached
from tg_bridge.agents.claude import session_finder
from tg_bridge.agents.claude.tmux_pane import TmuxPane, TmuxRunner, input_text, run_tmux, working
from tg_bridge.agents.claude.transcript_turns import TranscriptTurns
from tg_bridge.agents.claude.translate import ClaudeTranslator
from tg_bridge.agents.events import ApprovalRequest, Decision, SessionStarted
from tg_bridge.clock import Clock
from tg_bridge.logging_ import Logger

UNREACHABLE_ERROR = "the terminal session could not be reached"
ATTACHED_CAPABILITIES = Capabilities(live_mode=False, live_model=False, questions=False, cost_usd=False, images=False,
                                     preassigned_id=True, linger=False,  # the process belongs to the terminal
                                     readable_tail=False)  # and its output is a terminal screen, not a log
RESTORE_WAIT_SECONDS = 3.0  # how long after its Escape the bridge looks for its message put back in the box
RESTORE_CAP_SECONDS = 20.0  # the look is pushed on while the CLI works, but never past this long after the Escape
RESTORE_POLL_SECONDS = 0.25  # between two looks at the box while it is empty
# stop() waits this long for the running restores, so the next paste cannot meet their keys. A restore still
# running then is abandoned: it presses no further key.
RESTORE_JOIN_SECONDS = 30.0
ESCAPE_REPEAT_SECONDS = 3.0  # a turn with no paste of its own takes one Escape per this many seconds
PASTED_PLACEHOLDER = re.compile(r"\[Pasted text #\d+(?: \+(\d+) lines)?\]")  # a long paste's box text (M: group 1)


def pane_idle(socket: str, pane: str, run: TmuxRunner = run_tmux) -> bool:
    """True when the pane can take a message now (TmuxPane.ready: it exists, is in no tmux mode, its input box is empty and
    the CLI is not working). A pane tmux cannot show reads as not idle: nothing is pasted into a pane that is gone. The
    dispatcher polls this while it waits."""
    return TmuxPane(socket, pane, run=run).ready()


def pane_reason(socket: str, pane: str, run: TmuxRunner = run_tmux) -> str:
    """Why the pane cannot take a message now, one of the tmux_pane REASON_ values (REASON_IDLE when it can). The owner
    shows it in the stall notice while the turn waits for the pane."""
    return TmuxPane(socket, pane, run=run).reason()


def _no_requests(message: str, *, main_only: bool = False) -> list[str]:
    """The terminal's permission dialog is not in this process: no request of this session is ever pending."""
    return []


def _spawn_daemon(work: Callable[[], None]) -> None:
    """Runs work in a daemon thread of its own: the restore waits for the CLI, and the owner's thread does not."""
    threading.Thread(target=work, name="restored-prompt", daemon=True).start()


def is_restored(box: str, text: str) -> bool:
    """True when the input box holds the message the bridge pasted. The CLI shows a long paste as a placeholder:
    '[Pasted text #N +M lines]' when M is within one of the lines of text (the CLI counts them its own way), or
    '[Pasted text #N]' when text is one line. Otherwise the box holds the same characters as text, with all whitespace
    removed: the CLI wraps and indents the lines, and a long token such as a URL can wrap mid-word. A blank box holds
    nothing."""
    shown = box.strip()
    if not shown:
        return False
    placeholder = PASTED_PLACEHOLDER.fullmatch(shown)
    if placeholder is not None:
        if placeholder.group(1) is None:
            return "\n" not in text
        return abs(int(placeholder.group(1)) - (text.count("\n") + 1)) <= 1
    return "".join(box.split()) == "".join(text.split())


def _still_ours(box: str, text: str) -> bool:
    """True when a clear may still empty this box: it holds the pasted message (is_restored), or a part of it cut
    from the end. C-u deletes from the end, so a box that a clear has partly emptied is a prefix of the text. A
    placeholder is only ever the whole message (is_restored), never a prefix. Whitespace is not compared, as in
    is_restored."""
    if is_restored(box, text):
        return True
    shown = box.strip()
    if not shown or PASTED_PLACEHOLDER.fullmatch(shown):
        return False
    return "".join(text.split()).startswith("".join(shown.split()))


class AttachedSession(AgentSession):
    """One Claude Code session of a terminal tab, driven through its tmux pane and read from its transcript."""

    def __init__(self, spec: SessionSpec, sink: EventSink, *, socket: str, pane: str, uuid: str, config_dir: str,
                 cwd: str, logger: Logger, clock: Clock, run: TmuxRunner = run_tmux,
                 still_attached: Callable[[], bool] = always_attached,
                 spawn: Callable[[Callable[[], None]], None] = _spawn_daemon) -> None:
        self.session_id = uuid  # the claim's session; moves to the new one when a /clear or /resume in the tab moves it
        self._config_dir = config_dir
        self._still_attached = still_attached  # the owner's check that the tab's claim is still held (the CLI runs)
        self._spec = spec
        self._cwd = cwd
        self._emit = sink
        self._logger = logger
        self._clock = clock
        self._spawn = spawn  # starts the restore thread after an Escape
        self._pane_id = pane
        self._pane = TmuxPane(socket, pane, run=run, clock=clock)
        self._open = False
        self._pasted: tuple[str, int] | None = None  # the last paste and its turn's gen, until an Escape takes it
        self._paste_seq = 0  # counts the pastes: a restore is for the one it was started after
        self._escaped_seq = -1  # the _paste_seq an Escape was pressed for (-1: none yet)
        self._escaped_gen = -1  # that paste's turn: its Escape covers the paste only while the turn is open
        self._escaped_at: float | None = None  # monotonic time of the last Escape sent; None after a paste (send_user)
        self._restores = 0  # the restore threads running; guarded by _restores_idle, which stop() waits on
        self._restores_idle = threading.Condition()  # notified each time a restore ends
        self._stopped = False  # set by stop(), under _keys: no restore starts after it
        self._abandoned = False  # set by stop() when a restore outlived its wait: no further key is pressed
        self._keys = threading.Lock()  # the paste and an Escape never interleave: an Escape waits for the Enter of the paste
        self._turns = TranscriptTurns(config_dir=config_dir, cwd=cwd, session_id=uuid,
                                      translator=ClaudeTranslator(cwd=cwd), emit=sink, log=logger.log,
                                      logger=logger, clock=clock, deny_requests=_no_requests,
                                      foreign_interrupts=True, relocate=self._relocate, on_relocated=self._moved,
                                      strict_echo=True)

    # ---- lifecycle ------------------------------------------------------------------------------------

    def start(self) -> None:
        """Checks that the pane exists, then reports the session. Raises TerminalNotReady when the pane is gone: there is no
        process to start, and the owner waits for the terminal again (a claim that went takes the normal path)."""
        if not self._pane.exists():
            raise TerminalNotReady(f"the terminal session is not open: tmux pane {self._pane_id} is gone")
        self._open = True
        self._emit(SessionStarted(session_id=self.session_id, model=self._spec.model, mode=self._spec.mode,
                                  cwd=self._cwd, resumed=True))

    def stop(self, grace: float = 5.0) -> int | None:
        """Stops any restore from starting, then waits up to RESTORE_JOIN_SECONDS for the running ones. A restore still
        running when the wait ends is abandoned and logged: it presses no further key, so the next turn's paste cannot
        meet its keys. Then closes the transcript follow (its final drain runs first). Sends no key: the CLI keeps
        running in the pane. Returns None, as there is no process of this session to report a code for."""
        with self._keys:  # an Escape in flight finishes first: the restore it starts is then one of those waited for
            self._stopped = True
        with self._restores_idle:  # outside _keys: a restore's clear holds _keys while it types
            idle = self._restores_idle.wait_for(lambda: self._restores == 0, timeout=RESTORE_JOIN_SECONDS)
        if not idle:
            # a plain flag, not under _keys: a clear in flight stops at its next key instead of at its end
            self._abandoned = True
            self._logger.log(f"terminal: tmux pane {self._pane_id}: the restore after the Escape did not finish in "
                             f"{RESTORE_JOIN_SECONDS:g} s: abandoned, no further key is pressed")
        self._turns.mark_ending()
        self._turns.close()
        self._open = False
        return None

    def end_gracefully(self, timeout: float) -> bool:
        """Nothing is ended: the terminal's CLI belongs to the terminal. Marks the session ending (no self turn starts from
        here on) and returns True, so the turn runner does not wait for a process."""
        self._turns.mark_ending()
        return True

    def alive(self) -> bool:
        """True from a successful start() until stop(), and only while the pane exists and the tab still holds its claim
        for this turn (still_attached: the view pane releases the claim when the CLI exits, and the pane then shows its
        ready screen again). A gone pane or a released claim ends a turn that waits on it (TurnState.observe_alive)."""
        if not self._open or not self._pane.exists():
            return False
        return self._still_attached()

    def stderr_tail(self, n: int = 20) -> str:
        """No process, no stderr."""
        return ""

    # ---- turns ----------------------------------------------------------------------------------------

    def send_user(self, text: str, images: tuple[str, ...] = ()) -> None:
        """Pastes the message into the pane, after checking the pane is still ready: the owner may have typed in the tab or
        started a turn there since the wait saw it idle. Not ready raises TerminalNotReady before anything is begun or
        pasted. Otherwise begins a turn and follows the transcript from its current end. A failed paste ends the turn with
        an ERROR. The echo net is armed only when the transcript is followed."""
        if images:
            raise ValueError("attached sessions take text only: capabilities.images is False")
        if not self._pane.ready():
            self._logger.log(f"terminal: tmux pane {self._pane_id} is not ready, the message is not pasted")
            raise TerminalNotReady(f"the terminal session of tmux pane {self._pane_id} is busy or has text typed in it")
        gen, primary, follows = self._turns.begin(text)
        armed = self._turns.follow_turn(gen, primary, follows)
        with self._keys:
            pasted = self._pane.paste(text)
            if pasted:
                self._paste_seq += 1
                self._pasted = (text, gen)
                self._escaped_at = None  # a paste is typed into a busy CLI: its own turn gets its Escape, not a repeat
        if not pasted:
            self._logger.log(f"message not pasted into tmux pane {self._pane_id}")
            self._turns.finish(gen, error=UNREACHABLE_ERROR)
            return
        if armed:
            self._turns.watch_echo(gen, text)

    def _relocate(self, since: float, text: str, primary: str) -> tuple[str, int, str] | None:
        """The transcript of the session that holds the echo of text, written after since (TranscriptTurns asks after
        ATTACH_RELOCATE_SECONDS). The tab's claim decides which transcript is read:
        - the claim names another session than the one followed (a /clear or /resume moved the tab, and the hook told
          the claim): that session's transcript, and no other;
        - else a hooked tab: None, since the claim proves the followed session is the tab's;
        - else (a tab started without the hook): the folder's transcripts, the followed file excluded (its echo was
          not read there)."""
        current = self._claim_uuid()
        if current and current != self.session_id:
            found = session_finder.find_in_session(self._config_dir, current, since, text)
        elif self._spec.attach is not None and self._spec.attach.hooked:
            return None
        else:
            found = session_finder.find_relocated(self._config_dir, since, text, exclude=primary)
        return None if found is None else (found.path, found.offset, found.session_id)

    def _claim_uuid(self) -> str:
        """The session the tab's claim names now: '' when there is no claim or its read fails (a failed read names
        nothing)."""
        if self._spec.attach is None:
            return ""
        try:
            return self._spec.attach.claim_uuid() or ""
        except Exception:  # the claim is read from the tab's side: a read that fails must not end the turn
            return ""

    def _moved(self, session_id: str) -> None:
        """The turn moved to another session's transcript: that session is now this session. Its start is reported, so
        the turn runner records the new id as the topic's session."""
        self.session_id = session_id
        self._emit(SessionStarted(session_id=session_id, model=self._spec.model, mode=self._spec.mode, cwd=self._cwd,
                                  resumed=True))

    def interrupt(self) -> bool:
        """Escape in the pane (_press_escape). The interrupt line of the transcript, or else the fallback, ends the turn
        as INTERRUPTED. True when an Escape was sent, or one already covers the turn or was sent too recently to repeat;
        False when the session is stopped or the Escape could not be sent. The Escape may start a restore of the
        message pasted for the interrupted turn."""
        return self._turns.interrupt(self._press_escape)

    def _press_escape(self) -> bool:
        """Presses one Escape, by a rule that reads no screen. Under _keys: after stop(), nothing is sent and False is
        returned. When an Escape already covers the bridge's open paste turn, or the last Escape of any kind was sent
        less than ESCAPE_REPEAT_SECONDS ago (_covered), nothing is sent and True is returned. Otherwise the Escape is
        sent (_send_escape), and its result is returned: False when the send-keys failed."""
        with self._keys:
            if self._stopped:
                return False
            if self._covered(self._clock.monotonic()):
                return True
            return self._send_escape()

    def _covered(self, now: float) -> bool:
        """True when an Escape already covers the turn, so none is sent: the Escape sent for the last paste while that
        paste's turn is still open (one per pasted message), or an Escape of any kind sent less than
        ESCAPE_REPEAT_SECONDS before now, with no paste since (a second Escape that soon could open the CLI's Rewind
        dialog at an idle prompt; send_user clears the time at each paste). Called under _keys."""
        if self._escaped_seq == self._paste_seq and self._turns.is_open(self._escaped_gen):
            return True
        return self._escaped_at is not None and now - self._escaped_at < ESCAPE_REPEAT_SECONDS

    def _send_escape(self) -> bool:
        """Sends the Escape under _keys and records it for _covered. The restore's target is decided before the Escape
        (_restore_target): the transcript may close the turn while the Escape is in flight, and the restore still starts
        when the Escape was sent. The restore thread starts before _keys is released, so no paste comes in between. The
        target is taken once: a second Escape starts no second restore for the same paste."""
        target = self._restore_target()
        if not self._pane.escape():
            return False
        if self._pasted is not None:
            self._escaped_seq, self._escaped_gen = self._paste_seq, self._pasted[1]
        self._escaped_at = self._clock.monotonic()
        self._pasted = None
        if target is not None:
            self._start_restore(*target)
        return True

    def _restore_target(self) -> tuple[str, int] | None:
        """The paste an Escape now would restore: (text, seq) of the last paste while its turn is still open and
        stop() has not begun, else None. Called under _keys."""
        if self._stopped or self._pasted is None or not self._turns.is_open(self._pasted[1]):
            return None
        return self._pasted[0], self._paste_seq

    def _start_restore(self, text: str, seq: int) -> None:
        """Starts the restore thread for the paste seq, under _keys. The running count goes up before the spawn, so
        stop() waits for the restore. A spawn that raises is logged and the count goes back down at once; nothing is
        raised to the Escape's caller."""
        with self._restores_idle:
            self._restores += 1
        try:
            self._spawn(lambda: self._clear_restored(text, seq))
        except Exception as exc:
            self._restore_ended()
            self._logger.log(f"terminal: tmux pane {self._pane_id}: could not start the restore after the Escape: "
                             f"{exc}")

    def _restore_ended(self) -> None:
        """Counts one restore as over and wakes stop() when none is left."""
        with self._restores_idle:
            self._restores -= 1
            self._restores_idle.notify_all()

    def _clear_restored(self, text: str, seq: int) -> None:
        """Restore thread, after the bridge's Escape; seq is the paste it was started for. For RESTORE_WAIT_SECONDS it
        reads the input box. While the CLI works the window is pushed on, but never past RESTORE_CAP_SECONDS after the
        Escape. A box that holds the pasted text (is_restored) is emptied by _clear_box, unless a newer paste came
        since; a box with other text is left as it is, since the user typed it. Once stop() has abandoned the restore
        (_abandoned), the look ends and no key is pressed. The running count goes down when it returns, however it
        ends. Nothing raised here leaves the thread: a failure is logged."""
        try:
            start = self._clock.monotonic()
            deadline = start + RESTORE_WAIT_SECONDS
            while True:
                if self._abandoned:
                    return
                screen = self._pane.screen()
                box = input_text(screen)
                busy = working(screen)
                now = self._clock.monotonic()
                if busy:  # the CLI's answer may still put the message back, and no key is typed while it works: wait
                    deadline = min(now + RESTORE_WAIT_SECONDS, start + RESTORE_CAP_SECONDS)
                if box:
                    if not is_restored(box, text):
                        self._logger.log(f"terminal: tmux pane {self._pane_id} holds other text after the Escape; "
                                         "left as is")
                        return
                    if not busy:
                        self._clear_box(box, text, seq)
                        return
                if now >= deadline:
                    return
                self._clock.sleep(RESTORE_POLL_SECONDS)
        except Exception as exc:  # the restore is best effort: a failure is logged, the owner's thread is not affected
            self._logger.log(f"terminal: tmux pane {self._pane_id}: restoring the message after the Escape failed: "
                             f"{exc}")
        finally:
            self._restore_ended()

    def _owned_by(self, text: str) -> Callable[[str], bool]:
        """The owned check of clear_input for the pasted text: a box that still holds it (or a part of it) may be
        cleared, unless stop() abandoned the restore. Called under _keys, so it takes no lock."""
        return lambda shown: not self._abandoned and _still_ours(shown, text)

    def _clear_box(self, box: str, text: str, seq: int) -> None:
        """Empties the input box that holds the restored message, under _keys. Nothing is typed when a newer paste
        (seq) came since: the box is that paste's then. Each key checks the box again (_still_ours, in clear_input): a
        box that changed to other text gets no further key, and neither does a restore that stop() abandoned
        (_abandoned, read there as a plain bool: this method already holds _keys). A clear that fails is tried once more
        after one poll gap, and logged if it fails again."""
        with self._keys:
            if self._paste_seq != seq:
                return
            rounds = box.count("\n") + text.count("\n") + 2  # input_text drops blank lines: the box may hold more rows
            cleared = self._pane.clear_input(rounds, owned=self._owned_by(text))
            if not cleared:
                self._clock.sleep(RESTORE_POLL_SECONDS)
                cleared = self._pane.clear_input(rounds, owned=self._owned_by(text))
        outcome = "cleared the message the CLI put back after the Escape" if cleared else \
            "could not clear the message the CLI put back after the Escape"
        self._logger.log(f"terminal: tmux pane {self._pane_id}: {outcome}")

    def halt_turn(self) -> None:
        """A ⏹ tap or /stop: interrupt(), which presses one Escape under the rule of _press_escape, so the turn ends as
        interrupted. The session stays open for the next message, and the terminal's CLI is never closed."""
        self.interrupt()

    def answer(self, req: ApprovalRequest, decision: Decision, *, updated_input: dict | None = None,
               answers: dict | None = None, updated_permissions: list | None = None,
               deny_message: str | None = None) -> bool:
        """Always False: the request is answered in the terminal, which asks it. Nothing is sent."""
        return False

    def set_mode(self, mode: str) -> bool:
        """Always False: a mode change would be typed into the user's terminal. It applies to the CLI, not here."""
        return False

    def set_model(self, model: str) -> bool:
        """Always False: a model change would be typed into the user's terminal."""
        return False

    def idle(self) -> bool:
        """True when the pane can take a message now (TmuxPane.ready): it exists, is in no tmux mode, its input box is empty
        and the CLI is not working."""
        return self._pane.ready()
