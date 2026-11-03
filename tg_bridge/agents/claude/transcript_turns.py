"""TranscriptTurns: the turns of one Claude session, read from its transcript (the tui transport and the attached session).

A turn is read from the session's .jsonl: TranscriptTail follows the file and every line goes to the ClaudeTranslator, and
TuiTurn keeps the turn's usage, text and interrupt state. The turn ends at the first system/turn_duration after its message
was typed; an interrupt (Esc) writes none, so the turn ends itself after INTERRUPT_FALLBACK_SECONDS. A message whose line
never reaches the transcript ends its turn with an error after USER_ECHO_SECONDS (the echo watch; an attached session
looks for it in rounds and waits longer for a queued message). Fifteen seconds after the
send, a transcript that is not where it was expected is looked up and followed from there (TRANSCRIPT_WAIT_SECONDS).

The tail lives as long as the session, not only as long as a turn: after a turn it keeps reading, so background work the
CLI reports (BackgroundLedger) and a turn the CLI starts by itself (TurnBegan, then a new generation with its own
TurnResult) are both seen. A session marked ending (an intentional close) or closed starts no such turn.

foreign_interrupts is the attached session's mode: the Esc (or a denied permission) was pressed in the terminal, not by this
bridge, so no interrupt is armed here, and an interrupt line of the transcript ends the turn as INTERRUPTED on its own. Without
it (the tui transport), an interrupt line counts only while the bridge's own Esc is awaited.

strict_echo (also the attached session) makes the turn's echo its own user line only: until that line is read, the lines
before it belong to an earlier turn of the tab, so a turn_duration among them does not end the pasted turn and their
text and tool events are not sent to the topic.

Nothing here knows of processes, PTYs or tmux. The owner passes the transcript location (config_dir, cwd, session_id), its
translator and logger, and small callables: emit (an event goes out), log (a message), deny_requests (the owner's pending
permission requests, denied when an interrupt ends the main turn) and the clock.

Threads: the tail thread delivers every transcript line (on_transcript); the owner's thread begins turns, interrupts and
closes; the locate, echo and interrupt timers run on threads of their own. Lock map: _turn_lock guards the generations, the
tail, the ledger and TuiTurn; no event is emitted and no tail is stopped while it is held.
"""

import os
import threading
from collections.abc import Callable

from tg_bridge.agents.claude import history
from tg_bridge.agents.claude.background import BackgroundLedger
from tg_bridge.agents.claude.hookbridge import TURN_ENDED
from tg_bridge.agents.claude.tail import TranscriptTail
from tg_bridge.agents.claude.translate import ClaudeTranslator
from tg_bridge.agents.claude.tui_turn import (ATTACH_ECHO_MISSED_ERROR, ECHO_MISSED_ERROR, ECHO_MISSED_SUBTYPE,
                                                INTERRUPTED_SUBTYPE, TURN_DURATION, USER_ECHO_SECONDS, TuiTurn,
                                                is_interrupt, turn_start_origin)
from tg_bridge.agents.events import (BackgroundWork, SessionStarted, TurnBegan, TurnResult, TurnStatus, UsageEvent)
from tg_bridge.clock import Clock
from tg_bridge.logging_ import Logger

TRANSCRIPT_WAIT_SECONDS = 15.0  # the transcript must exist this soon after the message is typed
INTERRUPT_FALLBACK_SECONDS = 3.0  # without an interrupt line in the transcript, the session ends the turn itself
ATTACH_RELOCATE_SECONDS = 10.0  # an attached turn whose echo is not read this soon looks for the session's new transcript
ATTACH_QUEUED_MAX_SECONDS = 600.0  # a message the CLI queued (a compaction, a busy turn) may be echoed this late

# relocate(since, text, primary) -> (path, offset, session_id) of the transcript that holds the echo, or None
Relocate = Callable[[float, str, str], tuple[str, int, str] | None]


class TranscriptTurns:
    """The turns of one session read from its transcript. begin() opens a turn for a send, follow_turn() and watch_echo()
    follow the transcript and arm the echo net, finish() ends a turn once, and close() stops the tail for good."""

    def __init__(self, *, config_dir: str, cwd: str, session_id: str | None, translator: ClaudeTranslator,
                 emit: Callable[[object], None], log: Callable[[str], None], logger: Logger, clock: Clock,
                 deny_requests: Callable[..., object], foreign_interrupts: bool = False,
                 relocate: Relocate | None = None, on_relocated: Callable[[str], None] | None = None,
                 strict_echo: bool = False) -> None:
        self._config_dir = config_dir
        self._relocate = relocate  # set for an attached session: the echo is looked for in the newer transcripts too
        self._on_relocated = on_relocated  # told the new session id when the turn moved to its transcript
        self._foreign_interrupts = foreign_interrupts  # an interrupt line ends a turn even when no Esc of ours was sent
        self._strict_echo = strict_echo  # an attached turn counts only its own user line as its echo (TuiTurn strict)
        self._cwd = cwd
        self._session_id = session_id
        self._translator = translator
        self._emit = emit
        self._log = log
        self._logger = logger
        self._clock = clock
        self._deny_requests = deny_requests
        self._turn_lock = threading.Lock()  # guards everything below; never held while emitting or stopping a tail
        self._closed = False
        self._gen = 0                       # the turn generation: a line from an older turn's tail is ignored
        self._done = True                   # no turn is running yet
        self._tail: TranscriptTail | None = None
        self._tail_gen = 0                  # the tail generation: a line from a replaced tail is ignored
        self._ending = False                # set by an intentional close: no self turn starts from here on
        self._before: tuple[str | None, int] = (None, 0)  # the transcript found before the send, and its size
        self._sent_at = 0.0  # the wall-clock time the current turn began, before its message is typed
        self._turn = TuiTurn(strict=strict_echo)   # the turn's usage, text and interrupt state; guarded by _turn_lock
        self._ledger = BackgroundLedger()          # the background work still pending; guarded by _turn_lock

    # ---- owner's thread -------------------------------------------------------------------------------

    def open_unless_closed(self, open_: Callable[[], None]) -> bool:
        """Runs open_ (the owner's permission socket) under the turn lock, unless the session is closed. True when it ran.
        A close cannot come between the check and the open."""
        with self._turn_lock:
            if self._closed:  # a stop() came first: nothing is opened
                return False
            open_()
            return True

    def mark_ending(self) -> None:
        """An intentional close (stop, end_gracefully) begins: a transcript line that would start a turn is no longer a
        turn. A natural exit does not call this, so its final drain still sees a background update."""
        with self._turn_lock:
            self._ending = True

    def close(self) -> None:
        """Stops following the transcript. Safe to call more than once.

        The tail is stopped first: its final drain delivers the lines the CLI wrote before it exited (the last answer,
        turn_duration, and a background update written just before a natural exit), so transcript handling is still on
        while it runs; an intentional close marked _ending before this, so its drain starts no self turn. Only then is the
        session marked closed, which rejects the lines read after this point. The drain runs without _turn_lock held."""
        with self._turn_lock:
            tail = self._tail
        if tail is not None:
            tail.stop()
        with self._turn_lock:
            self._closed = True
            current, self._tail = self._tail, None
        if current is not None and current is not tail:  # a follow that started during the drain
            current.stop()

    def begin(self, text: str = "") -> tuple[int, str | None, bool]:
        """Opens a turn for a send: a new generation with a fresh TuiTurn, and the transcript as it is now. text is the
        message pasted for the turn (TuiTurn.expect: its echo is then the line that holds it). Returns the generation,
        the transcript path of this session (None without a session id) and whether a tail already follows."""
        primary = self._primary_path()
        before = None
        if self._session_id:
            before = history.transcript_path(self._config_dir, self._cwd, self._session_id)
        with self._turn_lock:
            follows = self._tail is not None  # a live session keeps the tail of its earlier turns
            self._gen += 1
            gen = self._gen
            self._done = False
            self._turn.reset()
            self._turn.expect(text)  # before the paste: the echo of this message is the only line that starts it
            self._before = (before, TranscriptTail.size_of(before) if before else 0)
            self._sent_at = self._clock.now()
        return gen, primary, follows

    def follow_turn(self, gen: int, primary: str | None, follows: bool) -> bool:
        """Follows the transcript for the turn gen (unless a tail already does). True when the echo net may be armed for it."""
        if primary is not None and not follows:
            self._follow(gen, primary, TranscriptTail.size_of(primary))  # the turn's lines start after what is there
            threading.Thread(target=self._locate, args=(gen, primary), name="transcript-locate", daemon=True).start()
        with self._turn_lock:  # the echo net is armed only where this turn's transcript is followed
            return self._tail is not None and self._gen == gen

    def watch_echo(self, gen: int, text: str = "") -> None:
        """Starts the echo net of turn gen: called right after its message (text) is typed. A relocation looks at the
        transcripts written since the turn began (begin), so an echo written while the message is pasted is found."""
        with self._turn_lock:
            since = self._sent_at
        threading.Thread(target=self._echo_watch, args=(gen, since, text), name="user-echo", daemon=True).start()

    def interrupt(self, press: Callable[[], bool]) -> bool:
        """Esc through press(). The interrupt line of the transcript, or else the fallback after INTERRUPT_FALLBACK_SECONDS,
        ends the turn with INTERRUPTED. Nothing is armed outside a turn."""
        with self._turn_lock:
            gen = self._gen
            armed = not self._done
            self._turn.interrupting = self._turn.interrupting or armed
        sent = press()
        if armed:
            threading.Thread(target=self._interrupt_fallback, args=(gen,), name="interrupt-fallback",
                             daemon=True).start()
        return sent

    # ---- tail thread ----------------------------------------------------------------------------------

    def on_transcript(self, tail_gen: int, line: dict) -> None:
        """Tail thread: one transcript line. A line that starts a turn while none is open (a task notification, a
        wakeup, or the bridge's own message seen late) opens a new generation, unless the session is ending."""
        origin = turn_start_origin(line)
        with self._turn_lock:
            if tail_gen != self._tail_gen or self._closed:
                return
            changed = self._ledger.note(line)
            pending = self._ledger.pending()
            began = self._done and not self._ending and origin is not None
            if began:
                self._gen += 1
                self._done = False
                self._turn.reset()
                self._turn.echoed = True  # the tab's own starting line is the echo: nothing is typed for this turn
            gen = self._gen
        if changed:
            self._emit(BackgroundWork(pending=pending))
        if began:
            self._emit(TurnBegan(origin=origin))
        if line.get("type") == "system" and line.get("subtype") == TURN_DURATION:
            with self._turn_lock:
                earlier = self._turn.waiting_for_echo  # this duration closes an earlier turn the owner started in the tab
            if not earlier:
                self.finish(gen, duration=line)
            return
        with self._turn_lock:
            if gen != self._gen or self._done:
                return
            self._turn.note(line)
            # an earlier turn's line: the ledger has it, the topic does not. The interrupt of our own Esc still ends the
            # turn: the CLI writes it when it stops, even when the pasted message's echo was not read first.
            own_interrupt = self._turn.interrupting and is_interrupt(line)
            if self._turn.waiting_for_echo and not own_interrupt:
                return
            interrupted = (self._turn.interrupting or self._foreign_interrupts) and is_interrupt(line)
        try:
            events = self._translator.translate(line)
        except Exception as exc:  # one bad line must not stop the turn
            self._log(f"translate: {exc!r}")
            events = []
        self._emit_turn(gen, events)
        if interrupted:
            self.finish(gen, interrupted=True)

    # ---- turn end -------------------------------------------------------------------------------------

    def finish(self, gen: int, *, duration: dict | None = None, interrupted: bool = False,
               error: str | None = None) -> None:
        """Ends the turn once: an open permission request is closed first, then the usage and the TurnResult. The tail
        keeps running: it belongs to the session, and closes with it. An error ends it only while its message is unread."""
        with self._turn_lock:
            if gen != self._gen or self._done:
                return
            if error is not None and (self._turn.echoed or self._closed):  # read after all, or the session is gone
                return
            self._done = True
            self._turn.interrupting = False
            usage = self._turn.usage_event()
            if interrupted:
                result = TurnResult(status=TurnStatus.INTERRUPTED, answer=None, subtype=INTERRUPTED_SUBTYPE)
            elif error is not None:
                result = TurnResult(status=TurnStatus.ERROR, answer=None, subtype=ECHO_MISSED_SUBTYPE, error=error)
            else:
                result = self._turn.result(duration)
        if interrupted:  # the main turn's requests still open after an Esc are denied before it is reported as ended;
            self._deny_requests(TURN_ENDED, main_only=True)  # a background subagent's ask stays open
        self._emit(usage)
        self._emit(result)

    # ---- internals ------------------------------------------------------------------------------------

    def _primary_path(self) -> str | None:
        if not self._session_id:
            return None
        folder = history.encoded_cwd(os.path.realpath(self._cwd))
        return os.path.join(self._config_dir, "projects", folder, self._session_id + ".jsonl")

    def _follow(self, gen: int, path: str, offset: int) -> None:
        """Replaces the tail with one reading path from offset. Each tail has its own generation, so the lines of a
        replaced tail are ignored."""
        with self._turn_lock:
            if gen != self._gen or self._done or self._closed:
                return
            self._tail_gen += 1
            tail_gen = self._tail_gen
        tail = TranscriptTail(path, lambda line: self.on_transcript(tail_gen, line), logger=self._logger,
                              start_offset=offset)
        with self._turn_lock:
            if tail_gen != self._tail_gen or self._closed:  # a newer follow or a close came first
                return
            old, self._tail = self._tail, tail
        tail.start()
        if old is not None:
            old.stop()

    def _locate(self, gen: int, primary: str) -> None:
        """Fifteen seconds after the send: when the transcript is still not where it was expected, it is looked up with
        history.transcript_path and followed from there. The lookup happens once and is logged either way."""
        self._clock.sleep(TRANSCRIPT_WAIT_SECONDS)
        with self._turn_lock:
            if gen != self._gen or self._done or self._closed:
                return
            before_path, before_size = self._before
        if os.path.isfile(primary):
            return
        found = history.transcript_path(self._config_dir, self._cwd, self._session_id or "")
        if found is None or found == primary:
            self._log(f"no transcript within {TRANSCRIPT_WAIT_SECONDS:g} s; the turn may not end on its own")
            return
        self._log(f"transcript found at {found}, following it")
        # A file that existed before the send is read from its size at that time; a new one from its start.
        self._follow(gen, found, before_size if found == before_path else 0)

    def _emit_turn(self, gen: int, events: list) -> None:
        with self._turn_lock:
            if gen != self._gen or self._done:
                return
        for event in events:
            if not isinstance(event, (SessionStarted, TurnResult, UsageEvent)):  # the session reports these itself
                self._emit(event)

    def _interrupt_fallback(self, gen: int) -> None:
        self._clock.sleep(INTERRUPT_FALLBACK_SECONDS)
        self.finish(gen, interrupted=True)

    def _echo_watch(self, gen: int, since: float, text: str) -> None:
        """Safety net for a typed message: when no line of this turn's message is read within USER_ECHO_SECONDS,
        the keys went to a dialog of the TUI, and the turn ends with an error instead of waiting for a
        turn_duration that never comes. An attached session works in rounds of ATTACH_RELOCATE_SECONDS: each
        round looks for the echo in the transcripts written since the paste (a /clear or /resume in the tab
        moved the session), and a message the CLI queued (a compaction, a busy turn) keeps the turn open for
        ATTACH_QUEUED_MAX_SECONDS without those lookups, since the followed transcript holds it. Nothing is
        sent again. finish does nothing when the message was read, or when the generation is no longer this
        turn."""
        if self._relocate is None:
            self._clock.sleep(USER_ECHO_SECONDS)
            self.finish(gen, error=ECHO_MISSED_ERROR)
            return
        waited = 0.0
        while True:
            self._clock.sleep(ATTACH_RELOCATE_SECONDS)
            waited += ATTACH_RELOCATE_SECONDS
            with self._turn_lock:
                if gen != self._gen or self._done or self._closed or self._turn.echoed:
                    return
                queued = self._turn.queued
                session = self._session_id or ""
            limit = ATTACH_QUEUED_MAX_SECONDS if queued else USER_ECHO_SECONDS
            if waited >= limit:
                self.finish(gen, error=ATTACH_ECHO_MISSED_ERROR.format(session=session, seconds=limit))
                return
            if not queued:  # a queued message is in the transcript being followed: other transcripts cannot hold it
                self._relocate_turn(gen, since, text, waited)

    def _relocate_turn(self, gen: int, since: float, text: str, waited: float) -> None:
        """Follows the turn to the transcript that holds its echo, from the echo line on, when the echo was not read in
        the claim's transcript. The session id moves with it (the next message follows that file too). waited is the time
        the echo has been looked for, for the log."""
        with self._turn_lock:
            if gen != self._gen or self._done or self._closed or self._turn.echoed:
                return
            primary = self._primary_path() or ""
        try:
            found = self._relocate(since, text, primary)
        except Exception as exc:  # the lookup is best-effort: the turn keeps its echo net
            self._log(f"session relocation failed: {exc!r}")
            return
        if found is None:
            self._log(f"no transcript holds the message after {waited:g} s")
            return
        path, offset, session_id = found
        with self._turn_lock:  # the lookup ran without the lock: the turn may have moved on meanwhile
            if gen != self._gen or self._done or self._closed or self._turn.echoed or self._turn.queued:
                self._log("session relocation dropped: the turn moved on")
                return
            self._session_id = session_id
        self._log(f"session moved to {session_id}: following {path}")
        if self._on_relocated is not None:  # before the follow: the lines read there may end the turn at once
            self._on_relocated(session_id)
        self._follow(gen, path, offset)
