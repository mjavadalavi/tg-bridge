"""AttachedSession: a Claude Code session that runs in a `tg-bridge view` tab (the attached transport, spec T2).

Nothing is spawned here. The CLI is the process of the terminal's own pane; this session types into that pane and reads
the session's transcript. start() checks that the pane exists and reports the session as resumed. send_user() pastes the
message into the pane (TmuxPane: bracketed paste, then Enter) and follows the transcript for the turn (TranscriptTurns: the
echo line, the answer, the turn_duration). interrupt() presses Escape in the pane. The permissions are asked in the terminal
itself, outside this process, so answer() is False. stop() closes the transcript follow and nothing else: the terminal's CLI
is never signalled or typed into.

The owner passes plain strings (the tmux socket, the pane id, the session uuid, the launcher's config folder, the topic's
folder), never a core type. idle() tells the owner whether a message can be pasted now (TmuxPane.ready).

Threads: the transcript tail thread delivers the lines (TranscriptTurns); the owner's thread calls the methods, and an
interrupt may come from another thread (the ⏹ tap). Lock map: _keys serialises the keys typed into the pane (the paste and
the Escape); the turn state is TranscriptTurns' own. No lock is held while a turn is opened or followed.
"""

import threading
from collections.abc import Callable

from tg_bridge.agents.base import AgentSession, Capabilities, EventSink, SessionSpec, TerminalNotReady, always_attached
from tg_bridge.agents.claude import session_finder
from tg_bridge.agents.claude.tmux_pane import TmuxPane, TmuxRunner, run_tmux
from tg_bridge.agents.claude.transcript_turns import TranscriptTurns
from tg_bridge.agents.claude.translate import ClaudeTranslator
from tg_bridge.agents.events import ApprovalRequest, Decision, SessionStarted
from tg_bridge.clock import Clock
from tg_bridge.logging_ import Logger

UNREACHABLE_ERROR = "the terminal session could not be reached"
ATTACHED_CAPABILITIES = Capabilities(live_mode=False, live_model=False, questions=False, cost_usd=False, images=False,
                                     preassigned_id=True, linger=False,  # the process belongs to the terminal
                                     readable_tail=False)  # and its output is a terminal screen, not a log


def pane_idle(socket: str, pane: str, run: TmuxRunner = run_tmux) -> bool:
    """True when the pane can take a message now (TmuxPane.ready: it exists, is in no tmux mode, its input box is empty and
    the CLI is not working). A pane tmux cannot show reads as not idle: nothing is pasted into a pane that is gone. The
    dispatcher polls this while it waits."""
    return TmuxPane(socket, pane, run=run).ready()


def _no_requests(message: str, *, main_only: bool = False) -> list[str]:
    """The terminal's permission dialog is not in this process: no request of this session is ever pending."""
    return []


class AttachedSession(AgentSession):
    """One Claude Code session of a terminal tab, driven through its tmux pane and read from its transcript."""

    def __init__(self, spec: SessionSpec, sink: EventSink, *, socket: str, pane: str, uuid: str, config_dir: str,
                 cwd: str, logger: Logger, clock: Clock, run: TmuxRunner = run_tmux,
                 still_attached: Callable[[], bool] = always_attached) -> None:
        self.session_id = uuid  # the claim's session; moves to the new one when a /clear or /resume in the tab moves it
        self._config_dir = config_dir
        self._still_attached = still_attached  # the owner's check that the tab's claim is still held (the CLI runs)
        self._spec = spec
        self._cwd = cwd
        self._emit = sink
        self._logger = logger
        self._pane_id = pane
        self._pane = TmuxPane(socket, pane, run=run)
        self._open = False
        self._keys = threading.Lock()  # the paste and an Escape never interleave: an Escape waits for the Enter of the paste
        self._turns = TranscriptTurns(config_dir=config_dir, cwd=cwd, session_id=uuid,
                                      translator=ClaudeTranslator(cwd=cwd), emit=sink, log=logger.log,
                                      logger=logger, clock=clock, deny_requests=_no_requests,
                                      foreign_interrupts=True, relocate=self._relocate, on_relocated=self._moved)

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
        """Closes the transcript follow (its final drain runs first). Sends no key: the CLI in the pane keeps running.
        Returns None, as there is no process of this session to report a code for."""
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
        gen, primary, follows = self._turns.begin()
        armed = self._turns.follow_turn(gen, primary, follows)
        with self._keys:
            pasted = self._pane.paste(text)
        if not pasted:
            self._logger.log(f"message not pasted into tmux pane {self._pane_id}")
            self._turns.finish(gen, error=UNREACHABLE_ERROR)
            return
        if armed:
            self._turns.watch_echo(gen, text)

    def _relocate(self, since: float, text: str, primary: str) -> tuple[str, int, str] | None:
        """The transcript of the session that holds the echo of text, written after since (TranscriptTurns asks after
        ATTACH_RELOCATE_SECONDS). The followed file is excluded: its echo was not read there."""
        found = session_finder.find_relocated(self._config_dir, since, text, exclude=primary)
        return None if found is None else (found.path, found.offset, found.session_id)

    def _moved(self, session_id: str) -> None:
        """The turn moved to another session's transcript: that session is now this session. Its start is reported, so
        the turn runner records the new id as the topic's session."""
        self.session_id = session_id
        self._emit(SessionStarted(session_id=session_id, model=self._spec.model, mode=self._spec.mode, cwd=self._cwd,
                                  resumed=True))

    def interrupt(self) -> bool:
        """Escape in the pane. The interrupt line of the transcript, or else the fallback, ends the turn as INTERRUPTED."""
        return self._turns.interrupt(self._press_escape)

    def _press_escape(self) -> bool:
        with self._keys:
            return self._pane.escape()

    def halt_turn(self) -> None:
        """A ⏹ tap or /stop: Escape in the pane, so the turn ends as interrupted. The session stays open for the next
        message, and the terminal's CLI is never closed."""
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
