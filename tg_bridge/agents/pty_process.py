"""PtyProcess: a child process on a pseudo-terminal, for an interactive TUI that cannot run on pipes.

Why a PTY: the interactive Claude TUI needs a terminal, and it draws its screen with escape codes. The child gets the
slave end of a pty as stdin, stdout and stderr. One reader thread drains the master end and keeps a bounded copy of
the screen text with the escape codes removed, so wait_for() can look for a dialog or a prompt. The reader also
answers the cursor-position query ESC[6n, because the TUI stalls without a reply.

Typing: text goes in as a bracketed paste (ESC[200~ ... ESC[201~), then a short pause, then CR. The pause is the one
the live spike used (PASTE_GAP_SECONDS). After ANSI stripping the spaces are not reliable, so callers match on
compact_screen(), which has all whitespace removed.

Writes are bounded: the master is non-blocking, and a write gives up after write_timeout seconds when the child stops
reading. So a full pty buffer cannot hold the write lock forever, and stop() can always take the lock to close the master.

Hooks, as in ProcessSession: _argv() (required), _on_output(text) on the reader thread, _on_spawned(),
_graceful_close() and _on_exited(rc). The kill ladder, the process group and the stderr file are the ChildGroup of
agents/process.py, so both channels share one set of signal rules. Process waits use real time, as there.
"""

import codecs
import fcntl
import os
import pty
import re
import select
import signal
import struct
import subprocess
import termios
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

from tg_bridge.agents.base import EventSink
from tg_bridge.agents.events import ProcessExited
from tg_bridge.agents.process import GROUP_WAIT, ChildGroup
from tg_bridge.clock import Clock
from tg_bridge.logging_ import Logger

SCREEN_CHARS = 16000        # the newest characters of the stripped screen kept in memory
PASTE_GAP_SECONDS = 0.6     # pause between the bracketed paste and CR, as in the live spike
POLL_SECONDS = 0.2          # wait_for() looks at the screen this often
WRITE_TIMEOUT_SECONDS = 10.0  # longest one write waits for the child to read; a timed-out write is not retried
CLOSE_WRITE_TIMEOUT_SECONDS = 1.0  # the writes of a close (Esc, /exit) give up this soon, so stop() is not held up
REDRAW_SETTLE_SECONDS = 0.8  # the pause after each size of a redraw: the TUI reads the size when it handles the SIGWINCH
REDRAW_MIN_COLS = 2         # a redraw narrows the pty by one column, so a pty narrower than this is not redrawn
READ_SIZE = 4096
ESC = "\x1b"
CR = "\r"
CURSOR_QUERY = "\x1b[6n"    # the TUI asks where the cursor is
CURSOR_REPLY = "\x1b[1;1R"  # row 1, column 1: enough for the TUI to go on
PASTE_START = "\x1b[200~"
PASTE_END = "\x1b[201~"
ARROW_DOWN = "\x1b[B"
_MAX_HELD = 512             # an unfinished escape longer than this is no longer held back
# CSI, then OSC ended by BEL or ST, then a two-character escape. The last one must not start with [ or ], because
# those open CSI and OSC and a partial one must stay unmatched until its end arrives.
_ESCAPE_SEQUENCE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b(?!\[|\])[ -/]*[0-~]")
_CONTROL = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")  # every control character except newline
_WHITESPACE = re.compile(r"\s+")


def _wait_fd(fd: int, event: int, seconds: float) -> int:
    """Sleep until fd is ready for event, or for seconds at most. Returns the events poll() reported, 0 on timeout.
    poll() instead of select(), so no fd number is too big."""
    poller = select.poll()
    poller.register(fd, event)
    ready = poller.poll(int(max(0.0, seconds) * 1000))
    return ready[0][1] if ready else 0


def _held_start(data: str) -> int:
    """Where an unfinished escape sequence at the end of data starts, or len(data) when there is none.

    The part from that point is held back until the next chunk completes it, so a query split across two reads is still
    seen whole and answered once.
    """
    i = data.rfind(ESC)
    if i < 0 or len(data) - i > _MAX_HELD:
        return len(data)
    return i if _ESCAPE_SEQUENCE.match(data, i) is None else len(data)


class PtyProcess(ChildGroup, ABC):
    """One child process on a pty, bound to one session. Subclasses implement _argv()."""

    def __init__(self, *, name: str, cwd: str, env: Mapping[str, str], errfile_dir: str,
                 sink: EventSink, logger: Logger, clock: Clock, rows: int = 50, cols: int = 200,
                 write_timeout: float = WRITE_TIMEOUT_SECONDS) -> None:
        super().__init__(name=name, errfile_dir=errfile_dir, sink=sink, logger=logger, clock=clock)
        self._cwd = cwd
        self._env = dict(env)
        self._rows = rows
        self._cols = cols
        self._write_timeout = write_timeout
        self._wlock = threading.Lock()  # guards the master fd: writes and its close
        self._master: int | None = None
        self._screen_lock = threading.Lock()
        self._screen = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")  # reader thread only
        self._held = ""  # reader thread only
        self._redraw_lock = threading.Lock()  # guards the two redraw flags below
        self._redrawing = False               # a redraw worker is running
        self._redraw_pending = False          # a redraw was asked for while a worker was running

    # ---- hooks for subclasses -------------------------------------------------------------------------

    @abstractmethod
    def _argv(self) -> list[str]:
        """The command line to spawn."""

    def _on_output(self, text: str) -> None:
        """Each decoded chunk, ANSI codes included, on the reader thread. An escape split across reads arrives whole."""

    def _on_spawned(self) -> None:
        """Called after the reader thread started. A handshake that needs the screen goes here or in start()."""

    def _graceful_close(self) -> None:
        """First step of stop(). The default does nothing, so the ladder waits the grace period first."""

    def _on_exited(self, returncode: int | None) -> None:
        """Reader thread, after the child was reaped and the master closed. The default reports ProcessExited."""
        self._emit(ProcessExited(returncode=returncode, stderr_tail=self.stderr_tail()))

    # ---- lifecycle ------------------------------------------------------------------------------------

    def start(self) -> None:
        """Open a pty, spawn the child on its slave end, start the reader thread, then call _on_spawned()."""
        if self._proc is not None:
            raise RuntimeError(f"session {self._name}: already started")
        argv = self._argv()
        errfile = self._new_errfile()
        master, slave = pty.openpty()
        try:
            os.set_blocking(master, False)  # writes give up on a full buffer; the reader polls for input
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", self._rows, self._cols, 0, 0))
            proc = subprocess.Popen(
                argv, cwd=self._cwd, env=self._env, stdin=slave, stdout=slave, stderr=slave,
                start_new_session=True, close_fds=True,
            )
        except BaseException:
            os.close(master)
            raise
        finally:
            os.close(slave)  # the child has its own copies; the parent must not keep the slave open
        self._errfile = errfile
        self._master = master
        self._proc = proc
        self._pgid = proc.pid  # new session, so the group id equals the pid
        self._reader_thread = threading.Thread(target=self._reader, args=(proc, master), daemon=True)
        self._reader_thread.start()
        self._on_spawned()

    def stop(self, grace: float = 5.0) -> int | None:
        """Graceful close, then wait grace seconds, then SIGINT, SIGTERM and SIGKILL, then empty the group and close
        the master. Returns the child's return code, or None when it was never started."""
        proc = self._proc
        if proc is None:
            return None
        if self.alive():
            self._graceful_close()
            if not self._wait(proc, grace):
                self._signal_ladder(proc)
        self._sweep_group()
        thread = self._reader_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=GROUP_WAIT)
            if thread.is_alive():  # the reader still owns the master; it closes it when it ends
                self._log("reader still running after the sweep; the master closes when it ends")
                return proc.returncode
        self._close_master()
        return proc.returncode

    def end_gracefully(self, timeout: float) -> bool:
        """Wait up to timeout seconds for the child to exit by itself. No key and no signal is sent here. True when the
        child has exited (or never ran)."""
        proc = self._proc
        if proc is None or not self.alive():
            return True
        return self._wait(proc, timeout)

    # ---- typing ---------------------------------------------------------------------------------------

    def write(self, data: str, timeout: float | None = None) -> bool:
        """Send raw text to the terminal. Waiting for the write lock and the write share one budget of timeout seconds
        (write_timeout when None). False when the child was never started, the master is closed, the budget ran out,
        the child stopped reading or the write failed. A timed-out write may have sent a prefix."""
        budget = self._write_timeout if timeout is None else timeout
        deadline = time.monotonic() + budget
        if not self._wlock.acquire(timeout=budget):
            self._log(f"write not sent: the write lock was busy for {budget}s")
            return False
        try:
            fd = self._master
            if fd is None:
                return False
            payload = data.encode("utf-8")
            while payload:
                try:
                    payload = payload[os.write(fd, payload):]
                except BlockingIOError:  # the pty buffer is full: the child is not reading
                    left = deadline - time.monotonic()
                    if left <= 0:
                        self._log(f"write timed out after {budget}s; the child is not reading")
                        return False
                    # A hang-up means the child side is gone; os.write would keep saying EAGAIN, so stop here.
                    hung_up = select.POLLHUP | select.POLLERR | select.POLLNVAL
                    if _wait_fd(fd, select.POLLOUT, left) & hung_up:
                        self._log("write failed: the child side of the pty is closed")
                        return False
            return True
        except OSError as e:  # EIO once the child side is gone
            self._log(f"write failed: {e!r}")
            return False
        finally:
            self._wlock.release()

    def type_text(self, text: str, timeout: float | None = None) -> bool:
        """Paste text as one bracketed paste, wait the paste gap, then press CR to submit it. A False after a partial
        paste leaves the child in paste mode, so the caller must end the session."""
        if not self.write(PASTE_START + text + PASTE_END, timeout):
            return False
        self._clock.sleep(PASTE_GAP_SECONDS)
        return self.write(CR, timeout)

    def press_esc(self, timeout: float | None = None) -> bool:
        return self.write(ESC, timeout)

    def press_down(self) -> bool:
        return self.write(ARROW_DOWN)

    def press_enter(self) -> bool:
        return self.write(CR)

    # ---- redraw ---------------------------------------------------------------------------------------

    def redraw(self) -> None:
        """Makes the TUI draw its whole screen again. The live spike showed that a SIGWINCH at the same size draws
        nothing, but a size one column narrower, then the real size, with a SIGWINCH after each, makes the TUI repaint
        everything. Returns at once: the steps run on a short worker thread, so the caller (the reader) is not held.
        Requests never overlap: a request during a running redraw makes the worker run one more round after it, so
        every request is followed by a full pair of sizes and the sizes come in order.
        The narrow size lasts REDRAW_SETTLE_SECONDS, so an attached user may see the live TUI flicker at each redraw.
        A pty narrower than REDRAW_MIN_COLS is not redrawn at all."""
        if self._cols < REDRAW_MIN_COLS:
            return
        with self._redraw_lock:
            self._redraw_pending = True
            if self._redrawing:
                return
            self._redrawing = True
        threading.Thread(target=self._redraw_worker, name="tui-redraw", daemon=True).start()

    def _redraw_worker(self) -> None:
        while True:
            with self._redraw_lock:
                if not self._redraw_pending:
                    self._redrawing = False
                    return
                self._redraw_pending = False
            self._redraw_steps()

    def _redraw_steps(self) -> None:
        self._set_size(max(1, self._cols - 1))  # a 1-column size must not become 0 columns
        self._signal_resize()
        time.sleep(REDRAW_SETTLE_SECONDS)  # real time, as the other waits of the pty
        self._set_size(self._cols)
        self._signal_resize()
        time.sleep(REDRAW_SETTLE_SECONDS)  # the next round must not change the size before the child has read this one

    def _set_size(self, cols: int) -> None:
        """Sets the pty to rows x cols under the write lock. A closed master is skipped."""
        with self._wlock:
            fd = self._master
            if fd is None:
                return
            try:
                fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", self._rows, cols, 0, 0))
            except OSError as e:
                self._log(f"resize: {e!r}")

    def _signal_resize(self) -> None:
        """SIGWINCH to the whole process group, unless the child has exited. A wrapper launcher runs claude as its child
        in the same group and does not forward the signal, and the pty has no controlling terminal to send one, so the
        group is signalled and not only the launcher."""
        if self._pgid is None or not self.alive():
            return
        try:
            os.killpg(self._pgid, signal.SIGWINCH)
        except (ProcessLookupError, PermissionError):
            pass

    # ---- screen ---------------------------------------------------------------------------------------

    def screen(self) -> str:
        """The newest SCREEN_CHARS characters of the output, escape codes and control characters removed."""
        with self._screen_lock:
            return self._screen

    def compact_screen(self) -> str:
        """screen() with all whitespace removed, the form to match against."""
        return _WHITESPACE.sub("", self.screen())

    def wait_for(self, predicate: Callable[[str], bool], timeout: float) -> bool:
        """True once predicate(compact_screen()) holds, False after timeout seconds or as soon as the child died."""
        deadline = time.monotonic() + timeout
        while True:
            if predicate(self.compact_screen()):
                return True
            if not self.alive() or time.monotonic() >= deadline:
                return False
            time.sleep(min(POLL_SECONDS, max(0.0, deadline - time.monotonic())))

    def stderr_tail(self, n: int = 20) -> str:
        """The last n lines of the screen. A pty has no separate stderr, so the screen stands in for it."""
        lines = self.screen().splitlines(keepends=True)
        return "".join(lines[-n:])

    # ---- reader ---------------------------------------------------------------------------------------

    def _reader(self, proc: subprocess.Popen, master: int) -> None:
        try:
            while True:
                try:
                    chunk = os.read(master, READ_SIZE)
                except BlockingIOError:  # nothing to read yet: the master is non-blocking
                    _wait_fd(master, select.POLLIN, POLL_SECONDS)
                    continue
                except OSError:  # EIO is how Linux ends a pty whose child side is closed
                    break
                if not chunk:
                    break
                try:
                    self._feed(chunk)
                except Exception as e:  # one bad chunk must not end the stream
                    self._log(f"output: {e!r}")
        except Exception as e:  # reading itself failed: the exit path below still runs
            self._log(f"reader: {e!r}")
        finally:  # always, exactly once: reap, keep the screen for a post-mortem, release the master, report
            rc = proc.wait()
            self._write_errfile()
            self._close_master()
            try:
                self._on_exited(rc)
            except Exception as e:
                self._log(f"exit callback: {e!r}")

    def _feed(self, chunk: bytes) -> None:
        data = self._held + self._decoder.decode(chunk)
        cut = _held_start(data)
        self._held = data[cut:]
        data = data[:cut]
        for _ in range(data.count(CURSOR_QUERY)):
            self.write(CURSOR_REPLY)  # under the write lock, so it never splits a paste
        if not data:
            return
        text = _CONTROL.sub("", _ESCAPE_SEQUENCE.sub("", data))
        with self._screen_lock:
            self._screen = (self._screen + text)[-SCREEN_CHARS:]
        try:
            self._on_output(data)
        except Exception as e:
            self._log(f"on_output: {e!r}")

    def _write_errfile(self) -> None:
        """Keep the screen tail in the per-session file, so a post-mortem survives the process."""
        if self._errfile is None:
            return
        try:
            with open(self._errfile, "w", encoding="utf-8") as f:
                f.write(self.screen())
        except OSError as e:
            self._log(f"errfile: {e!r}")

    def _close_master(self) -> None:
        with self._wlock:
            fd, self._master = self._master, None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
