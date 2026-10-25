"""ProcessSession: Template Method base for a backend that is one child process driven over stdio.

Both CLIs run the same way: spawn in a new session (so the whole group can be killed), stdout read line by line on a
reader thread, stdin written under one lock, stderr kept in a per-session file, and a kill ladder on stop(). What
differs is the argv, what a line means and what the handshake is, so those are hooks that a subclass overrides:

  _argv()            the command line to spawn (required)
  _on_line(line)     one stdout line, on the reader thread (required)
  _on_spawned()      after the reader thread started; a handshake goes here (default: nothing)
  _graceful_close()  first step of stop(); default closes stdin, Claude sends an interrupt first
  _on_exited(rc)     reader thread, after the process was reaped; default emits ProcessExited

The EventSink receives ProcessExited when the stream ends. A sink or _on_line exception is logged and never ends
the stream. An exception from reading stdout itself is logged and ends the reader, and ProcessExited still follows.
Deadlines for process waits use real time, because a child process cannot be driven by a fake clock; the Clock is
used only to stamp the stderr file name.
"""

import json
import os
import re
import signal
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

from tg_bridge.agents.base import EventSink
from tg_bridge.agents.events import ProcessExited
from tg_bridge.clock import Clock
from tg_bridge.logging_ import Logger

KEEP_ERRFILES = 50          # newest stderr files kept in the turns folder, the new one included
SIGNAL_STEP = 0.5           # seconds a child gets to exit after SIGINT or SIGTERM
KILL_WAIT = 1.0             # seconds to wait after SIGKILL
GROUP_WAIT = 1.0            # seconds the process group gets to empty after the leader exited
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")


def _group_alive(pgid: int) -> bool:
    """True while any process still belongs to process group pgid (the leader may already be gone)."""
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


class ProcessSession(ABC):
    """One child process bound to one session. Subclasses implement _argv() and _on_line()."""

    def __init__(self, *, name: str, cwd: str, env: Mapping[str, str], errfile_dir: str,
                 sink: EventSink, logger: Logger, clock: Clock) -> None:
        self._name = name
        self._cwd = cwd
        self._env = dict(env)
        self._errfile_dir = errfile_dir
        self._sink = sink
        self._logger = logger
        self._clock = clock
        self._wlock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._pgid: int | None = None
        self._reader_thread: threading.Thread | None = None
        self._errfile: str | None = None

    # ---- hooks for subclasses -------------------------------------------------------------------------

    @abstractmethod
    def _argv(self) -> list[str]:
        """The command line to spawn."""

    @abstractmethod
    def _on_line(self, line: str) -> None:
        """One stdout line without its newline. Runs on the reader thread and must not wait for a reply."""

    def _on_spawned(self) -> None:
        """Called after spawn and reader start. A handshake that needs replies goes here or in start()."""

    def _graceful_close(self) -> None:
        """First step of stop(): ask the child to finish. The default closes stdin (the child sees EOF)."""
        self.close_stdin()

    def _on_exited(self, returncode: int | None) -> None:
        """Reader thread, after the child was reaped. The default reports ProcessExited to the sink."""
        self._emit(ProcessExited(returncode=returncode, stderr_tail=self.stderr_tail()))

    def _iter_stdout(self, proc: subprocess.Popen[str]) -> Iterable[str]:
        """The stdout lines of proc. A seam: tests replace it to make reading itself fail."""
        return iter(proc.stdout)

    # ---- lifecycle ------------------------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the child in its own session, start the reader thread, then call _on_spawned()."""
        if self._proc is not None:
            raise RuntimeError(f"session {self._name}: already started")
        argv = self._argv()
        errfile = self._new_errfile()
        with open(errfile, "w", encoding="utf-8") as errh:
            proc = subprocess.Popen(
                argv, cwd=self._cwd, env=self._env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=errh, text=True, encoding="utf-8", errors="replace", bufsize=1,
                start_new_session=True,
            )
        self._errfile = errfile
        self._proc = proc
        self._pgid = proc.pid  # new session, so the group id equals the pid
        self._reader_thread = threading.Thread(target=self._reader, args=(proc,), daemon=True)
        self._reader_thread.start()
        self._on_spawned()

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self, grace: float = 5.0) -> int | None:
        """Graceful close, then wait grace seconds, then SIGINT, SIGTERM and SIGKILL, then empty the group.

        Returns the child's return code, or None when it was never started.
        """
        proc = self._proc
        if proc is None:
            return None
        if self.alive():
            self._graceful_close()
            if not self._wait(proc, grace):
                self._signal_ladder(proc)
        self._sweep_group()
        self.close_stdin()
        return proc.returncode

    def end_gracefully(self, timeout: float) -> bool:
        """Close stdin so the child sees EOF and finishes by itself, then wait up to timeout seconds. No interrupt and
        no signal is sent here. True when the child has exited (or never ran)."""
        proc = self._proc
        if proc is None or not self.alive():
            return True
        self.close_stdin()
        return self._wait(proc, timeout)

    def write(self, obj: dict) -> bool:
        """Write one JSON object as one line to stdin. False when there is no stdin to write to."""
        proc = self._proc
        if proc is None:
            return False
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        try:
            with self._wlock:
                proc.stdin.write(line)
                proc.stdin.flush()
            return True
        except (OSError, ValueError) as e:  # BrokenPipeError is an OSError
            self._log(f"write failed: {e!r}")
            return False

    def close_stdin(self) -> None:
        with self._wlock:
            try:
                if self._proc is not None:
                    self._proc.stdin.close()
            except Exception:
                pass

    def stderr_tail(self, n: int = 20) -> str:
        """The last n lines of this session's stderr file, or '' when there is none."""
        if not self._errfile:
            return ""
        try:
            with open(self._errfile, encoding="utf-8", errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return ""

    @property
    def pgid(self) -> int | None:
        return self._pgid

    def on_reader_thread(self) -> bool:
        """True when called on the reader thread (so inside _on_line). Waiting for a reply from there would deadlock."""
        return self._reader_thread is not None and threading.current_thread() is self._reader_thread

    # ---- reader ---------------------------------------------------------------------------------------

    def _reader(self, proc: subprocess.Popen[str]) -> None:
        try:
            for line in self._iter_stdout(proc):
                try:
                    self._on_line(line.rstrip("\r\n"))
                except Exception as e:  # one bad line must not end the stream
                    self._log(f"line: {e!r}")
        except Exception as e:  # reading itself failed: the exit path below still runs
            self._log(f"reader: {e!r}")
        finally:  # always, exactly once: reap, release stdout, then report the exit
            rc = proc.wait()
            try:
                proc.stdout.close()
            except Exception:
                pass
            try:
                self._on_exited(rc)
            except Exception as e:
                self._log(f"exit callback: {e!r}")

    def _emit(self, event: object) -> None:
        try:
            self._sink(event)  # type: ignore[arg-type]
        except Exception as e:
            self._log(f"sink: {e!r}")

    # ---- kill ladder ----------------------------------------------------------------------------------

    def _wait(self, proc: subprocess.Popen[str], timeout: float) -> bool:
        try:
            proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def _killpg(self, sig: signal.Signals) -> None:
        if self._pgid is None:
            return
        try:
            os.killpg(self._pgid, sig)
        except ProcessLookupError:
            pass

    def _signal_ladder(self, proc: subprocess.Popen[str]) -> None:
        for sig, wait in ((signal.SIGINT, SIGNAL_STEP), (signal.SIGTERM, SIGNAL_STEP), (signal.SIGKILL, KILL_WAIT)):
            self._killpg(sig)
            if self._wait(proc, wait):
                return
        self._log("child survived SIGKILL")

    def _sweep_group(self) -> None:
        """Stragglers such as a child's own children may outlive the leader; the group must end up empty."""
        if self._pgid is None:
            return
        self._killpg(signal.SIGTERM)
        if self._wait_group_gone():
            return
        self._killpg(signal.SIGKILL)
        self._wait_group_gone()

    def _wait_group_gone(self) -> bool:
        deadline = time.monotonic() + GROUP_WAIT
        while self._pgid is not None and _group_alive(self._pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
        return self._pgid is None or not _group_alive(self._pgid)

    # ---- helpers --------------------------------------------------------------------------------------

    def _new_errfile(self) -> str:
        """Create the per-session stderr file and drop the oldest ones so at most KEEP_ERRFILES remain."""
        d = self._errfile_dir
        os.makedirs(d, exist_ok=True)
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass
        paths = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".err")]
        paths.sort(key=_mtime)
        for old in paths[: max(0, len(paths) - (KEEP_ERRFILES - 1))]:
            try:
                os.remove(old)
            except OSError:
                pass
        safe = _UNSAFE_NAME.sub("_", self._name)
        return os.path.join(d, f"{safe}-{int(self._clock.now())}.err")

    def _log(self, msg: str) -> None:
        self._logger.log(f"session {self._name}: {msg}")


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0
