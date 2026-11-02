"""Terminal claims: the lock between `tg-bridge view` and the daemon for one topic's session (spec: the Lock section).

A claim is the file <view>/<thread>.term (JSON: pid, started). The view pane writes it while it runs the real CLI on its
own tty. A claim is live while its pid is alive. A claim whose pid is dead is stale: it is removed when it is read.

The topic's state lives in the meta file <view>/<thread>.json that ViewRecorder writes. The state "running" or
"lingering" means the bridge owns the session, so the pane must not claim it.

The turn mark <view>/<thread>.turn is the daemon's own answer to "is a process of this topic still running?". begin_turn
creates it and end_turn removes it, and only the daemon decides when: end_turn runs when the topic's process is really
gone and not kept (the dispatcher, or a hold that was released). The meta state is written by ViewRecorder at each event
(a TurnResult writes "idle" while the CLI still lives), so it cannot tell the pane that the process is gone. take_claim
refuses while the mark exists. A mark left by a killed daemon is removed by clear_turn_marks at start-up.

Check-and-set on both sides runs under a per-topic fcntl.flock on <view>/<thread>.lock. take_claim (the pane) and
begin_turn (the daemon, before any turn of the topic) read the other side's state and write their own under that lock.
Every public function takes the lock itself and never nests it, so callers do not have to. The flock is a file lock, so
two processes exclude each other. Nothing is called out while it is held: the callers log and send after it returns.

Threads: any. Lock map: none inside the process; the flock is held only for the file reads and writes below.
"""

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from tg_bridge.clock import Clock
from tg_bridge.core.view_recorder import META_SUFFIX, STATE_IDLE, STATE_LINGERING, STATE_RUNNING, TMP_SUFFIX
from tg_bridge.logging_ import Logger

CLAIM_SUFFIX = ".term"
LOCK_SUFFIX = ".lock"
TURN_SUFFIX = ".turn"
BUSY_STATES = (STATE_RUNNING, STATE_LINGERING)  # the bridge owns the session in these states
TERMINAL_NOTICE = "🖥 this session is open in the terminal (tg-bridge view); your message runs when you close it there."


@dataclass(frozen=True)
class Claim:
    """A terminal claim that names a live process."""

    pid: int
    started: float


def claim_path(view_dir: str, thread_id: int) -> str:
    return os.path.join(view_dir, f"{thread_id}{CLAIM_SUFFIX}")


def turn_path(view_dir: str, thread_id: int) -> str:
    return os.path.join(view_dir, f"{thread_id}{TURN_SUFFIX}")


def _meta_path(view_dir: str, thread_id: int) -> str:
    return os.path.join(view_dir, f"{thread_id}{META_SUFFIX}")


@contextmanager
def _locked(view_dir: str, thread_id: int) -> Iterator[None]:
    """Holds the topic's flock. Closing the descriptor drops it, also when the body raises."""
    fd = os.open(os.path.join(view_dir, f"{thread_id}{LOCK_SUFFIX}"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:  # os.kill(0, 0) would signal the whole process group
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # the process exists, another user owns it
        return True
    return True


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _read_claim(path: str) -> Claim | None:
    """The claim in the file, or None. A file that is not a claim is removed, so it cannot block the topic."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        _remove(path)
        return None
    pid = data.get("pid") if isinstance(data, dict) else None
    started = data.get("started") if isinstance(data, dict) else None
    if not isinstance(pid, int) or isinstance(pid, bool) or not isinstance(started, (int, float)):
        _remove(path)
        return None
    return Claim(pid=pid, started=float(started))


def _live_claim_locked(view_dir: str, thread_id: int) -> Claim | None:
    path = claim_path(view_dir, thread_id)
    claim = _read_claim(path)
    if claim is None:
        return None
    if _pid_alive(claim.pid):
        return claim
    _remove(path)  # stale: the pane that wrote it is gone
    return None


def _meta_state_locked(view_dir: str, thread_id: int) -> str:
    """The state in the topic's meta file, or '' when there is no readable meta."""
    try:
        with open(_meta_path(view_dir, thread_id), encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        return ""
    state = meta.get("state") if isinstance(meta, dict) else None
    return state if isinstance(state, str) else ""


def _turn_marked_locked(view_dir: str, thread_id: int) -> bool:
    return os.path.exists(turn_path(view_dir, thread_id))


def _set_state_locked(view_dir: str, thread_id: int, state: str, clock: Clock) -> None:
    """Writes the topic's meta with another state and keeps the other fields the recorder wrote."""
    path = _meta_path(view_dir, thread_id)
    try:
        with open(path, encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        meta = {}
    fields = dict(meta) if isinstance(meta, dict) else {}
    fields.update(state=state, updated=clock.now())
    tmp = path + TMP_SUFFIX
    with open(tmp, "wb") as handle:
        handle.write(json.dumps(fields, ensure_ascii=True).encode("utf-8"))
    os.replace(tmp, path)


def meta_state(view_dir: str, thread_id: int) -> str:
    """The topic's meta state, or '' when there is no readable meta. Read under the topic's flock, so the pane and the
    claim check see the same state."""
    with _locked(view_dir, thread_id):
        return _meta_state_locked(view_dir, thread_id)


def turn_marked(view_dir: str, thread_id: int) -> bool:
    """True while the daemon's turn mark of the topic exists: a process of the topic may still run."""
    with _locked(view_dir, thread_id):
        return _turn_marked_locked(view_dir, thread_id)


def live_claim(view_dir: str, thread_id: int) -> Claim | None:
    """The live claim of the topic, or None. A stale claim (dead pid) is removed here."""
    with _locked(view_dir, thread_id):
        return _live_claim_locked(view_dir, thread_id)


def _free_locked(view_dir: str, thread_id: int) -> bool:
    """True when the topic is free for a terminal session: no live claim, no turn mark, and a meta state that is neither
    running nor lingering. The mark decides over the meta: a TurnResult writes 'idle' while the CLI of the turn still
    runs, so an idle meta alone is no proof that the process is gone. Stale claims are removed on the way."""
    if _live_claim_locked(view_dir, thread_id) is not None:
        return False
    return not (_turn_marked_locked(view_dir, thread_id) or _meta_state_locked(view_dir, thread_id) in BUSY_STATES)


@contextmanager
def free_topic(view_dir: str, thread_id: int) -> Iterator[bool]:
    """Holds the topic's flock and yields whether the topic is free (the test take_claim makes). The body runs under the
    lock, so begin_turn of the daemon waits for it. The body must not call another function of this module for the same
    topic: the flock is not re-entrant."""
    with _locked(view_dir, thread_id):
        yield _free_locked(view_dir, thread_id)


def take_claim(view_dir: str, thread_id: int, pid: int, started: float) -> bool:
    """The pane's claim. False when another claim is live, the turn mark exists, or the meta says running or lingering:
    the pane then stays busy. True when the claim file was written for this pid."""
    with _locked(view_dir, thread_id):
        if not _free_locked(view_dir, thread_id):
            return False
        path = claim_path(view_dir, thread_id)
        tmp = path + TMP_SUFFIX
        with open(tmp, "wb") as handle:
            handle.write(json.dumps({"pid": pid, "started": started}).encode("utf-8"))
        os.replace(tmp, path)
        return True


def release_claim(view_dir: str, thread_id: int, pid: int) -> bool:
    """Removes the claim when it names this pid. False when there is none, or it names another process."""
    with _locked(view_dir, thread_id):
        path = claim_path(view_dir, thread_id)
        claim = _read_claim(path)
        if claim is None or claim.pid != pid:
            return False
        _remove(path)
        return True


def begin_turn(view_dir: str, thread_id: int, clock: Clock) -> bool:
    """The daemon's check before a turn of the topic. False while a live claim exists: nothing is marked and the
    caller waits. True when no claim is live; the turn mark is then written and the meta is 'running', both under the
    same lock, so the pane cannot claim between the check and the start. The mark stays until end_turn."""
    with _locked(view_dir, thread_id):
        if _live_claim_locked(view_dir, thread_id) is not None:
            return False
        with open(turn_path(view_dir, thread_id), "wb"):
            pass
        _set_state_locked(view_dir, thread_id, STATE_RUNNING, clock)
        return True


def end_turn(view_dir: str, thread_id: int, clock: Clock) -> None:
    """The end of the topic's process: removes the turn mark and turns a meta still 'running' into 'idle'. The caller
    calls it only when no process of the topic is left (the turn ended without a kept session, or its hold was
    released); a kept session keeps its mark until then. A 'lingering' meta is left as it is."""
    with _locked(view_dir, thread_id):
        _remove(turn_path(view_dir, thread_id))
        if _meta_state_locked(view_dir, thread_id) == STATE_RUNNING:
            _set_state_locked(view_dir, thread_id, STATE_IDLE, clock)


def clear_turn_marks(view_dir: str, logger: Logger) -> int:
    """Daemon start-up: every turn mark a killed daemon left behind belongs to a process that this daemon does not own,
    so it is removed. Runs before any turn, so it takes no topic lock. Returns how many marks were removed."""
    try:
        names = os.listdir(view_dir)
    except OSError as exc:
        logger.log(f"view: cannot list {view_dir}: {exc!r}")
        return 0
    removed = 0
    for name in names:
        if name.endswith(TURN_SUFFIX):
            try:
                _remove(os.path.join(view_dir, name))
            except OSError as exc:
                logger.log(f"view: cannot remove the turn mark {name}: {exc!r}")
                continue
            removed += 1
    if removed:
        logger.log(f"view: {removed} turn mark(s) left by a stopped daemon removed")
    return removed
