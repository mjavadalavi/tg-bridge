"""Process trees: the descendants of a pid, and one signal sent to a whole tree.

A tab's Claude Code may run behind a launcher: the pid the view started can be an interactive shell that ignores
SIGTERM, and the CLI is its grandchild. descendants() finds every process below a pid by scanning /proc, deepest first,
so the leaves get the signal before the parent that could outlive them. signal_tree() sends one signal to that list
and then to the pid itself. The caller and its ancestors are never signalled (_ancestry()): a bridge daemon started from
inside a tab's Claude Code is a descendant of that CLI, and a tree kill must not take the daemon down with it.

Only Linux /proc is read. An entry that vanishes or cannot be read during the scan is skipped. The scan is not atomic:
a process forked while it runs can be missed, which is acceptable when a session is being ended.

Threads: any. Lock map: none; the module holds no state.
"""

import os

PROC = "/proc"


def _ppid(stat: str) -> int:
    """The ppid from the text of a /proc/<pid>/stat line: field 4, read after the last ')' because comm may hold spaces
    and parentheses. Raises ValueError or IndexError when the line is garbled."""
    fields = stat[stat.rindex(")") + 1:].split()
    return int(fields[1])


def _parent_of(pid: int) -> int | None:
    """The ppid of pid read from /proc, or None when the entry is gone or unreadable."""
    try:
        with open(os.path.join(PROC, str(pid), "stat"), encoding="utf-8", errors="replace") as handle:
            return _ppid(handle.read())
    except (OSError, ValueError, IndexError):
        return None


def _children_by_parent() -> dict[int, list[int]]:
    """Maps each ppid to its pids, from a scan of /proc."""
    children: dict[int, list[int]] = {}
    for name in os.listdir(PROC):
        if not name.isdigit():
            continue
        parent = _parent_of(int(name))
        if parent is not None:
            children.setdefault(parent, []).append(int(name))
    return children


def _ancestry() -> set[int]:
    """The caller's pid and each ancestor up to pid 1. The walk stops at an unreadable entry or at a loop, and returns
    what it has reached."""
    chain: set[int] = set()
    pid = os.getpid()
    while pid > 0 and pid not in chain:
        chain.add(pid)
        parent = _parent_of(pid)
        if parent is None:
            break
        pid = parent
    return chain


def descendants(pid: int) -> list[int]:
    """Every pid below pid, deepest first (the children of the last generation come before their parents). pid itself
    is not listed. A pid <= 0 has no descendants here: 0 would name the whole kernel process tree."""
    if pid <= 0:
        return []
    children = _children_by_parent()
    generations: list[list[int]] = []
    frontier = [pid]
    seen = {pid}
    while frontier:
        found = [child for parent in frontier for child in children.get(parent, []) if child not in seen]
        seen.update(found)
        if found:
            generations.append(found)
        frontier = found
    return [child for generation in reversed(generations) for child in generation]


def _send(pid: int, sig: int) -> bool:
    """Sends sig to pid. False only when the kernel refuses with PermissionError; a pid that is gone counts as sent."""
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return True


def signal_tree(pid: int, sig: int) -> bool:
    """Sends sig to every descendant of pid, deepest first, then to pid. The caller and its ancestors (_ancestry()) are
    never signalled: a descendant in that set is skipped, and when pid itself is in it nothing is sent and False is
    returned. A descendant that cannot be signalled is skipped. False when pid itself is refused with PermissionError,
    when pid <= 0 (never signalled: 0 would name the caller's whole process group), or when pid is an ancestor of the
    caller. True otherwise, including when pid is already gone."""
    if pid <= 0:
        return False
    protected = _ancestry()
    if pid in protected:
        return False
    for child in descendants(pid):
        if child not in protected:
            _send(child, sig)
    return _send(pid, sig)
