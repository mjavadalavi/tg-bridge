"""session_finder: the transcript a Claude Code session moved to when /clear or /resume ran in its terminal tab.

A message pasted into an attached session is written by the CLI into the transcript of the session it runs. After /clear
or /resume <other> the CLI writes into another file, so the echo of the message never shows in the claim's transcript.
find_relocated() looks for that echo among the newest transcripts of a config folder: a file modified after the paste,
whose tail holds the echo of the message (tui_turn.is_echo_of: a user line with the start of the message, the attachment
of a message delivered mid-turn, or the queue line of a message that waits). It reads at most MAX_FILES files and the
last TAIL_BYTES of each, and writes nothing. An echo line timestamped before the paste is an older line of that
transcript, not the echo, so it is skipped; a line without a timestamp is taken as it is. The result names the file, its
session id (the file stem) and the byte offset of the echo line, so the caller follows the file from that line.
find_in_session() applies the same rules to one session (the claim's uuid), without the scan.

Pure apart from the reads: no clock, no threads. The caller passes the paste time (the mtime bound), the pasted text and the
file that is already followed (excluded: its echo was not read there).
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime

from tg_bridge.agents.claude.tui_turn import echo_needle, is_echo_of

MAX_FILES = 20  # the newest transcripts of the config folder, the only ones looked at
TAIL_BYTES = 64 * 1024  # the last bytes of each file that are read
MTIME_SLACK_SECONDS = 1.0  # file times are coarse (a kernel tick) and may read a little before the clock: a write just after the paste still counts
TRANSCRIPT_SUFFIX = ".jsonl"
NEWLINE = b"\n"


@dataclass(frozen=True)
class Relocated:
    """The transcript that holds the echo: its path, its session id (the file stem) and the offset of the echo line."""

    path: str
    session_id: str
    offset: int


def find_relocated(config_dir: str, since: float, text: str, *, exclude: str = "") -> Relocated | None:
    """The newest transcript under config_dir/projects/*/ modified after `since` whose tail holds the echo of `text`,
    or None. The echo is a line that is_echo_of finds the start of the text (echo_needle) in, and it is not timestamped
    before `since` (an older line of the transcript is not the echo)."""
    needle = echo_needle(text)
    if not config_dir or not needle:
        return None
    for _mtime, path in _candidates(config_dir, since, exclude)[:MAX_FILES]:
        offset = _echo_offset(path, needle, since)
        if offset is not None:
            stem = os.path.basename(path)[:-len(TRANSCRIPT_SUFFIX)]
            return Relocated(path=path, session_id=stem, offset=offset)
    return None


def find_in_session(config_dir: str, session_id: str, since: float, text: str) -> Relocated | None:
    """The transcript of the known session `session_id` under config_dir/projects/*/ (its newest copy by mtime) when
    its tail holds the echo of `text`, or None. The echo rules are those of find_relocated. A session id that is empty
    or holds a path separator or '..' is rejected; a missing or unreadable file finds nothing."""
    needle = echo_needle(text)
    if not config_dir or not needle or not _is_plain_session_id(session_id):
        return None
    path = _session_file(config_dir, session_id)
    if path is None:
        return None
    offset = _echo_offset(path, needle, since)
    if offset is None:
        return None
    return Relocated(path=path, session_id=session_id, offset=offset)


def _is_plain_session_id(session_id: str) -> bool:
    """True when the session id can only name a file in one folder: not empty, no path separator, no '..'."""
    separators = ("/", "\\", os.sep)
    if not session_id or ".." in session_id:
        return False
    return not any(sep in session_id for sep in separators)


def _session_file(config_dir: str, session_id: str) -> str | None:
    """The newest config_dir/projects/<folder>/<session_id>.jsonl by mtime (ties by path), or None. A copy whose stat
    fails is skipped. A session resumed from another cwd has a copy per folder; the tab writes the newest one."""
    projects = os.path.join(config_dir, "projects")
    try:
        folders = os.listdir(projects)
    except OSError:
        return None
    copies: list[tuple[float, str]] = []
    for folder in folders:
        path = os.path.join(projects, folder, session_id + TRANSCRIPT_SUFFIX)
        if not os.path.isfile(path):
            continue
        try:
            copies.append((os.path.getmtime(path), path))
        except OSError:
            continue
    return max(copies)[1] if copies else None


def _candidates(config_dir: str, since: float, exclude: str) -> list[tuple[float, str]]:
    """Every transcript modified after `since` (less MTIME_SLACK_SECONDS), newest first (ties by path). Unreadable folders and
    files are skipped."""
    projects = os.path.join(config_dir, "projects")
    try:
        folders = os.listdir(projects)
    except OSError:
        return []
    found: list[tuple[float, str]] = []
    for folder in folders:
        folder_path = os.path.join(projects, folder)
        try:
            names = os.listdir(folder_path)
        except OSError:
            continue
        for name in names:
            path = os.path.join(folder_path, name)
            if not name.endswith(TRANSCRIPT_SUFFIX) or path == exclude:
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime > since - MTIME_SLACK_SECONDS:
                found.append((mtime, path))
    return sorted(found, reverse=True)


def _echo_offset(path: str, needle: str, since: float) -> int | None:
    """Byte offset of the first echo line in the file's tail, or None. The tail's first line may be cut and is skipped; a
    last line without its newline is still being written and is skipped too. An echo line timestamped before `since`
    (less MTIME_SLACK_SECONDS) is skipped: it was written before the paste."""
    try:
        with open(path, "rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            start = max(0, size - TAIL_BYTES)
            handle.seek(start)
            data = handle.read()
    except OSError:
        return None
    position = start
    if start > 0:
        cut = data.find(NEWLINE) + 1
        if cut == 0:
            return None
        position += cut
        data = data[cut:]
    for raw in data.split(NEWLINE)[:-1]:
        line_offset = position
        position += len(raw) + len(NEWLINE)
        if _is_echo_of(raw, needle, since):
            return line_offset
    return None


def _is_echo_of(raw: bytes, needle: str, since: float) -> bool:
    """True when the raw line is the echo of the text (is_echo_of). A line timestamped before `since` is never the
    echo."""
    try:
        line = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(line, dict):
        return False
    written = _epoch(line.get("timestamp"))
    if written is not None and written < since - MTIME_SLACK_SECONDS:
        return False
    return is_echo_of(line, needle)


def _epoch(value: object) -> float | None:
    """The epoch seconds of an ISO-8601 timestamp such as "2026-11-02T18:54:48.129Z", or None when it is not one."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, OverflowError, OSError):
        return None
