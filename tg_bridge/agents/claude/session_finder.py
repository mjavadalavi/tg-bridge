"""session_finder: the transcript a Claude Code session moved to when /clear or /resume ran in its terminal tab.

A message pasted into an attached session is written by the CLI into the transcript of the session it runs. After /clear or
/resume <other> the CLI writes into another file, so the echo of the message never shows in the claim's transcript.
find_relocated() looks for that echo among the newest transcripts of a config folder: a file modified after the paste, whose
tail holds a user line with the start of the message. It reads at most MAX_FILES files and the last TAIL_BYTES of each, and
writes nothing. The result names the file, its session id (the file stem) and the byte offset of the echo line, so the caller
follows the file from that line.

Pure apart from the reads: no clock, no threads. The caller passes the paste time (the mtime bound), the pasted text and the
file that is already followed (excluded: its echo was not read there).
"""

import json
import os
from dataclasses import dataclass

from tg_bridge.agents.claude.tui_turn import is_echo

MAX_FILES = 20  # the newest transcripts of the config folder, the only ones looked at
TAIL_BYTES = 64 * 1024  # the last bytes of each file that are read
PREFIX_CHARS = 60  # the start of the pasted text the echo must contain
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
    """The newest transcript under config_dir/projects/*/ modified after `since` whose tail holds the echo of `text`, or None.
    The echo is a user line (is_echo) whose text contains the first PREFIX_CHARS characters of the stripped text."""
    needle = text.strip()[:PREFIX_CHARS]
    if not config_dir or not needle:
        return None
    for _mtime, path in _candidates(config_dir, since, exclude)[:MAX_FILES]:
        offset = _echo_offset(path, needle)
        if offset is not None:
            stem = os.path.basename(path)[:-len(TRANSCRIPT_SUFFIX)]
            return Relocated(path=path, session_id=stem, offset=offset)
    return None


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


def _echo_offset(path: str, needle: str) -> int | None:
    """Byte offset of the first echo line in the file's tail, or None. The tail's first line may be cut and is skipped; a
    last line without its newline is still being written and is skipped too."""
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
        if _is_echo_of(raw, needle):
            return line_offset
    return None


def _is_echo_of(raw: bytes, needle: str) -> bool:
    try:
        line = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(line, dict) or not is_echo(line):
        return False
    return needle in _user_text(line)


def _user_text(line: dict) -> str:
    message = line.get("message") if isinstance(line.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b["text"] for b in content if isinstance(b, dict) and isinstance(b.get("text"), str))
