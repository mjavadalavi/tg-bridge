"""On-disk session listing for Claude (the list a session picker shows).

A Claude session is a transcript at <config_dir>/projects/<encoded cwd>/<uuid>.jsonl. list_sessions finds those files for
one folder, sorts them newest first and reads a short title from the start of each one: the ai-title line, else the
summary line, else the first real user prompt. A file with no ai-title and no real prompt (an empty session) is skipped.
A transcript that records a different cwd is skipped too: two cwds can encode to the same folder name (/work/foo-bar and
/work/foo/bar). The bridge's own title calls run with --no-session-persistence, so they write no transcript; a title-call
transcript left by an older bridge may still be listed. Only the first TITLE_SCAN_LINES lines or TITLE_SCAN_BYTES bytes of
a file are read. At most limit * MAX_FILES_PER_SESSION files are opened per call: the newest candidates only, picked with
a bounded heap rather than a full sort. Ids passed in exclude are left out before the limit is applied. Nothing here
raises: a folder or file that cannot be read is skipped.
"""

import heapq
import json
import os
from collections.abc import Iterator

from tg_bridge.agents.base import DiskSession
from tg_bridge.agents.claude.history import TRANSCRIPT_ID, encoded_cwd

PROJECTS_DIR = "projects"
TRANSCRIPT_SUFFIX = ".jsonl"
TITLE_SCAN_LINES = 200         # lines read from the start of a transcript when looking for a title
TITLE_SCAN_BYTES = 262144      # bytes read from the start of a transcript (256 KiB)
TITLE_CHARS = 60               # the title is cut to this many characters
MAX_FILES_PER_SESSION = 5      # files opened per call are capped at limit * this, so unusable files cannot pile up
AI_TITLE_TYPE = "ai-title"
SUMMARY_TYPE = "summary"
USER_TYPE = "user"
TEXT_BLOCK = "text"
CWD_FIELD = "cwd"              # transcript lines carry the cwd the session ran in
COMMAND_PREFIX = "<"           # command wrappers such as <command-name> are not a prompt


def list_sessions(config_dir: str, cwd: str, limit: int,
                  exclude: frozenset[str] = frozenset()) -> tuple[DiskSession, ...]:
    """The usable sessions of one cwd under a Claude config folder, newest first, at most limit of them.

    The cwd is looked up under its realpath folder and its abspath folder (a symlinked cwd has both). A session id that
    appears in both is listed once, from the copy with the newer mtime. Ids in exclude are skipped before the limit is
    applied and do not count toward the file-open cap. An empty config_dir or cwd, or a limit below 1, gives ().
    """
    if not config_dir or not cwd or limit < 1:
        return ()
    real = os.path.realpath(os.path.expanduser(cwd))
    absolute = os.path.abspath(os.path.expanduser(cwd))
    bound = limit * MAX_FILES_PER_SESSION
    out: list[DiskSession] = []
    seen: set[str] = set()
    for modified, path, uuid in heapq.nlargest(bound, _candidates(config_dir, (real, absolute), exclude)):
        if uuid in seen:
            continue  # the newer copy came first, so the other copy of this id is not opened
        seen.add(uuid)
        try:
            title = _title(path, real)
        except OSError:
            continue
        if title is None:
            continue
        out.append(DiskSession(uuid=uuid, title=title, modified=modified))
        if len(out) >= limit:
            break
    return tuple(out)


def _candidates(config_dir: str, forms: tuple[str, ...], exclude: frozenset[str]) -> Iterator[tuple[float, str, str]]:
    """(mtime, path, uuid) of every top-level transcript under the folder of each cwd form, skipping ids in exclude."""
    folders = dict.fromkeys(os.path.join(config_dir, PROJECTS_DIR, encoded_cwd(form)) for form in forms)
    for folder in folders:
        yield from _folder_files(folder, exclude)


def _folder_files(folder: str, exclude: frozenset[str]) -> Iterator[tuple[float, str, str]]:
    """(mtime, path, uuid) of the regular <uuid>.jsonl files directly in one folder, one entry at a time. Sub-folders
    (subagents live in <uuid>/) are not listed. A folder that cannot be read yields nothing."""
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                item = _transcript(entry, exclude)
                if item is not None:
                    yield item
    except OSError:
        return


def _transcript(entry: os.DirEntry, exclude: frozenset[str]) -> tuple[float, str, str] | None:
    """(mtime, path, uuid) of one directory entry when it is a transcript name not in exclude, else None."""
    if not entry.name.endswith(TRANSCRIPT_SUFFIX):
        return None
    uuid = entry.name[:-len(TRANSCRIPT_SUFFIX)]
    if not TRANSCRIPT_ID.fullmatch(uuid) or uuid in exclude:
        return None
    try:
        if not entry.is_file():
            return None
        modified = entry.stat().st_mtime
    except OSError:
        return None
    return modified, entry.path, uuid


def _title(path: str, real: str) -> str | None:
    """The title of one transcript, or None when it has neither an ai-title nor a real prompt, or when it records a cwd
    other than real (the folder name is shared with another cwd)."""
    ai_title, summary, prompt, recorded = _scan(path)
    if recorded and not _same_folder(recorded, real):
        return None
    if not ai_title and not prompt:
        return None
    return (ai_title or summary or prompt)[:TITLE_CHARS].rstrip()


def _same_folder(recorded: str, real: str) -> bool:
    """True when a recorded cwd names the folder real, both through realpath so that a symlink alias matches."""
    try:
        return os.path.realpath(recorded) == real
    except ValueError:  # for example an embedded NUL byte: such a cwd is not one we can match
        return False


def _scan(path: str) -> tuple[str, str, str, str]:
    """(ai-title, summary, first real prompt, recorded cwd) found in the scan window of one transcript; '' for each not
    found."""
    with open(path, "rb") as handle:
        head = handle.read(TITLE_SCAN_BYTES)
    ai_title = summary = prompt = recorded = ""
    for line in head.decode("utf-8", errors="replace").split("\n")[:TITLE_SCAN_LINES]:
        obj = _parse(line)
        if obj is None:
            continue
        kind = obj.get("type")
        if kind == AI_TITLE_TYPE and not ai_title:
            ai_title = _collapse(obj.get("aiTitle"))
        elif kind == SUMMARY_TYPE and not summary:
            summary = _collapse(obj.get("summary"))
        elif kind == USER_TYPE and not prompt:
            prompt = _prompt(obj)
        if not recorded and isinstance(obj.get(CWD_FIELD), str):
            recorded = obj[CWD_FIELD]
        if ai_title and recorded:
            break  # the ai-title wins and the cwd is known, so nothing later can change either
    return ai_title, summary, prompt, recorded


def _parse(line: str) -> dict | None:
    """The line as a JSON object, or None for a line that is not one (a truncated last line, for example)."""
    try:
        obj = json.loads(line)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _prompt(obj: dict) -> str:
    """The text of a real user prompt line; '' for meta lines, command wrappers and tool-result-only content."""
    if obj.get("isMeta"):
        return ""
    msg = obj.get("message")
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, list):
        content = next((b.get("text") for b in content if isinstance(b, dict) and b.get("type") == TEXT_BLOCK), None)
    text = _collapse(content)
    return "" if text.startswith(COMMAND_PREFIX) else text


def _collapse(value: object) -> str:
    """The string with every run of whitespace turned into one space; '' for anything that is not a string."""
    return " ".join(value.split()) if isinstance(value, str) else ""
