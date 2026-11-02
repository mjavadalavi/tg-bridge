"""Transcript carry-over between Claude launchers (port of encoded_cwd, transcript_path, strip_thinking_line,
copy_transcript and carry_history from the live file).

A Claude session's transcript lives at <config_dir>/projects/<encoded cwd>/<sid>.jsonl. When a topic switches to
another Claude launcher with a different config folder, the transcript is copied into the new folder so `--resume`
finds it. Thinking blocks are stripped on the way (they are signed by the old account and are not needed to resume).
Nothing here raises into the caller: carry_history logs the failure and returns False.
"""

import json
import os
import re

from tg_bridge.logging_ import Logger

TRANSCRIPT_ID = re.compile(r"[A-Za-z0-9-]{1,64}")  # a session uuid: safe to use in a file name
THINKING_TYPES = ("thinking", "redacted_thinking")
EMPTY_THINKING_TEXT = "(…)"  # placeholder for a message whose only blocks were thinking
TMP_SUFFIX = "tmp"


def encoded_cwd(cwd: str) -> str:
    """The folder name Claude Code keeps a cwd's transcripts under: every character that is not [A-Za-z0-9] becomes '-'."""
    return re.sub(r"[^A-Za-z0-9]", "-", cwd)


def _valid(config_dir: str, cwd: str, sid: str) -> bool:
    return bool(config_dir and cwd and TRANSCRIPT_ID.fullmatch(sid or ""))


def cwd_transcript_path(config_dir: str, cwd: str, sid: str) -> str | None:
    """Session sid's transcript in the cwd's own folder only (the one `claude --resume` looks in): the realpath and the
    abspath encoded folders are both checked, and the newer copy wins. None when the id is malformed or no such copy
    exists under that cwd. Other folders of the config dir are never looked at."""
    if not _valid(config_dir, cwd, sid):
        return None
    projects = os.path.join(config_dir, "projects")
    own = [os.path.join(projects, folder, sid + ".jsonl") for folder in _cwd_folders(cwd)]
    present = [path for path in own if os.path.isfile(path)]
    return max(present, key=_newest_key) if present else None


def _cwd_folders(cwd: str) -> list[str]:
    """The encoded folder names of a cwd: the realpath's, then the abspath's when it differs."""
    real = os.path.realpath(os.path.expanduser(cwd))
    return list(dict.fromkeys([encoded_cwd(real), encoded_cwd(os.path.abspath(os.path.expanduser(cwd)))]))


def _any_folder_path(config_dir: str, sid: str) -> str | None:
    """The first folder of the config dir (sorted by name) that holds sid's transcript, or None."""
    projects = os.path.join(config_dir, "projects")
    try:
        folders = sorted(os.listdir(projects))
    except OSError:
        return None
    for folder in folders:
        path = os.path.join(projects, folder, sid + ".jsonl")
        if os.path.isfile(path):
            return path
    return None


def transcript_path(config_dir: str, cwd: str, sid: str) -> str | None:
    """Where session sid's transcript sits under a Claude config folder: the cwd's own folder first (cwd_transcript_path),
    else any folder holding that uuid. None when the id is malformed or the file is not there."""
    own = cwd_transcript_path(config_dir, cwd, sid)
    if own is not None or not _valid(config_dir, cwd, sid):
        return own
    return _any_folder_path(config_dir, sid)


def _newest_key(path: str) -> tuple[float, str]:
    """Ranks a transcript by mtime, ties by path: the same order sessions_index lists by, so both pick the same copy."""
    try:
        return os.path.getmtime(path), path
    except OSError:
        return float("-inf"), path


def strip_thinking_line(line: str) -> str:
    """One transcript line with its thinking blocks removed. A line that is not JSON, or that has nothing to remove, comes
    back unchanged. A message left without content gets a short text block instead."""
    try:
        obj = json.loads(line)
    except ValueError:
        return line
    msg = obj.get("message") if isinstance(obj, dict) else None
    content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, list):
        return line
    kept = [b for b in content if not (isinstance(b, dict) and b.get("type") in THINKING_TYPES)]
    if len(kept) == len(content):
        return line
    msg["content"] = kept or [{"type": "text", "text": EMPTY_THINKING_TEXT}]
    return json.dumps(obj) + ("\n" if line.endswith("\n") else "")


def copy_transcript(src: str, dst: str) -> bool:
    """Copy a transcript to dst with thinking blocks stripped, streamed line by line through a temp file. An existing dst
    is replaced only when it is shorter than the new copy. Returns True when dst holds the copy. Raises OSError or
    ValueError on failure, and the temp file never outlives the call."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.{os.getpid()}.{TMP_SUFFIX}"
    try:
        with open(src, encoding="utf-8", errors="surrogateescape", newline="") as fi, \
                open(tmp, "w", encoding="utf-8", errors="surrogateescape", newline="") as fo:
            for line in fi:
                fo.write(strip_thinking_line(line))
        if os.path.exists(dst) and os.path.getsize(dst) >= os.path.getsize(tmp):
            return True  # the existing copy is at least as long: keep it
        os.replace(tmp, dst)
        return True
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def carry_history(src_config_dir: str, dst_config_dir: str, src_cwd: str, sid: str, dst_cwd: str | None,
                  logger: Logger) -> bool:
    """Copy session sid's transcript from one Claude config folder into another, under the cwd's folder (dst_cwd,
    default src_cwd). True when the copy is in place. False when the source is missing or the copy failed; nothing is
    raised, the failure is logged."""
    try:
        src = transcript_path(src_config_dir, src_cwd, sid)
        if not src or not dst_config_dir:
            return False
        dst_folder = encoded_cwd(os.path.realpath(os.path.expanduser(dst_cwd or src_cwd)))
        return copy_transcript(src, os.path.join(dst_config_dir, "projects", dst_folder, sid + ".jsonl"))
    except Exception as exc:  # never raised into the caller's thread
        logger.log(f"history copy for {sid} failed: {exc}")
        return False
