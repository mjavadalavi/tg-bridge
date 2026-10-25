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


def transcript_path(config_dir: str, cwd: str, sid: str) -> str | None:
    """Where session sid's transcript sits under a Claude config folder: the cwd's own folder first, else any folder
    holding that uuid. None when the id is malformed or the file is not there."""
    if not config_dir or not cwd or not TRANSCRIPT_ID.fullmatch(sid or ""):
        return None
    projects = os.path.join(config_dir, "projects")
    real = os.path.realpath(os.path.expanduser(cwd))
    for folder in dict.fromkeys([encoded_cwd(real), encoded_cwd(os.path.abspath(os.path.expanduser(cwd)))]):
        path = os.path.join(projects, folder, sid + ".jsonl")
        if os.path.isfile(path):
            return path
    try:
        folders = sorted(os.listdir(projects))
    except OSError:
        return None
    for folder in folders:
        path = os.path.join(projects, folder, sid + ".jsonl")
        if os.path.isfile(path):
            return path
    return None


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
