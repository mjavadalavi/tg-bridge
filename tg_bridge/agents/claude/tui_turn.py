"""TuiTurn: the per-turn bookkeeping of the tui transport (agents/claude/tui_session.py).

While a turn runs, every assistant line of its transcript is noted here: the usage per message id, the text blocks per
message id (the TUI writes each content block as its own line under one id, so the blocks accumulate), and the last
assistant line, which tells an API error from an answer. At the end of the turn usage_event() and result() build the
UsageEvent and the TurnResult from what was noted. The interrupting flag records that an Esc was sent during the turn,
so the interrupt line of the transcript can end it. turn_start_origin() tells which transcript lines start a turn
(a human prompt, a task notification, a scheduled wakeup). Pure bookkeeping: no threads, no I/O.

Threads: the ledger holds no lock of its own. Its owner, ClaudeTuiSession, calls every method with its _turn_lock held, and
the flag interrupting is read and written only under that lock as well.
"""

import json
import re
from collections.abc import Mapping

from tg_bridge.agents.claude.translate import NO_CONVERSATION, RESULT_DETAIL_CHARS
from tg_bridge.agents.events import TurnResult, TurnStatus, UsageEvent

TURN_DURATION = "turn_duration"
INTERRUPT_MARK = "[Request interrupted by user"
INTERRUPTED_SUBTYPE = "interrupted"
SELF_ORIGINS = frozenset({"task_notification", "scheduled"})  # turns the CLI starts by itself
HUMAN_ORIGIN = "human"
TASK_NOTIFICATION_KIND = "task-notification"  # origin.kind of a task notification line
TASK_NOTIFICATION_ORIGIN = "task_notification"
USER_ECHO_SECONDS = 30.0       # after the message is typed, its transcript line must be read this soon, or a dialog has the keys
ECHO_MISSED_SUBTYPE = "message_not_echoed"
ECHO_MISSED_ERROR = ("the message did not reach Claude Code (a dialog may be open in the terminal session) — send it "
                     "again or /stop")
ATTACH_ECHO_MISSED_ERROR = ("the message was pasted into the terminal session, but its line never showed in transcript "
                            "{session} within {seconds:g} s — check the tab (the bridge did not send it again)")
QUEUE_OPERATION = "queue-operation"  # a transcript line the CLI writes when a pasted message waits behind a busy turn
ENQUEUE = "enqueue"
ATTACHMENT = "attachment"  # a transcript line the CLI writes when it delivers a message pasted mid-turn
QUEUED_COMMAND = "queued_command"  # the attachment kind of that line; its prompt is the pasted text
PREFIX_CHARS = 60  # the start of a pasted text that its echo must contain
COMMAND_NEEDLE = re.compile(r"(/\S*)(?:\s+(.*))?", re.DOTALL)  # a pasted slash command: its word, then the rest
COMMAND_ARGS = re.compile(r"<command-args>(.*?)</command-args>", re.DOTALL)  # the arguments the CLI stores of a command
STARTUP_DIALOG_ERROR = ("Claude Code is showing a dialog the bridge does not know — open the topic in tg-bridge view and "
                        "answer it, then send again")


def is_interrupt(line: dict) -> bool:
    return line.get("type") == "user" and INTERRUPT_MARK in json.dumps(line, ensure_ascii=False)


def turn_start_origin(line: dict) -> str | None:
    """The origin of a user prompt line that starts a turn, or None. A tool_result line, a local-command line (no
    turnOrigin) and a non-user line are not starts. 'human' is returned too: the caller treats a human line seen while no
    turn is open as a start (the bridge's own message processed late)."""
    if line.get("type") != "user":
        return None
    message = line.get("message") if isinstance(line.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, list) and content and all(isinstance(b, dict) and b.get("type") == "tool_result"
                                                    for b in content):
        return None
    origin = line.get("turnOrigin")
    if origin in SELF_ORIGINS or origin == HUMAN_ORIGIN:
        return origin
    kind = line.get("origin")
    if isinstance(kind, dict) and kind.get("kind") == TASK_NOTIFICATION_KIND:
        return TASK_NOTIFICATION_ORIGIN
    return None


def is_echo(line: dict) -> bool:
    """True when the line shows that the message typed for the turn reached Claude Code: a user prompt line with text that
    is not a tool result, not a meta line, and not a self turn (a task notification or a scheduled wakeup). The bridge's
    typed prompt is marked turnOrigin 'sdk' in a real transcript and an interactive prompt 'human'; a prompt with no
    turnOrigin at all counts too. Metadata lines (last-prompt, cost-state, custom-title, ...) do not count: the TUI writes
    them on resume even when the keys were swallowed by a dialog."""
    if line.get("type") != "user" or line.get("isMeta") is True:
        return False
    if line.get("turnOrigin") in SELF_ORIGINS:
        return False
    kind = line.get("origin")
    if isinstance(kind, dict) and kind.get("kind") == TASK_NOTIFICATION_KIND:
        return False
    message = line.get("message") if isinstance(line.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    return isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "text" for b in content)


def is_queued(line: dict) -> bool:
    """True when the CLI took the message into its queue (a message pasted while a turn is busy). The line proves the CLI
    holds the message; it is not an echo, because the message has not started its turn yet."""
    return line.get("type") == QUEUE_OPERATION and line.get("operation") == ENQUEUE


def echo_needle(text: str) -> str:
    """The start of a pasted text that its echo must hold: the first PREFIX_CHARS characters of the stripped text, or ''
    for a blank text."""
    return text.strip()[:PREFIX_CHARS]


def _queued_command_prompt(line: dict) -> str | None:
    """The prompt of the attachment the CLI writes when a message pasted mid-turn is delivered, or None for any other
    line."""
    attachment = line.get("attachment")
    if line.get("type") != ATTACHMENT or not isinstance(attachment, dict) or attachment.get("type") != QUEUED_COMMAND:
        return None
    prompt = attachment.get("prompt")
    return prompt if isinstance(prompt, str) else ""


def _user_text(line: dict) -> str:
    """All the text blocks of a user line, joined by new lines (a plain string content is the text itself)."""
    message = line.get("message") if isinstance(line.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b["text"] for b in content if isinstance(b, dict) and isinstance(b.get("text"), str))


def _command_echo_of(text: str, needle: str) -> bool:
    """True when the text is the CLI's record of the slash command that the needle starts with. A command with arguments
    is stored as <command-name>/word</command-name> and <command-args>...</command-args>, not as the typed text: the
    needle's rest (after its first whitespace) must start the stripped arguments. A needle with no rest needs only the
    command. A needle that does not start with a slash is not a command."""
    found = COMMAND_NEEDLE.match(needle)
    if found is None:
        return False
    command, rest = found.group(1), found.group(2) or ""
    if f"<command-name>{command}</command-name>" not in text:
        return False
    if not rest:
        return True
    args = COMMAND_ARGS.search(text)
    return args is not None and args.group(1).strip().startswith(rest)


def is_echo_of(line: dict, needle: str) -> bool:
    """True when the line shows that the message whose text starts with needle reached the CLI: a user line (is_echo)
    that holds it (or, for a slash command, its stored command tags: see _command_echo_of), the attachment the CLI
    writes when it delivers a message pasted mid-turn (its prompt holds it), or the queue line of a message that waits
    behind a busy turn (its content holds it). A blank needle matches nothing. The other user lines of the transcript
    (the owner's own prompts, an interrupt line, a task notification) do not match."""
    if not needle:
        return False
    if is_queued(line):
        content = line.get("content")
        return isinstance(content, str) and needle in content
    prompt = _queued_command_prompt(line)
    if prompt is not None:
        return needle in prompt
    if not is_echo(line):
        return False
    text = _user_text(line)
    return needle in text or _command_echo_of(text, needle)


def _texts(message: dict) -> list[str]:
    """The non-blank text blocks of a message, in order."""
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
            and isinstance(b.get("text"), str) and b["text"].strip()]


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0  # JSON numbers only; no bools


def _usage_event(usages: Mapping[str, dict]) -> UsageEvent:
    """The turn's tokens. usages holds one entry per message id, so a message written as several lines counts once."""
    fresh = output = cached = 0
    for usage in usages.values():
        fresh += _int(usage.get("input_tokens"))
        output += _int(usage.get("output_tokens"))
        cached += _int(usage.get("cache_read_input_tokens")) + _int(usage.get("cache_creation_input_tokens"))
    return UsageEvent(input_tokens=fresh, output_tokens=output, cached_tokens=cached, cost_usd=None, cost_known=False)


class TuiTurn:
    """The bookkeeping of one turn. reset() starts a new turn; note() is called for each transcript line of it.

    strict is for an attached turn (a message pasted into a tab the owner also types in): only the turn's own user line
    counts as its echo, and the assistant lines read before it belong to an earlier turn, so they are ignored. A
    non-strict turn (the bridge's own TUI process) also counts an assistant line as the echo, because the CLI works.
    expect(text) names the text the turn pastes: a strict turn with that text counts only the lines that hold it
    (is_echo_of), so another prompt of the owner in the same tab is neither its echo nor its queue line.

    No lock here: ClaudeTuiSession's _turn_lock guards this object, and every call (reset, note, result, usage_event,
    and the interrupting flag) happens with that lock held, exactly as the session's own turn state did before."""

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict       # kept across reset(): it describes the kind of turn, not one turn's state
        self._usage: dict[str, dict] = {}   # the turn's usage per message id
        self._answers: dict[str, list[str]] = {}  # the turn's text blocks per message id, the last written last
        self._last_assistant: dict | None = None
        self.interrupting = False  # an Esc was sent during this turn and its interrupt line is awaited
        self.echoed = False        # the turn's own message is in the transcript (its human line was read)
        self.queued = False        # the CLI queued the turn's message (its enqueue line was read); not an echo
        self._needle = ""          # the start of the pasted text, set by expect() for this turn only

    @property
    def waiting_for_echo(self) -> bool:
        """True while a strict turn has not seen its own user line yet: the lines before it are an earlier turn's."""
        return self.strict and not self.echoed

    def reset(self) -> None:
        """Clears the per-turn state. Caller holds the turn lock."""
        self.interrupting, self._last_assistant, self.echoed, self.queued = False, None, False, False
        self._usage, self._answers = {}, {}
        self._needle = ""

    def expect(self, text: str) -> None:
        """Names the text pasted for this turn. Call after reset(), when the turn begins: reset() clears it. Caller
        holds the turn lock."""
        self._needle = echo_needle(text)

    def note(self, line: dict) -> None:
        """Records an assistant line for the turn's usage, text and error state. Caller holds the lock."""
        if self.strict and self._needle:  # an attached turn with its pasted text: only the lines that hold it count
            if is_echo_of(line, self._needle):
                if is_queued(line):
                    self.queued = True  # the CLI holds the message in its queue: not yet its turn
                else:
                    self.echoed = True  # the typed message (or its delivery mid-turn) reached the transcript
        else:
            if is_queued(line):
                self.queued = True
            if is_echo(line):  # the typed message reached the transcript
                self.echoed = True
            elif line.get("type") == "assistant" and not self.strict:  # the CLI works (the bridge's own TUI only)
                self.echoed = True
        if line.get("type") != "assistant":
            return
        if self.strict and not self.echoed:  # an attached turn: this line is an earlier turn's, not this one's
            return
        message = line.get("message") if isinstance(line.get("message"), dict) else {}
        key = message.get("id") if isinstance(message.get("id"), str) else str(line.get("uuid") or id(line))
        usage = message.get("usage")
        if isinstance(usage, dict):
            self._usage[key] = usage
        texts = _texts(message)
        if texts:  # the TUI writes each content block as its own line under one id: the blocks accumulate
            self._answers[key] = self._answers.pop(key, []) + texts
        self._last_assistant = line

    def result(self, duration: dict | None) -> TurnResult:
        """The TurnResult of a finished turn. Caller holds the turn lock."""
        duration_ms = _int(duration.get("durationMs")) if duration else None
        last = self._last_assistant
        if last is not None and last.get("isApiErrorMessage") is True:
            message = last.get("message") if isinstance(last.get("message"), dict) else {}
            text = "\n".join(_texts(message))
            return TurnResult(status=TurnStatus.ERROR, answer=None, duration_ms=duration_ms, subtype=TURN_DURATION,
                              error=text[:RESULT_DETAIL_CHARS] or "no detail", resume_failed=NO_CONVERSATION in text)
        answer = "\n".join(list(self._answers.values())[-1]) if self._answers else None
        return TurnResult(status=TurnStatus.OK, answer=answer, duration_ms=duration_ms, subtype=TURN_DURATION)

    def usage_event(self) -> UsageEvent:
        return _usage_event(self._usage)
