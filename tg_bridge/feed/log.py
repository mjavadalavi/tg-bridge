"""The tool-log model of a turn feed: one log message's lines and its send bookkeeping (pure, no I/O).

A log is one Telegram message that shows the tool calls of a turn between two texts. Its lines are cut to
LOG_LINE_CHARS each, and a log that would pass LOG_CHARS freezes so the next tool call starts another message.
"""

from dataclasses import dataclass

from tg_bridge.feed.format import short

LOG_CHARS = 3800  # visible text of one tool-log message; a fuller log freezes (Telegram caps messages at 4096)
LOG_LINE_CHARS = 120  # one tool line is cut to this many characters
LOG_LABEL_CHARS = 300  # the backend's label is cut to this before it is stored
LOG_HEADER_RESERVE = 100  # characters kept free in a log for its header


@dataclass(slots=True)
class ToolLine:
    label: str
    parent: bool  # a subagent's tool call
    failed: bool = False  # its tool result was an error


class LogSegment:
    """One tool-log message of the turn: its lines and what its next edit shows."""

    def __init__(self) -> None:
        self.mid: int | None = None  # the Telegram message id once sent
        self.lines: list[ToolLine] = []
        self.tools = 0  # tool calls in this message
        self.closed = False  # no more lines: a 💬 text came after it, it froze when full, or the turn ended
        self.frozen = False  # closed because it was full: the header says 'continued ↓'
        self.final: str | None = None  # the header once the turn has ended
        self.dirty = True  # the text shown is out of date
        self.shown: str | None = None  # the header last sent
        self.last_edit = 0.0  # clock time of the last send or edit
        self.next_try = 0.0  # after a failed send or edit, none is retried before this clock time
        self.failures = 0  # failed sends or edits of this log

    def header_base(self) -> str | None:
        """The header a closed log shows (its final summary, or its segment count); None while it is open."""
        if self.final is not None:
            return self.final
        if self.closed:
            return f"🔧 {self.tools} tools" + (" · continued ↓" if self.frozen else "")
        return None


def line_text(line: ToolLine) -> str:
    prefix = "  ↳ " if line.parent else ""
    return short(prefix + ("❌" if line.failed else "🔧") + " " + line.label, LOG_LINE_CHARS)


def lines_text(seg: LogSegment) -> str:
    return "\n".join(line_text(x) for x in seg.lines)


def log_len(seg: LogSegment) -> int:
    return sum(len(line_text(x)) + 1 for x in seg.lines) + LOG_HEADER_RESERVE
