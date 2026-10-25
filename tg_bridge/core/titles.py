"""TitleService: the one-time auto-title of a topic (port of title_session and _submit_title, C09).

Once a topic's first answer is out, the title is asked from the cheap title launcher through AgentBackend.one_shot
stored once in sessions.title, and the topic is renamed to "<name> · <title>" with a
forced rename. A topic that already has a title is never titled again. When the call fails, times out or the backend
has no one_shot, the default title is the first message (white space collapsed, cut to 40 characters), or the session
name when that is empty.

submit() queues the job on a WorkerPool of one thread (the live _title_q), so title calls run one at a time and never
delay media or the send queue. title() does the work synchronously; the tests call it directly.
"""

from tg_bridge.agents.base import AgentBackend
from tg_bridge.core.workers import WorkerPool
from tg_bridge.logging_ import Logger
from tg_bridge.store.sessions import SessionRepo
from tg_bridge.telegram.messenger import Messenger

TITLE_TIMEOUT_SECONDS = 120.0  # the live TITLE_TIMEOUT: the title call may run this long before its group is killed
DEFAULT_TITLE_CHARS = 40  # the default title is the first message cut to this many characters
PROMPT_EXCERPT_CHARS = 300  # how much of the first message and of the first answer goes into the prompt
TITLE_SEPARATOR = " · "


def title_prompt(first_msg: str, first_answer: str) -> str:
    """The prompt that asks the title launcher for a 3-6 word title of the work of a session."""
    return ("Reply with ONLY a 3-6 word title for a conversation that starts with:\n"
            f"{first_msg[:PROMPT_EXCERPT_CHARS]}\n\nand whose first answer begins:\n"
            f"{str(first_answer)[:PROMPT_EXCERPT_CHARS]}\n"
            "The title names the WORK (what the session is for). No quotes, no punctuation at the end.")


def default_title(first_msg: str, name: str) -> str:
    """The title used when no model title is available: the first message in one line, else the session name."""
    return " ".join(first_msg.split())[:DEFAULT_TITLE_CHARS] or name


class TitleService:
    """Titles topics once. Every collaborator is injected; the only state is in the sessions table and the pool."""

    def __init__(
        self,
        *,
        backend: AgentBackend,
        sessions: SessionRepo,
        messenger: Messenger,
        pool: WorkerPool,
        logger: Logger,
        timeout: float = TITLE_TIMEOUT_SECONDS,
    ) -> None:
        self._backend = backend
        self._sessions = sessions
        self._messenger = messenger
        self._pool = pool
        self._logger = logger
        self._timeout = timeout

    def submit(self, chat_id: int | str, thread_id: int, name: str, first_msg: str, first_answer: str) -> None:
        """Queue title() on the title worker and return at once. Called after the answer has been sent."""
        self._pool.submit(self.title, chat_id, thread_id, name, first_msg, first_answer)

    def title(self, chat_id: int | str, thread_id: int, name: str, first_msg: str, first_answer: str) -> str | None:
        """Title the topic if it has no title yet. Returns the new topic name, or None when nothing was done."""
        row = self._sessions.get(thread_id)
        if row is None or row.title:
            return None
        title = self._ask(first_msg, first_answer) or default_title(first_msg, name)
        full = f"{name}{TITLE_SEPARATOR}{title}"
        self._sessions.set_title(thread_id, full)
        self._messenger.rename_topic(chat_id, thread_id, full, force=True)
        self._logger.log(f"titled thread {thread_id}: {full!r}")
        return full

    def _ask(self, first_msg: str, first_answer: str) -> str:
        """The model title, or '' when the call fails. A backend exception is logged, never raised."""
        try:
            return self._backend.one_shot(title_prompt(first_msg, first_answer), self._timeout) or ""
        except Exception as exc:
            self._logger.log(f"title: the title call failed: {exc!r}")
            return ""
