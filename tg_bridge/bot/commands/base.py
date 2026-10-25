"""Command, CommandRegistry and CommandContext: the bridge's slash-command framework (C19 handle_command).

A Command is a value: its name, its help line, optional aliases (each with its own help line) and run(ctx, args).
CommandRegistry owns the name table: it rejects a name or alias registered twice, parses '/cmd@BotName' and is
case-insensitive as the live file is, and gives the setMyCommands list. Text that is not a registered command returns
False, so the caller sends it to the agent as an ordinary message (unknown /xyz and Claude's own commands included).

CommandContext is built per message. It holds the services a command may use (CommandDeps) and sends its replies
through the ordered out-worker (a Submitter), never from the calling thread, so a command never waits on the Telegram
send budget. Registration happens once at start-up, before any message is handled, so the registry needs no lock.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from tg_bridge.agents.registry import BackendRegistry
from tg_bridge.config import Config
from tg_bridge.core.dispatcher import TopicQueues
from tg_bridge.core.live import LiveRegistry
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.rules import RuleRepo
from tg_bridge.store.sessions import SessionRepo
from tg_bridge.store.turns import TurnRepo
from tg_bridge.telegram.keyboards import CallbackData
from tg_bridge.telegram.messenger import Messenger

NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
ButtonRows = list[list[tuple[str, str | CallbackData]]]
RunFn = Callable[["CommandContext", str], None]


class Submitter(Protocol):
    """The ordered out-worker (WorkerPool.submit): runs fn(*args) later, in submission order."""

    def submit(self, fn: Callable[..., object], *args: object) -> None: ...


@dataclass(frozen=True)
class CommandDeps:
    """The services the commands use. Built once by the App and shared by every command."""

    config: Config
    sessions: SessionRepo
    turns: TurnRepo
    rules: RuleRepo
    kv: KvRepo
    live: LiveRegistry
    queues: TopicQueues
    backends: BackendRegistry
    messenger: Messenger
    out: Submitter


@dataclass(frozen=True)
class Alias:
    """Another word that runs a command, with the help line /help and the Telegram menu show for it."""

    name: str
    help: str


@dataclass(frozen=True)
class Command:
    """One slash command. run(ctx, args) gets the text after the command word, stripped ('' when there is none)."""

    name: str
    help: str
    run: RunFn
    aliases: tuple[Alias, ...] = ()


@dataclass(frozen=True)
class CommandContext:
    """What one command run needs: its services, the registry (for /help), the topic it runs in and the replies."""

    deps: CommandDeps
    registry: "CommandRegistry"
    chat_id: int | str
    thread_id: int
    topic: str

    def say(self, text: str) -> None:
        """A plain reply to the topic, sent through the ordered out-worker."""
        self.deps.out.submit(self.deps.messenger.send_text, self.chat_id, self.thread_id, text)

    def say_buttons(self, text: str, rows: ButtonRows) -> None:
        """A reply with an inline keyboard, sent through the ordered out-worker."""
        self.deps.out.submit(self.deps.messenger.send_buttons, self.chat_id, self.thread_id, text, rows)


class CommandRegistry:
    """The name table of the bridge commands. Words are unique across commands and aliases."""

    def __init__(self) -> None:
        self._by_word: dict[str, Command] = {}
        self._menu: list[tuple[str, str]] = []  # (word, help) in registration order: /help and setMyCommands
        self._order: dict[str, int] = {}        # the menu order set by set_menu_order; other words follow it

    def register(self, command: Command) -> None:
        """Adds a command and its aliases. ValueError for a malformed word or a word that is already registered."""
        words = [command.name, *(alias.name for alias in command.aliases)]
        for word in words:
            if not NAME_PATTERN.fullmatch(word):
                raise ValueError(f"command word {word!r} must be lowercase letters, digits or underscores")
            if word in self._by_word:
                raise ValueError(f"command /{word} is already registered")
        self._by_word[command.name] = command
        self._menu.append((command.name, command.help))
        for alias in command.aliases:
            self._by_word[alias.name] = command
            self._menu.append((alias.name, alias.help))

    def find(self, word: str) -> Command | None:
        """The command a word runs (an alias finds its command), or None."""
        return self._by_word.get(word)

    def set_menu_order(self, words: Sequence[str]) -> None:
        """Sets the order of /help and the setMyCommands menu: the given words first, in that order. Words not listed
        keep their registration order after them."""
        self._order = {word: index for index, word in enumerate(words)}

    def bridge_commands(self) -> tuple[tuple[str, str], ...]:
        """Every registered word with its help line: the menu order when one is set, else registration order (the
        setMyCommands list). The sort is stable, so unlisted words keep their registration order."""
        fallback = len(self._order)
        return tuple(sorted(self._menu, key=lambda item: self._order.get(item[0], fallback)))

    def dispatch(self, deps: CommandDeps, chat_id: int | str, thread_id: int, topic: str, text: str | None) -> bool:
        """Runs the command the text starts with. True when it was a registered command (it is then answered here and
        never queued); False for anything else, so it reaches the agent as text."""
        parts = (text or "").strip().split(None, 1)
        if not parts or not parts[0].startswith("/"):
            return False
        word = parts[0][1:].split("@", 1)[0].lower()
        command = self._by_word.get(word)
        if command is None:
            return False
        args = parts[1].strip() if len(parts) > 1 else ""
        ctx = CommandContext(deps, self, chat_id, thread_id, topic)
        command.run(ctx, args)
        return True
