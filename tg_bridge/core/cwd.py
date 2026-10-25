"""CwdService: the folder (cwd) of a topic, the /cwd buttons and name searches, and the new-topic folder card (C19 cwd).

Ported from cwd_choices, cwd_path_id, apply_cwd, the search and suggestion jobs and their cards, the search taps,
topic_keep_tap, the pending folder (cwd_pending:<thread> in kv) and cwd_picker of tg-bridge.py.

- Search and suggestion jobs run on `search_pool` (the live worker threads, never the poll thread). Their cards are
  posted on `out_pool`, the ordered sender, so replies keep their order.
- Taps do not answer Telegram themselves: search_tap() and keep_tap() return a TapOutcome, and the callback handler
  sends the answerCallbackQuery toast and, when card_text is set, the edit of the same card.
- The folder list of a search card is kept under (thread, message id) for its taps, at most CWD_FOUND_KEEP cards.
- Lock map: `_found_lock` guards `_found` only. Repositories take the database lock themselves, and no lock is held
  while calling Messenger, LiveRegistry or a worker pool.
"""

import os
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from tg_bridge.clock import Clock
from tg_bridge.config import Config, tilde
from tg_bridge.core.cwd_search import CWD_SEARCH_SECONDS, find_dirs, normalize_term, resolve_search_roots
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.workers import WorkerPool
from tg_bridge.logging_ import Logger
from tg_bridge.store.inbox import CwdRepo
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.sessions import SessionRepo
from tg_bridge.telegram.keyboards import PICKER_PREFIX, CallbackData, picker_rows
from tg_bridge.telegram.messenger import Messenger

Finder = Callable[[str, Sequence[str], float], list[str]]

CWD_CHOICES_MAX = 8  # buttons of the /cwd picker
CWD_FOUND_KEEP = 200  # search cards whose folder lists are kept for their taps
TOPIC_FOLDER_MIN = 3  # shortest topic name (after stripping emojis and punctuation) that gets a folder card
TOPIC_FOLDER_MAX = 4  # folder buttons on a new topic's card
PENDING_KEY = "cwd_pending:{thread}"  # kv key of a folder tapped before the topic's session row existed
EXPIRED_TEXT = "list expired, run /cwd again"
GONE_TEXT = "not in the list anymore"
BUSY_TEXT = "a turn is running — send /stop first"
KEEP_TEXT = "🏠 this topic keeps its default folder"
SEARCH_CARD_TEXT = '🔎 folders matching "{term}" — tap one to set it as the cwd:'
NO_MATCH_TEXT = 'no folder matching "{term}" — send a full path'
TOPIC_CARD_TEXT = "📁 Start this topic in one of these folders?"
PENDING_TEXT = "📁 {shown} — the first message in this topic starts there"
KEEP_BUTTON = "🏠 keep {shown}"
KEEP_KIND = "ck"
SEARCH_KIND = "cs"
PICKER_KIND = "cw"


def cwd_reply(shown: str, changed: bool) -> str:
    """The reply to a cwd change. A real change starts a fresh session (claude keeps transcripts per project dir, so a
    resume from another cwd fails); the same cwd changes nothing."""
    if changed:
        return f"📁 {shown} — the next message starts a fresh session there"
    return f"📁 {shown} — already the cwd, nothing changed"


def folder_term(name: str) -> str:
    """The folder search term of a topic name: its words without emojis and punctuation."""
    return " ".join(re.sub(r"[\W_]+", " ", name or "").split())


@dataclass(frozen=True)
class TapOutcome:
    """What a tap on a cwd card answers and shows. card_text, when set, replaces the card's text and drops its buttons."""

    toast: str
    card_text: str | None = None
    applied: bool = False


class CwdService:
    """Folder choices, searches and cards for topics. Every collaborator is injected; the only state is the found-card
    cache, and the folder of a topic lives in the sessions table."""

    def __init__(
        self,
        *,
        sessions: SessionRepo,
        kv: KvRepo,
        paths: CwdRepo,
        messenger: Messenger,
        live: LiveRegistry,
        search_pool: WorkerPool,
        out_pool: WorkerPool,
        logger: Logger,
        clock: Clock,
        handoff: HandoffService,
        finder: Finder = find_dirs,
    ) -> None:
        self._sessions = sessions
        self._kv = kv
        self._handoff = handoff
        self._paths = paths
        self._messenger = messenger
        self._live = live
        self._search_pool = search_pool
        self._out_pool = out_pool
        self._logger = logger
        self._clock = clock
        self._finder = finder
        self._found_lock = threading.Lock()
        self._found: dict[tuple[int, int | None], list[str]] = {}

    # --- folders of a topic and the /cwd picker

    def choices(self, cfg: Config) -> list[str]:
        """The directories the /cwd buttons offer: the config's cwd_shortcuts first, then the topics' distinct folders,
        only existing directories, at most CWD_CHOICES_MAX. Recomputed on every call, so a button's index is read
        against the list it was drawn from."""
        out: list[str] = []
        for p in [*cfg.cwd_shortcuts(), *self._sessions.cwds()]:
            path = os.path.realpath(os.path.expanduser(p))
            if path not in out and os.path.isdir(path):
                out.append(path)
        return out[:CWD_CHOICES_MAX]

    def path_id(self, path: str) -> int:
        """The stable id of a folder in the /cwd buttons (cwd_paths table)."""
        return self._paths.path_id(path)

    def path_of(self, path_id: int) -> str | None:
        """The folder a /cwd button's id stands for, or None when there is no such id."""
        return self._paths.path_of(path_id)

    def picker(self, cfg: Config, thread: int, current: str) -> tuple[str, list[list[tuple[str, CallbackData]]]]:
        """Text and keyboard of the /cwd picker: the current directory, one button per choice, a mark on the current one."""
        options = [(tilde(p, cfg.home), p, self.path_id(p)) for p in self.choices(cfg)]
        text = f"📁 cwd: {current}\nchange it with a button or /cwd <path>"
        return text, picker_rows(thread, PICKER_KIND, options, current)

    def apply(self, thread: int, path: str) -> bool:
        """Set the topic's folder to path (a real directory). True when it changed. A change forgets the session id
        (it stays reachable through /resume) and drops a pending handoff digest, since the next session does not continue
        the old conversation. False for an unknown topic or the folder it already has."""
        row = self._sessions.get(thread)
        if row is None or os.path.realpath(os.path.expanduser(row.cwd or "")) == path:
            return False
        self._sessions.set_cwd(thread, path)
        self._sessions.new_session(thread)
        self._handoff.discard(thread)
        return True

    # --- the pending folder of a new topic

    def pending_set(self, thread: int, path: str) -> None:
        """Keep a folder for the topic's first message ('' clears it)."""
        self._kv.set(PENDING_KEY.format(thread=thread), path)

    def pending_get(self, thread: int) -> str:
        """The folder kept for the first message, or '' when none."""
        return self._kv.get(PENDING_KEY.format(thread=thread), "") or ""

    def apply_pending(self, thread: int) -> bool:
        """Apply the kept folder once, when the topic's session row is created. The pending value is cleared either way;
        the folder is applied only when it still exists. True when the folder changed."""
        path = self.pending_get(thread)
        if not path:
            return False
        self.pending_set(thread, "")
        return os.path.isdir(path) and self.apply(thread, path)

    # --- the folder list of a search card

    def remember_found(self, thread: int, mid: int | None, found: Sequence[str]) -> None:
        """Keep a card's folder list under its message id, so its taps apply the folders their buttons showed."""
        with self._found_lock:
            self._found[(thread, mid)] = list(found)
            while len(self._found) > CWD_FOUND_KEEP:
                self._found.pop(next(iter(self._found)))

    def found_list(self, thread: int, mid: int | None) -> list[str] | None:
        """The folder list kept for a card, or None when the card is unknown or its list was forgotten."""
        with self._found_lock:
            found = self._found.get((thread, mid))
            return list(found) if found is not None else None

    def forget_found(self, thread: int, mid: int | None) -> None:
        with self._found_lock:
            self._found.pop((thread, mid), None)

    # --- name search (/cwd <name>) and its card

    def start_search(self, chat_id: int | str, thread: int, cfg: Config, term: str) -> None:
        """Queue a /cwd name search on the search pool and return at once. The card follows on the out pool."""
        self._search_pool.submit(self._search_job, chat_id, thread, cfg.search_roots(), cfg.home, term)

    def _search_job(self, chat_id: int | str, thread: int, roots: Sequence[str], home: str, term: str) -> None:
        found = self._run_finder(term, roots, "cwd search")
        self._out_pool.submit(self._post_search, chat_id, thread, home, term, found)

    def _post_search(self, chat_id: int | str, thread: int, home: str, term: str, found: list[str]) -> None:
        """Out-pool half of a name search: the no-result text, or a card with one button per folder."""
        if not found:
            self._messenger.send_text(chat_id, thread, NO_MATCH_TEXT.format(term=term))
            return
        rows = picker_rows(thread, SEARCH_KIND, [(tilde(p, home), p, i) for i, p in enumerate(found)], None)
        res = self._messenger.send_buttons(chat_id, thread, SEARCH_CARD_TEXT.format(term=term), rows)
        mid = (res or {}).get("message_id")
        if mid is not None:
            self.remember_found(thread, mid, found)

    def search_tap(self, cfg: Config, thread: int, mid: int | None, idx: int) -> TapOutcome:
        """A tap on a search button: the same change as /cwd <path>, refused while a turn runs. The folder comes from
        the list kept for this card. A topic whose session row does not exist yet keeps the folder for its first message."""
        found = self.found_list(thread, mid)
        if found is None or idx >= len(found):
            return TapOutcome(toast=EXPIRED_TEXT)
        path = found[idx]
        if not os.path.isdir(path):
            return TapOutcome(toast=GONE_TEXT)
        shown = tilde(path, cfg.home)
        if self._sessions.get(thread) is None:
            self.pending_set(thread, path)
            text = PENDING_TEXT.format(shown=shown)
            return TapOutcome(toast=text, card_text=text)
        if self._live.running(thread) is not None:
            return TapOutcome(toast=BUSY_TEXT)
        changed = self.apply(thread, path)
        text = cwd_reply(shown, changed)
        return TapOutcome(toast=text, card_text=text, applied=changed)

    # --- the new-topic folder card

    def suggest_for_new_topic(self, cfg: Config, chat_id: int | str, thread: int, name: str) -> None:
        """Poll-thread half of the folder suggestion for a new forum topic: cheap checks only, then the search runs on
        the search pool. Skipped for short names, topics whose config already sets a folder, spawn and pane topics,
        and names that would not get a plain session. Nothing waits for it."""
        term = folder_term(name)
        if len(term) < TOPIC_FOLDER_MIN:
            return
        alias, sep, sess_name = (name or "").partition(".")
        if sep and sess_name.strip() and cfg.spawn().get(alias):
            return
        cfgt = cfg.topics().get(name)
        if cfgt and cfgt.get("cwd"):
            return
        if cfgt is None:
            cfgt = cfg.default_topic()
        if not cfgt or cfgt.get("type") == "pane" or cfgt.get("pane"):
            return
        default = cfgt.get("cwd") or cfg.home
        self._search_pool.submit(self._topic_job, chat_id, thread, cfg.search_roots(), cfg.home, term, default)

    def _topic_job(
        self, chat_id: int | str, thread: int, roots: Sequence[str], home: str, term: str, default: str
    ) -> None:
        found = self._run_finder(term, roots, "topic folder search")
        if found:
            self._out_pool.submit(self._post_topic_card, chat_id, thread, home, default, found[:TOPIC_FOLDER_MAX])

    def _post_topic_card(self, chat_id: int | str, thread: int, home: str, default: str, found: list[str]) -> None:
        """Out-pool half of the folder suggestion: the question, one button per folder, and a keep button."""
        rows = picker_rows(thread, SEARCH_KIND, [(tilde(p, home), p, i) for i, p in enumerate(found)], None)
        keep = CallbackData(PICKER_PREFIX, (str(thread), KEEP_KIND, 0))
        rows.append([(KEEP_BUTTON.format(shown=tilde(default, home)), keep)])
        res = self._messenger.send_buttons(chat_id, thread, TOPIC_CARD_TEXT, rows)
        mid = (res or {}).get("message_id")
        if mid is not None:
            self.remember_found(thread, mid, found)

    def keep_tap(self, thread: int, mid: int | None) -> TapOutcome:
        """A tap on the keep button of a new topic's card: the topic stays in its default folder. The card is edited."""
        self.pending_set(thread, "")
        self.forget_found(thread, mid)
        return TapOutcome(toast=KEEP_TEXT, card_text=KEEP_TEXT)

    # --- helpers

    def _run_finder(self, term: str, roots: Sequence[str], what: str) -> list[str]:
        """The finder over the resolved roots under the search time limit. A failure is logged and means no folders."""
        deadline = self._clock.monotonic() + CWD_SEARCH_SECONDS
        try:
            return list(self._finder(normalize_term(term), resolve_search_roots(roots), deadline))
        except Exception as exc:
            self._logger.log(f"{what} failed: {str(exc)[:120]}")
            return []
