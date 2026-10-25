"""cwd: port of the /cwd and new-topic folder scenarios of fakecli/test_launcher_cwd.py (labels A to I7 and the
search-card, tap and pending-folder rules; the /resume folder rules J are in the sessions and /resume steps).

Everything runs in a TempHome: the folder trees are created inside it, the search finder is a recording stub (except
in the find_dirs tests, which walk only the temp tree), and Telegram, the clock and the worker pools are fakes or real
pools that the test joins. No process starts and no real state directory is touched.
"""

import json
import os
import threading
import time
import unittest

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tg_bridge.config import Config, tilde
from tg_bridge.core.cwd import (
    CWD_CHOICES_MAX,
    CWD_FOUND_KEEP,
    TOPIC_FOLDER_MAX,
    CwdService,
    cwd_reply,
    folder_term,
)
from tg_bridge.core.cwd_search import CWD_SEARCH_MAX, find_dirs, resolve_search_roots
from tg_bridge.agents.registry import BackendRegistry
from tg_bridge.core.handoff import HandoffService, handoff_key
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.workers import WorkerPool
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.db import Database
from tg_bridge.store.inbox import CwdRepo
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.sessions import SessionRepo
from tg_bridge.telegram.messenger import Messenger

CHAT = -100
THREAD = 9601
CLAUDE = "claude --alpha"
JOIN_SECONDS = 5
EXPIRED = "list expired, run /cwd again"


class CardTelegram(FakeTelegram):
    """FakeTelegram that also remembers the message id every sendMessage received, so tests can tap a card."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock=clock)
        self.sent: list[tuple[dict, int]] = []
        self._sent_lock = threading.Lock()

    def call(self, method: str, feed: bool = False, **params):
        result = super().call(method, feed, **params)
        if method == "sendMessage" and isinstance(result, dict):
            with self._sent_lock:
                self.sent.append((dict(params), int(result["message_id"])))
        return result


class StubFinder:
    """Records each search (term, roots, deadline). Returns `result`, raises `error` when set, and waits on `gate`
    (when set) before answering."""

    def __init__(self, result=None, error: Exception | None = None, gate: threading.Event | None = None) -> None:
        self.result = list(result or [])
        self.error = error
        self.gate = gate
        self.calls: list[tuple[str, tuple[str, ...], float]] = []
        self._lock = threading.Lock()

    def __call__(self, term: str, roots, deadline: float) -> list[str]:
        with self._lock:
            self.calls.append((term, tuple(roots), deadline))
        if self.gate is not None:
            self.gate.wait(JOIN_SECONDS)
        if self.error is not None:
            raise self.error
        return list(self.result)


class FakeTurn:
    """Stands in for a running turn in LiveRegistry; only its presence matters there."""


def callbacks(markup: str) -> list[str]:
    return [b["callback_data"] for row in json.loads(markup)["inline_keyboard"] for b in row]


def button_labels(markup: str) -> list[str]:
    return [b["text"] for row in json.loads(markup)["inline_keyboard"] for b in row]


class CwdCase(unittest.TestCase):
    """A TempHome with a migrated Database, a CwdService over fakes, and helpers to run the pools to the end."""

    def setUp(self) -> None:
        self._home = TempHome()
        self.home = self._home.__enter__().home
        self.clock = FakeClock(1000.0)
        self.tg = CardTelegram(self.clock)
        self.logger = Logger(os.path.join(self._home.state, "bridge.log"))
        self.messenger = Messenger(self.tg, self.logger, self.clock)
        self.db = Database(Paths.from_home(self.home))
        self.sessions = SessionRepo(self.db, self.clock)
        self.kv = KvRepo(self.db)
        self.paths = CwdRepo(self.db)
        self.live = LiveRegistry()
        self.search_pool = WorkerPool("cwd-search", 1, self.logger)
        self.out_pool = WorkerPool("out", 1, self.logger)
        self.finder = StubFinder()
        self.service = self._service(self.finder)
        self.cfg = self._cfg()

    def tearDown(self) -> None:
        self.search_pool.join(JOIN_SECONDS)
        self.out_pool.join(JOIN_SECONDS)
        self.db.close()
        self._home.__exit__(None, None, None)

    def _service(self, finder) -> CwdService:
        handoff = HandoffService(self.kv, BackendRegistry(self._cfg(), self.logger, self.clock), self.logger)
        return CwdService(
            sessions=self.sessions, kv=self.kv, paths=self.paths, messenger=self.messenger, live=self.live,
            search_pool=self.search_pool, out_pool=self.out_pool, logger=self.logger, clock=self.clock,
            handoff=handoff, finder=finder,
        )

    def _cfg(self, data: dict | None = None) -> Config:
        base = {
            "search_roots": [self.home],
            "allow_user_ids": [42],
            "topics": {},
            "spawn": {"budget": "/bin/true"},
            "default_topic": {"cwd": self.home},
        }
        base.update(data or {})
        return Config(base, home=self.home)

    def mk(self, *parts: str) -> str:
        """A real folder under the temp home; returns its path as given."""
        path = os.path.join(self.home, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    def drain(self) -> None:
        """Run every queued search and card to the end (the search pool first: it feeds the out pool)."""
        self.assertTrue(self.search_pool.join(JOIN_SECONDS))
        self.assertTrue(self.out_pool.join(JOIN_SECONDS))

    def row(self, thread: int = THREAD):
        return self.sessions.get(thread)

    def add_session(self, thread: int = THREAD, cwd: str | None = None, uuid: str = "OLDUUID") -> None:
        self.sessions.ensure(thread, "side", CLAUDE, cwd or self.home)
        if uuid:
            self.sessions.set_uuid(thread, uuid)

    def card(self, prefix: str, thread: int = THREAD) -> tuple[dict, int]:
        """The last sendMessage to thread whose text starts with prefix: its parameters and message id."""
        found = [(p, mid) for p, mid in self.tg.sent
                 if str(p.get("message_thread_id")) == str(thread) and str(p.get("text", "")).startswith(prefix)]
        self.assertTrue(found, f"no card starting {prefix!r} in thread {thread}")
        return found[-1]

    def start_search(self, term: str, thread: int = THREAD) -> int:
        """Run a /cwd name search end to end; returns the message id of its card (0 when no card was posted)."""
        self.service.start_search(CHAT, thread, self.cfg, term)
        self.drain()
        cards = [mid for p, mid in self.tg.sent
                 if str(p.get("message_thread_id")) == str(thread) and str(p.get("text", "")).startswith("🔎 folders matching")]
        return cards[-1] if cards else 0


class TestFindDirs(CwdCase):
    """A: find_dirs on a real temp tree (names, words, case, hidden/skip folders, depth, ranking, limit, deadline)."""

    def setUp(self) -> None:
        super().setUp()
        self.exact = self.mk("tree", "zz", "alpha")
        self.mk("tree", "alpha-tool")
        self.mk("tree", "x", "my-alpha-tool-old")
        self.mk("tree", ".hidden", "alpha-hidden")
        self.mk("tree", "node_modules", "alpha-pkg")
        self.mk("tree", "venv", "alpha-venv")
        self.five = self.mk("tree", "a", "b", "c", "d", "alpha-five")
        self.mk("tree", "a", "b", "c", "d", "e", "alpha-six")
        self.mk("tree", "alpha-word-tool")
        for i in range(10):
            self.mk("tree", f"alpha-many-{i}")
        self.tree = os.path.join(self.home, "tree")

    def far(self) -> float:
        return time.monotonic() + 30

    def test_A_exact_name_first(self) -> None:
        self.assertEqual(find_dirs("alpha", [self.tree], self.far())[0], self.exact)

    def test_A_at_most_eight_results(self) -> None:
        """Ported labels:
        A: at most 8 results
        """
        self.assertEqual(len(find_dirs("alpha", [self.tree], self.far())), CWD_SEARCH_MAX)

    def test_A_depth_five_found_and_depth_six_not_walked(self) -> None:
        """Ported labels:
        A: depth 5 found, depth 6 not walked
        """
        self.assertEqual(find_dirs("alpha-five", [self.tree], self.far()), [self.five])
        self.assertEqual(find_dirs("alpha-six", [self.tree], self.far()), [])

    def test_A_hidden_node_modules_and_venv_are_skipped(self) -> None:
        """Ported labels:
        A: hidden, node_modules and venv folders are skipped
        """
        found = find_dirs("alpha", [self.tree], self.far())
        self.assertFalse(any("hidden" in p or "node_modules" in p or "venv" in p for p in found))

    def test_A_every_word_matches_case_insensitive(self) -> None:
        """Ported labels:
        A: all words of the term appear, case-insensitive
        """
        names = [os.path.basename(p) for p in find_dirs("ALPHA  TOOL", [self.tree], self.far())]
        self.assertIn("alpha-tool", names)
        self.assertIn("alpha-word-tool", names)

    def test_A_passed_deadline_returns_nothing(self) -> None:
        self.assertEqual(find_dirs("alpha", [self.tree], time.monotonic() - 1), [])

    def test_A_no_match_gives_empty_list(self) -> None:
        """Ported labels:
        A: no match gives an empty list
        """
        self.assertEqual(find_dirs("no-such-name-xyz", [self.tree], self.far()), [])


class TestRootsAndTerms(CwdCase):
    """B: search roots are real and existing, and a root inside another root is dropped. Topic names give search terms."""

    def test_B_nested_and_repeated_roots_are_dropped(self) -> None:
        sub = self.mk("sub")
        roots = resolve_search_roots([self.home, sub, self.home, os.path.join(self.home, "no-such-root")])
        self.assertEqual(roots, [os.path.realpath(self.home)])

    def test_B_config_default_is_workspace_then_home(self) -> None:
        """Ported labels:
        B: default roots are ~/workspace and ~
        """
        self.assertEqual(Config({}, home=self.home).search_roots(), (os.path.join(self.home, "workspace"), self.home))

    def test_folder_term_strips_emojis_and_punctuation(self) -> None:
        self.assertEqual(folder_term("🚀 alpha-tool 🔥"), "alpha tool")
        self.assertEqual(folder_term("🔥🚀 ✨"), "")
        self.assertEqual(folder_term("ab"), "ab")

    def test_cwd_reply_texts(self) -> None:
        self.assertEqual(cwd_reply("~/x", True), "📁 ~/x — the next message starts a fresh session there")
        self.assertEqual(cwd_reply("~/x", False), "📁 ~/x — already the cwd, nothing changed")


class TestChoicesAndPicker(CwdCase):
    """The /cwd picker: shortcuts first, topics' folders next, only existing folders, at most eight, a stable id per
    folder, and the mark on the current one."""

    def test_shortcuts_come_first_then_topic_folders_and_only_existing(self) -> None:
        """Ported labels:
        T: shortcuts, then the topics' directories; repeats and missing ones left out
        """
        shortcut = self.mk("short")
        topic_dir = self.mk("topic-dir")
        self.add_session(THREAD, cwd=topic_dir)
        self.add_session(THREAD + 1, cwd=os.path.join(self.home, "gone"))
        cfg = self._cfg({"cwd_shortcuts": [shortcut, shortcut, os.path.join(self.home, "missing")]})
        self.assertEqual(self.service.choices(cfg), [os.path.realpath(shortcut), os.path.realpath(topic_dir)])

    def test_at_most_eight_choices(self) -> None:
        """Ported labels:
        T6: at most 8 buttons (8)
        """
        many = [self.mk(f"dir-{i}") for i in range(CWD_CHOICES_MAX + 3)]
        self.assertEqual(len(self.service.choices(self._cfg({"cwd_shortcuts": many}))), CWD_CHOICES_MAX)

    def test_path_id_is_stable_and_resolves_back(self) -> None:
        folder = self.mk("stable")
        pid = self.service.path_id(folder)
        self.assertEqual(self.service.path_id(folder), pid)
        self.assertEqual(self.service.path_of(pid), folder)
        self.assertIsNone(self.service.path_of(pid + 999))

    def test_picker_marks_current_and_uses_path_ids(self) -> None:
        other = self.mk("other-dir")
        self.add_session(THREAD, cwd=other)
        real = os.path.realpath(other)
        text, rows = self.service.picker(self.cfg, THREAD, real)
        self.assertTrue(text.startswith(f"📁 cwd: {real}\n"))
        markup = json.dumps({"inline_keyboard": [[{"text": t, "callback_data": d.encode()} for t, d in r] for r in rows]})
        self.assertTrue(any(label.startswith("✓ ") for label in button_labels(markup)))
        self.assertTrue(all(c.startswith(f"cm:{THREAD}:cw:") for c in callbacks(markup)))


class TestSearchCard(CwdCase):
    """C and F: /cwd <name> returns at once, the search runs on a worker and the card lists the folders; a single or
    empty result is still shown as a card or as text."""

    def test_C_search_card_lists_exact_match_first(self) -> None:
        """Ported labels:
        C: callbacks are cm:<thread>:cs:<index>
        C: card is in the topic
        C: first button is the exact match
        C: the search card is sent
        """
        exact = self.mk("alpha")
        self.finder.result = [exact, self.mk("alpha-tool")]
        self.start_search("alpha")
        params, _ = self.card('🔎 folders matching "alpha"')
        self.assertEqual(params["message_thread_id"], THREAD)
        self.assertEqual(callbacks(params["reply_markup"])[0], f"cm:{THREAD}:cs:0")
        self.assertEqual(button_labels(params["reply_markup"])[0], tilde(os.path.realpath(exact), self.home))
        self.assertTrue(all(c.startswith(f"cm:{THREAD}:cs:") for c in callbacks(params["reply_markup"])))

    def test_C_search_runs_with_resolved_roots_and_a_deadline(self) -> None:
        self.finder.result = [self.mk("alpha")]
        before = self.clock.monotonic()
        self.start_search("alpha")
        term, roots, deadline = self.finder.calls[0]
        self.assertEqual(term, "alpha")
        self.assertEqual(roots, (os.path.realpath(self.home),))
        self.assertEqual(deadline, before + 5.0)

    def test_C_card_keeps_its_folder_list_under_its_message_id(self) -> None:
        """Ported labels:
        C: the list is kept under the card
        """
        exact = self.mk("alpha")
        self.finder.result = [exact]
        mid = self.start_search("alpha")
        self.assertEqual(self.service.found_list(THREAD, mid), [os.path.realpath(exact)])

    def test_F_no_match_asks_for_a_full_path(self) -> None:
        """Ported labels:
        F: exact no-result text
        """
        self.finder.result = []
        self.service.start_search(CHAT, THREAD, self.cfg, "zzz-no-such-folder")
        self.drain()
        self.assertIn('no folder matching "zzz-no-such-folder" — send a full path', self.tg.texts(THREAD))

    def test_F_single_result_is_still_a_button_and_does_not_apply(self) -> None:
        """Ported labels:
        F: one result is still a button, not applied
        F: single result sends a card
        F: the search itself does not change the cwd
        """
        folder = self.mk("alpha-five")
        self.add_session(THREAD, cwd=self.mk("typed-before"))
        before = self.row().cwd
        self.finder.result = [folder]
        self.start_search("alpha-five")
        params, _ = self.card('🔎 folders matching "alpha-five"')
        self.assertEqual(len(callbacks(params["reply_markup"])), 1)
        self.assertEqual(self.row().cwd, before)

    def test_search_failure_is_logged_and_means_no_folders(self) -> None:
        self.finder = StubFinder(error=OSError("disk gone"))
        self.service = self._service(self.finder)
        self.service.start_search(CHAT, THREAD, self.cfg, "alpha")
        self.drain()
        self.assertIn('no folder matching "alpha" — send a full path', self.tg.texts(THREAD))
        with open(os.path.join(self._home.state, "bridge.log"), encoding="utf-8") as fh:
            self.assertIn("cwd search failed", fh.read())


class TestSearchTap(CwdCase):
    """D, E, G and I2: a tap on a search button applies the folder with a fresh session; the same folder changes
    nothing; an expired, gone or unknown card is refused; a running turn refuses; a topic without a session row keeps
    the folder for its first message."""

    def setUp(self) -> None:
        super().setUp()
        self.target = self.mk("target")
        self.add_session(THREAD, cwd=self.home, uuid="OLDUUID")
        self.finder.result = [self.target]
        self.card_mid = self.start_search("target")

    def test_D_tap_sets_cwd_and_starts_a_fresh_session(self) -> None:
        """Ported labels:
        D: tap sets the cwd and starts a fresh session
        D: toast says so
        """
        outcome = self.service.search_tap(self.cfg, THREAD, self.card_mid, 0)
        row = self.row()
        self.assertEqual(row.cwd, os.path.realpath(self.target))
        self.assertEqual(row.uuid, "")
        self.assertEqual(row.prev_uuid, "OLDUUID")
        self.assertEqual(outcome.toast, cwd_reply(tilde(os.path.realpath(self.target), self.home), True))
        self.assertEqual(outcome.card_text, outcome.toast)
        self.assertTrue(outcome.applied)

    def test_E_same_folder_changes_nothing_and_keeps_the_session(self) -> None:
        self.sessions.set_cwd(THREAD, os.path.realpath(self.target))
        self.sessions.set_uuid(THREAD, "KEEP")
        outcome = self.service.search_tap(self.cfg, THREAD, self.card_mid, 0)
        self.assertEqual(self.row().uuid, "KEEP")
        self.assertIn("already the cwd", outcome.toast)
        self.assertFalse(outcome.applied)

    def test_G_unknown_card_is_expired(self) -> None:
        """Ported labels:
        G: unknown card gives the expired toast
        """
        self.assertEqual(self.service.search_tap(self.cfg, THREAD, 999999, 0).toast, EXPIRED)

    def test_G_index_past_the_list_is_expired(self) -> None:
        self.assertEqual(self.service.search_tap(self.cfg, THREAD, self.card_mid, 5).toast, EXPIRED)

    def test_G_folder_removed_since_the_card_is_not_applied(self) -> None:
        os.rmdir(self.target)
        outcome = self.service.search_tap(self.cfg, THREAD, self.card_mid, 0)
        self.assertEqual(outcome.toast, "not in the list anymore")
        self.assertIsNone(outcome.card_text)

    def test_G_running_turn_refuses_the_tap(self) -> None:
        """Ported labels:
        G: a running turn refuses the tap
        G: and the cwd is unchanged
        """
        self.live.begin_turn(THREAD, FakeTurn(), object())
        before = self.row().cwd
        outcome = self.service.search_tap(self.cfg, THREAD, self.card_mid, 0)
        self.assertEqual(outcome.toast, "a turn is running — send /stop first")
        self.assertIsNone(outcome.card_text)
        self.assertEqual(self.row().cwd, before)

    def test_I2_tap_without_session_row_keeps_the_folder_pending(self) -> None:
        """Ported labels:
        I2: a tap does not create the session row
        I2: the tapped folder is kept as pending
        I2: toast says the first message starts there
        """
        new_thread = 9611
        mid = self.start_search("target", thread=new_thread)
        outcome = self.service.search_tap(self.cfg, new_thread, mid, 0)
        self.assertIsNone(self.row(new_thread))
        self.assertEqual(self.service.pending_get(new_thread), os.path.realpath(self.target))
        self.assertTrue(outcome.toast.endswith("the first message in this topic starts there"))

    def test_I3_pending_folder_applies_once_when_the_row_is_created(self) -> None:
        """Ported labels:
        I3: the pending folder is used once
        """
        new_thread = 9612
        self.service.pending_set(new_thread, os.path.realpath(self.target))
        self.sessions.ensure(new_thread, "fresh", CLAUDE, self.home)
        self.assertTrue(self.service.apply_pending(new_thread))
        self.assertEqual(self.row(new_thread).cwd, os.path.realpath(self.target))
        self.assertEqual(self.service.pending_get(new_thread), "")
        self.assertFalse(self.service.apply_pending(new_thread))

    def test_pending_folder_that_vanished_is_cleared_and_not_applied(self) -> None:
        new_thread = 9613
        gone = os.path.join(self.home, "vanished")
        self.service.pending_set(new_thread, gone)
        self.sessions.ensure(new_thread, "fresh", CLAUDE, self.home)
        self.assertFalse(self.service.apply_pending(new_thread))
        self.assertEqual(self.service.pending_get(new_thread), "")
        self.assertEqual(self.row(new_thread).cwd, self.home)

    def test_apply_is_refused_for_an_unknown_topic(self) -> None:
        self.assertFalse(self.service.apply(9999, os.path.realpath(self.target)))

    def test_a_folder_change_discards_a_pending_handoff_digest(self) -> None:
        """apply() with a real folder change drops the digest of an earlier cross-backend switch: the next session is
        not the continuation of that conversation (/cwd scenario of the handoff discard rule)."""
        self.kv.set(handoff_key(THREAD), "Context from the previous agent in this topic: User: old question")
        self.assertTrue(self.service.apply(THREAD, os.path.realpath(self.target)))
        self.assertIsNone(self.kv.get(handoff_key(THREAD)))

    def test_apply_with_the_same_folder_keeps_a_pending_handoff_digest(self) -> None:
        """A /cwd tap for the folder the topic already has changes nothing, so the digest stays for the next fresh start."""
        self.sessions.set_cwd(THREAD, os.path.realpath(self.target))
        self.kv.set(handoff_key(THREAD), "Context from the previous agent in this topic: User: old question")
        self.assertFalse(self.service.apply(THREAD, os.path.realpath(self.target)))
        self.assertIsNotNone(self.kv.get(handoff_key(THREAD)))


class TestNewTopicCard(CwdCase):
    """I1, I4, I5, I6 and I7: the new-topic folder card, keep, a tap with the row present, the skip rules, and a slow
    search that never holds the caller."""

    def suggest(self, name: str, cfg: Config | None = None, thread: int = 9621) -> None:
        self.service.suggest_for_new_topic(cfg or self.cfg, CHAT, thread, name)

    def topic_card(self, thread: int = 9621) -> tuple[dict, int]:
        return self.card("📁 Start this topic in one of these folders?", thread)

    def test_I1_card_has_folders_then_keep_and_short_callbacks(self) -> None:
        """Ported labels:
        I1: a new topic gets the folder card
        I1: every callback fits in 64 bytes
        I1: last button is the keep button
        I1: the folder list is kept under the card
        """
        folders = [self.mk(f"alpha-tool-{i}") for i in range(TOPIC_FOLDER_MAX + 2)]
        self.finder.result = folders
        thread = 9621
        self.suggest("🚀 alpha-tool 🔥", thread=thread)
        self.drain()
        params, mid = self.topic_card(thread)
        keys = callbacks(params["reply_markup"])
        self.assertEqual(keys[0], f"cm:{thread}:cs:0")
        self.assertEqual(len(keys) - 1, TOPIC_FOLDER_MAX)
        self.assertEqual(keys[-1], f"cm:{thread}:ck:0")
        self.assertTrue(all(len(k.encode()) <= 64 for k in keys))
        self.assertTrue(button_labels(params["reply_markup"])[-1].startswith("🏠 keep "))
        self.assertEqual(self.finder.calls[0][0], "alpha tool")
        self.assertEqual(self.service.found_list(thread, mid), [os.path.realpath(f) for f in folders[:TOPIC_FOLDER_MAX]])

    def test_I4_keep_edits_the_card_and_creates_no_row(self) -> None:
        """Ported labels:
        I4: keep creates no row
        I4: keep forgets the folder list
        """
        thread = 9622
        self.finder.result = [self.mk("alpha-tool")]
        self.suggest("🚀 alpha-tool 🔥", thread=thread)
        self.drain()
        _, mid = self.topic_card(thread)
        outcome = self.service.keep_tap(thread, mid)
        self.assertEqual(outcome.card_text, "🏠 this topic keeps its default folder")
        self.assertIsNone(self.row(thread))
        self.assertIsNone(self.service.found_list(thread, mid))
        self.assertEqual(self.service.pending_get(thread), "")

    def test_I5_tap_with_the_row_present_applies_with_a_fresh_session(self) -> None:
        """Ported labels:
        I5: a tap with the row present applies the folder and starts a fresh session
        I5: no pending folder is left behind
        """
        thread = 9623
        self.add_session(thread, cwd=self.home, uuid="OLD")
        target = self.mk("alpha-tool")
        self.finder.result = [target]
        self.suggest("alpha-tool", thread=thread)
        self.drain()
        _, mid = self.topic_card(thread)
        outcome = self.service.search_tap(self.cfg, thread, mid, 0)
        row = self.row(thread)
        self.assertEqual((row.cwd, row.uuid, row.prev_uuid), (os.path.realpath(target), "", "OLD"))
        self.assertTrue(outcome.applied)
        self.assertEqual(self.service.pending_get(thread), "")

    def test_I6_skips_post_no_card_and_no_search(self) -> None:
        """Ported labels:
        I6: no card for short name
        I6: no card for emojis only
        I6: no card for spawn topic
        I6: no card for config cwd
        I6: no card for pane topic
        I6: no card for no default topic
        """
        self.finder.result = [self.mk("alpha-tool")]
        skip_cases = [
            ("short name", "ab", self.cfg, 9631),
            ("emojis only", "🔥🚀 ✨", self.cfg, 9632),
            ("spawn topic", "budget.alpha-tool", self.cfg, 9633),
            ("config cwd", "alpha-tool", self._cfg({"topics": {"alpha-tool": {"cwd": self.home}}}), 9634),
            ("pane topic", "alpha-tool", self._cfg({"topics": {"alpha-tool": {"type": "pane", "pane": "%1"}}}), 9635),
            ("no default topic", "alpha-tool", self._cfg({"default_topic": {}}), 9636),
        ]
        for label, name, cfg, thread in skip_cases:
            with self.subTest(label):
                before = len(self.finder.calls)
                self.suggest(name, cfg=cfg, thread=thread)
                self.drain()
                self.assertEqual(len(self.finder.calls), before, f"{label}: a search ran")
                self.assertFalse([p for p, _ in self.tg.sent if str(p.get("message_thread_id")) == str(thread)])

    def test_I6_no_match_posts_no_card(self) -> None:
        """Ported labels:
        I6: no card for no match
        """
        self.finder.result = []
        self.suggest("zz qq nothing", thread=9637)
        self.drain()
        self.assertFalse([p for p, _ in self.tg.sent if str(p.get("message_thread_id")) == "9637"])

    def test_I7_a_slow_search_does_not_hold_the_caller_and_still_posts(self) -> None:
        """Ported labels:
        I7: a slow folder search does not hold the poll thread
        I7: the slow search still posts the card
        """
        gate = threading.Event()
        self.finder = StubFinder(result=[self.mk("alpha-tool")], gate=gate)
        self.service = self._service(self.finder)
        started = time.monotonic()
        self.suggest("alpha-tool slow", thread=9638)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertFalse(self.tg.sent)
        gate.set()
        self.drain()
        self.card("📁 Start this topic in one of these folders?", 9638)


class TestFoundCache(CwdCase):
    """The folder lists of search cards are kept for their taps, at most CWD_FOUND_KEEP, the oldest dropped first."""

    def test_oldest_card_is_forgotten_past_the_limit(self) -> None:
        for mid in range(1, CWD_FOUND_KEEP + 2):
            self.service.remember_found(THREAD, mid, [self.home])
        self.assertIsNone(self.service.found_list(THREAD, 1))
        self.assertEqual(self.service.found_list(THREAD, CWD_FOUND_KEEP + 1), [self.home])

    def test_forget_removes_one_card(self) -> None:
        self.service.remember_found(THREAD, 7, [self.home])
        self.service.forget_found(THREAD, 7)
        self.assertIsNone(self.service.found_list(THREAD, 7))


if __name__ == "__main__":
    unittest.main()
