"""Settings commands (STEP 48): /mode, /model, /launcher and /cwd, ported from fakecli/test_launcher.py (sections A-E,
H, I), fakecli/test_modes.py (the /mode and live-session parts) and fakecli/test_commands.py (sections C, D, F, K).

Scenarios: /mode alone shows the backend's picker with the stored mode marked; a typed mode is stored in any case and an
unknown word gets the usage line; a mode change during a turn reaches the live session through PermissionService and
says so, a backend that cannot switch inside a turn says it applies from the next message; /model alone shows the
configured models (or plain text when the backend lists none), a typed model or default is stored and sent to a
running session; /launcher alone shows the picker, an unknown name is refused with the list, the same launcher says so,
a running turn refuses, and a change settles the session through LauncherSwitch (fresh, not found, handoff); claude ->
codex -> claude keeps sessions.launcher a Claude command; /cwd alone shows the folder picker, a missing path is said
aloud and never searched, an existing folder is applied with the live reply, a bare name starts the search, and every
change is refused while a turn runs.
Every test runs against a migrated database in a TempHome with FakeTelegram and a FakeClock; no subprocess starts.
"""

import os
import shutil
import unittest

from tests.fakes.backend import FakeBackend, RecordingSink, fake_spec
from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tests.turn_support import wait_until
from tg_bridge.agents.claude.backend import ClaudeRules
from tg_bridge.agents.claude.history import encoded_cwd
from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.bot.commands.base import CommandDeps, CommandRegistry
from tg_bridge.bot.commands.settings import (
    BUSY_TEXT,
    MODE_USAGE_TEXT,
    NO_SESSION_TEXT,
    SettingsCommands,
)
from tg_bridge.bot.pickers import NOT_LIVE_NOTE
from tg_bridge.config import Config, tilde
from tg_bridge.core.cards import ApprovalCards
from tg_bridge.core.cwd import CwdService
from tg_bridge.core.topic_queues import TopicQueues
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.launcher_switch import COPIED_TEXT, FRESH_TEXT, HANDOFF_TEXT, NOT_FOUND_TEXT, SAME_TEXT, LauncherSwitch
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.permissions import PermissionService, TurnBinding
from tg_bridge.core.workers import WorkerPool
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.asks import AskRepo
from tg_bridge.store.db import Database
from tg_bridge.store.inbox import CwdRepo
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.rules import RuleRepo
from tg_bridge.store.sessions import HistoryRepo, SessionRepo
from tg_bridge.store.turns import TurnRepo
from tg_bridge.telegram.messenger import Messenger

THREAD = 9480
CHAT = -1009480
TOPIC = "SET"
CLAUDE_CMD = "wrap run"
TUI_CMD = "tui run"
SID = "1f2e3d4c-aaaa-4bbb-8ccc-000000000080"
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "claude_transcripts",
                       "handoff_session.jsonl")
MODELS = ["haiku", "opus"]


class FinderStub:
    """Stands in for the folder finder: records each search and returns the folders it was given."""

    def __init__(self, found=()) -> None:
        self.found = list(found)
        self.calls: list[str] = []

    def __call__(self, term, roots, deadline):
        self.calls.append(term)
        return list(self.found)


class WaitRecorder:
    def waiting_changed(self, thread_id: int, waiting: bool) -> None:
        pass


def config_data(project: str, codex_home: str, codex_models: list[str] | None = None, with_tui: bool = False) -> dict:
    codex = {"backend": "codex", "command": "codex", "config_dir": codex_home}
    if codex_models is not None:
        codex["models"] = codex_models
    launchers = {
        "wrap": {"cmd": CLAUDE_CMD, "config_dir": os.path.join(project, "cfg-a")},
        "plain": {"cmd": "plain run", "config_dir": os.path.join(project, "cfg-b")},
        "shared": {"cmd": "shared-cmd", "config_dir": os.path.join(project, "cfg-a")},
    }
    if with_tui:  # only the tui tests add it: the launcher picker lists every launcher of the config
        launchers["tui"] = {"cmd": TUI_CMD, "config_dir": os.path.join(project, "cfg-c"), "transport": "tui"}
    return {
        "launchers": launchers,
        "agent_launchers": {"codex": codex},
        "topics": {TOPIC: {"launcher": CLAUDE_CMD, "cwd": project}},
        "models": MODELS,
    }


class SettingsCase(unittest.TestCase):
    """The settings commands wired to real services and repositories under a TempHome."""

    CODEX_MODELS: list[str] | None = None  # the 'models' of the codex agent launcher (None: the config lists none)
    WITH_TUI = False                       # True adds a tui launcher ('tui') to the config

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.base = os.path.realpath(self.home.home)
        self.project = os.path.join(self.base, "project")
        os.makedirs(self.project, exist_ok=True)
        self.clock = FakeClock()
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.config = Config(config_data(self.project, os.path.join(self.base, "codex"), self.CODEX_MODELS,
                                         self.WITH_TUI), home=self.base, env={})
        self.backends = build_default_registry(self.config, self.logger, self.clock, title_launcher="")
        self.db = Database(Paths.from_home(self.base))
        self.addCleanup(self.db.close)
        self.sessions = SessionRepo(self.db, self.clock)
        self.history = HistoryRepo(self.db, self.clock)
        self.kv = KvRepo(self.db)
        self.live = LiveRegistry()
        self.telegram = FakeTelegram(self.clock)
        self.messenger = Messenger(self.telegram, self.logger, self.clock)
        self.out = WorkerPool("out", 1, self.logger)
        self.search = WorkerPool("search", 1, self.logger)
        self.finder = FinderStub()
        self.permissions = PermissionService(
            asks=AskRepo(self.db, self.clock), rules=RuleRepo(self.db, self.clock),
            cards=ApprovalCards(self.messenger, self.logger, self.clock), messenger=self.messenger,
            workers=self.out, waiting=WaitRecorder(), logger=self.logger)
        handoff = HandoffService(self.kv, self.backends, self.logger)
        self.cwd = CwdService(sessions=self.sessions, kv=self.kv, paths=CwdRepo(self.db), messenger=self.messenger,
                              live=self.live, search_pool=self.search, out_pool=self.out, logger=self.logger,
                              clock=self.clock, handoff=handoff, finder=self.finder)
        self.launchers = LauncherSwitch(self.config, self.sessions, self.history, handoff, self.backends)
        self.commands = SettingsCommands(config=self.config, sessions=self.sessions, live=self.live,
                                         backends=self.backends, permissions=self.permissions,
                                         launchers=self.launchers, cwd=self.cwd, logger=self.logger)
        self.registry = CommandRegistry()
        self.commands.register(self.registry)
        self.deps = CommandDeps(
            config=self.config, sessions=self.sessions, turns=TurnRepo(self.db), rules=RuleRepo(self.db, self.clock),
            kv=self.kv, live=self.live, queues=TopicQueues(), backends=self.backends,
            messenger=self.messenger, out=self.out,
        )
        self.sessions.ensure(THREAD, TOPIC, CLAUDE_CMD, self.project, "bypassPermissions")
        self.fake_backend = FakeBackend()

    def run_cmd(self, text: str) -> bool:
        handled = self.registry.dispatch(self.deps, CHAT, THREAD, TOPIC, text)
        self.assertTrue(self.drain(), "the replies did not go out")
        return handled

    def drain(self) -> bool:
        """Waits for the search worker and then the ordered out-worker, so every reply and card is in FakeTelegram."""
        return self.search.join(5) and self.out.join(5)

    def last(self) -> str:
        return self.telegram.texts(THREAD)[-1]

    def labels(self) -> list[str]:
        return [b["text"] for keyboard in self.telegram.buttons(THREAD) for row in keyboard for b in row]

    def callbacks(self) -> list[str]:
        return [b["callback_data"] for keyboard in self.telegram.buttons(THREAD) for row in keyboard for b in row]

    def row(self):
        return self.sessions.get(THREAD)

    def start_turn(self):
        """A live turn in the topic: a fake session registered in LiveRegistry and bound to the permission service."""
        session = self.fake_backend.open_session(fake_spec(thread_id=THREAD, chat_id=CHAT), RecordingSink())
        feed = object()
        self.assertTrue(self.live.begin_turn(THREAD, session, feed))
        binding = TurnBinding(thread_id=THREAD, chat_id=CHAT, cwd=self.project, rules=ClaudeRules(), mode="default")
        self.permissions.bind_turn(binding)
        self.permissions.attach(binding, session)
        return session, binding

    def put_claude_transcript(self, config_dir: str, sid: str) -> None:
        folder = os.path.join(config_dir, "projects", encoded_cwd(os.path.realpath(self.project)))
        os.makedirs(folder, exist_ok=True)
        shutil.copyfile(FIXTURE, os.path.join(folder, sid + ".jsonl"))

    def set_history(self, uuid: str, launcher: str = CLAUDE_CMD) -> None:
        self.sessions.set_uuid(THREAD, uuid)
        self.history.note_session(THREAD, uuid, "earlier question", self.project, launcher, "claude")


class ModeTests(SettingsCase):
    def test_mode_alone_shows_the_picker_with_the_stored_mode_marked(self) -> None:
        """Labels: E: /mode shows the picker with the stored mode marked; E: callbacks are md:<index>.

        Ported labels:
        C: /mode alone shows the mode
        E: /mode shows picker ('🛡 mode: bypassPermissions')
        """
        self.assertTrue(self.run_cmd("/mode"))
        self.assertEqual(self.last(), "🛡 mode: bypassPermissions")
        self.assertEqual(self.labels(), ["✓ bypass", "acceptEdits", "default", "plan"])
        self.assertEqual(self.callbacks(), [f"cm:{THREAD}:md:0", f"cm:{THREAD}:md:1", f"cm:{THREAD}:md:2",
                                            f"cm:{THREAD}:md:3"])

    def test_typed_modes_are_stored_in_any_case(self) -> None:
        """Labels: C: /mode acceptEdits, /mode plan are stored; E: typed /mode plan persists; the key works too.

        Ported labels:
        C: /mode acceptEdits is stored
        C: /mode plan is stored
        C: the bypass alias works in any case
        E: typed /mode plan still persists
        E: typed reply unchanged
        """
        self.run_cmd("/mode acceptEdits")
        self.assertEqual(self.row().mode, "acceptEdits")
        self.assertEqual(self.last(), "🛡 mode set to acceptEdits")
        self.run_cmd("/mode PLAN")
        self.assertEqual(self.row().mode, "plan")
        self.run_cmd("/mode bypassPermissions")
        self.assertEqual(self.row().mode, "bypassPermissions")
        self.run_cmd("/mode ACCEPTEDITS")
        self.assertEqual(self.row().mode, "acceptEdits")
        self.run_cmd("/MODE Bypass")  # the picker label 'bypass' names bypassPermissions, in any case
        self.assertEqual(self.row().mode, "bypassPermissions")

    def test_unknown_mode_gets_the_usage_line_and_changes_nothing(self) -> None:
        """Labels: C: /mode bogus says usage and keeps the mode.

        Ported labels:
        C: an unknown mode is refused and the mode stays
        """
        self.run_cmd("/mode bogus")
        self.assertEqual(self.last(), MODE_USAGE_TEXT)
        self.assertEqual(self.row().mode, "bypassPermissions")

    def test_mode_without_a_turn_is_only_stored(self) -> None:
        """Labels: D: a mode change with no live session stores the mode and sends nothing to a session."""
        self.run_cmd("/mode default")
        self.assertEqual(self.last(), "🛡 mode set to default")
        self.assertEqual(self.row().mode, "default")

    def test_mode_during_a_turn_reaches_the_live_session(self) -> None:
        """Labels: K: /mode bypass sends set_permission_mode to the running CLI; E: live session mode updated.

        Ported labels:
        K: /mode sends set_permission_mode
        K: the live mode and the stored mode both switch
        """
        session, binding = self.start_turn()
        self.run_cmd("/mode bypass")
        self.assertTrue(wait_until(lambda: ("set_mode", "bypassPermissions") in session.calls),
                        f"the live session got no set_mode ({session.calls})")
        self.assertEqual(binding.mode, "bypassPermissions")
        self.assertEqual(self.row().mode, "bypassPermissions")
        self.assertEqual(self.last(), "🛡 mode set to bypassPermissions — applied to the running turn")

    def test_mode_during_a_turn_that_cannot_switch_live_says_next_message(self) -> None:
        """A Codex topic cannot change its mode inside a running turn: the reply says the change applies next."""
        self.sessions.set_launcher_pick(THREAD, "codex")
        self.sessions.set_backend(THREAD, "codex")
        session, _ = self.start_turn()
        self.run_cmd("/mode plan")
        self.assertEqual(self.last(), "🛡 mode set to plan — applies from your next message")
        self.assertEqual(self.row().mode, "plan")
        self.assertTrue(wait_until(lambda: ("set_mode", "plan") in session.calls))

    def test_codex_mode_names_resolve_to_the_codex_modes(self) -> None:
        """A Codex topic offers the same four modes; the label 'bypass' stores bypassPermissions."""
        self.sessions.set_launcher_pick(THREAD, "codex")
        self.sessions.set_backend(THREAD, "codex")
        self.run_cmd("/mode bypass")
        self.assertEqual(self.row().mode, "bypassPermissions")
        self.run_cmd("/mode")
        self.assertIn("🛡 mode: bypassPermissions", self.last())

    def test_mode_without_a_session_row_says_so(self) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM sessions WHERE thread_id=?", (THREAD,))
        self.run_cmd("/mode plan")
        self.assertEqual(self.last(), NO_SESSION_TEXT)


class ModelTests(SettingsCase):
    def test_model_alone_shows_the_configured_models(self) -> None:
        """Labels: B: /model shows the picker with the models of the config; no model is marked when none is set.

        Ported labels:
        B: /model shows picker text ('🧠 model: default')
        B: one button per configured model
        B: no-arg /model stores nothing
        B2: config models list is used

        Ported labels:
        S3: model buttons carry the model name (['cm:9504:mo:sonnet'])
        """
        self.run_cmd("/model")
        self.assertEqual(self.last(), "🧠 model: default")
        self.assertEqual(self.labels(), MODELS)
        self.assertEqual(self.callbacks(), [f"cm:{THREAD}:mo:{m}" for m in MODELS])
        self.assertEqual(self.row().model, "", "a no-argument /model stores nothing")

    def test_model_alone_for_a_backend_without_a_model_list_is_plain_text(self) -> None:
        """A Codex topic whose launcher lists no models: the picker is plain text, with the not-live note."""
        self.sessions.set_launcher_pick(THREAD, "codex")
        self.sessions.set_backend(THREAD, "codex")
        self.run_cmd("/model")
        self.assertEqual(self.last(), "🧠 model: default\napplies from your next message")
        self.assertEqual(self.labels(), [])

    def test_typed_model_is_stored_and_default_clears_it(self) -> None:
        """Labels: C: typed /model haiku persists; C: /model Default clears the model; D: /model sonnet is stored.

        Ported labels:
        C: typed /model haiku persisted
        C: typed reply unchanged
        C: typed /model Default clears the model
        """
        self.run_cmd("/model haiku")
        self.assertEqual(self.row().model, "haiku")
        self.assertEqual(self.last(), "🧠 model set to haiku")
        self.run_cmd("/model Default")
        self.assertEqual(self.row().model, "")
        self.assertEqual(self.last(), "🧠 model set to default")

    def test_model_during_a_turn_is_sent_to_the_live_session(self) -> None:
        """Labels: K: /model opus sends set_model; K: /model default sends set_model default."""
        session, _ = self.start_turn()
        self.run_cmd("/model opus")
        self.assertTrue(wait_until(lambda: ("set_model", "opus") in session.calls), session.calls)
        self.run_cmd("/model default")
        self.assertTrue(wait_until(lambda: ("set_model", "default") in session.calls), session.calls)
        self.assertEqual(self.row().model, "")

    def test_model_without_a_session_row_says_so(self) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM sessions WHERE thread_id=?", (THREAD,))
        self.run_cmd("/model opus")
        self.assertEqual(self.last(), NO_SESSION_TEXT)


class CodexModelTests(SettingsCase):
    """The /model picker follows the topic's launcher: a Codex launcher's configured 'models' are its buttons."""

    CODEX_MODELS = ["gpt-6.1-sol"]

    def test_model_alone_on_a_codex_topic_shows_the_configured_codex_models(self) -> None:
        self.sessions.set_launcher_pick(THREAD, "codex")
        self.sessions.set_backend(THREAD, "codex")
        self.run_cmd("/model")
        self.assertEqual(self.last(), "🧠 model: default\napplies from your next message")
        self.assertEqual(self.labels(), ["gpt-6.1-sol"])
        self.assertEqual(self.callbacks(), [f"cm:{THREAD}:mo:gpt-6.1-sol"])

    def test_the_models_follow_the_launcher_not_the_stored_backend_column(self) -> None:
        self.sessions.set_launcher_pick(THREAD, "codex")  # launcher says codex, the stored backend is stale
        self.run_cmd("/model")
        self.assertEqual(self.labels(), ["gpt-6.1-sol"])
        self.sessions.set_launcher_pick(THREAD, "wrap")  # moved back to Claude: the column still says codex
        self.sessions.set_backend(THREAD, "codex")
        self.run_cmd("/model")
        newest = self.telegram.buttons(THREAD)[-1]  # the older picker is still in the topic: read the new one only
        self.assertEqual([b["text"] for row in newest for b in row], MODELS)


class LauncherTests(SettingsCase):
    def test_launcher_alone_shows_the_picker_with_the_current_launcher_marked(self) -> None:
        """Labels: E1: /launcher picker, check on the current launcher, callbacks cm:<thread>:la:<index>."""
        self.run_cmd("/launcher")
        self.assertEqual(self.last(), "🚀 launcher: wrap")
        self.assertEqual(self.labels(), ["✓ wrap", "plain", "shared", "codex · codex"])
        self.assertEqual(self.callbacks(), [f"cm:{THREAD}:la:{i}" for i in range(4)])

    def test_unknown_launcher_is_refused_with_the_list(self) -> None:
        """Labels: E2: an unknown name is refused with the choices and changes nothing."""
        self.set_history(SID)
        self.run_cmd("/launcher nope")
        self.assertTrue(self.last().startswith('unknown launcher "nope". choose one of: wrap, plain, shared'))
        self.assertEqual((self.row().launcher_pick, self.row().uuid), ("", SID))

    def test_same_launcher_says_so_and_keeps_the_pick_and_session(self) -> None:
        """Labels: E3: the same launcher says so; the pick and the session are kept."""
        self.set_history(SID)
        self.run_cmd("/launcher wrap")
        self.assertEqual(self.last(), SAME_TEXT.format(name="wrap"))
        self.assertEqual((self.row().launcher_pick, self.row().uuid), ("", SID))

    def test_a_running_turn_refuses_a_different_launcher(self) -> None:
        """Labels: E5: a running turn refuses a different launcher; the refusal changes nothing."""
        self.set_history(SID)
        self.start_turn()
        self.run_cmd("/launcher plain")
        self.assertEqual(self.last(), BUSY_TEXT)
        self.assertEqual((self.row().launcher_pick, self.row().uuid), ("", SID))

    def test_change_without_a_transcript_says_not_found_and_starts_fresh(self) -> None:
        """Labels: E4: a launcher change with no transcript to copy says not found; the id moves to prev_uuid."""
        self.set_history(SID)
        self.run_cmd("/launcher plain")
        self.assertEqual(self.last(), NOT_FOUND_TEXT.format(name="plain"))
        row = self.row()
        self.assertEqual((row.launcher_pick, row.uuid, row.prev_uuid), ("plain", "", SID))

    def test_change_with_nothing_to_carry_starts_fresh(self) -> None:
        self.run_cmd("/launcher plain")
        self.assertEqual(self.last(), FRESH_TEXT.format(name="plain"))
        self.assertEqual(self.row().launcher_pick, "plain")

    def test_change_to_a_folder_with_a_copy_keeps_the_id(self) -> None:
        """Labels: E4: another Claude folder gets a copy of the transcript and keeps the id."""
        self.set_history(SID)
        self.put_claude_transcript(os.path.join(self.project, "cfg-a"), SID)
        self.run_cmd("/launcher plain")
        self.assertEqual(self.last(), COPIED_TEXT.format(name="plain"))
        self.assertEqual(self.row().uuid, SID)

    def test_claude_codex_claude_keeps_the_launcher_a_claude_command(self) -> None:
        """Labels: claude -> codex -> claude: sessions.launcher stays a Claude command; the pick and backend follow."""
        self.set_history(SID)
        self.put_claude_transcript(os.path.join(self.project, "cfg-a"), SID)
        self.run_cmd("/launcher codex")
        self.assertEqual(self.last(), HANDOFF_TEXT.format(name="codex"))
        row = self.row()
        self.assertEqual((row.backend, row.launcher_pick, row.uuid, row.prev_uuid),
                         ("codex", "codex", "", SID))
        self.assertEqual(row.launcher, CLAUDE_CMD)
        self.run_cmd("/launcher wrap")
        row = self.row()
        self.assertEqual((row.backend, row.launcher_pick), ("claude", "wrap"))
        self.assertEqual(row.launcher, CLAUDE_CMD)
        self.assertEqual(self.history.find(THREAD, SID).launcher, CLAUDE_CMD)

    def test_launcher_without_a_session_row_says_so(self) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM sessions WHERE thread_id=?", (THREAD,))
        self.run_cmd("/launcher plain")
        self.assertEqual(self.last(), NO_SESSION_TEXT)


class CwdTests(SettingsCase):
    def test_cwd_alone_shows_the_folder_picker_with_the_current_folder_marked(self) -> None:
        """Labels: T: /cwd with no argument offers the topics' folders; the current one is marked.

        Ported labels:
        T: /cwd shows the current directory and buttons (send_buttons)
        T: labels are shortened paths, ✓ on the one in use (['~/cwd-a', '~/cwd-b', '✓ ~/cwd-c'])
        T: callbacks are cm:<thread>:cw:<path id>
        H: plain /cwd still shows the shortcuts picker
        """
        self.run_cmd("/cwd")
        self.assertIn("📁 cwd: " + self.project, self.last())
        self.assertIn("✓ " + tilde(self.project, self.base), self.labels())
        self.assertTrue(all(label.lstrip("✓ ").startswith("~") for label in self.labels()), self.labels())
        for data in self.callbacks():
            self.assertTrue(data.startswith(f"cm:{THREAD}:cw:"), data)
            self.assertIsNotNone(self.cwd.path_of(int(data.rsplit(":", 1)[1])), data)

    def test_missing_absolute_path_is_said_aloud_and_never_searched(self) -> None:
        """Labels: H1/H2/F: /cwd with a path that is not a directory says so, keeps the cwd, and searches nothing.

        Ported labels:
        F: /cwd with a missing path says not a directory, cwd kept
        H4: refused paths keep the cwd
        """
        missing = os.path.join(self.base, "no-such-dir-xyz")
        self.run_cmd("/cwd " + missing)
        self.assertEqual(self.last(), f"not a directory: {missing}")
        self.assertEqual(self.finder.calls, [])
        self.assertEqual(self.row().cwd, self.project)

    def test_existing_folder_is_applied_and_forgets_the_session(self) -> None:
        """Labels: F: /cwd stores the real path; the change starts a fresh session (the id moves to prev_uuid).

        Ported labels:
        E: a typed directory starts a fresh session
        E: typed change says so
        """
        other = os.path.join(self.base, "other")
        os.makedirs(other)
        self.set_history(SID)
        self.run_cmd("/cwd " + other)
        self.assertEqual(self.last(), f"📁 {os.path.realpath(other)} — the next message starts a fresh session there")
        row = self.row()
        self.assertEqual((row.cwd, row.uuid, row.prev_uuid), (os.path.realpath(other), "", SID))

    def test_same_folder_changes_nothing(self) -> None:
        """Ported labels:
        E: same cwd keeps the session
        E: same cwd says nothing changed
        """
        self.set_history(SID)
        self.run_cmd("/cwd " + self.project)
        self.assertEqual(self.last(), f"📁 {self.project} — already the cwd, nothing changed")
        self.assertEqual(self.row().uuid, SID)

    def test_bare_name_starts_the_folder_search_and_posts_the_card(self) -> None:
        """Labels: H3: a bare word still runs the folder-name search; the card lists the folders found."""
        found = os.path.join(self.base, "photos")
        os.makedirs(found)
        self.finder.found = [found]
        self.run_cmd("/cwd photos")
        self.assertEqual(self.finder.calls, ["photos"])
        self.assertEqual(self.last(), '🔎 folders matching "photos" — tap one to set it as the cwd:')
        self.assertEqual(self.row().cwd, self.project)

    def test_changes_are_refused_while_a_turn_runs(self) -> None:
        """Labels: /cwd while a turn runs (picker, path and name): refused, nothing changes."""
        other = os.path.join(self.base, "other")
        os.makedirs(other)
        self.start_turn()
        self.run_cmd("/cwd")
        self.assertEqual(self.last(), BUSY_TEXT)
        self.run_cmd("/cwd " + other)
        self.assertEqual(self.last(), BUSY_TEXT)
        self.run_cmd("/cwd photos")
        self.assertEqual(self.last(), BUSY_TEXT)
        self.assertEqual(self.finder.calls, [])
        self.assertEqual(self.row().cwd, self.project)

    def test_cwd_without_a_session_row_says_so(self) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM sessions WHERE thread_id=?", (THREAD,))
        self.run_cmd("/cwd " + self.base)
        self.assertEqual(self.last(), NO_SESSION_TEXT)


class RegistrationTests(SettingsCase):
    def test_the_four_settings_words_are_registered(self) -> None:
        """Ported labels:
        B6: /launcher is a bridge command
        """
        words = [word for word, _ in self.registry.bridge_commands()]
        self.assertEqual([w for w in words if w in {"mode", "model", "launcher", "cwd"}],
                         ["mode", "model", "launcher", "cwd"])

    def test_text_that_is_not_a_settings_command_is_left_to_the_agent(self) -> None:
        self.assertFalse(self.run_cmd("/compact"))
        self.assertFalse(self.run_cmd("use /mode here"))
        self.assertEqual(self.row().mode, "bypassPermissions")


class TuiLauncherTests(SettingsCase):
    """A tui launcher (transport tui) has no live mode or model switch: its replies say the change applies next."""

    WITH_TUI = True

    def test_mode_during_a_turn_on_a_tui_launcher_says_next_message(self) -> None:
        """A tui launcher has no live mode switch (its capabilities say so): the reply says the change applies next."""
        self.sessions.set_launcher_pick(THREAD, "tui")
        session, _ = self.start_turn()
        self.run_cmd("/mode plan")
        self.assertEqual(self.last(), "🛡 mode set to plan — applies from your next message")
        self.assertEqual(self.row().mode, "plan")
        self.assertTrue(wait_until(lambda: ("set_mode", "plan") in session.calls))

    def test_mode_picker_on_a_tui_launcher_carries_the_not_live_note(self) -> None:
        self.sessions.set_launcher_pick(THREAD, "tui")
        self.run_cmd("/mode")
        self.assertIn(NOT_LIVE_NOTE, self.last())

    def test_model_during_a_turn_on_a_tui_launcher_says_next_message(self) -> None:
        self.sessions.set_launcher_pick(THREAD, "tui")
        self.start_turn()
        self.run_cmd("/model opus")
        self.assertEqual(self.last(), "🧠 model set to opus — applies from your next message")
        self.assertEqual(self.row().model, "opus")


if __name__ == "__main__":
    unittest.main()
