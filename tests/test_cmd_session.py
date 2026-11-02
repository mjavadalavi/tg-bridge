"""Session commands (STEP 47): /stop, /new (alias /clear) and /resume, ported from fakecli/test_commands.py (sections
E, J, K) and fakecli/test_launcher_cwd.py (sections D, E, J).

Scenarios: /stop with no turn says so, with a live turn stops its session and marks the stop as the owner's; the stop
button path and /stop share SessionCommands.stop_turn; /new moves the id to prev_uuid (a second /new keeps it), /clear is
the same command, and /new is refused while a turn runs and offers the stop-and-new button; stop-and-new waits for the
turn to end (or says it is still stopping) and edits the same message; /resume rejects a non-uuid, binds a uuid
(upper case too), refuses a running turn, shows the picker across both backends with the ⚠ launcher mark; the
folder of a session becomes the cwd when it exists, a missing folder is said aloud and a session without a folder
leaves the cwd alone; a session of a launcher sharing the folder keeps the launcher; another Claude folder gets a copy
of the transcript; a missing transcript or a Codex session switches the launcher pick (and the backend) to the one
that made it; a launcher that is not in config refuses the resume; the resume buttons follow the same rules.
Every test runs against a migrated database in a TempHome, with a FakeClock and FakeTelegram; no subprocess starts.
"""

import json
import os
import threading
import unittest

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.linger_stub import StubKeptLinger
from tests.support import TempHome
from tests.turn_support import wait_until
from tg_bridge.agents.claude.history import encoded_cwd
from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.bot.commands.base import CommandDeps, CommandRegistry
from tg_bridge.bot.commands.session import (
    BOUND_ELSEWHERE_TEXT,
    LINGER_STOPPED_TEXT,
    NEW_BUSY_TEXT,
    NEW_DONE_TEXT,
    NEW_TEXT,
    NOT_IDLE_TAP_TEXT,
    NOT_IDLE_TEXT,
    NOTHING_RUNNING_TEXT,
    RESUME_USAGE_TEXT,
    STILL_STOPPING_TEXT,
    STOP_NEW_TOAST,
    SessionCommands,
    StopKind,
)
from tg_bridge.bot.pickers import STOP_CHOICE_TEXT, started_label
from tg_bridge.clock import utc_stamp
from tg_bridge.config import Config
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.linger import LingerStatus
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.topic_queues import TopicQueues
from tg_bridge.core.workers import WorkerPool
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.db import Database
from tg_bridge.store.kv import KvRepo
from tg_bridge.store.rules import RuleRepo
from tg_bridge.store.sessions import HistoryRepo, SessionRepo
from tg_bridge.store.turns import TurnRepo
from tg_bridge.telegram.messenger import Messenger

THREAD = 9470
CHAT = -1009470
MID = 777
TOPIC = "SX"
CLAUDE_CMD = "wrap run"
SHARED_CMD = "shared-cmd"
OTHER_CMD = "plain run"
HOME_CMD = "home run"
RAW_CMD = "raw run"
CODEX_CMD = "codex"
GONE_CMD = "retired-cmd"
U1 = "1f2e3d4c-aaaa-4bbb-8ccc-000000000001"
U2 = "1f2e3d4c-aaaa-4bbb-8ccc-000000000002"
U3 = "1f2e3d4c-aaaa-4bbb-8ccc-000000000003"
U4 = "1f2e3d4c-aaaa-4bbb-8ccc-000000000004"
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "claude_transcripts",
                       "handoff_session.jsonl")


class StubSession:
    """Stands in for an AgentSession: stop() is recorded and runs an optional hook (a test ends the turn there)."""

    def __init__(self) -> None:
        self.stops = 0
        self.stopped = threading.Event()
        self.on_stop = None

    def stop(self, grace: float = 5.0) -> None:
        self.stops += 1
        if self.on_stop is not None:  # the hook runs first, so 'stopped' means the turn has already ended
            self.on_stop()
        self.stopped.set()

    def halt_turn(self) -> None:  # the ⏹ and Now taps end the turn with halt_turn, which is stop() for this stand-in
        self.stop()


class EndingClock(FakeClock):
    """A FakeClock whose sleep() runs a hook: the test ends the turn while finish_new waits."""

    def __init__(self) -> None:
        super().__init__()
        self.on_tick = None

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        if self.on_tick is not None:
            self.on_tick()


def config_data(project: str, codex_home: str) -> dict:
    return {
        "launchers": {
            "wrap": {"cmd": CLAUDE_CMD, "config_dir": os.path.join(project, "cfg-a")},
            "plain": {"cmd": OTHER_CMD, "config_dir": os.path.join(project, "cfg-b")},
            "shared": {"cmd": SHARED_CMD, "config_dir": os.path.join(project, "cfg-a")},
        },
        "agent_launchers": {"codex": {"backend": "codex", "command": CODEX_CMD, "config_dir": codex_home}},
        "topics": {TOPIC: {"launcher": CLAUDE_CMD, "cwd": project}},
    }


class SessionCase(unittest.TestCase):
    """The session commands wired to real repositories, a real WorkerPool and FakeTelegram under a TempHome."""

    clock_class = FakeClock

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.base = os.path.realpath(self.home.home)
        self.project = os.path.join(self.base, "project")
        os.makedirs(self.project, exist_ok=True)
        self.clock = self.clock_class()
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.config = Config(config_data(self.project, os.path.join(self.base, "codex")), home=self.base, env={})
        self.backends = build_default_registry(self.config, self.logger, self.clock, title_launcher="")
        self.db = Database(Paths.from_home(self.base))
        self.addCleanup(self.db.close)
        self.sessions = SessionRepo(self.db, self.clock)
        self.history = HistoryRepo(self.db, self.clock)
        self.live = LiveRegistry()
        self.telegram = FakeTelegram(self.clock)
        self.messenger = Messenger(self.telegram, self.logger, self.clock)
        self.out = WorkerPool("out", 1, self.logger)
        self.commands = self.build_commands(self.clock)
        self.registry = CommandRegistry()
        self.commands.register(self.registry)
        self.deps = CommandDeps(
            config=self.config, sessions=self.sessions, turns=TurnRepo(self.db), rules=RuleRepo(self.db, self.clock),
            kv=KvRepo(self.db), live=self.live, queues=TopicQueues(), backends=self.backends,
            messenger=self.messenger, out=self.out,
        )
        self.sessions.ensure(THREAD, TOPIC, CLAUDE_CMD, self.project, "bypassPermissions")

    def build_commands(self, clock, stop_wait: float = 30.0) -> SessionCommands:
        return SessionCommands(config=self.config, sessions=self.sessions, history=self.history, live=self.live,
                               backends=self.backends, messenger=self.messenger, clock=clock, logger=self.logger,
                               handoff=HandoffService(KvRepo(self.db), self.backends, self.logger),
                               stop_wait=stop_wait)

    def run_cmd(self, text: str) -> bool:
        handled = self.registry.dispatch(self.deps, CHAT, THREAD, TOPIC, text)
        self.assertTrue(self.out.join(5), "the replies did not go out")
        return handled

    def last(self) -> str:
        return self.telegram.texts(THREAD)[-1]

    def button_callbacks(self) -> list[str]:
        return [b["callback_data"] for keyboard in self.telegram.buttons(THREAD) for row in keyboard for b in row]

    def row(self):
        return self.sessions.get(THREAD)

    def start_turn(self, session: StubSession | None = None) -> StubSession:
        session = session or StubSession()
        self.feed = object()
        self.assertTrue(self.live.begin_turn(THREAD, session, self.feed))
        return session

    def end_turn(self, session: StubSession) -> None:
        self.live.end_turn(THREAD, session, self.feed)

    def note(self, sid: str, launcher: str, backend: str = "claude", cwd: str | None = None,
             text: str = "earlier question") -> None:
        self.history.note_session(THREAD, sid, text, cwd or self.project, launcher, backend)


class StopTests(SessionCase):
    def test_stop_with_no_turn_says_so(self) -> None:
        """Labels: J: /stop with nothing running says so."""
        self.assertTrue(self.run_cmd("/stop"))
        self.assertEqual(self.last(), NOTHING_RUNNING_TEXT)

    def test_stop_with_a_live_turn_stops_its_session_and_marks_the_owner_stop(self) -> None:
        """Labels: J: /stop stops the running turn; J: the stop is marked as the owner stop."""
        session = self.start_turn()
        self.assertTrue(self.run_cmd("/stop@MyBot"))
        self.assertTrue(session.stopped.wait(2), "/stop did not stop the session")
        flags = self.live.consume_abort(THREAD)
        self.assertTrue(flags.aborted)
        self.assertTrue(flags.stop_tap)
        self.assertEqual(self.telegram.texts(THREAD), [], "a successful stop sends no reply")

    def test_stop_turn_is_the_one_stop_path(self) -> None:
        """Labels: the st button and /stop call the same stop_turn method."""
        self.assertFalse(self.commands.stop_turn(THREAD))
        session = self.start_turn()
        self.assertTrue(self.commands.stop_turn(THREAD))
        self.assertTrue(session.stopped.wait(2))

    def test_stop_keeps_the_session_row(self) -> None:
        self.sessions.set_uuid(THREAD, U1)
        self.start_turn()
        self.run_cmd("/stop")
        self.assertEqual(self.row().uuid, U1)


class NewTests(SessionCase):
    def test_new_moves_the_uuid_to_prev_and_a_second_new_keeps_prev(self) -> None:
        """Labels: E: /new moves the uuid to prev_uuid; E: a second /new keeps prev_uuid."""
        self.sessions.set_uuid(THREAD, U1)
        self.run_cmd("/new")
        self.assertEqual((self.row().uuid, self.row().prev_uuid), ("", U1))
        self.assertEqual(self.last(), NEW_TEXT)
        self.run_cmd("/new")
        self.assertEqual(self.row().prev_uuid, U1)

    def test_clear_is_the_same_command_as_new(self) -> None:
        self.sessions.set_uuid(THREAD, U1)
        self.assertTrue(self.run_cmd("/clear"))
        self.assertEqual((self.row().uuid, self.row().prev_uuid), ("", U1))
        self.assertEqual(self.last(), NEW_TEXT)

    def test_new_while_a_turn_runs_is_refused_with_the_stop_and_new_button(self) -> None:
        """Labels: K: /new is refused while a turn runs."""
        self.sessions.set_uuid(THREAD, U1)
        self.start_turn()
        self.run_cmd("/new")
        self.assertEqual(self.last(), NEW_BUSY_TEXT)
        self.assertIn(f"cm:{THREAD}:nw:0", self.button_callbacks())
        self.assertEqual(self.row().uuid, U1)

    def test_stop_and_new_edits_the_message_once_the_stopped_turn_has_ended(self) -> None:
        """Labels: stop-and-new: the message says the new session started once the turn has ended."""
        clock = EndingClock()
        commands = self.build_commands(clock)
        session = self.start_turn()
        self.sessions.set_uuid(THREAD, U1)
        session.on_stop = lambda: self.end_turn(session)  # the stop ends the turn, as the real process does
        self.assertEqual(commands.stop_and_new(THREAD, CHAT, MID), STOP_NEW_TOAST)
        self.assertTrue(session.stopped.wait(2))
        self.assertEqual(commands.finish_new(THREAD, CHAT, MID), NEW_DONE_TEXT)
        self.assertEqual(self.telegram.edits(MID)[-1], NEW_DONE_TEXT)
        self.assertEqual((self.row().uuid, self.row().prev_uuid), ("", U1))

    def test_finish_new_sees_the_turn_end_during_its_wait(self) -> None:
        clock = EndingClock()
        commands = self.build_commands(clock)
        session = self.start_turn()
        self.sessions.set_uuid(THREAD, U1)
        clock.on_tick = lambda: self.end_turn(session)
        self.assertEqual(commands.finish_new(THREAD, CHAT, MID), NEW_DONE_TEXT)
        self.assertEqual(self.row().uuid, "")

    def test_finish_new_says_still_stopping_when_the_turn_outlives_the_wait(self) -> None:
        clock = FakeClock()
        commands = self.build_commands(clock, stop_wait=2.0)
        self.start_turn()
        self.sessions.set_uuid(THREAD, U1)
        self.assertEqual(commands.finish_new(THREAD, CHAT, MID), STILL_STOPPING_TEXT.format(wait=2))
        self.assertEqual(self.row().uuid, U1, "a turn still running keeps the session")

    def test_a_refused_edit_is_logged_and_the_session_still_starts_fresh(self) -> None:
        self.telegram.fail_next("editMessageText", code=400)
        self.sessions.set_uuid(THREAD, U1)
        self.assertEqual(self.commands.finish_new(THREAD, CHAT, MID), NEW_DONE_TEXT)
        self.assertEqual(self.row().uuid, "")


class ResumeCommandTests(SessionCase):
    def test_resume_rejects_a_non_uuid(self) -> None:
        """Labels: E: /resume rejects a non-uuid."""
        self.run_cmd("/resume not-a-uuid")
        self.assertEqual(self.last(), "usage: /resume <session uuid>")
        self.assertEqual(self.row().uuid, "")

    def test_resume_binds_the_uuid_and_moves_the_old_one_to_prev(self) -> None:
        """Labels: E: /resume binds the topic to that uuid; E: /resume moves the old uuid to prev_uuid.

        Ported labels:
        Q5: typed /resume <uuid> still binds and replies as before
        """
        self.note(U2, CLAUDE_CMD)
        self.sessions.set_uuid(THREAD, U1)
        self.run_cmd(f"/resume {U2}")
        self.assertEqual((self.row().uuid, self.row().prev_uuid), (U2, U1))
        self.assertEqual(self.last(), f"↩️ the next message resumes session {U2}")
        self.note(U3, CLAUDE_CMD)
        self.run_cmd(f"/resume {U3.upper()}")
        self.assertEqual((self.row().uuid, self.row().prev_uuid), (U3, U2))

    def test_resume_with_no_history_and_no_argument_shows_usage(self) -> None:
        """Ported labels:
        Q6: no history says so
        """
        self.run_cmd("/resume")
        self.assertEqual(self.last(), RESUME_USAGE_TEXT)

    def test_resume_refused_while_a_turn_runs(self) -> None:
        self.start_turn()
        self.note(U2, CLAUDE_CMD)
        self.run_cmd(f"/resume {U2}")
        self.assertEqual(self.last(), NOT_IDLE_TEXT)
        self.assertEqual(self.row().uuid, "")

    def test_no_argument_shows_the_picker_with_the_session_in_use_marked(self) -> None:
        """Ported labels:
        Q: /resume shows the picker (↩️ the next message resumes the ✓ session)
        """
        self.note(U1, CLAUDE_CMD, text="first question")
        self.note(U2, CLAUDE_CMD, text="second question")
        self.sessions.set_uuid(THREAD, U2)
        self.run_cmd("/resume")
        self.assertIn("✓ ", " ".join(b["text"] for kb in self.telegram.buttons(THREAD) for row in kb for b in row))
        newest = self.history.list_for(THREAD, "claude")[0].id
        self.assertIn(f"cm:{THREAD}:rs:{newest}", self.button_callbacks())
        self.assertIn("↩️ the next message resumes the ✓ session", self.telegram.texts(THREAD)[-1])


class ResumeFolderTests(SessionCase):
    """The folder a session ran in becomes the topic's cwd, and what happens when it is gone."""

    def setUp(self) -> None:
        super().setUp()
        self.dir_a = os.path.join(self.base, "j-a")
        self.dir_gone = os.path.join(self.base, "j-gone")
        os.makedirs(self.dir_a, exist_ok=True)

    def test_a_session_folder_becomes_the_cwd_and_the_reply_names_it(self) -> None:
        """Labels: J2: /resume <uuid> returns the topic to the session folder.

        Ported labels:
        J2: the reply names the folder
        """
        self.note(U1, CLAUDE_CMD, cwd=self.dir_a)
        self.run_cmd(f"/resume {U1}")
        self.assertEqual(self.row().cwd, os.path.realpath(self.dir_a))
        self.assertEqual(self.last(), f"↩️ the next message resumes session {U1} in ~/j-a")

    def test_a_missing_folder_keeps_the_cwd_and_says_so(self) -> None:
        """Labels: J4: a missing folder keeps the current cwd; the uuid is still set."""
        self.note(U2, CLAUDE_CMD, cwd=self.dir_gone)
        self.run_cmd(f"/resume {U2}")
        self.assertEqual(self.row().cwd, self.project)
        self.assertEqual(self.row().uuid, U2)
        self.assertIn("no longer exists, so it runs in the current one", self.last())

    def test_the_resume_button_says_a_missing_folder_is_gone(self) -> None:
        """Ported labels:
        J4: the toast says the folder is gone
        """
        self.note(U2, CLAUDE_CMD, cwd=self.dir_gone)
        hid = self.history.find(THREAD, U2).id
        outcome = self.commands.resume_tap(THREAD, hid)
        self.assertEqual(outcome.toast, "↩️ the next message resumes it — its folder ~/j-gone no longer exists, "
                                        "so it runs in the current one")
        self.assertEqual(self.row().cwd, self.project)
        self.assertEqual(self.row().uuid, U2)

    def test_a_session_in_the_current_folder_gets_a_plain_reply(self) -> None:
        """Labels: J5: same folder, plain reply."""
        self.note(U1, CLAUDE_CMD, cwd=self.project)
        self.run_cmd(f"/resume {U1}")
        self.assertEqual(self.last(), f"↩️ the next message resumes session {U1}")

    def test_a_session_without_a_recorded_folder_leaves_the_cwd_alone(self) -> None:
        """Labels: J6: a session without a recorded folder leaves the cwd alone."""
        self.db.execute("INSERT INTO session_history(thread_id, uuid, started) VALUES(?,?,?)",
                        (THREAD, U4, "2026-10-01T00:00:00"))
        self.run_cmd(f"/resume {U4}")
        self.assertEqual(self.row().cwd, self.project)
        self.assertEqual(self.row().uuid, U4)

    def test_the_resume_button_returns_to_the_folder_and_shows_the_picker(self) -> None:
        """Labels: J3: a /resume button tap returns the topic to its folder.

        Ported labels:
        J3: the toast names the folder
        """
        self.note(U1, CLAUDE_CMD, cwd=self.dir_a)
        hid = self.history.find(THREAD, U1).id
        outcome = self.commands.resume_tap(THREAD, hid)
        self.assertEqual(outcome.toast, "↩️ the next message resumes it in ~/j-a")
        self.assertIsNotNone(outcome.picker)
        self.assertEqual(self.row().uuid, U1)
        self.assertEqual(self.row().cwd, os.path.realpath(self.dir_a))

    def test_the_resume_button_refuses_a_running_turn_and_an_unknown_row(self) -> None:
        self.note(U1, CLAUDE_CMD)
        hid = self.history.find(THREAD, U1).id
        self.start_turn()
        self.assertEqual(self.commands.resume_tap(THREAD, hid).toast, NOT_IDLE_TAP_TEXT)
        self.assertEqual(self.commands.resume_tap(THREAD, 999999).toast, "not in the list anymore")
        self.assertEqual(self.row().uuid, "")


class ResumeLauncherTests(SessionCase):
    """A session made by another launcher: the current launcher is kept, its transcript is copied, or the pick switches."""

    def test_the_copied_session_is_recorded_for_the_current_launcher_and_the_old_id_moves_to_prev(self) -> None:
        """Ported labels:
        9d: session_history now records the current launcher
        9e: the previous uuid moved to prev_uuid as usual
        """
        cfg_b = os.path.join(self.project, "cfg-b")
        folder = os.path.join(cfg_b, "projects", encoded_cwd(os.path.realpath(self.project)))
        os.makedirs(folder, exist_ok=True)
        with open(FIXTURE, encoding="utf-8") as handle, open(os.path.join(folder, U1 + ".jsonl"), "w",
                                                             encoding="utf-8") as out:
            out.write(handle.read())
        self.sessions.set_uuid(THREAD, U2)
        self.note(U1, OTHER_CMD)
        self.run_cmd(f"/resume {U1}")
        self.assertEqual(self.history.find(THREAD, U1).launcher, CLAUDE_CMD)
        self.assertEqual((self.row().uuid, self.row().prev_uuid), (U1, U2))

    def test_a_launcher_sharing_the_folder_keeps_the_current_launcher(self) -> None:
        """Ported labels:
        9f: a launcher on the same folder keeps the pick (' — lb shares its history folder with la')
        """
        self.note(U1, SHARED_CMD)
        self.run_cmd(f"/resume {U1}")
        self.assertEqual(self.last(), f"↩️ the next message resumes session {U1} — wrap shares its history "
                                      "folder with shared")
        self.assertEqual(self.row().launcher_pick, "")
        self.assertEqual(self.history.find(THREAD, U1).launcher, CLAUDE_CMD, "the row is adopted by the launcher")

    def test_another_claude_folder_gets_a_copy_of_the_transcript(self) -> None:
        """Ported labels:
        9a: the pick stays on the current launcher
        9b: the reply says the history was copied (' — history copied from launcher la into lb')
        9c: the transcript is copied into the current launcher folder
        """
        cfg_b = os.path.join(self.project, "cfg-b")
        folder = os.path.join(cfg_b, "projects", encoded_cwd(os.path.realpath(self.project)))
        os.makedirs(folder, exist_ok=True)
        with open(FIXTURE, encoding="utf-8") as handle, open(os.path.join(folder, U1 + ".jsonl"), "w",
                                                             encoding="utf-8") as out:
            out.write(handle.read())
        self.note(U1, OTHER_CMD)
        self.run_cmd(f"/resume {U1}")
        self.assertEqual(self.last(), f"↩️ the next message resumes session {U1} — history copied from launcher "
                                      "plain into wrap")
        copied = os.path.join(self.project, "cfg-a", "projects", encoded_cwd(os.path.realpath(self.project)), U1 + ".jsonl")
        self.assertTrue(os.path.exists(copied))
        self.assertEqual(self.row().launcher_pick, "", "a copy keeps the current launcher")

    def test_a_missing_transcript_switches_the_pick_to_the_launcher_that_made_it(self) -> None:
        """Ported labels:
        9g: no source transcript keeps the old switch (' — launcher switched to la, which made this session')
        """
        self.note(U2, OTHER_CMD)
        self.run_cmd(f"/resume {U2}")
        self.assertEqual(self.last(), f"↩️ the next message resumes session {U2} — launcher switched to plain, "
                                      "which made this session")
        self.assertEqual(self.row().launcher_pick, "plain")
        self.assertEqual(self.row().backend, "claude")

    def test_a_codex_session_switches_the_pick_and_the_backend(self) -> None:
        self.note(U3, CODEX_CMD, backend="codex")
        self.run_cmd(f"/resume {U3}")
        self.assertEqual(self.last(), f"↩️ the next message resumes session {U3} — launcher switched to codex, "
                                      "which made this session")
        self.assertEqual((self.row().launcher_pick, self.row().backend), ("codex", "codex"))

    def test_a_topic_on_codex_resumes_a_claude_session_by_switching_back(self) -> None:
        self.sessions.set_launcher_pick(THREAD, "codex")
        self.sessions.set_backend(THREAD, "codex")
        self.note(U4, CLAUDE_CMD, backend="claude")
        self.run_cmd(f"/resume {U4}")
        self.assertEqual((self.row().launcher_pick, self.row().backend), ("wrap", "claude"))
        self.assertIn("launcher switched to wrap", self.last())

    def test_a_session_of_a_launcher_not_in_config_is_refused(self) -> None:
        """Ported labels:
        9h: a session of a launcher not in config is refused as before
        """
        self.note(U1, GONE_CMD)
        self.run_cmd(f"/resume {U1}")
        self.assertEqual(self.last(), f"session {U1} belongs to launcher {GONE_CMD}, which is not in config — it cannot "
                                      "run here")
        self.assertEqual(self.row().uuid, "", "a refused resume changes nothing")

    def reconfigure(self, data: dict) -> None:
        """Rebuilds the config, the backends and the commands on new config data (the wiring of setUp)."""
        self.config = Config(data, home=self.base, env={})
        self.backends = build_default_registry(self.config, self.logger, self.clock, title_launcher="")
        self.commands = self.build_commands(self.clock)

    def test_an_unnamed_launcher_sharing_the_claude_folder_keeps_its_own_session(self) -> None:
        """The topic runs a command that is not in config (an unnamed launcher on ~/.claude), and the named launcher
        'home' uses that same folder. The transcript is in that folder, so resuming it switches nothing."""
        data = config_data(self.project, os.path.join(self.base, "codex"))
        data["launchers"]["home"] = {"cmd": HOME_CMD, "config_dir": os.path.join(self.base, ".claude")}
        data["topics"][TOPIC]["launcher"] = RAW_CMD
        self.reconfigure(data)
        folder = os.path.join(self.base, ".claude", "projects", encoded_cwd(os.path.realpath(self.project)))
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, U1 + ".jsonl"), "w", encoding="utf-8") as out:
            out.write(json.dumps({"type": "ai-title", "aiTitle": "own"}) + "\n")
        self.note(U1, RAW_CMD)
        self.assertEqual(self.commands.resume_session(THREAD, U1), (True, ""))
        self.assertEqual((self.row().uuid, self.row().launcher_pick), (U1, ""),
                         "the topic keeps its unnamed launcher, not 'home'")

    def test_a_transcript_in_the_current_folder_is_not_switched_by_the_history_row(self) -> None:
        """The current launcher's own folder holds the transcript while the history row names launcher 'plain'. Nothing
        is copied or switched: the topic is bound to the session under its current launcher, and the row follows it."""
        folder = os.path.join(self.project, "cfg-a", "projects", encoded_cwd(os.path.realpath(self.project)))
        os.makedirs(folder, exist_ok=True)
        with open(FIXTURE, encoding="utf-8") as handle, open(os.path.join(folder, U1 + ".jsonl"), "w",
                                                             encoding="utf-8") as out:
            out.write(handle.read())
        self.note(U1, OTHER_CMD)
        self.run_cmd(f"/resume {U1}")
        self.assertEqual(self.last(), f"↩️ the next message resumes session {U1}")
        self.assertEqual((self.row().uuid, self.row().launcher_pick, self.row().backend), (U1, "", "claude"))
        self.assertEqual(self.history.find(THREAD, U1).launcher, CLAUDE_CMD, "the row is re-tagged to the current launcher")

    def test_a_held_resume_without_a_history_row_creates_no_row(self) -> None:
        """The current folder holds the transcript but no history row exists: the re-tag only re-labels an existing row,
        so nothing is inserted into the history."""
        folder = os.path.join(self.project, "cfg-a", "projects", encoded_cwd(os.path.realpath(self.project)))
        os.makedirs(folder, exist_ok=True)
        with open(FIXTURE, encoding="utf-8") as handle, open(os.path.join(folder, U1 + ".jsonl"), "w",
                                                             encoding="utf-8") as out:
            out.write(handle.read())
        self.run_cmd(f"/resume {U1}")
        self.assertEqual(self.row().uuid, U1)
        self.assertIsNone(self.history.find(THREAD, U1))

    def test_the_picker_lists_both_backends_newest_first_with_the_launcher_mark(self) -> None:
        self.note(U1, CLAUDE_CMD, text="claude one")
        self.note(U2, CODEX_CMD, backend="codex", text="codex two")
        self.run_cmd("/resume")
        labels = [b["text"] for kb in self.telegram.buttons(THREAD) for row in kb for b in row]
        self.assertEqual(len(labels), 2)
        self.assertIn("codex two", labels[0], "newest first")
        self.assertIn("⚠ codex · ", labels[0])
        self.assertNotIn("⚠", labels[1])


class DiskSessionsTests(SessionCase):
    """Sessions the CLI left on disk for the topic's folder show in /resume, tagged with their launcher."""

    def write_transcript(self, cfg: str, sid: str, mtime: float, title: str = "") -> str:
        """A transcript under a config folder for the topic's cwd, with its modification time set. Returns its path."""
        folder = os.path.join(cfg, "projects", encoded_cwd(os.path.realpath(self.project)))
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, sid + ".jsonl")
        with open(path, "w", encoding="utf-8") as out:
            out.write(json.dumps({"type": "ai-title", "aiTitle": title}) + "\n")
        os.utime(path, (mtime, mtime))
        return path

    def labels(self) -> list[str]:
        return [b["text"] for keyboard in self.telegram.buttons(THREAD) for row in keyboard for b in row]

    def cfg_a(self) -> str:
        return os.path.join(self.project, "cfg-a")

    def cfg_b(self) -> str:
        return os.path.join(self.project, "cfg-b")

    def test_a_transcript_on_disk_under_the_current_launcher_is_listed_once(self) -> None:
        """A transcript in the folder of 'wrap' (and 'shared', which uses the same folder) is listed once, under 'wrap'."""
        stamp = self.clock.now() + 100
        self.write_transcript(self.cfg_a(), U1, stamp, "fix login")
        self.run_cmd("/resume")
        self.assertEqual(self.labels(), [f"wrap · {started_label(utc_stamp(stamp))} · fix login…"])
        self.assertEqual(self.button_callbacks(), [f"cm:{THREAD}:rs:{self.history.find(THREAD, U1).id}"])
        self.assertEqual(self.history.find(THREAD, U1).cwd, os.path.realpath(self.project))

    def test_a_shared_folder_is_attributed_to_the_launcher_in_use(self) -> None:
        self.sessions.set_launcher_pick(THREAD, "shared")
        self.write_transcript(self.cfg_a(), U1, self.clock.now() + 100, "fix login")
        picker = self.commands.resume_picker(THREAD)
        self.assertEqual(picker.rows[0][0][0], f"shared · {started_label(utc_stamp(self.clock.now() + 100))} · fix login…")
        self.assertEqual(self.history.find(THREAD, U1).launcher, SHARED_CMD)

    def test_a_transcript_under_another_launcher_is_tagged_and_tapping_copies_it(self) -> None:
        """A transcript under cfg-b shows with ⚠ and 'plain'; tapping it copies the transcript into cfg-a and binds it."""
        stamp = self.clock.now() + 100
        with open(FIXTURE, encoding="utf-8") as handle:
            text = handle.read()
        cfg_b_file = self.write_transcript(self.cfg_b(), U1, stamp)
        with open(cfg_b_file, "w", encoding="utf-8") as out:
            out.write(text)
        os.utime(cfg_b_file, (stamp, stamp))
        self.run_cmd("/resume")
        self.assertEqual(self.labels(), [f"⚠ plain · {started_label(utc_stamp(stamp))} · Login fix…"])
        outcome = self.commands.resume_tap(THREAD, self.history.find(THREAD, U1).id)
        self.assertIn("history copied from launcher plain into wrap", outcome.toast)
        copied = os.path.join(self.cfg_a(), "projects", encoded_cwd(os.path.realpath(self.project)), U1 + ".jsonl")
        self.assertTrue(os.path.exists(copied), "the transcript was not copied into the current launcher's folder")
        self.assertEqual((self.row().uuid, self.history.find(THREAD, U1).launcher), (U1, CLAUDE_CMD))

    def test_a_session_already_in_history_is_not_duplicated(self) -> None:
        """The history row keeps its text and start; the disk file adds no second row."""
        self.note(U1, CLAUDE_CMD, text="from history")
        self.write_transcript(self.cfg_a(), U1, self.clock.now() + 500, "disk title")
        self.run_cmd("/resume")
        rows = [r for r in self.history.list_for(THREAD, "claude", limit=None) if r.uuid == U1]
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].first_text, rows[0].started), ("from history", utc_stamp(self.clock.now())))
        self.assertEqual(len(self.button_callbacks()), 1)

    def test_disk_sessions_sort_by_start_time_among_history_rows(self) -> None:
        """Newest first by start time: a disk session newer than the history row leads, an older one comes last."""
        now = self.clock.now()
        self.note(U2, CLAUDE_CMD, text="history two")
        self.write_transcript(self.cfg_a(), U1, now - 3600, "older on disk")
        self.write_transcript(self.cfg_a(), U3, now + 3600, "newer on disk")
        self.run_cmd("/resume")
        expected = [self.history.find(THREAD, uuid).id for uuid in (U3, U2, U1)]
        self.assertEqual(self.button_callbacks(), [f"cm:{THREAD}:rs:{rid}" for rid in expected])

    def test_at_most_ten_disk_sessions_are_listed_newest_first(self) -> None:
        sids = [f"1f2e3d4c-aaaa-4bbb-8ccc-0000000001{n:02d}" for n in range(12)]
        for n, sid in enumerate(sids):
            self.write_transcript(self.cfg_a(), sid, self.clock.now() + n * 10, f"disk {n}")
        picker = self.commands.resume_picker(THREAD)
        self.assertEqual(len(picker.rows), 10)
        self.assertTrue(picker.rows[0][0][0].endswith("· disk 11…"))
        self.assertIsNone(self.history.find(THREAD, sids[0]), "the two oldest are beyond the limit")

    def test_a_session_bound_to_another_topic_is_not_offered(self) -> None:
        """Another topic is bound to U1 right now: this topic must not list it, while its own U2 still shows."""
        other = THREAD + 1
        self.sessions.ensure(other, "other", CLAUDE_CMD, self.project)
        self.sessions.set_uuid(other, U1)
        self.write_transcript(self.cfg_a(), U1, self.clock.now() + 200, "bound elsewhere")
        self.write_transcript(self.cfg_a(), U2, self.clock.now() + 100, "free one")
        self.commands.resume_picker(THREAD)
        self.assertIsNone(self.history.find(THREAD, U1), "a session another topic is bound to was offered")
        self.assertIsNotNone(self.history.find(THREAD, U2))

    def test_this_topics_own_session_is_still_offered(self) -> None:
        self.sessions.set_uuid(THREAD, U1)
        self.write_transcript(self.cfg_a(), U1, self.clock.now() + 100, "my own")
        self.commands.resume_picker(THREAD)
        self.assertIsNotNone(self.history.find(THREAD, U1))

    def test_no_topic_row_means_no_disk_sessions_and_no_crash(self) -> None:
        self.write_transcript(self.cfg_a(), U1, self.clock.now() + 100, "fix login")
        self.db.execute("DELETE FROM sessions WHERE thread_id=?", (THREAD,))
        self.assertEqual(self.commands.resume_picker(THREAD).rows, [])
        self.assertIsNone(self.history.find(THREAD, U1))


class DiskFixesTests(SessionCase):
    """The /resume fixes: the launcher follows the transcript file, a session bound to another topic is neither listed nor
    resumed, the picker is ordered by a listed file's mtime, and the disk listing skips the bound ids before its limit."""

    def write_transcript(self, cfg: str, sid: str, mtime: float, title: str = "") -> str:
        """A transcript under a config folder for the topic's cwd, with its modification time set. Returns its path."""
        folder = os.path.join(cfg, "projects", encoded_cwd(os.path.realpath(self.project)))
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, sid + ".jsonl")
        with open(path, "w", encoding="utf-8") as out:
            out.write(json.dumps({"type": "ai-title", "aiTitle": title}) + "\n")
        os.utime(path, (mtime, mtime))
        return path

    def write_fixture(self, cfg: str, sid: str) -> None:
        """Copies the handoff fixture transcript into a config folder for the topic's cwd, as session sid."""
        folder = os.path.join(cfg, "projects", encoded_cwd(os.path.realpath(self.project)))
        os.makedirs(folder, exist_ok=True)
        with open(FIXTURE, encoding="utf-8") as handle, open(os.path.join(folder, sid + ".jsonl"), "w",
                                                             encoding="utf-8") as out:
            out.write(handle.read())

    def cfg_a(self) -> str:
        return os.path.join(self.project, "cfg-a")

    def cfg_b(self) -> str:
        return os.path.join(self.project, "cfg-b")

    def bind_elsewhere(self, sid: str, thread: int = THREAD + 1) -> None:
        """Another topic is bound to sid right now."""
        self.sessions.ensure(thread, f"other {thread}", CLAUDE_CMD, self.project)
        self.sessions.set_uuid(thread, sid)

    def labels(self) -> list[str]:
        return [b["text"] for keyboard in self.telegram.buttons(THREAD) for row in keyboard for b in row]

    def test_a_blank_launcher_row_with_its_transcript_only_in_another_folder_is_copied_on_resume(self) -> None:
        """An older history row has no launcher, and the transcript exists only under cfg-b: the resume looks in the
        other folders, copies the transcript into the current one and binds the topic to it."""
        self.write_fixture(self.cfg_b(), U1)
        self.note(U1, "")
        self.run_cmd(f"/resume {U1}")
        self.assertEqual(self.last(), f"↩️ the next message resumes session {U1} — history copied from launcher plain "
                                      "into wrap")
        copied = os.path.join(self.cfg_a(), "projects", encoded_cwd(os.path.realpath(self.project)), U1 + ".jsonl")
        self.assertTrue(os.path.exists(copied), "the transcript was not copied into the current launcher's folder")
        self.assertEqual((self.row().uuid, self.history.find(THREAD, U1).launcher), (U1, CLAUDE_CMD))

    def test_a_transcript_in_the_current_folder_is_not_copied_from_elsewhere(self) -> None:
        """The current launcher's own folder holds the transcript: nothing is copied or switched, even when another folder
        holds a copy too."""
        self.write_fixture(self.cfg_a(), U1)
        self.write_fixture(self.cfg_b(), U1)
        self.note(U1, "")
        self.run_cmd(f"/resume {U1}")
        self.assertEqual(self.last(), f"↩️ the next message resumes session {U1}")
        self.assertEqual(self.row().launcher_pick, "")

    def test_a_session_bound_to_another_topic_is_refused_by_uuid(self) -> None:
        self.bind_elsewhere(U1)
        self.note(U1, CLAUDE_CMD)
        self.run_cmd(f"/resume {U1}")
        self.assertEqual(self.last(), BOUND_ELSEWHERE_TEXT.format(sid=U1))
        self.assertEqual(self.row().uuid, "", "the refused session was bound anyway")

    def test_a_history_row_of_a_session_bound_to_another_topic_is_not_listed(self) -> None:
        self.bind_elsewhere(U1)
        self.note(U1, CLAUDE_CMD, text="taken elsewhere")
        self.note(U2, CLAUDE_CMD, text="free one")
        self.run_cmd("/resume")
        self.assertEqual(self.button_callbacks(), [f"cm:{THREAD}:rs:{self.history.find(THREAD, U2).id}"])

    def test_this_topics_own_current_session_is_listed_even_when_another_topic_holds_it(self) -> None:
        self.sessions.set_uuid(THREAD, U1)
        self.bind_elsewhere(U1)
        self.note(U1, CLAUDE_CMD, text="my own")
        self.run_cmd("/resume")
        self.assertEqual(self.button_callbacks(), [f"cm:{THREAD}:rs:{self.history.find(THREAD, U1).id}"])
        self.assertTrue(self.labels()[0].startswith("✓ "), self.labels())

    def test_a_recently_modified_disk_session_moves_to_the_top(self) -> None:
        """U1 started first and U2 started later (so U2 leads by start time); U1's file was modified after both, so the
        listing puts U1 first. The row still shows its own start time."""
        now = self.clock.now()
        self.note(U1, CLAUDE_CMD, text="older start")
        self.note(U2, CLAUDE_CMD, text="newer start")
        self.write_transcript(self.cfg_a(), U1, now + 500, "touched again")
        self.run_cmd("/resume")
        expected = [self.history.find(THREAD, U1).id, self.history.find(THREAD, U2).id]
        self.assertEqual(self.button_callbacks(), [f"cm:{THREAD}:rs:{rid}" for rid in expected])
        self.assertTrue(self.labels()[0].endswith("· older start…"), self.labels())  # the row keeps its own text and start

    def test_a_session_found_on_disk_is_listed_even_when_its_old_row_is_outside_the_window(self) -> None:
        """Ten later sessions fill the bounded history read; U1's row is older than all of them, but its file was
        touched just now, so the listing still finds it and the picker puts it first."""
        now = self.clock.now()
        self.note(U1, CLAUDE_CMD, text="old row")
        self.clock.advance(100)
        for n in range(10):
            self.note(f"1f2e3d4c-aaaa-4bbb-8ccc-0000000003{n:02d}", CLAUDE_CMD, text=f"later {n}")
        self.write_transcript(self.cfg_a(), U1, now + 500, "touched")
        self.run_cmd("/resume")
        self.assertEqual(self.button_callbacks()[0], f"cm:{THREAD}:rs:{self.history.find(THREAD, U1).id}")
        self.assertEqual(len(self.button_callbacks()), 10)

    def test_the_eleventh_disk_session_is_listed_when_the_newest_ten_are_bound_elsewhere(self) -> None:
        now = self.clock.now()
        sids = [f"1f2e3d4c-aaaa-4bbb-8ccc-0000000002{n:02d}" for n in range(11)]
        for n, sid in enumerate(sids):
            self.write_transcript(self.cfg_a(), sid, now + n * 10, f"disk {n}")
        for n in range(1, 11):
            self.bind_elsewhere(sids[n], thread=THREAD + n)
        picker = self.commands.resume_picker(THREAD)
        self.assertEqual(len(picker.rows), 1)
        self.assertIsNotNone(self.history.find(THREAD, sids[0]), "the eleventh session was cut by the limit")
        self.assertTrue(picker.rows[0][0][0].endswith("· disk 0…"))


class BackgroundStopTests(SessionCase):
    def test_a_short_wait_is_reported_with_its_decimal_seconds(self) -> None:
        """Ported labels:
        N: a turn that never ends is reported ('⏹ still stopping after 0.6 s — send /new again in a moment')
        N: /new is not applied while the turn still runs
        """
        commands = self.build_commands(FakeClock(), stop_wait=0.6)
        session = self.start_turn()
        self.sessions.set_uuid(THREAD, U1)
        self.assertEqual(commands.finish_new(THREAD, CHAT, MID),
                         "⏹ still stopping after 0.6 s — send /new again in a moment")
        self.assertEqual(self.row().uuid, U1)
        self.end_turn(session)

    def test_a_turn_that_never_ends_gets_the_still_stopping_edit(self) -> None:
        """Labels: stop-and-new: a turn that outlives the wait says so, and the session is kept."""
        session = self.start_turn()
        self.sessions.set_uuid(THREAD, U1)
        self.assertEqual(self.commands.stop_and_new(THREAD, CHAT, MID), STOP_NEW_TOAST)
        self.assertTrue(session.stopped.wait(2), "the stop did not reach the session")
        expected = STILL_STOPPING_TEXT.format(wait=30)
        self.assertTrue(wait_until(lambda: self.telegram.edits(MID), timeout=5), "the wait never edited the message")
        self.assertEqual(self.telegram.edits(MID)[-1], expected)
        self.assertEqual(self.row().uuid, U1)
        self.end_turn(session)


class KeptStopTests(SessionCase):
    """Stops with a kept session (linger, section 5 D2): /stop asks which stop during a background update, closes an
    idle kept session at once, and stops a user turn as it always did."""

    def setUp(self) -> None:
        super().setUp()
        self.linger = StubKeptLinger()
        self.commands = self.build_kept_commands(self.linger)
        self.registry = CommandRegistry()
        self.commands.register(self.registry)

    def build_kept_commands(self, linger: StubKeptLinger) -> SessionCommands:
        return SessionCommands(config=self.config, sessions=self.sessions, history=self.history, live=self.live,
                               backends=self.backends, messenger=self.messenger, clock=self.clock, logger=self.logger,
                               handoff=HandoffService(KvRepo(self.db), self.backends, self.logger), linger=linger)

    def test_stop_closes_a_kept_session_and_says_so(self) -> None:
        """Nothing runs but a session is kept: /stop asks it to close and says so."""
        self.linger.release_answer = True
        self.assertTrue(self.run_cmd("/stop"))
        self.assertEqual(self.last(), LINGER_STOPPED_TEXT)
        self.assertEqual(self.linger.requests, [("release", THREAD)])

    def test_stop_with_nothing_kept_says_nothing_runs(self) -> None:
        """Nothing runs and nothing is kept: /stop says so and asks for no release."""
        self.linger.release_answer = False
        self.assertTrue(self.run_cmd("/stop"))
        self.assertEqual(self.last(), NOTHING_RUNNING_TEXT)
        self.assertEqual(self.linger.requests, [("release", THREAD)])

    def test_stop_during_a_self_turn_asks_which_stop(self) -> None:
        """A background update runs: /stop posts the choice card and stops nothing yet."""
        self.linger.status_value = LingerStatus(pending=1, seconds_left=60.0, self_turn=True, hold_id=3, self_seq=2)
        session = self.start_turn()
        self.assertTrue(self.run_cmd("/stop"))
        self.assertEqual(self.last(), STOP_CHOICE_TEXT)
        self.assertIn(f"cm:{THREAD}:sq:u:3:2", self.button_callbacks())
        self.assertIn(f"cm:{THREAD}:sq:c:3:2", self.button_callbacks())
        self.assertFalse(session.stopped.is_set(), "the choice card must not stop the update before the tap")
        self.assertEqual(self.linger.requests, [])

    def test_stop_turn_during_a_self_turn_returns_choose(self) -> None:
        """The shared stop path reports CHOOSE for a background update, and nothing was stopped."""
        self.linger.status_value = LingerStatus(pending=1, seconds_left=60.0, self_turn=True)
        session = self.start_turn()
        result = self.commands.stop_turn(THREAD)
        self.assertEqual(result.kind, StopKind.CHOOSE)
        self.assertTrue(result)
        self.assertEqual(session.stops, 0)

    def test_stop_during_a_user_turn_stops_it_as_before(self) -> None:
        """A user turn on a kept session: /stop stops the turn and its session, with no card."""
        self.linger.status_value = LingerStatus(pending=0, seconds_left=60.0, self_turn=False)
        session = self.start_turn()
        self.assertTrue(self.run_cmd("/stop"))
        self.assertTrue(session.stopped.wait(2), "/stop did not stop the session")
        self.assertEqual(self.telegram.texts(THREAD), [], "a stopped user turn sends no reply and no card")
        self.assertEqual(self.linger.requests, [])

    def test_interrupt_update_asks_the_session_to_interrupt_only(self) -> None:
        """The card's update button interrupts the update through the linger service and stops nothing."""
        session = self.start_turn()
        self.linger.interrupt_answer = True
        self.assertTrue(self.commands.interrupt_update(THREAD))
        self.assertEqual(self.linger.requests, [("interrupt", THREAD)])
        self.assertEqual(session.stops, 0)
        self.linger.interrupt_answer = False
        self.assertFalse(self.commands.interrupt_update(THREAD))

    def test_stop_and_new_during_a_self_turn_stops_the_session_without_a_card(self) -> None:
        """The stop-and-new button stops the update and its session at once: starting fresh drops both."""
        self.linger.status_value = LingerStatus(pending=1, seconds_left=60.0, self_turn=True)
        session = self.start_turn()
        session.on_stop = lambda: self.end_turn(session)
        self.assertEqual(self.commands.stop_and_new(THREAD, CHAT, MID), STOP_NEW_TOAST)
        self.assertTrue(session.stopped.wait(2), "stop-and-new did not stop the session")
        self.assertEqual(session.stops, 1)
        self.assertEqual(self.linger.requests, [])

    def test_stop_now_during_a_self_turn_stops_the_session(self) -> None:
        """The card's close button: the update and the session are stopped, as /stop did before D2."""
        session = self.start_turn()
        self.linger.status_value = LingerStatus(pending=1, seconds_left=60.0, self_turn=True)
        self.assertEqual(self.commands.stop_now(THREAD).kind, StopKind.STOPPED)
        self.assertTrue(session.stopped.wait(2), "the close button did not stop the session")
        self.assertEqual(session.stops, 1)


if __name__ == "__main__":
    unittest.main()
