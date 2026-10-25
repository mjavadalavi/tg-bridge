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

import os
import threading
import unittest

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.support import TempHome
from tests.turn_support import wait_until
from tg_bridge.agents.claude.history import encoded_cwd
from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.bot.commands.base import CommandDeps, CommandRegistry
from tg_bridge.bot.commands.session import (
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
)
from tg_bridge.config import Config
from tg_bridge.core.dispatcher import TopicQueues
from tg_bridge.core.handoff import HandoffService
from tg_bridge.core.live import LiveRegistry
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

    def test_the_picker_lists_both_backends_newest_first_with_the_launcher_mark(self) -> None:
        self.note(U1, CLAUDE_CMD, text="claude one")
        self.note(U2, CODEX_CMD, backend="codex", text="codex two")
        self.run_cmd("/resume")
        labels = [b["text"] for kb in self.telegram.buttons(THREAD) for row in kb for b in row]
        self.assertEqual(len(labels), 2)
        self.assertIn("codex two", labels[0], "newest first")
        self.assertIn("⚠ codex · ", labels[0])
        self.assertNotIn("⚠", labels[1])


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


if __name__ == "__main__":
    unittest.main()
