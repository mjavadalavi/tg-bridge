"""Port of fakecli/test_launcher.py, sections C to K (the scenarios the other suites do not state in these words).

The scenarios run through the real LauncherSwitch, SettingsCommands, SessionCommands, pickers and tap handlers, with the
old suite's launcher names (tests/port_launcher_support.py). Section A, B and the blank-launcher part of D live in
test_config and test_turn_runner; the labels of those are listed in their docstrings.
"""

import json
import os
import sqlite3
import unittest

from tests.port_launcher_support import OLD_LAUNCHERS, LauncherCwdPortCase
from tests.support import TempHome
from tests.turn_support import CHAT, TurnHarness, join_turn
from tests.fakes.backend import FakeBackend
from tg_bridge.bot.commands.session import REFUSED_TEXT, RESUMED_TAP_TEXT, RESUMED_TEXT, SWITCHED_TEXT
from tg_bridge.bot.commands.settings import UNKNOWN_LAUNCHER_TEXT
from tg_bridge.core.launcher_switch import BUSY_TEXT, NOT_FOUND_TEXT, SAME_TEXT
from tg_bridge.paths import Paths
from tg_bridge.store import migrations
from tg_bridge.store.db import Database
from tg_bridge.agents.events import TurnResult, TurnStatus

THREAD = 9601
TOPIC = "CB"
U1 = "11111111-1111-4111-8111-111111111111"
U_OLD = "61111111-1111-4111-8111-111111111111"  # made under wrap
U_CL = "62222222-2222-4222-8222-222222222222"  # made under claude
U_LEG = "63333333-3333-4333-8333-333333333333"  # recorded before launchers were stored
U_ORPHAN = "64444444-4444-4444-8444-444444444444"  # made by a launcher that is not in config
U_NEW = "65555555-5555-4555-8555-555555555555"
OK = TurnResult(TurnStatus.OK, answer="ok:allow", duration_ms=5, num_turns=1, subtype="success")
NOT_FOUND_OLD = NOT_FOUND_TEXT.format(name="plain")


class LauncherSchemaPort(unittest.TestCase):
    """C and K1: the migration adds launcher_pick to an old sessions table, once; the history table has launcher."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())

    def test_old_table_gains_launcher_pick_and_a_second_migration_adds_nothing(self) -> None:
        """Ported labels:
        C1: old table gains launcher_pick
        C2: a second migration adds nothing
        C3: the live db has the column
        """
        path = os.path.join(self.home.home, "old.db")
        conn = sqlite3.connect(path, isolation_level=None)
        conn.executescript("CREATE TABLE sessions(thread_id INTEGER PRIMARY KEY, name TEXT, uuid TEXT, launcher TEXT, "
                           "cwd TEXT, created TEXT, title TEXT DEFAULT '');")
        conn.close()
        conn = sqlite3.connect(path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            migrations.migrate(conn)
            columns = [r["name"] for r in conn.execute("PRAGMA table_info(sessions)")]
            self.assertEqual(columns.count("launcher_pick"), 1)
            migrations.migrate(conn)
            again = [r["name"] for r in conn.execute("PRAGMA table_info(sessions)")]
            self.assertEqual((again.count("launcher_pick"), len(again)), (1, len(columns)))
        finally:
            conn.close()
        db = Database(Paths.from_home(self.home.home))
        self.addCleanup(db.close)
        with db.tx() as tx:
            live = [r["name"] for r in tx.execute("PRAGMA table_info(sessions)")]
        self.assertIn("launcher_pick", live)

    def test_session_history_has_the_launcher_and_cwd_columns(self) -> None:
        """Ported labels:
        K1: session_history has a launcher column
        J1: session_history has a cwd column
        """
        db = Database(Paths.from_home(self.home.home))
        self.addCleanup(db.close)
        with db.tx() as tx:
            columns = {r["name"] for r in tx.execute("PRAGMA table_info(session_history)")}
        self.assertTrue({"launcher", "cwd"} <= columns)


class LauncherCommandPort(LauncherCwdPortCase):
    """E, G, H and K3 to K8 through the typed commands, the pickers and the /resume buttons."""

    def setUp(self) -> None:
        super().setUp()
        self.set_row(launcher="wrap run")

    # --- E: /launcher

    def test_launcher_alone_sends_the_picker_with_the_current_launcher_marked(self) -> None:
        """Ported labels:
        E1: /launcher with no argument sends the picker, current from the command string
        E1: callbacks are cm:<thread>:la:<index>
        E1: callbacks fit in 64 bytes
        E1: check mark on the current launcher (['✓ wrap', 'plain', 'claude'])
        """
        self.run_cmd("/launcher")
        self.assertEqual(self.sent_buttons()[-1]["text"], "🚀 launcher: wrap")
        self.assertEqual(self.card_labels(), ["✓ wrap", "plain", "claude"])
        self.assertEqual(self.card_callbacks(), [f"cm:{THREAD}:la:{i}" for i in range(3)])
        self.assertTrue(all(len(c.encode()) <= 64 for c in self.card_callbacks()))

    def test_unknown_name_is_refused_with_the_list_and_changes_nothing(self) -> None:
        """Ported labels:
        E2: unknown name lists the names ('unknown launcher "nope". choose one of: wrap, plain, claude (or /launcher alone for buttons)')
        E2: unknown name changes nothing
        """
        self.sessions.set_uuid(THREAD, U1)
        self.run_cmd("/launcher nope")
        self.assertEqual(self.last_text(), UNKNOWN_LAUNCHER_TEXT.format(name="nope", names="wrap, plain, claude"))
        self.assertEqual((self.row().launcher_pick, self.row().uuid), ("", U1))

    def test_same_launcher_says_so_and_keeps_pick_and_session(self) -> None:
        """Ported labels:
        E3: same launcher says so
        E3: same launcher keeps pick and session
        """
        self.sessions.set_uuid(THREAD, U1)
        self.run_cmd("/launcher wrap")
        self.assertEqual(self.last_text(), "🚀 launcher: wrap — already the launcher, nothing changed")
        self.assertEqual((self.row().launcher_pick, self.row().uuid), ("", U1))

    def test_change_without_a_transcript_starts_fresh_and_keeps_the_id_for_resume(self) -> None:
        """Ported labels:
        E4: a change with no transcript to copy says the next message starts fresh
        E4: pick stored and the session moved to prev_uuid (kept for /resume)
        """
        self.sessions.set_uuid(THREAD, U1)
        self.run_cmd("/launcher plain")
        self.assertEqual(self.last_text(), NOT_FOUND_OLD)
        row = self.row()
        self.assertEqual((row.launcher_pick, row.uuid, row.prev_uuid), ("plain", "", U1))

    def test_a_running_turn_refuses_a_different_launcher(self) -> None:
        """Ported labels:
        E5: refusal changes nothing
        E5: a running turn refuses a different launcher
        """
        self.sessions.set_uuid(THREAD, U1)
        session = self.start_turn()
        self.run_cmd("/launcher plain")
        self.assertEqual(self.last_text(), BUSY_TEXT)
        self.assertEqual((self.row().launcher_pick, self.row().uuid), ("", U1))
        self.end_turn(session)

    def test_the_picker_marks_the_launcher_that_was_picked(self) -> None:
        """Ported labels:
        E6: the picked launcher is the one with the check
        """
        self.run_cmd("/launcher plain")
        self.run_cmd("/launcher")
        self.assertEqual(self.sent_buttons()[-1]["text"], "🚀 launcher: plain")
        self.assertEqual(self.card_labels()[1], "✓ plain")

    def test_a_pick_missing_from_config_falls_back_to_the_topic_launcher_in_the_picker(self) -> None:
        """Ported labels:
        E7: a pick missing from config falls back to the topic launcher in the picker
        """
        from tg_bridge.bot.commands.info import launcher_state
        from tg_bridge.bot.pickers import launcher_picker
        from tg_bridge.config import Config
        cfg_one = Config({"launchers": {"a": "alpha --x", "b": "beta"}, "topics": {TOPIC: {"launcher": "claude"}}},
                         home=self.base, env={})
        self.set_row(launcher_pick="plain", launcher="claude")
        name, command = launcher_state(cfg_one, self.row(), TOPIC)
        picker = launcher_picker(THREAD, list(cfg_one.launcher_specs().values()), name, command)
        self.assertEqual(picker.text, "🚀 launcher: claude")
        self.assertEqual([label for row in picker.rows for label, _ in row], ["a", "b"])

    # --- G: /status and its shortcut

    def test_status_shows_the_launcher_line_and_the_launcher_button(self) -> None:
        """Ported labels:
        G1: /status has the launcher line ('launcher: wrap (wrap run)')
        G1: /status has the 🚀 Launcher button
        """
        self.run_cmd("/status")
        self.assertIn("launcher: wrap (wrap run)", self.sent_buttons()[-1]["text"])
        self.assertIn(f"cm:{THREAD}:sl:0", self.card_callbacks())
        self.assertIn("🚀 Launcher", self.card_labels())

    def test_the_launcher_shortcut_sends_the_picker_and_edits_nothing(self) -> None:
        """Ported labels:
        G2: the shortcut sends the launcher picker as a new message
        G2: the shortcut answers and edits nothing
        """
        self.tap(f"cm:{THREAD}:sl:0")
        self.assertEqual(self.answers(), ["🚀 launcher"])
        self.assertTrue(self.sent_buttons()[-1]["text"].startswith("🚀 launcher: wrap"))
        self.assertEqual(self.edits(), [])

    # --- H: /cwd paths that are not folders

    def test_an_absolute_path_that_is_not_a_folder_is_said_aloud_and_never_searched(self) -> None:
        """Ported labels:
        H1: /cwd with an absolute non-directory says so ('not a directory: /definitely/not/here-xyz')
        H4: refused paths keep the cwd
        """
        self.run_cmd("/cwd /definitely/not/here-xyz")
        self.assertEqual(self.last_text(), "not a directory: /definitely/not/here-xyz")
        self.assertEqual(self.finder.calls, [])
        self.assertEqual(self.row().cwd, self.project)

    def test_a_tilde_path_that_is_not_a_folder_is_said_aloud_and_never_searched(self) -> None:
        """Ported labels:
        H2: /cwd with a ~ path that is not a directory says so, no search
        """
        self.run_cmd("/cwd ~/no-such-folder-xyz")
        self.assertEqual(self.last_text(), "not a directory: " + os.path.join(self.base, "no-such-folder-xyz"))
        self.assertEqual(self.finder.calls, [])

    def test_no_live_session_is_left_behind_by_the_settings_commands(self) -> None:
        """Ported labels:
        I: no live session left behind
        """
        self.run_cmd("/launcher plain")
        self.run_cmd("/cwd /definitely/not/here-xyz")
        self.assertIsNone(self.live.running(THREAD))

    # --- K3: the /resume picker marks sessions made by another launcher

    def _history_rows(self) -> None:
        """The stored history of the old suite, in insertion order (the picker lists the newest start first). Times are
        the owner's clock: 10-24 19:50 is 16:20 UTC, 10-01 00:00 is 20:30 UTC of the day before, 10-02 00:00 the 1st."""
        rows = [
            (U_OLD, "2026-10-24T16:20:00+00:00", "old wrap chat", "wrap run"),
            (U_CL, "2026-10-24T16:20:00+00:00", "claude chat", "claude"),
            (U_LEG, "2026-09-30T20:30:00+00:00", "legacy chat", ""),
            (U_ORPHAN, "2026-10-01T20:30:00+00:00", "orphan chat", "gone-cli --x"),
        ]
        with self.db.tx() as tx:
            tx.execute("DELETE FROM session_history WHERE thread_id=?", (THREAD,))
            for uuid, started, text, launcher in rows:
                tx.execute("INSERT INTO session_history(thread_id, uuid, started, first_text, cwd, launcher, backend) "
                           "VALUES(?,?,?,?,?,?,?)", (THREAD, uuid, started, text, self.project, launcher, "claude"))

    def test_the_picker_marks_only_sessions_of_other_launchers(self) -> None:
        """Ported labels:
        K3: a session of another launcher is marked with its name ('⚠ wrap · 10-24 19:50 · old wrap chat…')
        K3: a session of an unknown launcher is marked 'other launcher'
        K3: the picker explains the mark

        Every row now names its launcher, and the rows are sorted by start time (the two 16:20 UTC rows keep id order).
        """
        self.set_row(launcher="claude")
        self._history_rows()
        self.run_cmd("/resume")
        self.assertEqual(self.card_labels(), [
            "claude · 10-24 19:50 · claude chat…",
            "⚠ wrap · 10-24 19:50 · old wrap chat…",
            "⚠ other launcher · 10-02 00:00 · orphan chat…",
            "10-01 00:00 · legacy chat…",
        ])
        self.assertIn("⚠ = made by another launcher", self.sent_buttons()[-1]["text"])

    # --- K2, K4 to K8: /resume and its taps

    def test_the_first_launcher_of_a_session_is_kept_when_it_is_noted_again(self) -> None:
        """Ported labels:
        K2: _note_session keeps the launcher the session was made under
        """
        self.note(U_OLD, "wrap run")
        self.note(U_OLD, "claude")
        self.assertEqual(self.history.find(THREAD, U_OLD).launcher, "wrap run")

    def test_resume_of_an_wrap_session_switches_the_launcher_to_wrap(self) -> None:
        """Ported labels:
        K4: /resume of an wrap session switches the launcher to wrap (61111111-1111-4111-8111-111111111111, 'wrap')
        K4: the reply says the launcher switched ('↩️ the next message resumes session 61111111-1111-4111-8111-111111111111 — launcher switched to wrap, which made this session')
        """
        self.set_row(launcher="claude")
        self.note(U_OLD, "wrap run")
        self.run_cmd(f"/resume {U_OLD}")
        self.assertEqual((self.row().uuid, self.row().launcher_pick), (U_OLD, "wrap"))
        self.assertEqual(self.last_text(), RESUMED_TEXT.format(sid=U_OLD, suffix=SWITCHED_TEXT.format(name="wrap")))

    def test_resume_of_a_session_with_no_recorded_launcher_leaves_the_pick_alone(self) -> None:
        """Ported labels:
        K5: an unknown-launcher session is resumed without a switch
        """
        self.set_row(launcher_pick="wrap")
        with self.db.tx() as tx:
            tx.execute("INSERT INTO session_history(thread_id, uuid, started, first_text, cwd, launcher, backend) "
                       "VALUES(?,?,?,?,?,?,?)", (THREAD, U_LEG, "2026-10-01T20:30:00+00:00", "legacy chat",
                                                 self.project, "", "claude"))
        self.run_cmd(f"/resume {U_LEG}")
        self.assertEqual((self.row().uuid, self.row().launcher_pick), (U_LEG, "wrap"))

    def test_resume_of_a_session_of_a_launcher_not_in_config_is_refused(self) -> None:
        """Ported labels:
        K6: a session of a launcher not in config is refused ('session 64444444-4444-4444-8444-444444444444 belongs to launcher gone-cli --x, which is not in config — it cannot run here')
        """
        self.set_row(uuid=U_LEG)
        self.note(U_ORPHAN, "gone-cli --x")
        self.run_cmd(f"/resume {U_ORPHAN}")
        self.assertEqual(self.last_text(), REFUSED_TEXT.format(sid=U_ORPHAN, launcher="gone-cli --x"))
        self.assertEqual(self.row().uuid, U_LEG)

    def test_a_resume_button_switches_to_the_claude_launcher_and_edits_the_picker(self) -> None:
        """Ported labels:
        K7: a /resume tap switches to the claude launcher ('claude')
        K7: the same picker message is edited
        K7: the toast says so (['↩️ the next message resumes it — launcher switched to claude, which made this session'])
        """
        self.set_row(launcher_pick="wrap", uuid=U_NEW)
        self.note(U_CL, "claude", text="claude chat")
        self.tap(f"cm:{THREAD}:rs:{self.history_id(U_CL)}")
        self.assertEqual((self.row().uuid, self.row().launcher_pick), (U_CL, "claude"))
        self.assertEqual(self.answers(), [RESUMED_TAP_TEXT.format(suffix=SWITCHED_TEXT.format(name="claude"))])
        self.assertEqual(len(self.edits()), 1)

    def test_a_refused_resume_button_says_why_and_edits_nothing(self) -> None:
        """Ported labels:
        K8: a refused /resume button answers with the reason and edits nothing
        """
        self.set_row(uuid=U_CL)
        self.note(U_ORPHAN, "gone-cli --x")
        self.tap(f"cm:{THREAD}:rs:{self.history_id(U_ORPHAN)}")
        self.assertEqual(self.answers(), [REFUSED_TEXT.format(sid=U_ORPHAN, launcher="gone-cli --x")])
        self.assertEqual(self.edits(), [])
        self.assertEqual(self.row().uuid, U_CL)


class LauncherTapPort(LauncherCwdPortCase):
    """F: the launcher picker's taps go through the real LauncherSwitch and edit the same message."""

    def setUp(self) -> None:
        super().setUp()
        self.set_row(launcher="claude", uuid=U1)

    def test_a_tap_stores_the_launcher_its_button_shows_and_edits_the_picker(self) -> None:
        """Ported labels:
        F1: a tap stores the launcher its button shows
        F1: the tap answers with the reply
        F1: the same message is edited
        F1: the picker moves the check (['✓ wrap', 'plain', 'claude'])
        """
        self.tap(f"cm:{THREAD}:la:0")
        self.assertEqual(self.row().launcher_pick, "wrap")
        reply = "🚀 launcher: wrap — previous history not found, the next message starts a fresh session"
        self.assertEqual(self.answers(), [reply])
        edit = self.edits()[-1]
        self.assertEqual((edit["text"], edit["message_id"]), (reply, 777))
        self.assertEqual(self.labels_of(edit["reply_markup"]), ["✓ wrap", "plain", "claude"])

    def test_tapping_the_current_launcher_changes_nothing(self) -> None:
        """Ported labels:
        F2: tapping the current launcher changes nothing
        """
        self.set_row(launcher_pick="wrap")
        self.tap(f"cm:{THREAD}:la:0")
        self.assertEqual(self.answers(), [SAME_TEXT.format(name="wrap")])
        self.assertEqual(self.row().launcher_pick, "wrap")

    def test_a_running_turn_refuses_the_tap_without_an_edit(self) -> None:
        """Ported labels:
        F3: a running turn refuses the tap, no edit
        F3: refusal keeps the pick
        """
        self.set_row(launcher_pick="wrap")
        session = self.start_turn()
        self.tap(f"cm:{THREAD}:la:2")
        self.assertEqual(self.answers(), [BUSY_TEXT])
        self.assertEqual(self.edits(), [])
        self.assertEqual(self.row().launcher_pick, "wrap")
        self.end_turn(session)

    def test_a_tap_past_the_list_or_from_a_stranger_or_in_a_topic_without_session(self) -> None:
        """Ported labels:
        F4: an index past the list is refused
        F5: a tap from a user not on the allow-list changes nothing
        F6: a tap in a topic without a session is refused
        """
        self.set_row(launcher_pick="wrap")
        self.tap(f"cm:{THREAD}:la:9")
        self.assertEqual(self.answers()[-1], "not in the list anymore")
        self.tap(f"cm:{THREAD}:la:1", tapper=999)
        self.assertEqual(self.answers()[-1], "not allowed")
        self.assertEqual(self.row().launcher_pick, "wrap")
        self.tap(f"cm:{THREAD + 1}:la:0")
        self.assertEqual(self.answers()[-1], "no session in this topic")


class LauncherTurnPort(unittest.TestCase):
    """D: the launcher a turn runs is the pick while it is in config, else the topic's, else the stored one (a real
    TurnRunner with the FakeBackend records the SessionSpec it was given)."""

    OLD_CONFIG = {
        "launchers": {"wrap": {"cmd": "wrap run", "config_dir": "~/.a"},
                      "plain": {"cmd": "/opt/bin/oc --x", "config_dir": "~/.b"}},
        "topics": {TOPIC: {"launcher": "wrap run", "cwd": "~"}},
    }

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "claude", self.home.home, "bypassPermissions")

    def run_turn(self, config: dict):
        self.h.set_config(**config)
        self.backend = FakeBackend([[OK]] * 3)
        runner = self.h.runner(self.h.registry_for(self.backend))
        reply = runner.run(thread_id=THREAD, chat_id=CHAT, topic=TOPIC, text="hello", sender="tester", slot=None)
        return reply

    def test_no_pick_runs_the_topic_launcher(self) -> None:
        """Ported labels:
        D1: no pick -> the topic launcher
        """
        self.run_turn(self.OLD_CONFIG)
        self.assertEqual(self.backend.specs[-1].launcher.name, "wrap")

    def test_a_pick_runs_its_multi_word_launcher(self) -> None:
        """Ported labels:
        D2: a pick runs its multi-word launcher
        """
        self.h.sessions.set_launcher_pick(THREAD, "wrap")
        self.run_turn(self.OLD_CONFIG)
        self.assertEqual(self.backend.specs[-1].launcher.argv, ("wrap", "run"))

    def test_the_pick_uses_the_command_from_config(self) -> None:
        """Ported labels:
        D3: the pick uses the command from config
        """
        self.h.sessions.set_launcher_pick(THREAD, "plain")
        self.run_turn(self.OLD_CONFIG)
        self.assertEqual(self.backend.specs[-1].launcher.argv, ("/opt/bin/oc", "--x"))

    def test_a_pick_no_longer_in_config_falls_back_to_the_topic_launcher(self) -> None:
        """Ported labels:
        D4: a pick no longer in config falls back to the topic launcher
        """
        self.h.sessions.set_launcher_pick(THREAD, "gone")
        self.run_turn(self.OLD_CONFIG)
        self.assertEqual(self.backend.specs[-1].launcher.name, "wrap")

    def test_a_blank_launcher_gives_a_clear_error_and_stores_no_id(self) -> None:
        """Ported labels:
        D5: a blank launcher gives a clear error, nothing spawned ('launcher failed: the launcher is empty: set it in config (topics or launchers)')
        D5: no session id stored for a blank launcher
        """
        with self.h.db.tx() as tx:
            tx.execute("UPDATE sessions SET launcher='' WHERE thread_id=?", (THREAD,))
        reply = self.run_turn({"topics": {TOPIC: {"launcher": "  "}}})
        self.assertEqual(reply.err, "launcher failed: the launcher is empty: set it in config (topics or launchers)")
        self.assertEqual(self.backend.specs, [])
        self.assertEqual(self.h.sessions.get(THREAD).uuid, "")


if __name__ == "__main__":
    unittest.main()
