"""Info commands and the command framework (STEP 45), ported from fakecli/test_commands.py.

Scenarios: plain text, unknown /xyz and Claude's own commands are not bridge commands; /status@BotName shows the topic,
backend, mode, queue and rules, the Stop button only while a turn runs, the resume hint and tool count of the running
turn; the launcher line follows the pick, then the topic, then the stored command; /cost totals with the unknown-cost
star and a split by backend; /queue lists items with their buttons and labels a media placeholder; /rules and /forget;
/help lists every registered command and the stored Claude commands; the registry rejects duplicate words and runs
aliases with their arguments. Every test runs against a real migrated database in a TempHome; replies are read from
FakeTelegram after the ordered out-worker has drained.
"""

import json
import os
import subprocess
import sys
import unittest
from dataclasses import replace
from unittest import mock

from tests.fakes.clock import FakeClock
from tests.fakes.telegram import FakeTelegram
from tests.linger_stub import StubKeptLinger
from tests.support import TempHome
from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.bot.commands.base import Alias, Command, CommandDeps, CommandRegistry
from tg_bridge.bot.commands.info import (
    LINGER_QUIET_STATUS_TEXT,
    LINGER_STATUS_TEXT,
    QUEUE_BUTTON_ITEMS,
    QUEUE_LIST_CHARS,
    SELF_TURN_STATUS_TEXT,
    launcher_state,
    register_info_commands,
)
from tg_bridge.config import Config
from tg_bridge.core.linger import LingerStatus
from tg_bridge.core import terminal_claim
from tg_bridge.core.live import LiveRegistry
from tg_bridge.core.topic_queues import Admission, TopicQueues
from tg_bridge.core.turn_listeners import slash_key
from tg_bridge.core.workers import WorkerPool
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.store.db import Database
from tg_bridge.store.kv import KvRepo
from tg_bridge.bot.pickers import rules_picker, status_rows
from tg_bridge.store.rules import RuleRepo
from tg_bridge.store.sessions import SessionRepo
from tg_bridge.store.turns import TurnRepo
from tg_bridge.telegram.messenger import Messenger

THREAD = 9401
CHAT = -1009401
TOPIC = "side"
CLAUDE_CMD = "wrap run"
UUID = "1f2e3d4c-aaaa-4bbb-8ccc-000000000401"
UNKNOWN_CMD = "some-other-cli --flag"


class FakeFeed:
    """The progress feed of a running turn: only its tool count is read by /status."""

    tools = 3


def config_for(project: str) -> dict:
    return {
        "launchers": {"wrap": {"cmd": CLAUDE_CMD, "config_dir": os.path.join(project, "cfg")}},
        "agent_launchers": {"codex": {"backend": "codex", "command": "codex",
                                      "config_dir": os.path.join(project, "cfg-codex")}},
        "topics": {TOPIC: {"launcher": CLAUDE_CMD, "cwd": project}},
    }


class InfoCase(unittest.TestCase):
    """A real config, database and repositories; the commands run through a real WorkerPool and FakeTelegram."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.base = os.path.realpath(self.home.home)
        self.project = os.path.join(self.base, "project")
        os.makedirs(self.project, exist_ok=True)
        self.clock = FakeClock()
        self.logger = Logger(os.path.join(self.home.state, "bridge.log"))
        self.config = Config(config_for(self.project), home=self.base, env={})
        self.db = Database(Paths.from_home(self.base))
        self.addCleanup(self.db.close)
        self.sessions = SessionRepo(self.db, self.clock)
        self.turns = TurnRepo(self.db)
        self.rules = RuleRepo(self.db, self.clock)
        self.kv = KvRepo(self.db)
        self.live = LiveRegistry()
        self.queues = TopicQueues()
        self.backends = build_default_registry(self.config, self.logger, self.clock, title_launcher="")
        self.telegram = FakeTelegram(self.clock)
        self.messenger = Messenger(self.telegram, self.logger, self.clock)
        self.out = WorkerPool("out", 1, self.logger)
        self.commands = CommandRegistry()
        register_info_commands(self.commands)
        self.deps = CommandDeps(
            config=self.config, sessions=self.sessions, turns=self.turns, rules=self.rules, kv=self.kv,
            live=self.live, queues=self.queues, backends=self.backends, messenger=self.messenger, out=self.out,
        )
        self.sessions.ensure(THREAD, TOPIC, CLAUDE_CMD, self.project, "")

    def run_cmd(self, text: str | None, thread: int = THREAD) -> bool:
        handled = self.commands.dispatch(self.deps, CHAT, thread, TOPIC, text)
        self.assertTrue(self.out.join(5), "the replies did not go out")
        return handled

    def replies(self) -> list[str]:
        return self.telegram.texts(THREAD)

    def last(self) -> str:
        return self.replies()[-1]

    def button_labels(self) -> list[str]:
        return [b["text"] for keyboard in self.telegram.buttons(THREAD) for row in keyboard for b in row]

    def button_callbacks(self) -> list[str]:
        return [b["callback_data"] for keyboard in self.telegram.buttons(THREAD) for row in keyboard for b in row]

    def insert_turn(self, cost: float | None, cost_known: bool, tin: int, tout: int, backend: str = "claude") -> None:
        self.turns.insert(THREAD, started="", ended="", status="ok", cost=cost, cost_known=cost_known,
                          in_tok=tin, out_tok=tout, backend=backend)


class StatusTests(InfoCase):
    def test_status_with_bot_name_shows_topic_backend_mode_and_queue(self) -> None:
        """Labels: B: /status@BotName is handled; B: status shows topic, mode and queue.
        """
        self.assertTrue(self.run_cmd("/status@MyBridgeBot"), "/status@BotName is handled")
        text = self.last()
        for line in ("topic: side", "backend: claude", "mode: bypassPermissions", "model: default", "turn: idle",
                     "queued: 0", "always-allow rules: 0"):
            self.assertIn(line, text)

    def test_status_idle_has_no_stop_button_but_the_shortcuts(self) -> None:
        """Ported labels:
        H: /status sends the status text with buttons (send_buttons)
        """
        self.run_cmd("/status")
        labels = self.button_labels()
        self.assertNotIn("⏹ Stop", labels)
        self.assertEqual(labels, ["🆕 New", "🛡 Mode", "🧠 Model", "💰 Cost", "🚀 Launcher"])

    def test_status_running_turn_shows_tools_stop_button_and_resume_hint(self) -> None:
        """Ported labels:
        H: running shows ⏹ Stop first
        """
        self.sessions.set_uuid(THREAD, UUID)
        self.live.begin_turn(THREAD, object(), FakeFeed())  # only the feed's tool count and the presence matter here
        self.run_cmd("/status")
        text = self.last()
        self.assertIn(f"in a terminal: claude --resume {UUID}", text)
        self.assertIn("turn: running, 3 tool calls so far", text)
        self.assertIn("⏹ Stop", self.button_labels())
        self.assertIn(f"cm:{THREAD}:st:0", self.button_callbacks())

    def test_status_without_a_session_row_uses_the_defaults(self) -> None:
        self.run_cmd("/status", thread=THREAD + 1)
        text = self.telegram.texts(THREAD + 1)[-1]
        self.assertIn("mode: bypassPermissions", text)
        self.assertIn("session: none yet (the next message starts one)", text)
        self.assertNotIn("in a terminal", text)

    def test_status_launcher_line_follows_pick_then_topic_then_stored_command(self) -> None:
        self.sessions.set_launcher_pick(THREAD, "wrap")
        self.run_cmd("/status")
        self.assertIn(f"launcher: wrap ({CLAUDE_CMD})", self.last())

        self.sessions.set_launcher_pick(THREAD, "")
        self.run_cmd("/status")
        self.assertIn(f"launcher: wrap ({CLAUDE_CMD})", self.last(), "the topic's configured launcher matches a name")

        self.sessions.set_launcher_pick(THREAD, "gone")  # a pick that is no longer in config is ignored
        self.run_cmd("/status")
        self.assertIn(f"launcher: wrap ({CLAUDE_CMD})", self.last())

    def test_launcher_state_shows_a_command_no_launcher_has_by_itself(self) -> None:
        data = config_for(self.project)
        data["topics"][TOPIC]["launcher"] = UNKNOWN_CMD
        config = Config(data, home=self.base, env={})
        row = self.sessions.get(THREAD)
        self.assertEqual(launcher_state(config, row, TOPIC), (None, UNKNOWN_CMD))
        self.assertEqual(launcher_state(self.config, row, TOPIC), ("wrap", CLAUDE_CMD))

    def test_status_shows_the_backend_of_the_launcher_not_the_stored_column(self) -> None:
        self.sessions.set_launcher_pick(THREAD, "wrap")  # moved back to Claude; the stored backend is stale
        self.sessions.set_backend(THREAD, "codex")
        self.run_cmd("/status")
        self.assertIn("backend: claude", self.last())
        self.assertNotIn("backend: codex", self.last())

    def test_status_codex_session_shows_its_backend_and_no_terminal_hint(self) -> None:
        self.sessions.set_uuid(THREAD, UUID)
        self.sessions.set_launcher_pick(THREAD, "codex")
        self.sessions.set_backend(THREAD, "codex")
        self.run_cmd("/status")
        text = self.last()
        self.assertIn("backend: codex", text)
        self.assertIn(f"session: {UUID}", text)
        self.assertNotIn("in a terminal", text, "the Codex backend has no resume hint in this step")


class TerminalStatusTests(InfoCase):
    """/status and the terminal claim (spec: the Lock section). A live claim of tg-bridge view shows 'open in the
    terminal'; a stale claim is removed and ignored; with views off, or a view folder that cannot be read, the claim is
    not a claim and /status still answers."""

    def setUp(self) -> None:
        super().setUp()
        self.view = Paths.from_home(self.base).view
        os.makedirs(self.view, mode=0o700, exist_ok=True)
        self.deps = replace(self.deps, view_dir=self.view)

    def write_claim(self, pid: int) -> None:
        with open(terminal_claim.claim_path(self.view, THREAD), "w", encoding="utf-8") as handle:
            json.dump({"pid": pid, "started": 1.0}, handle)

    def test_a_live_claim_shows_the_turn_open_in_the_terminal(self) -> None:
        self.write_claim(os.getpid())
        self.assertTrue(self.run_cmd("/status"))
        self.assertIn("turn: open in the terminal", self.last())

    def test_no_claim_shows_idle(self) -> None:
        self.assertTrue(self.run_cmd("/status"))
        self.assertIn("turn: idle", self.last())
        self.assertNotIn("open in the terminal", self.last())

    def test_a_stale_claim_is_removed_and_shows_idle(self) -> None:
        proc = subprocess.Popen([sys.executable, "-I", "-c", "pass"])
        proc.wait()  # reaped: no process has its pid now
        self.write_claim(proc.pid)
        self.assertTrue(self.run_cmd("/status"))
        self.assertIn("turn: idle", self.last())
        self.assertFalse(os.path.exists(terminal_claim.claim_path(self.view, THREAD)), "stale claim not removed")

    def test_views_off_never_reads_the_view_folder(self) -> None:
        self.write_claim(os.getpid())
        self.deps = replace(self.deps, view_dir="")
        self.assertTrue(self.run_cmd("/status"))
        self.assertIn("turn: idle", self.last())
        self.assertTrue(os.path.exists(terminal_claim.claim_path(self.view, THREAD)), "claim touched with views off")

    def test_an_unreadable_view_folder_is_no_claim_and_status_still_answers(self) -> None:
        self.deps = replace(self.deps, view_dir=os.path.join(self.base, "missing-view"))
        self.assertTrue(self.run_cmd("/status"))
        self.assertIn("turn: idle", self.last())


class SharedPickerTests(InfoCase):
    def test_status_and_rules_draw_their_keyboards_with_the_picker_builders(self) -> None:
        """The /status shortcuts and the /rules list come from bot/pickers.py, the same builders the taps redraw with."""
        with mock.patch("tg_bridge.bot.commands.info.status_rows", wraps=status_rows) as status_spy:
            self.run_cmd("/status")
        with mock.patch("tg_bridge.bot.commands.info.rules_picker", wraps=rules_picker) as rules_spy:
            self.run_cmd("/rules")
        status_spy.assert_called_once()
        rules_spy.assert_called_once()


class CostTests(InfoCase):
    def test_cost_totals_with_the_unknown_cost_star(self) -> None:
        """Labels: H: /cost totals with * for unknown cost.
        """
        self.insert_turn(0.5, True, 1000, 200)
        self.insert_turn(None, False, 500, 100)
        self.run_cmd("/cost")
        text = self.last()
        self.assertIn("2 turns", text)
        self.assertIn("↑1.5k", text)
        self.assertIn("↓300", text)
        self.assertIn("$0.50*", text)
        self.assertIn("* some turns ran on a model without a known price", text)

    def test_cost_splits_by_backend_when_a_topic_ran_on_two(self) -> None:
        self.insert_turn(0.25, True, 400, 40, backend="claude")
        self.insert_turn(None, False, 600, 60, backend="codex")
        self.run_cmd("/cost")
        text = self.last()
        self.assertIn("claude: 1 turns · ↑400 ↓40 tokens · cost $0.25", text)
        self.assertIn("codex: 1 turns · ↑600 ↓60 tokens · cost unknown", text)

    def test_cost_with_no_turns_says_so(self) -> None:
        self.run_cmd("/cost")
        self.assertIn("0 turns · ↑0 ↓0 tokens · cost unknown", self.last())


class QueueTests(InfoCase):
    def test_empty_queue_says_so(self) -> None:
        """Labels: G: an empty queue says so.
        """
        self.run_cmd("/queue")
        self.assertEqual(self.last(), "the queue is empty.")

    def queue(self, texts: list[str]) -> None:
        """Queues the texts behind a running turn: the first starts it, the rest wait in order."""
        for index, text in enumerate(texts, 1):
            self.queues.deliver(THREAD, CHAT, "tester", index, text)

    def test_queue_is_one_message_with_a_media_placeholder_label(self) -> None:
        """Labels: G: one line per queued item; G: a media placeholder is labelled, not shown as empty.
        """
        started = self.queues.deliver(THREAD, CHAT, "tester", 1, "first queued")  # idle topic: it starts
        self.assertEqual(started.admission, Admission.STARTED)
        self.queues.reserve(THREAD, CHAT, "tester", 2)  # a voice note still being transcribed, behind a turn
        self.assertEqual([item.text for item in self.queues.snapshot(THREAD)], ["first queued", None])
        self.run_cmd("/queue")
        nid = self.queues.snapshot(THREAD)[0].nid
        self.assertEqual(len(self.replies()), 1)
        self.assertEqual(self.replies()[0].splitlines(),
                         ["📥 2 queued:", "1. first queued", "2. (voice or photo, still processing)"])
        self.assertEqual(len(self.telegram.buttons(THREAD)), 1)
        self.assertIn(f"now:{THREAD}:{nid}", self.button_callbacks())
        self.assertIn(f"jump:{THREAD}:{nid}", self.button_callbacks())
        self.assertIn(f"drop:{THREAD}:{nid}", self.button_callbacks())
        self.assertIn("⏭ 1", self.button_labels())
        self.assertIn("⏩ 2", self.button_labels())

    def test_thirty_items_are_one_message_with_ten_button_rows(self) -> None:
        self.queue([f"item {n}" for n in range(30)])
        self.run_cmd("/queue")
        self.assertEqual(len(self.replies()), 1, "/queue sends one message, whatever the queue length")
        lines = self.replies()[0].splitlines()
        self.assertEqual(lines[0], "📥 30 queued:")
        self.assertEqual(lines[1:], [f"{n}. item {n - 1}" for n in range(1, 31)])
        keyboards = self.telegram.buttons(THREAD)
        self.assertEqual(len(keyboards), 1)
        rows = keyboards[0]
        self.assertEqual(len(rows), QUEUE_BUTTON_ITEMS)
        self.assertEqual([button["text"] for button in rows[0]], ["⏭ 1", "⏩ 1", "❌ 1"])
        self.assertEqual([button["text"] for button in rows[-1]], ["⏭ 10", "⏩ 10", "❌ 10"])
        nid = self.queues.snapshot(THREAD)[9].nid
        self.assertEqual([button["callback_data"] for button in rows[-1]],
                         [f"now:{THREAD}:{nid}", f"jump:{THREAD}:{nid}", f"drop:{THREAD}:{nid}"])

    def test_a_list_over_the_limit_ends_with_the_more_line(self) -> None:
        self.queue([f"long item {n} " + "x" * 200 for n in range(100)])
        self.run_cmd("/queue")
        text = self.replies()[0]
        self.assertLessEqual(len(text), QUEUE_LIST_CHARS)
        lines = text.splitlines()
        listed = len(lines) - 2  # the header and the more line
        self.assertGreater(listed, 0)
        self.assertEqual(lines[0], "📥 100 queued:")
        self.assertEqual(lines[-1], f"… and {100 - listed} more")
        self.assertTrue(lines[listed].startswith(f"{listed}. long item"))
        self.assertEqual(len(self.telegram.buttons(THREAD)[0]), QUEUE_BUTTON_ITEMS)

    def test_a_list_that_fits_has_no_more_line(self) -> None:
        self.queue(["a", "b"])
        self.run_cmd("/queue")
        self.assertNotIn("more", self.replies()[0])
        self.assertEqual(self.replies()[0].splitlines()[-1], "2. b")


class RulesTests(InfoCase):
    def add_rules(self) -> list[int]:
        self.rules.add(THREAD, "Bash(git status *)")
        self.rules.add(THREAD, "Edit")
        return [r.id for r in self.rules.for_thread(THREAD)]

    def test_rules_none_says_so_without_buttons(self) -> None:
        """Ported labels:
        R5: /rules with none is a plain reply
        """
        self.run_cmd("/rules")
        self.assertEqual(self.last(), "no always-allow rules in this topic.")
        self.assertEqual(self.telegram.buttons(THREAD), [])

    def test_rules_list_with_numbers_hits_and_buttons(self) -> None:
        """Labels: I: /rules lists the rules.
                

        Ported labels:
        R: /rules lists the rules with buttons
        """
        first, second = self.add_rules()
        self.run_cmd("/rules")
        text = self.last()
        self.assertIn("Bash(git status *)", text)
        self.assertIn("Edit", text)
        self.assertIn("(used 0×)", text)
        callbacks = self.button_callbacks()
        self.assertIn(f"cm:{THREAD}:rf:{first}", callbacks)
        self.assertIn(f"cm:{THREAD}:rf:{second}", callbacks)
        self.assertIn(f"cm:{THREAD}:ra:0", callbacks)

    def test_forget_by_number_removes_only_that_rule(self) -> None:
        """Labels: I: /forget <n> removes that one rule.
        """
        first, second = self.add_rules()
        self.run_cmd(f"/forget {second}")
        self.assertIn("removed 1 rule(s)", self.last())
        self.assertEqual([r.id for r in self.rules.for_thread(THREAD)], [first])

    def test_forget_unknown_number_and_huge_number_say_no_such_rule(self) -> None:
        """Labels: I: /forget of an unknown number says so.
        """
        self.add_rules()
        self.run_cmd("/forget 999")
        self.assertEqual(self.last(), "no such rule — see /rules")
        self.run_cmd("/forget 99999999999999999999999")
        self.assertEqual(self.last(), "no such rule — see /rules")
        self.assertEqual(len(self.rules.for_thread(THREAD)), 2)

    def test_forget_all_removes_the_rest(self) -> None:
        """Labels: I: /forget all removes the rest.
        """
        self.add_rules()
        self.run_cmd("/forget all")
        self.assertIn("removed 2 rule(s)", self.last())
        self.assertEqual(self.rules.for_thread(THREAD), [])

    def test_forget_alone_and_a_word_show_usage(self) -> None:
        """Labels: I: /forget alone shows usage.
        """
        self.run_cmd("/forget")
        self.assertIn("usage: /forget <number> or /forget all", self.last())
        self.run_cmd("/forget last")
        self.assertIn("usage: /forget <number> or /forget all", self.last())


class HelpTests(InfoCase):
    def test_help_lists_every_registered_command_and_no_claude_section(self) -> None:
        """Labels: L: /help lists the bridge commands.
        """
        self.assertTrue(self.run_cmd("/help@MyBot"))
        text = self.last()
        self.assertTrue(text.startswith("Bridge commands:"))
        for word in ("status", "cost", "queue", "rules", "forget", "help"):
            self.assertIn(f"/{word} — ", text)
        self.assertNotIn("Claude commands", text)

    def test_help_lists_the_stored_claude_commands(self) -> None:
        """Ported labels:
        L: /help lists Claude commands from init
        """
        self.kv.set(slash_key(THREAD), json.dumps(["compact", "review"]))
        self.run_cmd("/help")
        text = self.last()
        self.assertIn("Claude commands (sent to Claude as text):", text)
        self.assertIn("/compact", text)
        self.assertIn("/review", text)

    def test_help_ignores_a_damaged_stored_list(self) -> None:
        self.kv.set(slash_key(THREAD), "not json")
        self.run_cmd("/help")
        self.assertNotIn("Claude commands", self.last())

    def test_help_lists_a_command_registered_later(self) -> None:
        self.commands.register(Command("ping", "a test command", lambda ctx, args: ctx.say("pong")))
        self.run_cmd("/help")
        self.assertIn("/ping — a test command", self.last())
        self.run_cmd("/ping")
        self.assertEqual(self.last(), "pong")


class RegistryTests(InfoCase):
    def test_duplicate_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            register_info_commands(self.commands)

    def test_alias_that_clashes_with_a_word_is_rejected(self) -> None:
        clash = Command("other", "x", lambda ctx, args: None, aliases=(Alias("status", "x"),))
        with self.assertRaises(ValueError):
            self.commands.register(clash)
        self.assertIsNone(self.commands.find("other"), "a refused command registers nothing")

    def test_malformed_word_is_rejected(self) -> None:
        for word in ("Bad Name", "", "9lives", "a-b"):
            with self.subTest(word=word), self.assertRaises(ValueError):
                CommandRegistry().register(Command(word, "x", lambda ctx, args: None))

    def test_alias_runs_its_command_with_the_arguments_and_the_bot_name_form_works(self) -> None:
        seen: list[str] = []
        registry = CommandRegistry()
        registry.register(Command("new", "start fresh", lambda ctx, args: seen.append(args),
                                  aliases=(Alias("clear", "same as /new"),)))
        self.assertTrue(registry.dispatch(self.deps, CHAT, THREAD, TOPIC, "/CLEAR@MyBot  some words "))
        self.assertEqual(seen, ["some words"])
        self.assertIs(registry.find("clear"), registry.find("new"))

    def test_bridge_commands_lists_words_and_aliases_in_registration_order(self) -> None:
        registry = CommandRegistry()
        registry.register(Command("status", "show status", lambda ctx, args: None))
        registry.register(Command("new", "start fresh", lambda ctx, args: None, aliases=(Alias("clear", "same"),)))
        self.assertEqual(registry.bridge_commands(), (("status", "show status"), ("new", "start fresh"),
                                                     ("clear", "same")))

    def test_setmycommands_list_covers_every_info_word(self) -> None:
        """Labels: N: setMyCommands registers every bridge command (words and aliases).
        """
        words = {word for word, _ in self.commands.bridge_commands()}
        self.assertEqual(words, {"status", "cost", "queue", "rules", "forget", "help"})

    def test_unknown_and_plain_text_are_not_handled(self) -> None:
        """Labels: A: '', '/', '/compact', '/xyz', 'hello there', 'use /mode here' is not a bridge command.

        Ported labels:
        A: '/compact' is not a bridge command
        A: '/xyz' is not a bridge command
        A: 'hello there' is not a bridge command
        """
        for text in ("hello there", "/xyz", "/compact", "/", "use /mode here", "", None):
            with self.subTest(text=text):
                self.assertFalse(self.run_cmd(text))
        self.assertEqual(self.telegram.texts(THREAD), [], "nothing was sent for text the bridge does not handle")


class KeptStatusTests(InfoCase):
    """/status of a topic whose session is kept open for background work (C9a): the turn line, the stop button."""

    def setUp(self) -> None:
        super().setUp()
        self.linger = StubKeptLinger()
        self.deps = replace(self.deps, linger=self.linger)
        self.sessions.set_uuid(THREAD, UUID)

    def test_status_shows_kept_session_with_pending_and_minutes(self) -> None:
        """Background work is pending: the turn line says how many tasks run and the minutes left (rounded up)."""
        self.linger.status_value = LingerStatus(pending=2, seconds_left=1000.0, self_turn=False)
        self.run_cmd("/status")
        expected = LINGER_STATUS_TEXT.format(pending=2, minutes=17)
        self.assertIn(expected, self.last().split("\n"))
        self.assertIn("turn: idle — session kept open for background work (2 running, closes in 17 min; /stop closes it)",
                      self.last())

    def test_status_shows_quiet_kept_session(self) -> None:
        """No background work is pending any more: the line says the session closes in a few seconds."""
        self.linger.status_value = LingerStatus(pending=0, seconds_left=300.0, self_turn=False)
        self.run_cmd("/status")
        self.assertIn(LINGER_QUIET_STATUS_TEXT, self.last().split("\n"))
        self.assertNotIn("turn: idle", self.last().split("\n"))

    def test_status_shows_background_update_running(self) -> None:
        """A self turn of the kept session runs: the turn line names it as a background update, with its tool count."""
        self.linger.status_value = LingerStatus(pending=1, seconds_left=300.0, self_turn=True)
        self.live.begin_turn(THREAD, object(), FakeFeed())
        self.run_cmd("/status")
        self.assertIn(SELF_TURN_STATUS_TEXT.format(tools=3), self.last().split("\n"))
        self.assertNotIn("turn: running, 3 tool calls so far", self.last())

    def test_status_of_a_user_turn_on_a_kept_session_is_a_plain_running_turn(self) -> None:
        """A user turn on the kept session is a running turn like any other, not a background update."""
        self.linger.status_value = LingerStatus(pending=1, seconds_left=300.0, self_turn=False)
        self.live.begin_turn(THREAD, object(), FakeFeed())
        self.run_cmd("/status")
        self.assertIn("turn: running, 3 tool calls so far", self.last().split("\n"))
        self.assertNotIn(SELF_TURN_STATUS_TEXT.format(tools=3), self.last())

    def test_status_has_stop_button_while_kept(self) -> None:
        """A kept session with nothing running still gets the ⏹ Stop button, which closes it (/stop does the same)."""
        self.linger.status_value = LingerStatus(pending=0, seconds_left=300.0, self_turn=False)
        self.run_cmd("/status")
        self.assertIn("⏹ Stop", self.button_labels())
        self.assertIn(f"cm:{THREAD}:st:0", self.button_callbacks())

    def test_status_without_a_kept_session_is_as_before(self) -> None:
        """Nothing kept and nothing running: the line is 'turn: idle' and there is no stop button."""
        self.linger.status_value = None
        self.run_cmd("/status")
        self.assertIn("turn: idle", self.last().split("\n"))
        self.assertNotIn("⏹ Stop", self.button_labels())


if __name__ == "__main__":
    unittest.main()
