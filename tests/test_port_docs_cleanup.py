"""Port of fakecli/test_docs_cleanup.py: the RULES text, the dead code of the old per-turn path, the subcommands that
stay, the package docs and the CLI routing. The old suite read the live file's source; these checks read V3.

The old check that the live module no longer defines a name is a module-level scan of every tg_bridge module, so a
name that came back in any module fails here. The old names that V3 renamed are checked under their V3 names, and the
label text stays in the docstring so the port audit can find it.
"""

import contextlib
import importlib
import io
import pkgutil
import unittest

import tg_bridge
from tg_bridge import cli
from tg_bridge.core.turn_text import format_rules

OLD_PER_TURN_NAMES = ("run_turn", "mcp_server", "WRITE_TOOLS", "_current")
KEPT_NAMES = (
    ("cmd_send", "send_to_topic", cli),
    ("cmd_sendfile", "send_file_to_topic", cli),
    ("type_to_pane", "type_to_pane", None),
    ("spawn_seat", "spawn_seat", None),
    ("main", "main", cli),
    ("Turn", "TurnRunner", None),
    ("_run_turn_body", "TurnRunner", None),
)
LIVE_SUBCOMMANDS = ("daemon", "send <topic>", "sendfile <topic>", "type <pane>", "spawn <launcher>")


def _tg_bridge_modules():
    yield tg_bridge
    for info in pkgutil.walk_packages(tg_bridge.__path__, prefix="tg_bridge."):
        yield importlib.import_module(info.name)


class _Recorder:
    """Runners for cli.route that record which one was called, so routing can be checked without any work."""

    def __init__(self) -> None:
        self.called: list[str] = []

    def _call(self, name: str):
        def run(*args):
            self.called.append(name)
            return 0
        return run

    def runners(self) -> cli.Runners:
        return cli.Runners(daemon=self._call("daemon"), send=self._call("send"), sendfile=self._call("sendfile"),
                           type_text=self._call("type"), spawn=self._call("spawn"),
                           hook_permission=self._call("hook"), view=self._call("view"),
                           view_sync=self._call("view-sync"), view_pane=self._call("view-pane"),
                           view_delete=self._call("view-delete"))


class DocsCleanupPort(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = format_rules("side", "Ali")

    def test_rules_text_has_the_headless_checkpoint_and_final_message_sentences(self) -> None:
        """1a: headless / live feed sentence in RULES; 1b: final-message sentence in RULES;
        1c: checkpoint sentence in RULES."""
        self.assertIn("You run headless; the owner sees your tool calls as a live feed and your final message as the "
                      "answer.", self.rules)
        self.assertIn("Put the result in the final message unless asked for a file.", self.rules)
        self.assertIn("Finish or checkpoint within ~20 minutes; if the task is bigger, stop at a checkpoint and say "
                      "what remains.", self.rules)

    def test_rules_have_no_unfilled_placeholders(self) -> None:
        """1d: no unformatted placeholders left in the formatted RULES."""
        self.assertNotIn("{", self.rules)
        self.assertNotIn("}", self.rules)

    def test_the_headless_paragraph_comes_after_the_old_rules(self) -> None:
        """1e: new paragraph is appended after the old rules."""
        self.assertGreater(self.rules.index("You run headless"), self.rules.index("do not use tools."))

    def test_the_old_per_turn_names_are_gone_from_every_module(self) -> None:
        """2: WRITE_TOOLS removed; 2: _current removed; 2: mcp_server removed; 2: run_turn removed."""
        found = {}
        for module in _tg_bridge_modules():
            for name in OLD_PER_TURN_NAMES:
                if hasattr(module, name):
                    found.setdefault(name, []).append(module.__name__)
        self.assertEqual(found, {})

    def test_the_subcommand_names_that_stay_still_exist(self) -> None:
        """3: cmd_send still defined; 3: cmd_sendfile still defined; 3: type_to_pane still defined;
        3: spawn_seat still defined; 3: main still defined; 3: Turn still defined (v3: TurnRunner);
        3: _run_turn_body still defined (v3: TurnRunner); 3: TopicSession still defined (v3: ClaudeSession)."""
        from tg_bridge.agents.claude.session import ClaudeSession
        from tg_bridge.core.turn_runner import TurnRunner
        from tg_bridge.daemon.panes import Panes

        self.assertTrue(callable(cli.send_to_topic))
        self.assertTrue(callable(cli.send_file_to_topic))
        self.assertTrue(callable(cli.type_into_pane))
        self.assertTrue(callable(Panes.spawn_seat))
        self.assertTrue(callable(Panes.type_to_pane))
        self.assertTrue(callable(cli.main))
        self.assertIsNotNone(TurnRunner)
        self.assertIsNotNone(ClaudeSession)

    def test_the_package_docs_list_the_live_subcommands_and_the_feed_and_commands(self) -> None:
        """4: docstring describes the live feed and commands; 4: docstring lists daemon; 4: docstring lists send <topic>;
        4: docstring lists sendfile <topic>; 4: docstring lists type <pane>; 4: docstring lists spawn <launcher>."""
        doc = tg_bridge.__doc__ or ""
        for command in LIVE_SUBCOMMANDS:
            self.assertIn(command, doc)
        self.assertIn("stream-json", doc)
        self.assertIn("/mode", doc)
        self.assertIn("/status", doc)

    def test_the_package_docs_describe_the_codex_launcher(self) -> None:
        """Design section 1: the spec is the live text plus the Codex lines (agent_launchers, codex app-server)."""
        doc = tg_bridge.__doc__ or ""
        for phrase in ("agent_launchers", "codex app-server", "/launcher codex", "$CODEX_HOME"):
            self.assertIn(phrase, doc)

    def test_the_package_docs_do_not_mention_the_removed_mcp_subcommand(self) -> None:
        """4: docstring does not mention the removed mcp subcommand."""
        self.assertNotIn("mcp", (tg_bridge.__doc__ or "").lower())

    def test_routing_has_no_mcp_branch(self) -> None:
        """5a: no mcp branch in main or in the mcp config: 'mcp' is not a command, it prints the spec and exits 0."""
        recorder = _Recorder()
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(cli.route(["mcp", "x"], recorder.runners()), 0)
        self.assertEqual(recorder.called, [])
        self.assertIn("Subcommands:", printed.getvalue())

    def test_routing_keeps_the_type_spawn_and_sendfile_branches(self) -> None:
        """5b: type, spawn and sendfile branches kept; send-file is the same branch as sendfile."""
        recorder = _Recorder()
        runners = recorder.runners()
        self.assertEqual(cli.route(["type", "%1", "hi"], runners), 0)
        self.assertEqual(cli.route(["spawn", "wrap run", "name", "text"], runners), 0)
        self.assertEqual(cli.route(["sendfile", "side", "/tmp/x"], runners), 0)
        self.assertEqual(cli.route(["send-file", "side", "/tmp/x"], runners), 0)
        self.assertEqual(recorder.called, ["type", "spawn", "sendfile", "sendfile"])


if __name__ == "__main__":
    unittest.main()
