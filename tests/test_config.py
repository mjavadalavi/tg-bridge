"""Config and LauncherSpec: launcher backend types, name clashes, rejected entries, precedence, typed getters.

Port of the launcher checks in scratchpad/fakecli/test_launcher.py and test_launcher_cwd.py (the backend type
part) plus the config defaults.
"""

import json
import os
import unittest

from tests.support import TempHome
from tg_bridge.config import (
    BACKEND_CLAUDE,
    BACKEND_CODEX,
    CLAUDE_CONFIG_DIR,
    MODEL_CHOICES,
    SECTION_AGENT,
    SECTION_CLAUDE,
    TRANSPORT_PRINT,
    TRANSPORT_TUI,
    TRANSPORTS,
    Config,
    LauncherError,
    LauncherSpec,
    default_config,
    default_launchers,
    expand_home,
    split_command,
    tilde,
)
from tg_bridge.paths import Paths



class _Log:
    """Collects log lines so tests can count them."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, message: str) -> None:
        self.lines.append(message)


def _config(home: str, data: dict, log: _Log | None = None, env: dict | None = None) -> Config:
    return Config(data, home=home, env={} if env is None else env, log=log or _Log())


class StringAndDictLauncherTests(unittest.TestCase):
    def test_string_launcher_is_claude_with_default_config_dir(self) -> None:
        """Ported labels:
        8a: a string entry gets ~/.claude, a dict keeps its folder
        8d: a string entry resolves to ~/.claude
        """
        with TempHome() as home:
            cfg = _config(home.home, {"launchers": {"plain": "plaincmd --flag"}})
            spec = cfg.launcher_specs()["plain"]
            self.assertEqual(spec.backend, BACKEND_CLAUDE)
            self.assertEqual(spec.command, "plaincmd --flag")
            self.assertEqual(spec.argv, ("plaincmd", "--flag"))
            self.assertEqual(spec.config_dir, os.path.join(home.home, ".claude"))

    def test_dict_launcher_is_claude_with_expanded_config_dir(self) -> None:
        """Ported labels:
        8a: a string entry gets ~/.claude, a dict keeps its folder
        8c: ~ is expanded
        """
        with TempHome() as home:
            cfg = _config(home.home, {"launchers": {"wrap": {"cmd": "wrap run",
                                                               "config_dir": "~/.local/share/x"}}})
            spec = cfg.launcher_specs()["wrap"]
            self.assertEqual(spec.backend, BACKEND_CLAUDE)
            self.assertEqual(spec.command, "wrap run")
            self.assertEqual(spec.argv, ("wrap", "run"))
            self.assertEqual(spec.config_dir, os.path.join(home.home, ".local", "share", "x"))
            self.assertEqual(cfg.launcher_config_dir("wrap"), spec.config_dir)

    def test_launcher_choices_and_folders_for_both_entry_forms(self) -> None:
        """Ported labels:
        8b: launcher_choices gives commands for both forms
        8e: an unknown name has no folder
        """
        with TempHome() as home:
            cfg = _config(home.home, {"launchers": {"x": "cmd x --y", "y": {"cmd": "cmd y", "config_dir": "~/foo-dir"}}})
            self.assertEqual(cfg.launcher_choices(), {"x": "cmd x --y", "y": "cmd y"})
            self.assertIsNone(cfg.launcher_config_dir("nope"))

    def test_default_launcher_folder_is_the_claude_default(self) -> None:
        """Ported labels:
        8h: the default folder of the default launcher is ~/.claude
        """
        with TempHome() as home:
            cfg = _config(home.home, {})
            self.assertEqual(cfg.launcher_config_dir("claude"), os.path.join(home.home, ".claude"))

    def test_dict_launcher_without_config_dir_uses_claude_default(self) -> None:
        with TempHome() as home:
            cfg = _config(home.home, {"launchers": {"c": {"cmd": "claude"}}})
            self.assertEqual(cfg.launcher_specs()["c"].config_dir, os.path.join(home.home, ".claude"))

    def test_default_launchers_are_used_when_launchers_missing_or_empty(self) -> None:
        """Ported labels:
        B1: default launchers when the key is absent
        B2: an empty launchers dict falls back to the default
        8f: no config keeps the default commands
        """
        with TempHome() as home:
            for data in ({}, {"launchers": {}}, {"launchers": "not-a-dict"}):
                with self.subTest(data=data):
                    cfg = _config(home.home, data)
                    self.assertEqual(sorted(cfg.launcher_choices()), ["claude"])
                    self.assertEqual(cfg.launcher_choices(), {"claude": "claude"})

    def test_launcher_argv_is_shell_split(self) -> None:
        """Ported labels:
        A1: multi-word launcher is shlex-split
        A2: single-word launcher unchanged
        A3: quotes respected
        A4: blank launcher '   ' gives a clear error
        A4: blank launcher '' gives a clear error
        A4: blank launcher None gives a clear error
        """
        self.assertEqual(split_command("wrap run --x 'a b'"), ("wrap", "run", "--x", "a b"))
        self.assertEqual(split_command("claude"), ("claude",))
        self.assertEqual(split_command('/opt/my bin/x --a "b c"'), ("/opt/my", "bin/x", "--a", "b c"))
        for blank in ("   ", "", None):
            with self.subTest(blank=blank):
                with self.assertRaises(ValueError) as caught:
                    split_command(blank)
                self.assertIn("empty", str(caught.exception))
        with self.assertRaises(ValueError):
            split_command("claude 'unclosed")
        with self.assertRaises(LauncherError):  # a blank entry never becomes a spec, so no turn can run it
            LauncherSpec.parse("blank", "   ", section=SECTION_CLAUDE, home="/h", codex_home="/c")

    def test_the_package_docstring_documents_the_launchers(self) -> None:
        """Ported labels:
        B5: module docstring documents /launcher, launchers and the multi-word note
        """
        import tg_bridge
        self.assertIn("/launcher", tg_bridge.__doc__)
        self.assertIn("launchers", tg_bridge.__doc__)
        self.assertIn("split like a shell line", tg_bridge.__doc__)

    def test_default_config_carries_the_default_launchers(self) -> None:
        """Ported labels:
        B4: DEFAULT_CONFIG carries the launchers key
        """
        self.assertEqual(default_config()["launchers"], default_launchers())




class AgentLauncherTests(unittest.TestCase):
    def test_agent_launcher_is_codex_with_codex_home_default(self) -> None:
        with TempHome() as home:
            cfg = _config(home.home, {"agent_launchers": {"codex": {"backend": "codex", "command": "codex",
                                                                   "models": ["gpt-x"]}}})
            spec = cfg.launcher_specs()["codex"]
            self.assertEqual(spec.backend, BACKEND_CODEX)
            self.assertEqual(spec.command, "codex")
            self.assertEqual(spec.config_dir, os.path.join(home.home, ".codex"))
            self.assertEqual(cfg.codex_home, os.path.join(home.home, ".codex"))
            self.assertEqual(cfg.launcher_models("codex"), ("gpt-x",))

    def test_codex_home_comes_from_env_and_config_dir_overrides_it(self) -> None:
        with TempHome() as home:
            env = {"CODEX_HOME": os.path.join(home.home, "from-env")}
            cfg = _config(home.home, {"agent_launchers": {
                "codex": {"backend": "codex", "command": "codex"},
                "codex2": {"backend": "codex", "command": "codex", "config_dir": "~/own"},
            }}, env=env)
            self.assertEqual(cfg.launcher_specs()["codex"].config_dir, os.path.join(home.home, "from-env"))
            self.assertEqual(cfg.launcher_specs()["codex2"].config_dir, os.path.join(home.home, "own"))

    def test_agent_launchers_are_not_listed_in_launchers_section(self) -> None:
        with TempHome() as home:
            cfg = _config(home.home, {"agent_launchers": {"codex": {"backend": "codex", "command": "codex"}}})
            self.assertIn("codex", cfg.launcher_choices())
            self.assertEqual(cfg.launcher_choices()["codex"], "codex")

    def test_unknown_backend_is_hidden_with_one_log_line(self) -> None:
        with TempHome() as home:
            log = _Log()
            cfg = _config(home.home, {"agent_launchers": {"weird": {"backend": "gemini", "command": "g"}}}, log)
            self.assertNotIn("weird", cfg.launcher_specs())
            self.assertEqual(len(log.lines), 1)
            self.assertIn("weird", log.lines[0])

    def test_name_clash_keeps_the_launchers_entry(self) -> None:
        with TempHome() as home:
            log = _Log()
            cfg = _config(home.home, {
                "launchers": {"shared": "claude-shared"},
                "agent_launchers": {"shared": {"backend": "codex", "command": "codex"}},
            }, log)
            spec = cfg.launcher_specs()["shared"]
            self.assertEqual(spec.backend, BACKEND_CLAUDE)
            self.assertEqual(spec.command, "claude-shared")
            self.assertEqual(len(log.lines), 1)

    def test_agent_launcher_with_blank_command_is_rejected(self) -> None:
        with TempHome() as home:
            log = _Log()
            cfg = _config(home.home, {"agent_launchers": {"codex": {"backend": "codex", "command": "  "}}}, log)
            self.assertNotIn("codex", cfg.launcher_specs())
            self.assertEqual(len(log.lines), 1)


class RejectedLauncherTests(unittest.TestCase):
    def test_launchers_dict_with_backend_key_is_rejected_without_echoing_the_value(self) -> None:
        with TempHome() as home:
            log = _Log()
            secret = "SECRET-VALUE-123"
            cfg = _config(home.home, {"launchers": {
                "bad": {"backend": "codex", "command": "codex", "token": secret},
                "good": "goodcmd",
            }}, log)
            self.assertNotIn("bad", cfg.launcher_specs())
            self.assertIn("good", cfg.launcher_specs())
            self.assertEqual(len(log.lines), 1)
            self.assertIn("bad", log.lines[0])
            self.assertNotIn(secret, log.lines[0])
            self.assertNotIn("{", log.lines[0])

    def test_launchers_dict_without_cmd_is_rejected_with_one_log_line(self) -> None:
        with TempHome() as home:
            log = _Log()
            cfg = _config(home.home, {"launchers": {"nocmd": {"config_dir": "~/x"}}}, log)
            self.assertNotIn("nocmd", cfg.launcher_specs())
            self.assertEqual(len(log.lines), 1)

    def test_blank_command_raises_value_error_in_parse(self) -> None:
        with self.assertRaises(ValueError):
            LauncherSpec.parse("blank", "   ", section=SECTION_CLAUDE, home="/h", codex_home="/h/.codex")
        with self.assertRaises(LauncherError):
            LauncherSpec.parse("blank", {"cmd": ""}, section=SECTION_CLAUDE, home="/h", codex_home="/h/.codex")

    def test_unknown_section_and_non_mapping_agent_entry_are_rejected(self) -> None:
        with self.assertRaises(LauncherError):
            LauncherSpec.parse("x", "cmd", section="other", home="/h", codex_home="/h/.codex")
        with self.assertRaises(LauncherError):
            LauncherSpec.parse("x", "codex", section=SECTION_AGENT, home="/h", codex_home="/h/.codex")

    def test_broken_quoting_is_rejected_at_parse(self) -> None:
        with self.assertRaises(LauncherError):
            LauncherSpec.parse("q", "claude 'open", section=SECTION_CLAUDE, home="/h", codex_home="/h/.codex")


class PrecedenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = TempHome()
        self.home.__enter__()
        self.cfg = _config(self.home.home, {
            "launchers": {"wrap": {"cmd": "wrap run", "config_dir": "~/.a"}, "plain": "plain"},
            "agent_launchers": {"codex": {"backend": "codex", "command": "codex"}},
        })

    def tearDown(self) -> None:
        self.home.__exit__(None, None, None)

    def test_pick_in_config_wins_over_topic_and_stored(self) -> None:
        spec = self.cfg.resolve_launcher("plain", "wrap run", "claude")
        self.assertEqual(spec.name, "plain")

    def test_pick_not_in_config_is_ignored(self) -> None:
        spec = self.cfg.resolve_launcher("gone", "wrap run", "claude")
        self.assertEqual(spec.name, "wrap")
        self.assertEqual(spec.config_dir, os.path.join(self.home.home, ".a"))

    def test_topic_wins_over_stored_and_matches_a_visible_command(self) -> None:
        spec = self.cfg.resolve_launcher("", "wrap run", "plain")
        self.assertEqual(spec.name, "wrap")

    def test_stored_used_when_pick_and_topic_are_empty(self) -> None:
        self.assertEqual(self.cfg.resolve_launcher("", "", "plain").name, "plain")

    def test_unmatched_command_runs_as_unnamed_claude_launcher(self) -> None:
        spec = self.cfg.resolve_launcher("", "altclaude --x", "")
        self.assertEqual(spec.name, "altclaude --x")
        self.assertEqual(spec.backend, BACKEND_CLAUDE)
        self.assertEqual(spec.argv, ("altclaude", "--x"))
        self.assertEqual(spec.config_dir, os.path.join(self.home.home, ".claude"))

    def test_unmatched_command_is_marked_unconfigured(self) -> None:
        self.assertFalse(self.cfg.resolve_launcher("", "altclaude --x", "").configured)

    def test_visible_launcher_is_marked_configured(self) -> None:
        self.assertTrue(self.cfg.resolve_launcher("", "wrap run", "").configured)
        self.assertTrue(self.cfg.resolve_launcher("plain", "", "").configured)

    def test_codex_command_resolves_to_the_codex_spec(self) -> None:
        self.assertEqual(self.cfg.resolve_launcher("", "codex", "").backend, BACKEND_CODEX)

    def test_all_empty_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.cfg.resolve_launcher("", "", "")


class TypedGetterTests(unittest.TestCase):
    def test_typed_getter_defaults(self) -> None:
        with TempHome() as home:
            cfg = _config(home.home, {})
            self.assertEqual(cfg.max_turn_seconds(), 1800)
            self.assertEqual(cfg.hard_cap_seconds(), 21600)
            self.assertEqual(cfg.poll_seconds(), 50)
            self.assertEqual(cfg.max_output_chunks(), 4)
            self.assertEqual(cfg.progress_flush_seconds(), 3)
            self.assertEqual(cfg.idle_ping_seconds(), 60)
            self.assertEqual(cfg.answer_doc_threshold(), 9000)
            self.assertEqual(cfg.allow_user_ids(), [])
            self.assertEqual(cfg.topics(), {})
            self.assertIsNone(cfg.default_topic())
            self.assertEqual(cfg.spawn(), {})
            self.assertEqual(cfg.cwd_shortcuts(), ())
            self.assertIsNone(cfg.egress_proxy())
            self.assertEqual(cfg.models(), MODEL_CHOICES)
            self.assertEqual(cfg.search_roots(), (os.path.join(home.home, "workspace"), home.home))

    def test_default_mode_fallback(self) -> None:
        """Ported labels:
        A: DEFAULT_CONFIG carries default_mode
        A: no config key -> bypassPermissions
        A: config default_mode honoured
        """
        self.assertEqual(default_config()["default_mode"], "bypassPermissions")
        with TempHome() as home:
            self.assertEqual(_config(home.home, {}).default_mode(), "bypassPermissions")
            self.assertEqual(_config(home.home, {"default_mode": ""}).default_mode(), "bypassPermissions")
            self.assertEqual(_config(home.home, {"default_mode": "plan"}).default_mode(), "plan")

    def test_topic_falls_back_to_default_topic_then_empty(self) -> None:
        with TempHome() as home:
            cfg = _config(home.home, {"topics": {"a": {"cwd": "/a"}}, "default_topic": {"cwd": "/d"}})
            self.assertEqual(cfg.topic("a"), {"cwd": "/a"})
            self.assertEqual(cfg.topic("zzz"), {"cwd": "/d"})
            self.assertEqual(_config(home.home, {}).topic("zzz"), {})
            self.assertEqual(_config(home.home, {"topics": {"bad": "not-an-object"}}).topic("bad"), {})

    def test_getters_return_copies_not_the_internal_data(self) -> None:
        with TempHome() as home:
            cfg = _config(home.home, {"allow_user_ids": [1]})
            cfg.allow_user_ids().append(2)
            self.assertEqual(cfg.allow_user_ids(), [1])

    def test_configured_values_are_returned(self) -> None:
        with TempHome() as home:
            cfg = _config(home.home, {"models": ["m1"], "cwd_shortcuts": ["/x"], "egress_proxy": "http://p:1",
                                      "search_roots": ["/r"], "max_turn_seconds": 60})
            self.assertEqual(cfg.models(), ("m1",))
            self.assertEqual(cfg.cwd_shortcuts(), ("/x",))
            self.assertEqual(cfg.egress_proxy(), "http://p:1")
            self.assertEqual(cfg.search_roots(), ("/r",))
            self.assertEqual(cfg.max_turn_seconds(), 60)

    def test_default_config_is_a_fresh_copy(self) -> None:
        first = default_config()
        first["topics"]["x"] = {"cwd": "changed"}
        self.assertEqual(default_config()["topics"], {})

    def test_input_data_is_not_mutated_by_construction(self) -> None:
        with TempHome() as home:
            data = {"launchers": {"x": "xcmd"}}
            _config(home.home, data).launcher_specs()
            self.assertEqual(data, {"launchers": {"x": "xcmd"}})


class ConfigFileTests(unittest.TestCase):
    def test_load_missing_file_uses_defaults_and_writes_nothing(self) -> None:
        with TempHome() as home:
            paths = Paths.from_home(home.home)
            cfg = Config.load(paths, env={}, log=_Log())
            self.assertEqual(sorted(cfg.launcher_choices()), ["claude"])
            self.assertFalse(os.path.exists(paths.config))

    def test_load_reads_the_file_and_leaves_it_byte_identical(self) -> None:
        """Ported labels:
        B3: config launchers override the defaults
        """
        with TempHome() as home:
            paths = Paths.from_home(home.home)
            with open(paths.config, "w", encoding="utf-8") as handle:
                json.dump({"launchers": {"mine": "minecmd"}, "default_mode": "plan"}, handle)
            with open(paths.config, "rb") as handle:
                before = handle.read()
            cfg = Config.load(paths, env={}, log=_Log())
            self.assertEqual(list(cfg.launcher_choices()), ["mine"])
            self.assertEqual(cfg.default_mode(), "plan")
            with open(paths.config, "rb") as handle:
                self.assertEqual(handle.read(), before)

    def test_non_object_config_is_refused(self) -> None:
        with TempHome() as home:
            paths = Paths.from_home(home.home)
            with open(paths.config, "w", encoding="utf-8") as handle:
                handle.write("[1, 2]")
            with self.assertRaises(ValueError):
                Config.load(paths, env={}, log=_Log())


class LingerConfigTests(unittest.TestCase):
    def test_defaults_when_absent(self) -> None:
        with TempHome() as home:
            cfg = _config(home.home, {})
            self.assertEqual(cfg.linger_seconds(), 1800)
            self.assertEqual(cfg.linger_max_sessions(), 3)

    def test_zero_turns_linger_off(self) -> None:
        with TempHome() as home:
            cfg = _config(home.home, {"linger_seconds": 0, "linger_max_sessions": 0})
            self.assertEqual(cfg.linger_seconds(), 0)
            self.assertEqual(cfg.linger_max_sessions(), 0)

    def test_bad_values_fall_back(self) -> None:
        with TempHome() as home:
            for bad in (-1, "x", True, None, 1.5):
                with self.subTest(value=bad):
                    cfg = _config(home.home, {"linger_seconds": bad, "linger_max_sessions": bad})
                    self.assertEqual(cfg.linger_max_sessions(), 3)
                    if bad == 1.5:
                        self.assertEqual(cfg.linger_seconds(), 1.5)
                    else:
                        self.assertEqual(cfg.linger_seconds(), 1800)
            self.assertEqual(_config(home.home, {"linger_seconds": 12.5}).linger_seconds(), 12.5)

    def test_non_finite_values_fall_back(self) -> None:
        """json accepts NaN and Infinity; neither is a usable number of seconds or of sessions."""
        with TempHome() as home:
            for bad in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(value=bad):
                    cfg = _config(home.home, {"linger_seconds": bad, "linger_max_sessions": bad})
                    self.assertEqual(cfg.linger_seconds(), 1800)
                    self.assertEqual(cfg.linger_max_sessions(), 3)

    def test_example_config_loads_both_keys(self) -> None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "config.example.json"), encoding="utf-8") as handle:
            data = json.load(handle)
        with TempHome() as home:
            cfg = _config(home.home, data)
            self.assertEqual((cfg.linger_seconds(), cfg.linger_max_sessions()), (1800, 3))


class HomeHelperTests(unittest.TestCase):
    def test_expand_home_and_tilde_round_trip(self) -> None:
        self.assertEqual(expand_home("~/a/b", "/h"), os.path.join("/h", "a", "b"))
        self.assertEqual(expand_home("~", "/h"), "/h")
        self.assertEqual(expand_home("/abs", "/h"), "/abs")
        self.assertEqual(tilde("/h/a", "/h"), "~/a")
        self.assertEqual(tilde("/h", "/h"), "~")
        self.assertEqual(tilde("/hx/a", "/h"), "/hx/a")

    def test_claude_config_dir_constant_is_home_relative(self) -> None:
        self.assertEqual(CLAUDE_CONFIG_DIR, "~/.claude")
        self.assertEqual(expand_home(CLAUDE_CONFIG_DIR, "/h"), os.path.join("/h", ".claude"))


class TransportTests(unittest.TestCase):
    """The per-launcher transport: print (default) or tui, accepted for claude only, invalid values hide the entry."""

    def test_transport_constants(self) -> None:
        self.assertEqual(TRANSPORT_PRINT, "print")
        self.assertEqual(TRANSPORT_TUI, "tui")
        self.assertEqual(TRANSPORTS, ("print", "tui"))

    def test_string_and_plain_object_launchers_default_to_print(self) -> None:
        with TempHome() as home:
            cfg = _config(home.home, {"launchers": {"plain": "plaincmd", "obj": {"cmd": "objcmd"}}})
            self.assertEqual(cfg.launcher_specs()["plain"].transport, TRANSPORT_PRINT)
            self.assertEqual(cfg.launcher_specs()["obj"].transport, TRANSPORT_PRINT)

    def test_spec_default_transport_is_print(self) -> None:
        spec = LauncherSpec(name="c", backend=BACKEND_CLAUDE, command="c", argv=("c",), config_dir="/x")
        self.assertEqual(spec.transport, TRANSPORT_PRINT)

    def test_tui_is_accepted_in_launchers_section(self) -> None:
        with TempHome() as home:
            log = _Log()
            cfg = _config(home.home, {"launchers": {"teamclaude": {"cmd": "teamclaude", "transport": "tui"}}}, log)
            spec = cfg.launcher_specs()["teamclaude"]
            self.assertEqual(spec.transport, TRANSPORT_TUI)
            self.assertEqual(spec.backend, BACKEND_CLAUDE)
            self.assertEqual(log.lines, [])

    def test_tui_is_accepted_for_a_claude_agent_launcher(self) -> None:
        with TempHome() as home:
            log = _Log()
            cfg = _config(home.home, {"agent_launchers": {
                "teamclaude": {"backend": "claude", "command": "teamclaude", "transport": "tui"},
            }}, log)
            self.assertEqual(cfg.launcher_specs()["teamclaude"].transport, TRANSPORT_TUI)
            self.assertEqual(log.lines, [])

    def test_codex_agent_launcher_accepts_print_and_rejects_tui(self) -> None:
        with TempHome() as home:
            log = _Log()
            cfg = _config(home.home, {"agent_launchers": {
                "ok": {"backend": "codex", "command": "codex", "transport": "print"},
                "bad": {"backend": "codex", "command": "codex", "transport": "tui"},
            }}, log)
            self.assertEqual(cfg.launcher_specs()["ok"].transport, TRANSPORT_PRINT)
            self.assertNotIn("bad", cfg.launcher_specs())
            self.assertEqual(len(log.lines), 1)
            self.assertIn("bad", log.lines[0])
            self.assertIn("only a claude launcher", log.lines[0])

    def test_invalid_transport_hides_launcher_with_one_log_line_and_no_value(self) -> None:
        with TempHome() as home:
            log = _Log()
            secret = "SECRET-TRANSPORT-987"
            cfg = _config(home.home, {"launchers": {
                "pty": {"cmd": "pty", "transport": secret},
                "num": {"cmd": "num", "transport": 5},
                "good": "goodcmd",
            }}, log)
            self.assertNotIn("pty", cfg.launcher_specs())
            self.assertNotIn("num", cfg.launcher_specs())
            self.assertIn("good", cfg.launcher_specs())
            self.assertEqual(len(log.lines), 2)
            for line in log.lines:
                self.assertIn("transport must be one of: print, tui", line)
                self.assertNotIn(secret, line)
            self.assertIn("'pty'", log.lines[0])

    def test_invalid_transport_on_agent_launcher_hides_it(self) -> None:
        with TempHome() as home:
            log = _Log()
            cfg = _config(home.home, {"agent_launchers": {
                "teamclaude": {"backend": "claude", "command": "teamclaude", "transport": "pipe"},
            }}, log)
            self.assertNotIn("teamclaude", cfg.launcher_specs())
            self.assertEqual(len(log.lines), 1)
            self.assertIn("transport must be one of: print, tui", log.lines[0])

    def test_parse_passes_transport_to_the_spec(self) -> None:
        spec = LauncherSpec.parse("t", {"cmd": "t run", "transport": "tui"},
                                  section=SECTION_CLAUDE, home="/h", codex_home="/h/.codex")
        self.assertEqual(spec.transport, TRANSPORT_TUI)
        self.assertEqual(spec.argv, ("t", "run"))

    def test_unnamed_launcher_from_a_bare_command_stays_print(self) -> None:
        with TempHome() as home:
            cfg = _config(home.home, {"launchers": {"teamclaude": {"cmd": "teamclaude", "transport": "tui"}}})
            spec = cfg.resolve_launcher("", "some-other-claude --x", "")
            self.assertFalse(spec.configured)
            self.assertEqual(spec.transport, TRANSPORT_PRINT)
            matched = cfg.resolve_launcher("", "teamclaude", "")
            self.assertEqual(matched.transport, TRANSPORT_TUI)


if __name__ == "__main__":
    unittest.main()
