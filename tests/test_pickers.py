"""Picker builders (STEP 46), ported from fakecli/test_launcher.py, test_modes.py and test_buttons.py.

Labels of the old checks are kept in the test docstrings (A, B, D, E, F, G, H, Q, R, T of test_buttons.py; E, G of
test_launcher.py; B, E, F of test_modes.py). Every picker is built for two launchers: a Claude launcher ('wrap')
and a Codex launcher ('gpt', command 'codex'), with the real ClaudeBackend and CodexBackend from the default registry,
so the mode list and the live flags are the ones the bridge uses. Nothing here touches Telegram, the database or the
filesystem beyond a TempHome that holds the config paths.
"""

import json
import unittest

from tests.fakes.clock import FakeClock
from tests.support import TempHome
from tg_bridge.agents.defaults import build_default_registry
from tg_bridge.bot.pickers import (
    NOT_LIVE_NOTE,
    Picker,
    launcher_picker,
    mode_picker,
    model_picker,
    resume_picker,
    rules_picker,
    started_label,
    status_rows,
)
from tg_bridge.config import Config
from tg_bridge.logging_ import Logger
from tg_bridge.store.rules import RuleRecord
from tg_bridge.store.sessions import HistoryRecord
from tg_bridge.telegram.keyboards import CALLBACK_DATA_LIMIT

THREAD = 9301
MODELS = ("opus", "sonnet", "haiku", "fable", "default")
CLAUDE_UUID = "11111111-2222-3333-4444-555555555555"
CODEX_UUID = "019b0000-aaaa-bbbb-cccc-000000000001"


def _labels(picker: Picker) -> list[list[str]]:
    return [[b["text"] for b in row] for row in json.loads(picker.markup())["inline_keyboard"]]


def _callbacks(picker: Picker) -> list[str]:
    return [b["callback_data"] for row in json.loads(picker.markup())["inline_keyboard"] for b in row]


def _flat_labels(picker: Picker) -> list[str]:
    return [label for row in _labels(picker) for label in row]


def _config(home: str) -> Config:
    """Two Claude launchers and one Codex launcher ('gpt', command 'codex') with its own model list."""
    return Config(
        {
            "launchers": {"wrap": "wrap run", "plain": "plain", "claude": "claude"},
            "agent_launchers": {"gpt": {"backend": "codex", "command": "codex", "models": ["gpt-6.1-sol"]}},
        },
        home=home,
        env={},
    )


class PickerCase(unittest.TestCase):
    """Builds the config and both real backends in a TempHome for every test."""

    def setUp(self) -> None:
        home = TempHome()
        self.addCleanup(home.__exit__, None, None, None)
        home.__enter__()
        self.config = _config(home.home)
        logger = Logger(f"{home.state}/picker-test.log")
        self.registry = build_default_registry(self.config, logger, FakeClock(), title_launcher="")
        self.claude = self.registry.get("claude")
        self.codex = self.registry.get("codex")
        self.launchers = tuple(self.config.launcher_specs().values())
        self.names = {spec.command: spec.name for spec in self.launchers}


class ModelPickerTests(PickerCase):
    def test_model_rows_and_current_mark(self) -> None:
        """A: 5 models -> rows of 2,2,1; ✓ marks only the current value; callback is cm:<thread>:mo:<model>.

        Ported labels:
        A: 5 models -> rows of 2,2,1 ([2, 2, 1])
        A: callback is cm:<thread>:mo:<model name>
        """
        picker = model_picker(THREAD, MODELS, "sonnet", live=True)
        self.assertEqual([len(row) for row in _labels(picker)], [2, 2, 1])
        self.assertEqual(_flat_labels(picker), ["opus", "✓ sonnet", "haiku", "fable", "default"])
        self.assertEqual(_callbacks(picker), [f"cm:{THREAD}:mo:{m}" for m in MODELS])
        self.assertEqual(picker.text, "🧠 model: sonnet")

    def test_unset_model_marks_default(self) -> None:
        """B: no model set -> the text says default and ✓ sits on default.

        Ported labels:
        B: default is marked ✓ when unset
        B: one button per configured model
        """
        picker = model_picker(THREAD, MODELS, None, live=True)
        self.assertEqual(picker.text, "🧠 model: default")
        self.assertEqual(_labels(picker)[-1], ["✓ default"])
        self.assertEqual(len(_callbacks(picker)), 5)

    def test_colon_in_model_name_is_kept(self) -> None:
        """A model name may contain ':' (qwen3:8b); it is the last callback field, so it round-trips."""
        picker = model_picker(THREAD, ("qwen3:8b",), None, live=True)
        self.assertEqual(_callbacks(picker), [f"cm:{THREAD}:mo:qwen3:8b"])

    def test_oversized_model_is_left_out_and_others_stay(self) -> None:
        """A: every callback stays within 64 bytes; a model whose callback would not fit is omitted, not an error.

        Ported labels:
        A: callback data stays within 64 bytes
        """
        too_long = "m" * 70
        picker = model_picker(THREAD, (too_long, "opus"), "opus", live=True)
        self.assertEqual(_flat_labels(picker), ["✓ opus"])
        for data in _callbacks(picker):
            self.assertLessEqual(len(data.encode()), CALLBACK_DATA_LIMIT)

    def test_not_live_model_says_applies_from_next_message(self) -> None:
        """B2: a backend whose model change is not live (Codex) shows the note under the text."""
        picker = model_picker(THREAD, ("gpt-6.1-sol",), "gpt-6.1-sol", live=self.codex.capabilities.live_model)
        self.assertEqual(picker.text, f"🧠 model: gpt-6.1-sol\n{NOT_LIVE_NOTE}")
        self.assertEqual(_flat_labels(picker), ["✓ gpt-6.1-sol"])

    def test_live_model_has_no_note(self) -> None:
        """B2: a live backend (Claude) shows the plain text only."""
        picker = model_picker(THREAD, MODELS, "opus", live=self.claude.capabilities.live_model)
        self.assertEqual(picker.text, "🧠 model: opus")
        self.assertNotIn(NOT_LIVE_NOTE, picker.text)


class ModePickerTests(PickerCase):
    def test_mode_rows_for_claude(self) -> None:
        """E: /mode picker rows are [✓ bypass, acceptEdits] and [default, plan], callbacks md:<index>.

        Ported labels:
        E: four modes, bypass is ✓ (the stored value)
        """
        modes = self.claude.modes()
        picker = mode_picker(THREAD, modes, "bypassPermissions", live=self.claude.capabilities.live_mode)
        self.assertEqual(_labels(picker), [["✓ bypass", "acceptEdits"], ["default", "plan"]])
        self.assertEqual(_callbacks(picker), [f"cm:{THREAD}:md:{i}" for i in range(4)])
        self.assertEqual(picker.text, "🛡 mode: bypassPermissions")

    def test_mode_rows_for_codex_have_the_codex_list(self) -> None:
        """E: the Codex picker is built from the Codex backend's own mode list, with the same four keys."""
        modes = self.codex.modes()
        picker = mode_picker(THREAD, modes, "acceptEdits", live=self.codex.capabilities.live_mode)
        self.assertEqual(_labels(picker), [["bypass", "✓ acceptEdits"], ["default", "plan"]])
        self.assertEqual([m.key for m in modes], ["bypassPermissions", "acceptEdits", "default", "plan"])

    def test_claude_mode_change_has_no_note(self) -> None:
        """E2: a live backend (Claude) says nothing about when the mode applies."""
        picker = mode_picker(THREAD, self.claude.modes(), "acceptEdits", live=self.claude.capabilities.live_mode)
        self.assertEqual(picker.text, "🛡 mode: acceptEdits")

    def test_codex_mode_change_says_applies_from_next_message(self) -> None:
        """E2: Codex cannot change the mode inside a running turn, so its picker says so."""
        picker = mode_picker(THREAD, self.codex.modes(), "default", live=self.codex.capabilities.live_mode)
        self.assertEqual(picker.text, f"🛡 mode: default\n{NOT_LIVE_NOTE}")


class LauncherPickerTests(PickerCase):
    def test_launcher_labels_and_check_for_claude(self) -> None:
        """E1: ✓ on the current Claude launcher; callbacks cm:<thread>:la:<index>; every callback fits 64 bytes."""
        picker = launcher_picker(THREAD, self.launchers, "wrap", "wrap run")
        self.assertEqual(picker.text, "🚀 launcher: wrap")
        self.assertEqual(_flat_labels(picker), ["✓ wrap", "plain", "claude", "gpt · codex"])
        self.assertEqual(_callbacks(picker), [f"cm:{THREAD}:la:{i}" for i in range(4)])
        for data in _callbacks(picker):
            self.assertLessEqual(len(data.encode()), CALLBACK_DATA_LIMIT)

    def test_codex_launcher_carries_backend_tag(self) -> None:
        """A Codex launcher's button shows its backend tag; the Claude buttons keep their plain names."""
        picker = launcher_picker(THREAD, self.launchers, "gpt", "codex")
        self.assertEqual(_flat_labels(picker), ["wrap", "plain", "claude", "✓ gpt · codex"])
        self.assertEqual(picker.text, "🚀 launcher: gpt")
        self.assertEqual(_callbacks(picker)[3], f"cm:{THREAD}:la:3")

    def test_text_falls_back_to_the_command_then_none(self) -> None:
        """G1: with no launcher name the text shows the command; with neither, 'none' and no ✓."""
        self.assertEqual(launcher_picker(THREAD, self.launchers, None, "plain").text, "🚀 launcher: plain")
        empty = launcher_picker(THREAD, self.launchers, None, "")
        self.assertEqual(empty.text, "🚀 launcher: none")
        self.assertNotIn("✓", "".join(_flat_labels(empty)))


class ResumePickerTests(PickerCase):
    def _history(self) -> list[HistoryRecord]:
        """Ten sessions, newest first. Ids 10..1; the launcher alternates between wrap and codex."""
        records = []
        for n in range(10, 0, -1):
            launcher = "codex" if n % 2 else "wrap run"
            records.append(
                HistoryRecord(
                    id=n,
                    thread_id=THREAD,
                    uuid=f"uuid-{n:02d}-xxxxxxxx",
                    started=f"2026-10-24T06:{n:02d}:00+00:00",
                    first_text=f"topic talk number {n}",
                    cwd="/srv/app",
                    launcher=launcher,
                    backend="codex" if n % 2 else "claude",
                )
            )
        return records

    def test_last_eight_newest_first_with_tehran_time(self) -> None:
        """Q: the last 8 sessions, one per row, callback cm:<thread>:rs:<id>; the label is MM-DD HH:MM Tehran time.

        Ported labels:
        Q: newest first, callback cm:<thread>:rs:<rowid>
        """
        picker = resume_picker(THREAD, self._history(), "", "wrap run", self.names)
        self.assertEqual(len(picker.rows), 8)
        self.assertTrue(all(len(row) == 1 for row in picker.rows))
        self.assertEqual(_callbacks(picker)[0], f"cm:{THREAD}:rs:10")
        self.assertEqual(_flat_labels(picker)[0], "10-24 09:40 · topic talk number 10…")
        # the fixture has sessions of the other launcher, so the ⚠ explanation follows the fresh-session text
        self.assertEqual(picker.text, "↩️ the next message starts a fresh session\n"
                         "⚠ = made by another launcher: tapping it switches the launcher to that one")

    def test_current_session_is_marked_and_text_says_resumes(self) -> None:
        """Q: the session in use gets ✓; the text says the next message resumes it.

        Ported labels:
        Q: only the session in use is ✓ ('✓ 10-24 09:37 · topic talk number 7…')
        """
        history = self._history()
        picker = resume_picker(THREAD, history, history[2].uuid, "wrap run", self.names)
        labels = _flat_labels(picker)
        self.assertEqual(sum(label.startswith("✓ ") for label in labels), 1)
        self.assertTrue(labels[2].startswith("✓ "))
        self.assertEqual(picker.text, "↩️ the next message resumes the ✓ session\n"
                         "⚠ = made by another launcher: tapping it switches the launcher to that one")

    def test_session_made_by_another_launcher_is_marked(self) -> None:
        """K3: a session from another launcher gets ⚠ and its launcher's name; the text explains the mark."""
        picker = resume_picker(THREAD, self._history(), "", "wrap run", self.names)
        labels = _flat_labels(picker)
        self.assertIn("⚠ gpt · 10-24 09:39 · topic talk number 9…", labels)
        self.assertEqual(sum("⚠" in label for label in labels), 4)   # sessions 9, 7, 5, 3 of the last eight
        self.assertIn("⚠ = made by another launcher", picker.text)

    def test_unknown_launcher_name_falls_back_to_other(self) -> None:
        """K3: a launcher that is no longer in config is named 'other launcher' on its ⚠ button."""
        records = [HistoryRecord(1, THREAD, "uuid-gone-00000000", "2026-10-24T06:00:00+00:00", "", "/x",
                                 "gone-cli --x", "claude")]
        picker = resume_picker(THREAD, records, "", "wrap run", self.names)
        self.assertEqual(_flat_labels(picker), ["⚠ other launcher · 10-24 09:30 · uuid-gon"])

    def test_bad_timestamp_and_empty_history(self) -> None:
        """Q: an unreadable start time shows '--'; no history gives the fresh-session text and no keyboard."""
        self.assertEqual(started_label("not a time"), "--")
        empty = resume_picker(THREAD, [], "", "wrap run", self.names)
        self.assertEqual(empty.rows, [])
        self.assertEqual(empty.text, "↩️ the next message starts a fresh session")


class RulesPickerTests(PickerCase):
    def test_rules_text_and_rows(self) -> None:
        """R: one line per rule, rf buttons, then 🗑 all as cm:<thread>:ra:0.

        Ported labels:
        R: one 🗑 button per rule plus 🗑 all ([['🗑 1', '🗑 2'], ['🗑 all']])
        R: callbacks are cm:<thread>:rf:<rule id> and cm:<thread>:ra:0
        """
        rules = [
            RuleRecord(1, THREAD, "Bash(git status *)", 3, "2026-10-24T00:00:00+00:00"),
            RuleRecord(2, THREAD, "Edit", 0, "2026-10-24T00:00:00+00:00"),
        ]
        picker = rules_picker(THREAD, rules)
        self.assertIn("1. Bash(git status *) (used 3×)", picker.text)
        self.assertIn("2. Edit (used 0×)", picker.text)
        self.assertEqual(_labels(picker), [["🗑 1", "🗑 2"], ["🗑 all"]])
        self.assertEqual(_callbacks(picker), [f"cm:{THREAD}:rf:1", f"cm:{THREAD}:rf:2", f"cm:{THREAD}:ra:0"])

    def test_more_than_four_rules_wrap(self) -> None:
        """R: four buttons per row, the rest on the next row, then 🗑 all."""
        rules = [RuleRecord(n, THREAD, f"Tool{n}", 0, "") for n in range(1, 6)]
        picker = rules_picker(THREAD, rules)
        self.assertEqual([len(row) for row in _labels(picker)], [4, 1, 1])

    def test_no_rules_is_text_only(self) -> None:
        """R5: no rules -> the plain notice and no keyboard."""
        picker = rules_picker(THREAD, [])
        self.assertEqual(picker.text, "no always-allow rules in this topic.")
        self.assertEqual(picker.markup(), '{"inline_keyboard": []}')


class StatusRowsTests(PickerCase):
    def test_idle_status_has_no_stop(self) -> None:
        """H: idle shows no Stop; the shortcut callbacks are sn, sm, so, sc, sl.

        Ported labels:
        H: idle callbacks are cm:<thread>:<kind>:0
        """
        rows = status_rows(THREAD, running=False)
        labels = [[text for text, _ in row] for row in rows]
        self.assertEqual(labels, [["🆕 New", "🛡 Mode"], ["🧠 Model", "💰 Cost"], ["🚀 Launcher"]])
        picker = Picker("", rows)
        self.assertEqual(_callbacks(picker),
                         [f"cm:{THREAD}:sn:0", f"cm:{THREAD}:sm:0", f"cm:{THREAD}:so:0",
                          f"cm:{THREAD}:sc:0", f"cm:{THREAD}:sl:0"])

    def test_running_status_shows_stop_first(self) -> None:
        """H: a running turn shows ⏹ Stop first, callback cm:<thread>:st:0.

        Ported labels:
        H: Stop callback is cm:<thread>:st:0
        H: running shows ⏹ Stop first ([['⏹ Stop'], ['🆕 New', '🛡 Mode'], ['🧠 Model', '💰 Cost'], ['🚀 Launcher']])
        """
        picker = Picker("", status_rows(THREAD, running=True))
        self.assertEqual(_labels(picker)[0], ["⏹ Stop"])
        self.assertEqual(len(_labels(picker)), 4)
        self.assertEqual(_callbacks(picker)[0], f"cm:{THREAD}:st:0")


class ClaudeAndCodexPickersTests(PickerCase):
    def test_every_builder_stays_within_64_bytes(self) -> None:
        """Every callback of every picker, for both launchers and both backends, fits the Telegram limit."""
        pickers = [
            model_picker(THREAD, MODELS, "opus", live=True),
            mode_picker(THREAD, self.claude.modes(), "plan", live=True),
            mode_picker(THREAD, self.codex.modes(), "plan", live=False),
            launcher_picker(THREAD, self.launchers, "gpt", "codex"),
            resume_picker(THREAD, [], "", "", self.names),
            rules_picker(THREAD, [RuleRecord(7, THREAD, "Edit", 0, "")]),
            Picker("", status_rows(THREAD, running=True)),
        ]
        for picker in pickers:
            for data in _callbacks(picker):
                self.assertLessEqual(len(data.encode()), CALLBACK_DATA_LIMIT, data)

    def test_codex_launcher_picker_uses_the_codex_spec(self) -> None:
        """The launcher table has the Codex entry with backend codex, so its button is tagged."""
        codex = [spec for spec in self.launchers if spec.backend == "codex"]
        self.assertEqual([spec.name for spec in codex], ["gpt"])
        self.assertEqual(codex[0].command, "codex")
        self.assertEqual(self.codex.models(self.config), ())
        self.assertEqual(self.config.launcher_models("gpt"), ("gpt-6.1-sol",))


if __name__ == "__main__":
    unittest.main()
