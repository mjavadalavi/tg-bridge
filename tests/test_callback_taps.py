"""Picker taps and approval taps (STEP 49), ported from fakecli/test_buttons.py (sections A-F, Q, R, S, T) and the
permission-card taps of fakecli/test_p2b_cards.py.

Scenarios: a model tap stores the model (default clears it), switches a running session on a thread, edits the same
message and moves the check mark; a model name with ':' round-trips; an unknown name is refused; a mode tap reads its
index at tap time and switches a running turn; a launcher tap goes through LauncherSwitch and edits its message only
when the switch was not refused by a busy turn; a resume tap binds the session and refuses a running turn; the rules
buttons delete one rule or all, and redraw the list; the folder buttons apply a directory, refuse a stale id and a
running turn, and the search and keep cards are edited without buttons; the approval buttons answer PermissionService
exactly once, and an ask that already closed gets the expired toast with the card's text kept.
"""

import os
import unittest

from tests.callback_support import CHAT, CLAUDE_CMD, MID, THREAD, U1, U2, CallbackCase
from tests.turn_support import wait_until
from tg_bridge.bot.callbacks.approvals import ALLOWED_TEXT, BYPASS_TEXT, DENIED_TEXT, EXPIRED_TEXT as APPROVAL_EXPIRED
from tg_bridge.bot.callbacks.approvals import OTHER_TEXT, RECORDED_TEXT
from tg_bridge.bot.callbacks.picker_taps import MODE_SET_TEXT, MODEL_SET_TEXT, NOT_IN_LIST_TEXT, PickerTaps
from tg_bridge.bot.callbacks.picker_taps import RULE_GONE_TEXT, RULE_REMOVED_TEXT, ALL_RULES_REMOVED_TEXT
from tg_bridge.bot.callbacks.picker_taps import NO_RULES_LEFT_TEXT
from tg_bridge.bot.callbacks.router import CallbackRouter
from tg_bridge.bot.pickers import NO_RULES_TEXT, RESUME_LIVE_TEXT
from tg_bridge.core.cwd import EXPIRED_TEXT as FOLDER_EXPIRED_TEXT
from tg_bridge.core.cwd import KEEP_TEXT, cwd_reply
from tg_bridge.config import Config, tilde
from tests.fakes.backend import FAKE_CAPABILITIES, FakeBackend, RecordingSink, fake_spec
from tg_bridge.agents.claude.backend import ClaudeRules
from tg_bridge.agents.events import ApprovalRequest, Decision
from tg_bridge.bot.callbacks.approvals import ApprovalTaps
from tg_bridge.core.cards import ApprovalCards
from tg_bridge.core.permissions import PermissionService, TurnBinding
from tg_bridge.store.asks import AskRepo

EMPTY_MARKUP = '{"inline_keyboard": []}'
NONCE = "a1b2c3d4"
OLD_UUID = "5f5f5f5f-0000-4000-8000-00000000000a"
OTHER_THREAD = THREAD + 1
CLAUDE = "claude"


class ApprovalTapTests(CallbackCase):
    """pa / pd / pA / pt / q / o: each tap is one PermissionService call on the card's nonce."""

    def test_allow_once_and_deny_answer_the_ask(self) -> None:
        """Ported labels:
        A: callback toast Allowed (['Allowed ✅'])
        """
        self.tap(f"pa:{NONCE}")
        self.assertEqual(self.permissions.calls[-1], ("answer_ask", NONCE, True))
        self.assertEqual(self.answers()[-1], ALLOWED_TEXT)
        self.tap(f"pd:{NONCE}")
        self.assertEqual(self.permissions.calls[-1], ("answer_ask", NONCE, False))
        self.assertEqual(self.answers()[-1], DENIED_TEXT)
        self.assertEqual(self.edits(), [], "an answered card is edited by its own turn, not by the tap")

    def test_always_and_bypass_answer_their_own_calls(self) -> None:
        """Ported labels:
        A: toast (['Always allowed ♾'])
        D: toast shown
        """
        self.tap(f"pA:{NONCE}")
        self.assertEqual(self.permissions.calls[-1], ("answer_always", NONCE))
        self.assertEqual(self.answers()[-1], "Always allowed ♾")
        self.tap(f"pt:{NONCE}")
        self.assertEqual(self.permissions.calls[-1], ("answer_bypass", NONCE))
        self.assertEqual(self.answers()[-1], BYPASS_TEXT)

    def test_question_option_and_other_answer_the_question(self) -> None:
        """Ported labels:
        F: toast Recorded
        K: toast (['Type your answer in this topic ✏️'])
        """
        self.tap(f"q:{NONCE}:1:2")
        self.assertEqual(self.permissions.calls[-1], ("answer_question", NONCE, 1, 2))
        self.assertEqual(self.answers()[-1], RECORDED_TEXT)
        self.tap(f"o:{NONCE}:0")
        self.assertEqual(self.permissions.calls[-1], ("answer_other", NONCE, 0))
        self.assertEqual(self.answers()[-1], OTHER_TEXT)

    def test_a_closed_ask_gets_the_expired_toast_and_keeps_the_card_text(self) -> None:
        """The second tap on the same card (double tap) finds nothing open: expired, card kept without buttons.

        Ported labels:
        E: stale card keyboard stripped
        E: stale tap toast expired (['expired — that turn already ended'])
        """
        self.permissions.result = True
        self.tap(f"pa:{NONCE}", text="Bash: git status")
        self.permissions.result = False
        self.tap(f"pa:{NONCE}", text="Bash: git status")
        self.assertEqual(self.answers()[-1], APPROVAL_EXPIRED)
        self.assertEqual(self.edits()[-1]["text"], "Bash: git status")
        self.assertEqual(self.edits()[-1]["reply_markup"], EMPTY_MARKUP)

    def test_stale_always_and_bypass_taps_are_expired(self) -> None:
        """Ported labels:
        I: stale pA tap is expired
        I: stale pt tap is expired
        """
        self.permissions.result = False
        for prefix in ("pA", "pt"):
            self.tap(f"{prefix}:{NONCE}")
            self.assertEqual(self.answers()[-1], APPROVAL_EXPIRED)

    def test_malformed_approval_callbacks_change_nothing(self) -> None:
        for data in ("pa:", "pa:a:b", "q:nonce:x:2", "q:nonce:1", "o:nonce:x", "o:nonce"):
            self.tap(data)
        self.assertEqual(self.permissions.calls, [])
        self.assertEqual(self.answers(), [])

    def test_a_stranger_cannot_answer_an_ask(self) -> None:
        self.tap(f"pa:{NONCE}", tapper=7)
        self.assertEqual(self.permissions.calls, [])
        self.assertEqual(self.answers(), ["not allowed"])


class ModelAndModeTapTests(CallbackCase):
    """cm:<thread>:mo:<name> and cm:<thread>:md:<index> (fakecli A, D, D2, E2, F, S3)."""

    def test_model_tap_stores_switches_and_edits_the_same_message(self) -> None:
        """Ported labels:
        D: tap persists sonnet
        D: live session switched ([('set_model', {'model': 'sonnet'})])
        D: toast answers the tap
        D: the same picker message is edited
        D: edited text shows the value ('🧠 model: sonnet')
        D: ✓ moved to sonnet
        D: no new message in the topic
        """
        session = self.start_turn()
        self.tap(f"cm:{THREAD}:mo:sonnet")
        self.assertEqual(self.sessions.get(THREAD).model, "sonnet")
        self.assertTrue(wait_until(lambda: session.models == ["sonnet"], 5), "the running session did not switch")
        self.assertEqual(self.answers(), [MODEL_SET_TEXT.format(model="sonnet")])
        edit = self.edits()[-1]
        self.assertEqual(edit["text"], "🧠 model: sonnet")
        self.assertIn(f"cm:{THREAD}:mo:sonnet", [b for b in self.callbacks_of(edit["reply_markup"])])
        self.assertEqual(self.labels_of(edit["reply_markup"])[1], "✓ sonnet")
        self.assertEqual([c for c in self.calls("sendMessage")], [], "no new message in the topic")

    def test_default_model_tap_clears_the_stored_model_without_a_session(self) -> None:
        """Ported labels:
        D2: default tap clears the stored model
        D2: ✓ on default after the tap
        """
        self.sessions.set_model(THREAD, "haiku")
        self.tap(f"cm:{THREAD}:mo:default")
        self.assertEqual(self.sessions.get(THREAD).model, "")
        self.assertEqual(self.answers(), [MODEL_SET_TEXT.format(model="default")])
        self.assertEqual(self.edits()[-1]["text"], "🧠 model: default")
        self.assertIn("✓ default", self.labels_of(self.edits()[-1]["reply_markup"]))

    def test_a_model_name_with_a_colon_round_trips(self) -> None:
        self.tap(f"cm:{THREAD}:mo:qwen3:8b")
        self.assertEqual(self.sessions.get(THREAD).model, "qwen3:8b")

    def test_a_model_not_in_the_list_is_refused(self) -> None:
        self.tap(f"cm:{THREAD}:mo:gpt-x")
        self.assertEqual(self.answers(), [NOT_IN_LIST_TEXT])
        self.assertEqual(self.sessions.get(THREAD).model, "")

    def test_mode_tap_reads_its_index_and_switches_the_running_turn(self) -> None:
        """Ported labels:
        E2: tap persists acceptEdits
        E2: live set_permission_mode sent ([('set_permission_mode', {'mode': 'acceptEdits'})])
        E2: picker edited with ✓ on acceptEdits
        """
        index = next(i for i, m in enumerate(self.backends.get(CLAUDE).modes()) if m.key == "acceptEdits")
        self.start_turn()
        self.tap(f"cm:{THREAD}:md:{index}")
        self.assertEqual(self.sessions.get(THREAD).mode, "acceptEdits")
        self.assertTrue(wait_until(lambda: self.permissions.live_modes == [(THREAD, "acceptEdits", "mode button")], 5))
        self.assertEqual(self.answers(), [MODE_SET_TEXT.format(mode="acceptEdits")])
        edit = self.edits()[-1]
        self.assertEqual(edit["text"], "🛡 mode: acceptEdits")
        self.assertEqual(self.labels_of(edit["reply_markup"])[:2], ["bypass", "✓ acceptEdits"])

    def test_mode_index_out_of_range_is_refused(self) -> None:
        """Ported labels:
        F: out-of-range index refused with a toast
        """
        self.tap(f"cm:{THREAD}:md:99")
        self.assertEqual(self.answers(), [NOT_IN_LIST_TEXT])
        self.assertEqual(self.sessions.get(THREAD).mode, "bypassPermissions")

    def test_a_tap_for_a_topic_without_session_is_refused(self) -> None:
        """Ported labels:
        F: tap for a topic without a session gets a toast
        """
        self.tap("cm:9699:mo:sonnet")
        self.assertEqual(self.answers(), ["no session in this topic"])

    def test_a_stranger_cannot_change_the_model(self) -> None:
        """Ported labels:
        F: tap from a non-allowlisted user changes nothing
        F: and does not edit the picker
        """
        self.tap(f"cm:{THREAD}:mo:sonnet", tapper=7)
        self.assertEqual(self.sessions.get(THREAD).model, "")
        self.assertEqual(self.edits(), [])

    def test_a_model_no_longer_in_the_config_is_refused_not_swapped(self) -> None:
        """Ported labels:
        S3: a model no longer in the config is refused, not swapped for another model at the same index
        S3: a listed model is applied by name
        """
        data = self._config_data()
        data["models"] = ["haiku", "opus"]
        router = self._router_for(Config(data, home=self.base, env={}))
        router.dispatch(self.query(f"cm:{THREAD}:mo:sonnet"))
        self.assertTrue(self.out.join(5))
        self.assertEqual(self.answers(), [NOT_IN_LIST_TEXT])
        self.assertEqual(self.sessions.get(THREAD).model, "")
        router.dispatch(self.query(f"cm:{THREAD}:mo:haiku"))
        self.assertTrue(self.out.join(5))
        self.assertEqual(self.sessions.get(THREAD).model, "haiku")

    def _router_for(self, config: Config) -> CallbackRouter:
        """A router with only the picker taps, reading the given config (the config changed since a picker was drawn)."""
        router = CallbackRouter(config=config, client=self.telegram, messenger=self.messenger, out=self.out,
                                logger=self.logger)
        PickerTaps(config=config, sessions=self.sessions, live=self.live, backends=self.backends,
                   permissions=self.permissions, launchers=self.launchers, stops=self.stops, rules=self.rules,
                   folders=self.folders, logger=self.logger).register(router)
        return router


class CodexModelTapTests(CallbackCase):
    """cm:<thread>:mo:<name> on a Codex topic: the names are the configured codex launcher's models, not Claude's."""

    def _config_data(self) -> dict:
        data = super()._config_data()
        codex = {"backend": "codex", "command": "codex", "config_dir": os.path.join(self.project, "cfg-codex"),
                 "models": ["gpt-6.1-sol"]}
        data["agent_launchers"] = {"codex": codex}
        return data

    def setUp(self) -> None:
        super().setUp()
        self.sessions.set_launcher_pick(THREAD, "codex")
        self.sessions.set_backend(THREAD, "codex")

    def test_a_configured_codex_model_is_stored_and_the_picker_edited(self) -> None:
        self.tap(f"cm:{THREAD}:mo:gpt-6.1-sol")
        self.assertEqual(self.sessions.get(THREAD).model, "gpt-6.1-sol")
        self.assertEqual(self.edits()[-1]["text"], "🧠 model: gpt-6.1-sol\napplies from your next message")

    def test_a_claude_model_is_refused_on_a_codex_topic(self) -> None:
        self.tap(f"cm:{THREAD}:mo:sonnet")
        self.assertEqual(self.answers(), [NOT_IN_LIST_TEXT])
        self.assertEqual(self.sessions.get(THREAD).model, "")


class LauncherTapTests(CallbackCase):
    """cm:<thread>:la:<index>: the launcher change goes through LauncherSwitch, the same message shows the result."""

    def test_launcher_tap_goes_through_the_switch_and_edits_the_message(self) -> None:
        specs = list(self.config.launcher_specs().values())
        target = specs[1].name
        self.tap(f"cm:{THREAD}:la:1")
        self.assertEqual(self.launchers.calls[-1], (THREAD, self.sessions.get(THREAD).name, None, target))
        self.assertEqual(self.answers(), [self.launchers.reply])
        self.assertEqual(self.edits()[-1]["text"], self.launchers.reply)

    def test_a_busy_refusal_leaves_the_message_as_it_is(self) -> None:
        self.launchers.kind, self.launchers.reply = "busy", "a turn is running — send /stop first, then try again."
        self.tap(f"cm:{THREAD}:la:1")
        self.assertEqual(self.answers(), [self.launchers.reply])
        self.assertEqual(self.edits(), [])

    def test_a_launcher_index_out_of_range_is_refused(self) -> None:
        self.tap(f"cm:{THREAD}:la:99")
        self.assertEqual(self.answers(), [NOT_IN_LIST_TEXT])
        self.assertEqual(self.launchers.calls, [])


class ResumeTapTests(CallbackCase):
    """cm:<thread>:rs:<history id>: the same change as /resume <uuid>, refused while a turn runs (fakecli Q2, Q3)."""

    def record_one(self) -> int:
        self.history.note_session(THREAD, U1, "first question", cwd=self.project, launcher=self.sessions.get(THREAD).launcher)
        return self.history.list_for(THREAD, CLAUDE)[0].id

    def test_resume_tap_binds_the_session_and_marks_it(self) -> None:
        """Ported labels:
        Q2: tap binds the topic to that session, the old one to prev_uuid
        Q2: toast answers the tap
        Q2: the same picker message is edited
        Q2: ✓ moved to the tapped session ('✓ 10-24 09:33 · topic talk number 3…')
        Q2: no new message in the topic
        """
        history_id = self.record_one()
        self.sessions.set_uuid(THREAD, OLD_UUID)
        self.tap(f"cm:{THREAD}:rs:{history_id}")
        row = self.sessions.get(THREAD)
        self.assertEqual((row.uuid, row.prev_uuid), (U1, OLD_UUID))
        self.assertEqual(self.answers(), ["↩️ the next message resumes it"])
        edit = self.edits()[-1]
        self.assertEqual(edit["text"], RESUME_LIVE_TEXT)
        first = self.labels_of(edit["reply_markup"])[0]
        self.assertTrue(first.startswith("✓ ") and first.endswith("first question…"), first)
        self.assertEqual(self.calls("sendMessage"), [], "no new message in the topic")

    def test_resume_tap_while_running_is_refused(self) -> None:
        """Ported labels:
        Q3: tap while a turn runs is refused with a toast
        Q3: and changes nothing
        """
        history_id = self.record_one()
        self.sessions.set_uuid(THREAD, OLD_UUID)
        self.start_turn()
        self.tap(f"cm:{THREAD}:rs:{history_id}")
        self.assertEqual(self.answers(), ["a turn is running — send /stop first"])
        self.assertEqual(self.sessions.get(THREAD).uuid, OLD_UUID)

    def test_resume_tap_for_an_unknown_record_is_refused(self) -> None:
        self.tap(f"cm:{THREAD}:rs:99999")
        self.assertEqual(self.answers(), ["not in the list anymore"])

    def test_a_record_of_another_topic_is_refused_and_a_stranger_changes_nothing(self) -> None:
        """Ported labels:
        Q4: a row of another topic is refused
        Q4: a non-allowlisted tap changes nothing
        """
        self.history.note_session(OTHER_THREAD, U1, "other topic", cwd=self.project, launcher=CLAUDE_CMD)
        other_id = self.history.list_for(OTHER_THREAD, CLAUDE)[0].id
        self.sessions.set_uuid(THREAD, OLD_UUID)
        self.tap(f"cm:{THREAD}:rs:{other_id}")
        self.assertEqual(self.answers(), ["not in the list anymore"])
        self.assertEqual(self.sessions.get(THREAD).uuid, OLD_UUID)
        self.history.note_session(THREAD, U2, "mine", cwd=self.project, launcher=CLAUDE_CMD)
        mine_id = self.history.list_for(THREAD, CLAUDE)[0].id
        self.tap(f"cm:{THREAD}:rs:{mine_id}", tapper=7)
        self.assertEqual(self.answers()[-1], "not allowed")
        self.assertEqual(self.sessions.get(THREAD).uuid, OLD_UUID)


class RuleTapTests(CallbackCase):
    """cm:<thread>:rf:<rule id> removes one rule, cm:<thread>:ra:0 removes all (fakecli R)."""

    def add_rules(self) -> tuple[int, int]:
        self.rules.add(THREAD, "Bash(git status *)")
        self.rules.add(THREAD, "Edit(src/*)")
        first, second = self.rules.for_thread(THREAD)
        return first.id, second.id

    def test_delete_one_rule_and_redraw_the_list(self) -> None:
        """Ported labels:
        R2: tapping one 🗑 removes only that rule
        R2: the same message now lists what is left
        R2: its buttons follow the list
        R2: toast answers the tap
        """
        first, second = self.add_rules()
        self.tap(f"cm:{THREAD}:rf:{first}")
        self.assertEqual(self.answers(), [RULE_REMOVED_TEXT])
        self.assertEqual(len(self.rules.for_thread(THREAD)), 1)
        edit = self.edits()[-1]
        self.assertIn("Edit(src/*)", edit["text"])
        self.assertNotIn("git status", edit["text"])
        self.assertEqual(self.labels_of(edit["reply_markup"]), [f"🗑 {second}", "🗑 all"])

    def test_a_rule_of_another_topic_is_not_deleted(self) -> None:
        """Ported labels:
        R3: a rule of another topic is not deleted
        """
        self.rules.add(OTHER_THREAD, "Write")
        other_rule = self.rules.for_thread(OTHER_THREAD)[0].id
        self.add_rules()
        self.tap(f"cm:{THREAD}:rf:{other_rule}")
        self.assertEqual(self.answers(), [RULE_GONE_TEXT])
        self.assertEqual(len(self.rules.for_thread(OTHER_THREAD)), 1)

    def test_a_stranger_cannot_delete_rules(self) -> None:
        """Ported labels:
        R3: a non-allowlisted tap changes nothing
        """
        self.add_rules()
        self.tap(f"cm:{THREAD}:ra:0", tapper=7)
        self.assertEqual(len(self.rules.for_thread(THREAD)), 2)
        self.assertEqual(self.answers(), ["not allowed"])

    def test_deleting_an_unknown_rule_says_so(self) -> None:
        self.add_rules()
        self.tap(f"cm:{THREAD}:rf:999")
        self.assertEqual(self.answers(), [RULE_GONE_TEXT])

    def test_delete_all_empties_the_list_and_says_no_rules(self) -> None:
        """Ported labels:
        R4: 🗑 all removes this topic's rules
        R4: the message says so and drops the buttons
        R4: the other topic keeps its rules
        """
        self.rules.add(OTHER_THREAD, "Write")
        self.add_rules()
        self.tap(f"cm:{THREAD}:ra:0")
        self.assertEqual(self.answers(), [ALL_RULES_REMOVED_TEXT])
        self.assertEqual(self.rules.for_thread(THREAD), [])
        self.assertEqual(self.edits()[-1]["text"], NO_RULES_TEXT)
        self.assertEqual(self.edits()[-1]["reply_markup"], EMPTY_MARKUP)
        self.assertEqual(len(self.rules.for_thread(OTHER_THREAD)), 1)
        self.tap(f"cm:{THREAD}:ra:0")
        self.assertEqual(self.answers()[-1], NO_RULES_LEFT_TEXT)


class FolderTapTests(CallbackCase):
    """cm:<thread>:cw:<path id> (the /cwd picker), cs (search card), ck (keep card) (fakecli T, S, S2)."""

    def test_folder_button_applies_the_directory_and_edits_the_picker(self) -> None:
        """Ported labels:
        T2: tap sets the topic cwd, as /cwd <path> does
        T2: toast answers the tap
        T2: the same picker is edited with ✓ moved
        """
        path_id = self.folders.path_id(self.dir_b)
        self.tap(f"cm:{THREAD}:cw:{path_id}")
        self.assertEqual(self.sessions.get(THREAD).cwd, os.path.realpath(self.dir_b))
        self.assertEqual(self.answers(), [cwd_reply(tilde(os.path.realpath(self.dir_b), self.base), True)])
        self.assertTrue(self.answers()[0].startswith("📁 ") and "fresh session" in self.answers()[0])
        edit = self.edits()[-1]
        self.assertTrue(edit["text"].startswith("📁 cwd: "), edit["text"])
        self.assertTrue(any(label.startswith("✓ ") and label.endswith("cwd-b") for label in self.labels_of(edit["reply_markup"])))

    def test_unknown_folder_id_is_refused(self) -> None:
        """Ported labels:
        T3: an unknown path id is refused
        """
        self.tap(f"cm:{THREAD}:cw:99999")
        self.assertEqual(self.answers(), [NOT_IN_LIST_TEXT])

    def test_folder_button_while_a_turn_runs_is_refused(self) -> None:
        """Ported labels:
        T4: tap while a turn runs is refused
        """
        path_id = self.folders.path_id(self.dir_b)
        self.start_turn()
        self.tap(f"cm:{THREAD}:cw:{path_id}")
        self.assertEqual(self.answers(), ["a turn is running — send /stop first"])
        self.assertEqual(self.sessions.get(THREAD).cwd, self.project)

    def test_a_stranger_cannot_move_the_folder(self) -> None:
        """Ported labels:
        T5: a non-allowlisted tap changes nothing
        """
        self.tap(f"cm:{THREAD}:cw:{self.folders.path_id(self.dir_b)}", tapper=7)
        self.assertEqual(self.sessions.get(THREAD).cwd, self.project)
        self.assertEqual(self.answers(), ["not allowed"])

    def test_a_button_for_a_folder_that_is_gone_is_refused(self) -> None:
        """Ported labels:
        S2: a button for a directory that is gone is refused
        S2: and nothing changes
        """
        path_id = self.folders.path_id(self.dir_b)
        os.rmdir(self.dir_b)
        self.tap(f"cm:{THREAD}:cw:{path_id}")
        self.assertEqual(self.answers(), [NOT_IN_LIST_TEXT])
        self.assertEqual(self.sessions.get(THREAD).cwd, self.project)

    def test_a_button_applies_the_folder_it_showed_after_another_topic_moved(self) -> None:
        """The harness names the folders cwd-a and cwd-b; the old labels call them stale-a and stale-b.

        Ported labels:
        S: another topic changed its cwd
        S: picker drawn with three directories (['stale-a', 'stale-b', 'stale-c'])
        S1: the stale button applies stale-a, the directory it showed (not stale-c at the shifted index)
        S1: toast names the tapped directory ('📁 ~/stale-a — the next message starts a fresh session there')
        S1: the same picker is edited with ✓ on stale-a
        """
        self.assertEqual(len(self.folders.choices(self.config)), 3)
        drawn = self.folders.path_id(self.dir_b)
        new_dir = os.path.join(self.project, "cwd-c")
        os.makedirs(new_dir)
        self.sessions.ensure(OTHER_THREAD, "OTHER", CLAUDE_CMD, self.project, "bypassPermissions")
        self.sessions.set_cwd(OTHER_THREAD, new_dir)
        self.tap(f"cm:{THREAD}:cw:{drawn}")
        self.assertEqual(self.sessions.get(THREAD).cwd, os.path.realpath(self.dir_b))
        self.assertIn("cwd-b", self.answers()[-1])
        self.assertNotIn("cwd-c", self.answers()[-1])
        labels = self.labels_of(self.edits()[-1]["reply_markup"])
        self.assertTrue(any(label.startswith("✓ ") and label.endswith("cwd-b") for label in labels), labels)

    def test_search_card_button_applies_its_folder_and_drops_the_buttons(self) -> None:
        """Ported labels:
        D: the same card is edited, buttons removed
        """
        self.folders.remember_found(THREAD, MID, [self.dir_a, self.dir_b])
        self.tap(f"cm:{THREAD}:cs:1")
        self.assertEqual(self.sessions.get(THREAD).cwd, os.path.realpath(self.dir_b))
        self.assertTrue(self.answers()[-1].startswith("📁 "))
        self.assertEqual(self.edits()[-1]["text"], self.answers()[-1])
        self.assertEqual(self.edits()[-1]["reply_markup"], EMPTY_MARKUP)

    def test_search_card_tap_after_the_list_expired(self) -> None:
        self.tap(f"cm:{THREAD}:cs:0", mid=900)
        self.assertEqual(self.answers(), [FOLDER_EXPIRED_TEXT])
        self.assertEqual(self.edits(900), [])

    def test_keep_button_keeps_the_default_folder_and_edits_the_card(self) -> None:
        """Ported labels:
        I4: keep edits the card to confirm, buttons removed
        """
        self.folders.pending_set(THREAD, self.dir_a)
        self.tap(f"cm:{THREAD}:ck:0")
        self.assertEqual(self.answers(), [KEEP_TEXT])
        self.assertEqual(self.edits()[-1], {"chat_id": CHAT, "message_id": MID, "text": KEEP_TEXT,
                                            "reply_markup": EMPTY_MARKUP})
        self.assertEqual(self.folders.pending_get(THREAD), "")


class _RecordingTimer:
    def __init__(self, seconds, fire) -> None:
        self.seconds = seconds

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        pass


class _InlineWorkers:
    def submit(self, fn, *args) -> None:
        fn(*args)


class _WaitRecorder:
    def waiting_changed(self, thread_id: int, waiting: bool) -> None:
        pass


class RealApprovalTapTests(CallbackCase):
    """The approval buttons over the real PermissionService (the other tests of this module use a stub)."""

    def setUp(self) -> None:
        super().setUp()
        self.asks = AskRepo(self.db, self.clock)
        cards = ApprovalCards(self.messenger, self.logger, self.clock, timer_factory=_RecordingTimer)
        nonces = iter(f"{n:08x}" for n in range(1, 10_000))
        self.service = PermissionService(asks=self.asks, rules=self.rules, cards=cards, messenger=self.messenger,
                                         workers=_InlineWorkers(), waiting=_WaitRecorder(), logger=self.logger,
                                         nonce_factory=lambda: next(nonces))
        self.router = CallbackRouter(config=self.config, client=self.telegram, messenger=self.messenger,
                                     out=self.out, logger=self.logger)
        ApprovalTaps(permissions=self.service).register(self.router)
        self.session = FakeBackend(capabilities=FAKE_CAPABILITIES).open_session(
            fake_spec(thread_id=THREAD, chat_id=CHAT), RecordingSink())
        binding = TurnBinding(thread_id=THREAD, chat_id=CHAT, cwd=self.project, rules=ClaudeRules(), mode="default")
        self.listener = self.service.bind_turn(binding)
        self.service.attach(binding, self.session)

    def test_a_deny_tap_sets_the_row_to_deny_and_toasts_denied(self) -> None:
        """Ported labels:
        B: row deny, toast Denied
        """
        self.listener.on_event(ApprovalRequest(
            approval_id="r1", kind="command", tool="Bash", input={"command": "git status --short"}, summary="Bash",
            choices=frozenset({Decision.ALLOW_ONCE, Decision.DENY})))
        self.tap("pd:00000001")
        self.assertEqual(self.answers(), [DENIED_TEXT])
        self.assertEqual(self.asks.by_nonce("00000001").answer, "deny")


if __name__ == "__main__":
    unittest.main()
