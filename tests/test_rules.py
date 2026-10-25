"""Shared rule grammar (agents/rules.py) and the Claude always-rule mapper (agents/claude/rules.py).

Ports: fakecli/test_permissions.py sections H and J (rule_matches), fakecli/test_p2b_cards.py sections A, B, C
(synthesized and suggested always rules, Edit alias, the next turn asks nothing).
"""

import json
import os
import unittest

from tests.support import TempHome
from tg_bridge.agents.claude.rules import ClaudeRuleMapper, always_rule
from tg_bridge.agents.claude.translate import ClaudeTranslator
from tg_bridge.agents.events import ApprovalRequest, Decision
from tg_bridge.agents.rules import EDIT_TOOLS, FILE_TOOLS, RuleGrammar, rule_matches

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "claude_lines")
ALWAYS_CHOICES = frozenset({Decision.ALLOW_ONCE, Decision.ALLOW_ALWAYS, Decision.ALLOW_TURN, Decision.DENY})


def approval(tool, inp, choices=ALWAYS_CHOICES, raw=None, kind="tool"):
    return ApprovalRequest(approval_id="req-1", kind=kind, tool=tool, input=inp, summary=tool,
                           choices=choices, raw=raw or {})


def control_raw(suggestions=None):
    """The raw control_request event the translator keeps in ApprovalRequest.raw."""
    request = {"subtype": "can_use_tool", "tool_name": "x", "input": {}}
    if suggestions is not None:
        request["permission_suggestions"] = suggestions
    return {"type": "control_request", "request_id": "req-1", "request": request}


class RuleMatchesTests(unittest.TestCase):
    # --- old H) rule_matches unit cases
    def test_bash_glob_matches_bare_prefix_and_longer_command(self) -> None:
        """Ported labels:
        H: git status * matches bare git status
        H: git status * matches git status --short
        """
        self.assertTrue(rule_matches("Bash(git status *)", "Bash", {"command": "git status"}))
        self.assertTrue(rule_matches("Bash(git status *)", "Bash", {"command": "git status --short"}))

    def test_bash_glob_rejects_other_commands(self) -> None:
        """Ported labels:
        H: git status * does not match git push
        """
        self.assertFalse(rule_matches("Bash(git status *)", "Bash", {"command": "git push"}))

    def test_compound_and_redirected_commands_never_match_a_pattern_rule(self) -> None:
        """Ported labels:
        H: compound/redirected commands do not match pattern rule
        """
        for cmd in ["git status; curl x", "git status && rm -rf /", "git status | sh", "git status $(id)",
                    "git status > /etc/x", "git status\nrm x", "git status `id`", "git status < /etc/passwd",
                    "git status\rrm x"]:
            with self.subTest(cmd=cmd):
                self.assertFalse(rule_matches("Bash(git status *)", "Bash", {"command": cmd}))

    def test_bare_bash_rule_covers_compound_command(self) -> None:
        """Ported labels:
        H: bare Bash matches compound command
        """
        self.assertTrue(rule_matches("Bash", "Bash", {"command": "a; b"}))

    def test_edit_rule_covers_write_multiedit_notebookedit(self) -> None:
        """Ported labels:
        H: Edit matches MultiEdit/NotebookEdit
        H: Edit matches Write
        """
        self.assertTrue(rule_matches("Edit", "Write", {"file_path": "x"}))
        self.assertTrue(rule_matches("Edit", "MultiEdit", {}))
        self.assertTrue(rule_matches("Edit", "NotebookEdit", {}))

    def test_bare_tool_rule_matches_only_that_tool(self) -> None:
        """Ported labels:
        H: WebFetch matches WebFetch
        H: unrelated tools do not match
        """
        self.assertTrue(rule_matches("WebFetch", "WebFetch", {"url": "https://x"}))
        self.assertFalse(rule_matches("WebFetch", "Bash", {}))
        self.assertFalse(rule_matches("Write", "Edit", {}))

    def test_edit_path_glob_matches_file_path(self) -> None:
        """Ported labels:
        H: Edit(path) glob on file_path
        """
        self.assertTrue(rule_matches("Edit(/a/*)", "Write", {"file_path": "/a/b.py"}))
        self.assertFalse(rule_matches("Edit(/a/*)", "Write", {"file_path": "/z"}))

    # --- old J) file-pattern rules match the real path
    def test_file_glob_matches_absolute_path(self) -> None:
        """Ported labels:
        J: Edit(/tmp/x/*) matches /tmp/x/a.py
        """
        self.assertTrue(rule_matches("Edit(/tmp/x/*)", "Edit", {"file_path": "/tmp/x/a.py"}))

    def test_file_glob_does_not_match_dotdot_escape(self) -> None:
        """Ported labels:
        J: Edit(/tmp/x/*) does not match /tmp/x/../../etc/passwd
        """
        self.assertFalse(rule_matches("Edit(/tmp/x/*)", "Edit", {"file_path": "/tmp/x/../../etc/passwd"}))

    def test_file_glob_does_not_match_relative_path(self) -> None:
        """Ported labels:
        J: Edit(/tmp/x/*) does not match relative x/a.py
        """
        self.assertFalse(rule_matches("Edit(/tmp/x/*)", "Edit", {"file_path": "x/a.py"}))

    def test_empty_file_path_never_matches_a_pattern_rule(self) -> None:
        """Ported labels:
        J: empty path never matches a pattern rule
        """
        self.assertFalse(rule_matches("Edit(/tmp/x/*)", "Write", {"file_path": ""}))

    def test_symlink_is_resolved_before_matching(self) -> None:
        with TempHome() as home:
            base = os.path.realpath(home.home)
            real_dir = os.path.join(base, "real")
            os.makedirs(real_dir)
            link = os.path.join(base, "link")
            os.symlink(real_dir, link)
            target = os.path.join(link, "a.py")
            self.assertTrue(rule_matches(f"Edit({os.path.join(real_dir, '*')})", "Edit", {"file_path": target}))
            self.assertFalse(rule_matches(f"Edit({os.path.join(link, '*')})", "Edit",
                                          {"file_path": os.path.join(real_dir, "a.py")}))

    def test_read_is_a_file_tool_for_path_rules(self) -> None:
        self.assertIn("Read", FILE_TOOLS)
        self.assertTrue(rule_matches("Read(/srv/data/*)", "Read", {"file_path": "/srv/data/x.csv"}))

    def test_other_tools_match_json_of_input(self) -> None:
        self.assertTrue(rule_matches("Grep(*todo*)", "Grep", {"pattern": "todo"}))

    def test_none_input_is_treated_as_empty(self) -> None:
        self.assertTrue(rule_matches("WebSearch", "WebSearch", None))
        self.assertFalse(rule_matches("WebSearch(x*)", "WebSearch", None))

    def test_malformed_rules_do_not_match(self) -> None:
        self.assertFalse(rule_matches("Bash(git status *", "Bash", {"command": "git status"}))
        self.assertFalse(rule_matches("Bash(git status *)", "Edit", {"command": "git status"}))

    def test_edit_tools_constant(self) -> None:
        self.assertEqual(set(EDIT_TOOLS), {"Edit", "Write", "MultiEdit", "NotebookEdit"})


class RuleGrammarTests(unittest.TestCase):
    def test_matches_uses_the_request_tool_and_input_and_ignores_cwd(self) -> None:
        grammar = RuleGrammar()
        req = approval("Bash", {"command": "npm test"})
        self.assertTrue(grammar.matches("Bash(npm test)", req, "/anywhere"))
        self.assertTrue(grammar.matches("Bash(npm test)", req, ""))
        self.assertFalse(grammar.matches("Bash(npm run)", req, "/anywhere"))


class SynthesizedAndSuggestedAlwaysRuleTests(unittest.TestCase):
    """Ports of test_p2b_cards.py sections A, B and C (the rule text and the session permission update)."""

    def setUp(self) -> None:
        self.mapper = ClaudeRuleMapper()

    def test_a_synthesized_bash_rule(self) -> None:
        """Ported labels:
        A: updatedPermissions carries the synthesized session rule
        """
        req = approval("Bash", {"command": "git status --short"}, raw=control_raw())
        self.assertEqual(self.mapper.rule_for(req), "Bash(git status *)")
        self.assertEqual(self.mapper.updated_permissions(req), [
            {"type": "addRules", "rules": [{"toolName": "Bash", "ruleContent": "git status *"}],
             "behavior": "allow", "destination": "session"}])

    def test_b_cli_suggestion_is_echoed_with_destination_session(self) -> None:
        """Ported labels:
        B: suggestion echoed with destination session (the setMode entry is not used)
        B: label from the addRules suggestion ('♾ Always: Bash(npm test)')
        """
        sugg = [{"type": "addRules", "rules": [{"toolName": "Bash", "ruleContent": "npm test"}],
                 "behavior": "allow", "destination": "localSettings"},
                {"type": "setMode", "mode": "acceptEdits", "destination": "session"}]
        req = approval("Bash", {"command": "npm test"}, raw=control_raw(sugg))
        self.assertEqual(self.mapper.rule_for(req), "Bash(npm test)")
        self.assertEqual(self.mapper.updated_permissions(req), [
            {"type": "addRules", "rules": [{"toolName": "Bash", "ruleContent": "npm test"}],
             "behavior": "allow", "destination": "session"}])

    def test_b_next_turn_with_the_saved_rule_asks_nothing(self) -> None:
        """Ported labels:
        B: next turn with the rule asks nothing ('ok:allow {"updatedInput": {"command": "npm test"}}')
        """
        self.assertTrue(RuleGrammar().matches("Bash(npm test)", approval("Bash", {"command": "npm test"}), ""))
        self.assertFalse(RuleGrammar().matches("Bash(npm test)", approval("Bash", {"command": "npm again"}), ""))

    def test_c_write_maps_to_the_bare_edit_rule(self) -> None:
        """Ported labels:
        C: Write maps to a session Edit rule
        """
        req = approval("Write", {"file_path": "/tmp/p2b/a.txt", "content": "x"}, raw=control_raw())
        self.assertEqual(self.mapper.rule_for(req), "Edit")
        self.assertEqual(self.mapper.updated_permissions(req), [
            {"type": "addRules", "rules": [{"toolName": "Edit"}], "behavior": "allow", "destination": "session"}])

    def test_other_tool_uses_its_name(self) -> None:
        req = approval("WebFetch", {"url": "https://x"}, raw=control_raw())
        self.assertEqual(self.mapper.rule_for(req), "WebFetch")
        self.assertEqual(self.mapper.updated_permissions(req)[0]["rules"], [{"toolName": "WebFetch"}])

    def test_bash_without_command_has_no_glob(self) -> None:
        req = approval("Bash", {}, raw=control_raw())
        self.assertEqual(self.mapper.rule_for(req), "Bash")

    def test_suggestion_without_rule_content_is_a_bare_rule(self) -> None:
        sugg = [{"type": "addRules", "rules": [{"toolName": "Grep"}], "behavior": "allow"}]
        req = approval("Grep", {"pattern": "x"}, raw=control_raw(sugg))
        self.assertEqual(self.mapper.rule_for(req), "Grep")

    def test_suggestion_rules_entry_that_is_not_a_dict_is_tolerated(self) -> None:
        sugg = [{"type": "addRules", "rules": [None], "behavior": "allow"}]
        self.assertEqual(always_rule("Bash", {"command": "ls -la"}, sugg)[0], "Bash")

    def test_first_add_rules_suggestion_wins_and_non_add_rules_are_skipped(self) -> None:
        sugg = [{"type": "setMode", "mode": "plan"},
                {"type": "addRules", "rules": [{"toolName": "Bash", "ruleContent": "make *"}]},
                {"type": "addRules", "rules": [{"toolName": "Bash", "ruleContent": "npm *"}]}]
        self.assertEqual(always_rule("Bash", {"command": "make"}, sugg)[0], "Bash(make *)")

    def test_empty_or_bad_suggestions_fall_back_to_synthesis(self) -> None:
        for bad in [None, [], "nope", [{"type": "addRules", "rules": []}], [{"type": "addRules", "rules": "x"}]]:
            with self.subTest(suggestions=bad):
                rule, update = always_rule("Bash", {"command": "git log -1"}, bad)
                self.assertEqual(rule, "Bash(git log *)")
                self.assertEqual(update["destination"], "session")

    def test_question_request_has_no_always_rule(self) -> None:
        req = approval("AskUserQuestion", {}, choices=frozenset({Decision.DENY}), kind="question", raw=control_raw())
        self.assertIsNone(self.mapper.rule_for(req))
        self.assertEqual(self.mapper.updated_permissions(req), [])

    def test_missing_raw_payload_still_synthesizes(self) -> None:
        req = approval("Bash", {"command": "ls"}, raw={})
        self.assertEqual(self.mapper.rule_for(req), "Bash(ls *)")

    def test_translator_fixture_round_trip(self) -> None:
        with open(os.path.join(FIXTURES, "permission_ask.jsonl"), encoding="utf-8") as handle:
            line = json.loads(handle.readline())
        (req,) = [e for e in ClaudeTranslator().translate(line) if isinstance(e, ApprovalRequest)]
        self.assertEqual(self.mapper.rule_for(req), "Bash(touch probe_ok.txt)")
        self.assertTrue(RuleGrammar().matches("Bash(touch probe_ok.txt)", req, ""))
        self.assertEqual(self.mapper.updated_permissions(req)[0]["destination"], "session")


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
