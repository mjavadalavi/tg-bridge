"""Codex approvals and modes (agents/codex/approvals.py, agents/codex/rules.py).

Scenarios: every Decision mapping for command, file, permissions and elicitation requests; the live 0.162.1 command
approval shape (availableDecisions without decline, so DENY is cancel and aborts the turn); the SCHEMA default when
availableDecisions is absent; unsafe execpolicy prefixes give no Always button; the mode table for thread/start and
turn/start; a fresh topic's default mode (bypassPermissions -> never + danger-full-access); serverRequest/resolved; a
compound command never matching a saved Bash rule, including the zsh-wrapped form; the Edit rule for one file.
"""

import os
import unittest

from tests.support import TempHome
from tg_bridge.agents.codex import approvals
from tg_bridge.agents.codex.approvals import (APPROVALS_REVIEWER, COMMAND_METHOD, ELICITATION_METHOD, FILE_METHOD,
                                              PERMISSIONS_METHOD, UnknownMode)
from tg_bridge.agents.codex.rules import CodexRuleMapper, always_rule, prefix_text, unwrap_shell
from tg_bridge.agents.events import ApprovalRequest, ApprovalWithdrawn, Decision
from tg_bridge.config import DEFAULT_MODE, Config

LIVE_COMMAND = "/usr/bin/zsh -lc 'touch probe_ok.txt'"
LIVE_DECISIONS = ["accept", {"acceptWithExecpolicyAmendment": {"execpolicy_amendment": ["touch", "probe_ok.txt"]}},
                  "cancel"]
CWD = "/home/x/proj"


def command_params(command=LIVE_COMMAND, decisions=LIVE_DECISIONS, amendment=("touch", "probe_ok.txt")):
    params = {"kind": "command", "threadId": "t1", "turnId": "u1", "itemId": "exec-1", "command": command,
              "cwd": CWD, "startedAtMs": 1, "environmentId": "local", "commandActions": []}
    if decisions is not None:
        params["availableDecisions"] = decisions
    if amendment is not None:
        params["proposedExecpolicyAmendment"] = list(amendment)
    return params


def command_request(command=LIVE_COMMAND, decisions=LIVE_DECISIONS, amendment=("touch", "probe_ok.txt"), rpc=10):
    return approvals.request_from_frame(COMMAND_METHOD, command_params(command, decisions, amendment), rpc, cwd=CWD)


def file_request(paths=("src/a.py",), rpc=11, extra=None):
    params = {"threadId": "t1", "turnId": "u1", "itemId": "fc-1", "startedAtMs": 1, "reason": None,
              "grantRoot": None}
    params.update(extra or {})
    changes = [{"path": p, "kind": {"type": "update", "move_path": None}, "diff": "@@"} for p in paths]
    return approvals.request_from_frame(FILE_METHOD, params, rpc, changes=changes, cwd=CWD)


class CommandDecisionTests(unittest.TestCase):
    def test_live_shape_offers_once_turn_always_and_cancel_only(self):
        req = command_request()
        self.assertEqual(req.kind, "command")
        self.assertEqual(req.tool, "commandExecution")
        self.assertEqual(req.approval_id, "rpc:10")
        self.assertEqual(req.choices, frozenset({Decision.ALLOW_ONCE, Decision.ALLOW_TURN,
                                                 Decision.ALLOW_ALWAYS, Decision.DENY}))
        self.assertTrue(req.deny_aborts_turn, "decline is not offered live, so Deny cancels the turn")
        self.assertEqual(req.always_hint, "Bash(touch probe_ok.txt *)")
        self.assertEqual(req.input, {"command": LIVE_COMMAND, "cwd": CWD, "prefix": "touch probe_ok.txt"})

    def test_once_turn_and_always_are_all_accept(self):
        req = command_request()
        for decision in (Decision.ALLOW_ONCE, Decision.ALLOW_TURN, Decision.ALLOW_ALWAYS):
            self.assertEqual(approvals.result_for(req, decision), {"decision": "accept"}, decision)

    def test_deny_is_cancel_when_decline_not_offered(self):
        req = command_request()
        self.assertEqual(approvals.result_for(req, Decision.DENY), {"decision": "cancel"})

    def test_deny_is_decline_when_decline_offered(self):
        req = command_request(decisions=["accept", "acceptForSession", "decline", "cancel"])
        self.assertFalse(req.deny_aborts_turn)
        self.assertEqual(approvals.result_for(req, Decision.DENY), {"decision": "decline"})

    def test_schema_default_when_available_decisions_absent(self):
        req = command_request(decisions=None)
        self.assertFalse(req.deny_aborts_turn)
        self.assertIn(Decision.ALLOW_ALWAYS, req.choices)
        self.assertEqual(approvals.result_for(req, Decision.DENY), {"decision": "decline"})

    def test_always_hidden_when_accept_not_offered(self):
        req = command_request(decisions=["cancel"])
        self.assertEqual(req.choices, frozenset({Decision.DENY}))
        self.assertIsNone(req.always_hint)
        self.assertEqual(approvals.result_for(req, Decision.DENY), {"decision": "cancel"})
        with self.assertRaises(ValueError):
            approvals.result_for(req, Decision.ALLOW_ONCE)

    def test_always_hidden_without_a_safe_prefix(self):
        for amendment in ([], None, ["rm", "*"], ["a b"], ["x(y)"], ["touch;", "x"], [1, 2]):
            req = command_request(amendment=amendment)
            self.assertNotIn(Decision.ALLOW_ALWAYS, req.choices, amendment)
            self.assertIsNone(req.always_hint, amendment)
            self.assertIsNone(always_rule("command", req.input), amendment)

    def test_compound_command_keeps_full_text_and_its_prefix_rule(self):
        req = command_request(command="git status && rm -rf x", amendment=["git", "status"])
        self.assertEqual(req.input["command"], "git status && rm -rf x")
        self.assertEqual(req.always_hint, "Bash(git status *)")

    def test_raw_keeps_request_id_for_the_answer(self):
        req = command_request(rpc=42)
        self.assertEqual(approvals.rpc_id(req), 42)


class FileDecisionTests(unittest.TestCase):
    def test_schema_defaults_for_file_change(self):
        req = file_request(paths=("src/a.py",))
        self.assertEqual(req.kind, "file")
        self.assertEqual(req.tool, "fileChange")
        self.assertFalse(req.deny_aborts_turn)
        self.assertEqual(req.input["paths"], [os.path.realpath(os.path.join(CWD, "src/a.py"))])
        self.assertEqual(req.choices, frozenset(Decision))
        self.assertEqual(approvals.result_for(req, Decision.DENY), {"decision": "decline"})
        self.assertEqual(approvals.result_for(req, Decision.ALLOW_ONCE), {"decision": "accept"})

    def test_always_is_edit_of_the_one_absolute_path(self):
        req = file_request(paths=("src/a.py",))
        self.assertEqual(req.always_hint, "Edit(%s)" % os.path.realpath(os.path.join(CWD, "src/a.py")))

    def test_several_files_offer_no_always(self):
        req = file_request(paths=("a.py", "b.py"))
        self.assertNotIn(Decision.ALLOW_ALWAYS, req.choices)
        self.assertIsNone(req.always_hint)

    def test_no_changes_known_means_no_paths_and_no_always(self):
        req = approvals.request_from_frame(FILE_METHOD, {"reason": "need write"}, 12, changes=(), cwd=CWD)
        self.assertEqual(req.input["paths"], [])
        self.assertEqual(req.summary, "✏️ files")
        self.assertNotIn(Decision.ALLOW_ALWAYS, req.choices)

    def test_cancel_only_file_request_aborts_on_deny(self):
        narrow = approvals.request_from_frame(
            FILE_METHOD, {"availableDecisions": ["accept", "cancel"]}, 13,
            changes=[{"path": "a.py"}], cwd=CWD)
        self.assertTrue(narrow.deny_aborts_turn)
        self.assertEqual(approvals.result_for(narrow, Decision.DENY), {"decision": "cancel"})
        self.assertEqual(approvals.result_for(file_request(), Decision.DENY), {"decision": "decline"})


class PermissionsAndElicitationTests(unittest.TestCase):
    def test_permissions_grant_once_for_the_turn_and_deny_grants_nothing(self):
        params = {"threadId": "t1", "turnId": "u1", "itemId": "p-1", "environmentId": None, "startedAtMs": 1,
                  "cwd": CWD, "reason": "write outside", "permissions": {
                      "network": None, "fileSystem": {"write": ["/tmp/x"]}}}
        req = approvals.request_from_frame(PERMISSIONS_METHOD, params, 20, cwd=CWD)
        self.assertEqual(req.kind, "permissions")
        self.assertEqual(req.choices, frozenset({Decision.ALLOW_ONCE, Decision.DENY}))
        self.assertIsNone(req.always_hint)
        self.assertEqual(approvals.result_for(req, Decision.ALLOW_ONCE),
                         {"permissions": {"fileSystem": {"write": ["/tmp/x"]}}, "scope": "turn"})
        self.assertEqual(approvals.result_for(req, Decision.DENY), {"permissions": {}, "scope": "turn"})
        with self.assertRaises(ValueError):
            approvals.result_for(req, Decision.ALLOW_ALWAYS)

    def test_elicitation_accept_and_decline_shapes(self):
        params = {"threadId": "t1", "turnId": None, "serverName": "docs", "mode": "form",
                  "message": "Pick a region\nsecond line", "requestedSchema": {}, "_meta": None}
        req = approvals.request_from_frame(ELICITATION_METHOD, params, 21, cwd=CWD)
        self.assertEqual(req.kind, "elicitation")
        self.assertEqual(req.choices, frozenset({Decision.ALLOW_ONCE, Decision.DENY}))
        self.assertEqual(req.input["server"], "docs")
        self.assertNotIn("\n", req.summary)
        self.assertEqual(approvals.result_for(req, Decision.ALLOW_ONCE), {"action": "accept", "content": {}})
        self.assertEqual(approvals.result_for(req, Decision.DENY), {"action": "decline", "content": None})


class RequestFilterTests(unittest.TestCase):
    def test_non_approval_server_requests_are_not_ours(self):
        for method in ("item/tool/requestUserInput", "item/tool/call", "account/chatgptAuthTokens/refresh",
                       "attestation/generate", "applyPatchApproval", "unknown/method"):
            self.assertFalse(approvals.is_approval(method), method)
            self.assertIsNone(approvals.request_from_frame(method, {}, 3), method)

    def test_request_id_must_be_an_int(self):
        self.assertIsNone(approvals.request_from_frame(COMMAND_METHOD, command_params(), "7", cwd=CWD))
        self.assertIsNone(approvals.request_from_frame(COMMAND_METHOD, command_params(), True, cwd=CWD))

    def test_withdrawn_names_the_same_approval_id(self):
        self.assertEqual(approvals.withdrawn({"threadId": "t1", "requestId": 10}),
                         ApprovalWithdrawn(approval_id="rpc:10"))
        self.assertIsNone(approvals.withdrawn({"requestId": "10"}))
        self.assertIsNone(approvals.withdrawn({}))


class ModeTableTests(unittest.TestCase):
    def test_thread_params_per_mode(self):
        expected = {
            "default": ("untrusted", "workspace-write"),
            "acceptEdits": ("on-request", "workspace-write"),
            "bypassPermissions": ("never", "danger-full-access"),
            "plan": ("on-request", "read-only"),
        }
        for mode, (policy_name, sandbox) in expected.items():
            params = approvals.thread_params(mode)
            self.assertEqual(params, {"approvalPolicy": policy_name, "sandbox": sandbox,
                                      "approvalsReviewer": APPROVALS_REVIEWER}, mode)

    def test_turn_params_carry_the_sandbox_object(self):
        self.assertEqual(approvals.turn_params("bypassPermissions"),
                         {"approvalPolicy": "never", "sandboxPolicy": {"type": "dangerFullAccess"},
                          "approvalsReviewer": "user"})
        plan = approvals.turn_params("plan")
        self.assertEqual(plan["sandboxPolicy"], {"type": "readOnly", "networkAccess": False})
        default = approvals.turn_params("default")
        self.assertEqual(default["sandboxPolicy"]["type"], "workspaceWrite")
        self.assertEqual(default["sandboxPolicy"]["writableRoots"], [])

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(UnknownMode):
            approvals.thread_params("yolo")
        with self.assertRaises(UnknownMode):
            approvals.turn_params("")

    def test_fresh_codex_topic_defaults_to_bypass_never_full_access(self):
        with TempHome() as home:
            config = Config({}, home=home.home, env={})
            mode = config.default_mode()
            self.assertEqual(mode, DEFAULT_MODE)
            self.assertEqual(mode, "bypassPermissions")
            self.assertEqual(approvals.thread_params(mode)["approvalPolicy"], "never")
            self.assertEqual(approvals.thread_params(mode)["sandbox"], "danger-full-access")
            self.assertEqual(approvals.turn_params(mode)["sandboxPolicy"], {"type": "dangerFullAccess"})


class RuleMatchingTests(unittest.TestCase):
    def setUp(self):
        self.mapper = CodexRuleMapper()

    def test_saved_rule_matches_the_next_identical_request(self):
        req = command_request()
        rule = self.mapper.rule_for(req)
        self.assertEqual(rule, "Bash(touch probe_ok.txt *)")
        self.assertTrue(self.mapper.matches(rule, req, CWD))
        self.assertTrue(self.mapper.matches(rule, command_request(command="touch probe_ok.txt"), CWD))

    def test_rule_for_is_none_when_always_not_offered(self):
        req = command_request(decisions=["cancel"])
        self.assertIsNone(self.mapper.rule_for(req))

    def test_compound_commands_never_match_a_saved_bash_rule(self):
        rule = "Bash(git status *)"
        self.assertTrue(self.mapper.matches(rule, command_request(command="git status", amendment=["git", "status"]),
                                            CWD))
        for command in ("git status && rm -rf x", "git status; rm x", "git status | sh", "git status $(rm x)",
                        "git status `rm x`", "git status > out", "git status\nrm x",
                        "/usr/bin/zsh -lc 'git status && rm -rf x'", "/usr/bin/zsh -lc 'git status; rm x'"):
            req = command_request(command=command, amendment=["git", "status"])
            self.assertFalse(self.mapper.matches(rule, req, CWD), command)

    def test_zsh_wrapped_command_matches_the_inner_rule(self):
        rule = "Bash(touch probe_ok.txt *)"
        self.assertTrue(self.mapper.matches(rule, command_request(command=LIVE_COMMAND), CWD))
        self.assertEqual(unwrap_shell(LIVE_COMMAND), "touch probe_ok.txt")
        self.assertEqual(unwrap_shell("ls -la"), "ls -la")

    def test_a_different_command_does_not_match(self):
        rule = "Bash(touch probe_ok.txt *)"
        self.assertFalse(self.mapper.matches(rule, command_request(command="touch other.txt"), CWD))

    def test_edit_rule_matches_only_that_file(self):
        req = file_request(paths=("src/a.py",))
        rule = self.mapper.rule_for(req)
        self.assertTrue(self.mapper.matches(rule, req, CWD))
        other = file_request(paths=("src/b.py",))
        self.assertFalse(self.mapper.matches(rule, other, CWD))

    def test_bare_edit_rule_covers_multi_file_changes(self):
        req = file_request(paths=("a.py", "b.py"))
        self.assertTrue(self.mapper.matches("Edit", req, CWD))
        self.assertFalse(self.mapper.matches("Edit(/home/x/proj/a.py)", req, CWD))

    def test_bare_bash_rule_matches_codex_command(self):
        self.assertTrue(self.mapper.matches("Bash", command_request(), CWD))

    def test_always_rule_text_guards_unsafe_input(self):
        self.assertIsNone(always_rule("file", {"paths": ["relative/a.py"]}))
        self.assertIsNone(always_rule("file", {"paths": ["/x/*.py"]}))
        self.assertIsNone(always_rule("file", {"paths": ["/x/a.py", "/x/b.py"]}))
        self.assertIsNone(always_rule("command", {"prefix": None}))
        self.assertIsNone(always_rule("permissions", {"permissions": {}}))

    def test_prefix_text_accepts_plain_words_only(self):
        self.assertEqual(prefix_text(["git", "status"]), "git status")
        self.assertIsNone(prefix_text(["git", "st*"]))
        self.assertIsNone(prefix_text(["a b"]))
        self.assertIsNone(prefix_text("git status"))


if __name__ == "__main__":
    unittest.main()
