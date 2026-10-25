"""Claude tool labels (agents/claude/tools.py), ported from the _tool_label cases in fakecli/test_feed.py.

Every case of the old suite is kept with its label.
"""

import unittest

from tg_bridge.agents.claude.tools import EDIT_TOOLS, FILE_TOOLS, rel_path, tool_label

CWD = "/home/x/proj"

# (name, input) -> label, as in fakecli/test_feed.py section A
CASES = [
    (("Read", {"file_path": "/home/x/proj/src/a.py"}), "Read src/a.py"),
    (("Edit", {"file_path": "/etc/hosts"}), "Edit /etc/hosts"),
    (("NotebookEdit", {"notebook_path": "/home/x/proj/n.ipynb"}), "NotebookEdit n.ipynb"),
    (("Bash", {"command": "git status --short", "description": "Show status"}), "Bash Show status"),
    (("Bash", {"command": "ls\n-la   /tmp"}), "Bash ls -la /tmp"),
    (("Bash", {"command": "x" * 100}), "Bash " + "x" * 60),
    (("Grep", {"pattern": "foo", "path": "/home/x/proj/src"}), 'Grep "foo" in src'),
    (("Grep", {"pattern": "bar"}), 'Grep "bar"'),
    (("Glob", {"pattern": "**/*.py"}), "Glob **/*.py"),
    (("WebFetch", {"url": "https://docs.example.com/a/b?x=1"}), "WebFetch docs.example.com"),
    (("WebSearch", {"query": "python sqlite"}), "WebSearch python sqlite"),
    (("Agent", {"description": "explore repo", "prompt": "long"}), "Agent explore repo"),
    (("Skill", {"skill": "simplify"}), "Skill simplify"),
    (("mcp__srv__tool", {}), "srv:tool"),
    (("TodoWrite", {"todos": []}), "TodoWrite"),
]


class ToolLabelTests(unittest.TestCase):
    def test_every_label_case_from_the_old_suite(self) -> None:
        """Ported labels:
        A: label Agent -> 'Agent explore repo'
        A: label Bash -> 'Bash Show status'
        A: label Bash -> 'Bash ls -la /tmp'
        A: label Bash -> 'Bash xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        A: label Edit -> 'Edit /etc/hosts'
        A: label Glob -> 'Glob **/*.py'
        A: label Grep -> 'Grep "bar"'
        A: label Grep -> 'Grep "foo" in src'
        A: label NotebookEdit -> 'NotebookEdit n.ipynb'
        A: label Read -> 'Read src/a.py'
        A: label Skill -> 'Skill simplify'
        A: label TodoWrite -> 'TodoWrite'
        A: label WebFetch -> 'WebFetch docs.example.com'
        A: label WebSearch -> 'WebSearch python sqlite'
        A: label mcp__srv__tool -> 'srv:tool'
        """
        for (name, inp), want in CASES:
            with self.subTest(name=name, inp=inp):
                self.assertEqual(tool_label(name, inp, CWD), want)

    def test_file_path_outside_cwd_is_kept_absolute(self) -> None:
        self.assertEqual(tool_label("Read", {"file_path": "/etc/hosts"}, CWD), "Read /etc/hosts")

    def test_relative_path_is_kept_as_given(self) -> None:
        self.assertEqual(tool_label("Edit", {"file_path": "src/a.py"}, CWD), "Edit src/a.py")

    def test_missing_path_leaves_the_bare_name(self) -> None:
        self.assertEqual(tool_label("Write", {}, CWD), "Write")
        self.assertEqual(tool_label("Grep", {}, CWD), 'Grep ""')

    def test_bash_description_wins_over_command_and_collapses_whitespace(self) -> None:
        self.assertEqual(tool_label("Bash", {"command": "rm -rf x", "description": "  Clean \n up "}, CWD),
                         "Bash Clean up")

    def test_bash_without_command_or_description_is_just_bash(self) -> None:
        self.assertEqual(tool_label("Bash", {}, CWD), "Bash")

    def test_non_dict_input_is_treated_as_empty(self) -> None:
        self.assertEqual(tool_label("Glob", "oops", CWD), "Glob")
        self.assertEqual(tool_label("Bash", None, CWD), "Bash")

    def test_mcp_name_without_tool_part_is_kept_whole(self) -> None:
        self.assertEqual(tool_label("mcp__srv", {}, CWD), "mcp__srv")

    def test_unknown_tool_is_its_name(self) -> None:
        self.assertEqual(tool_label("AskUserQuestion", {"questions": []}, CWD), "AskUserQuestion")

    def test_rel_path_without_cwd_keeps_the_path(self) -> None:
        self.assertEqual(rel_path("/home/x/proj/a.py", None), "/home/x/proj/a.py")
        self.assertEqual(rel_path("", CWD), "")

    def test_rel_path_sibling_with_shared_prefix_is_not_inside_cwd(self) -> None:
        self.assertEqual(rel_path("/home/x/proj2/a.py", CWD), "/home/x/proj2/a.py")

    def test_file_tool_sets(self) -> None:
        self.assertEqual(set(EDIT_TOOLS), {"Edit", "Write", "MultiEdit", "NotebookEdit"})
        self.assertTrue(set(EDIT_TOOLS) <= set(FILE_TOOLS))
        self.assertIn("Read", FILE_TOOLS)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
