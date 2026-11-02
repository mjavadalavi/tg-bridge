"""Fake launcher for the view pane tests (tests/test_view_pane.py): stands in for claude-work, claude-alt or claude.

The first argument is the launcher's kind; the rest is the argv the pane passed on. It appends one JSON line to the file
FAKE_VIEW_OUT: {"kind", "argv" (the arguments after this script, the kind first), "cwd" (where it runs), "claim" (whether
the claim file named by FAKE_VIEW_CLAIM exists while it runs)}. Then it exits 0. No Claude process is started.
"""

import json
import os
import sys

argv = sys.argv[1:]
claim = os.environ.get("FAKE_VIEW_CLAIM", "")
record = {
    "kind": argv[0] if argv else "",
    "argv": argv,
    "cwd": os.getcwd(),
    "claim": bool(claim) and os.path.exists(claim),
}
with open(os.environ["FAKE_VIEW_OUT"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")
