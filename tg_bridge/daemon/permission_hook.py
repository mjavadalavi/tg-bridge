"""The `tg-bridge hook permission` client, reachable from the command line.

The CLI may import only app, daemon and telegram (tests/check_layers.py), and the hook client lives with the agent's
permission bridge in agents/claude/hookbridge.py. This module is that one import. The CLI's hook route runs the client
without building the App, so it stays fast and never touches Telegram or the daemon lock.
"""

from tg_bridge.agents.claude.hookbridge import run_hook_client

__all__ = ("run_hook_client",)
