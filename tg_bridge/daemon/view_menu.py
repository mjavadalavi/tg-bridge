"""The right-click menu of a topic's tab in the `tg-bridge view` session (card D4). tmux binds MouseDown3Status server-wide,
so the menu is changed by a config file that makes the binding an if-shell on the session name: in the tg-bridge session
the menu is tmux's own menu with a "Delete topic" item added on top; every other session keeps the original command,
copied verbatim from list-keys. The item asks with confirm-before and, on y, runs view-delete with the thread id of the
window (the window option @tg_thread) through run-shell -b. A window without a thread id (the sync window) does nothing.

The file is written by Viewer.ensure_session. When the original binding is not a display-menu with a -y target, nothing is
bound: the menu then stays tmux's and the Delete item is simply absent.
"""

import re

MENU_KEY = "MouseDown3Status"  # the status-line right click
MENU_FILE = "menu.conf"        # written into Paths.view
DELETE_MARKER = "view-delete"  # in the binding once the file is sourced: the marker that makes ensure_session skip it
_BINDING = re.compile(r"^bind-key\s+-T\s+root\s+MouseDown3Status\s+(.*?)\s*$", re.MULTILINE)
_MENU = re.compile(r"(display-menu\b.*? -y \S+)\s+(.*)", re.DOTALL)  # the menu's flags, then its items


def original_binding(listing: str) -> str | None:
    """The command of the root MouseDown3Status binding, as list-keys printed it after the key. None when the listing has
    no such binding (nothing printed, or a listing of another key)."""
    match = _BINDING.search(listing)
    return match.group(1) if match else None


def menu_config(original: str, command: str, session: str) -> str | None:
    """The config file text: one bind-key that runs our menu in the session and the original command in every other one.
    command is the view-delete command line without its thread id. None when the original is not a display-menu with a
    -y target, since its items cannot be copied then."""
    match = _MENU.match(original)
    if match is None:
        return None
    ours = f"{match.group(1)} {_delete_item(command)} '' {match.group(2)}"
    guard = "#{==:#{session_name}," + session + "}"
    return (f"bind-key -T root {MENU_KEY} if-shell -F '{guard}' {{ {ours} }} {{ {original} }}\n")


def _delete_item(command: str) -> str:
    """The menu item "Delete topic" (key d): only for a tagged window, then the confirmation, then view-delete <thread>.
    No window or topic name is put in the command text: tmux substitutes formats before it parses the command, so a name
    with quotes or '#(' would break the item or run a shell command. Only #{@tg_thread} (digits, set by the bridge) is."""
    run = 'run-shell -b "' + _escape(command) + ' #{@tg_thread}"'
    ask = 'confirm-before -p "Delete this topic here and in Telegram? (y/n)" { ' + run + ' }'
    return '"Delete topic" d { if-shell -F "#{@tg_thread}" { ' + ask + ' } }'


def _escape(text: str) -> str:
    """The text inside a tmux double-quoted string: backslash and double quote are escaped."""
    return text.replace("\\", "\\\\").replace('"', '\\"')
