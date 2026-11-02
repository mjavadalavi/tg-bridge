# tg-bridge

Talk to headless coding-agent sessions from a Telegram forum supergroup — one topic per session, with the
same feel as working in a terminal: a live tool feed while the agent works, permission cards outside bypass
mode, a queue for messages sent mid-turn, and per-topic launchers, models, modes and working folders.

Built for the [Claude Code](https://claude.com/claude-code) CLI and the OpenAI Codex CLI. Standard library
only — no third-party Python packages.

## How it works

- **Topics are sessions.** The first message in a forum topic starts a headless `claude -p` (or Codex) turn;
  every later turn resumes the same session by its UUID, which is stored before the process starts, so a
  crash or restart keeps the context.
- **Launchers are configurable.** A launcher names the command a topic runs (anything on `PATH`, split like a
  shell line) plus the config folder its transcripts live in. You define launchers in `config.json`; the only
  built-in default is `claude`. Switching a topic between launchers carries the conversation over (a
  transcript copy when the config folder differs).
- **Two backends.** Claude runs over the stream-json stdio protocol; Codex runs over `codex app-server`
  (JSON-RPC). Both get the same live feed, approval cards, modes, model picker and Stop button.
- **Live feed.** Claude's intermediate text goes out as its own messages. Tool calls between two texts share
  one message that is edited as the turn goes on: a header with the tool count, elapsed time, idle time and
  API retries, the calls in an expandable quote, a ❌ on failures, and a Stop button while it runs. The final
  header holds the summary (tools, duration, tokens, cost).
- **Answers** are Telegram HTML in up to three messages, or a `.md` document when longer.
- **Permissions** per topic: `bypassPermissions` (default), `acceptEdits`, `default`, `plan`. Outside bypass,
  each permission request becomes a card — Allow once / Always (saved as a rule) / Deny / Bypass this turn.
  `AskUserQuestion` becomes option buttons.
- **Queue.** Messages sent during a turn queue per topic (Now / Next / Drop). The inbox row is stored before
  any work starts, so a restart replays whatever was pending.
- **Folders.** `/cwd <name>` searches your folders (5 levels deep, at most 5 s) and shows matches as buttons.
  A new topic whose name matches a folder gets a "start here?" card. Changing the folder starts a fresh
  session, since transcripts are per project directory; `/resume` still reaches the old one.

## Requirements

- Python 3.11+ (standard library only)
- The Claude Code CLI (`claude`) on `PATH`, logged in
- Optional: the Codex CLI (`codex`) for Codex launchers
- Optional: a Soniox API key for voice-note transcription
- Optional: tmux, for `tg-bridge view`

## Setup

1. Create a bot with [@BotFather](https://t.me/BotFather) and add it to a forum (supergroup with topics) as an
   administrator. Send any message in the group once so the bridge learns the group id.
2. Store the bot token at `~/.local/state/tg-bridge/token` (mode 600):
   ```sh
   install -m 600 /dev/null ~/.local/state/tg-bridge/token   # then paste the token inside
   ```
3. Copy `config.example.json` to `~/.local/state/tg-bridge/config.json` and edit it. At minimum put your own
   Telegram user id in `allow_user_ids` — only listed ids are ever served.
4. Run the daemon:
   ```sh
   tg-bridge daemon        # or: python3 tg-bridge.py daemon
   ```
5. Open a topic and send a message. `/help` lists the bridge commands.

### systemd (user service)

```ini
[Unit]
Description=tg-bridge
Wants=network-online.target
After=network-online.target

[Service]
ExecStart=%h/.local/bin/tg-bridge daemon
Restart=always
RestartSec=5
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin
KillMode=mixed
TimeoutStopSec=60

[Install]
WantedBy=default.target
```

## Launchers (config.json)

Everything a topic runs is a launcher you configure. The `launchers` section maps a name to a command string
or a `{cmd, config_dir}` object; `agent_launchers` adds non-Claude backends:

```json
{
  "launchers": {
    "claude": {"cmd": "claude", "config_dir": "~/.claude"},
    "fast":   {"cmd": "claude --model haiku", "config_dir": "~/.claude"},
    "work":   {"cmd": "myclaude run", "config_dir": "~/.claude-work"}
  },
  "agent_launchers": {
    "codex": {"backend": "codex", "command": "codex", "models": ["gpt-5.1-codex"], "config_dir": "~/.codex"}
  }
}
```

- `config_dir` is where that launcher keeps its session transcripts (`~/.claude` for Claude). Launchers that
  share a folder share sessions; switching to a launcher with a different folder copies the transcript across
  so the conversation continues.
- `transport` is optional and only for Claude launchers. `print` (the default) runs one `claude -p` per turn over
  stream-json. `tui` runs the interactive TUI in a pseudo-terminal, for services that refuse non-interactive use,
  e.g. a team gateway that answers `claude -p` with `API Error: 403 Non-interactive use isn't supported`.
- A Codex entry turns its name into a Codex launcher. `/launcher codex` switches a topic to it; a switch
  between Claude and Codex starts a fresh session with a summary of the earlier conversation. A Codex
  `config_dir` is the Codex home (default `$CODEX_HOME`, else `~/.codex`).
- Auto-generated topic titles use `TG_TITLE_LAUNCHER` (env var, default `claude`); set it to a cheap seat or
  to an empty string to turn titles off.

### The `tui` transport

With `"transport": "tui"`, each turn runs the interactive Claude TUI in a pseudo-terminal:

- One TUI process per turn. The message is typed into it, and when the turn ends the bridge closes it with `/exit`.
- The live feed comes from the session transcript file, the same JSON lines the print transport reads.
- Permission cards come from a `PermissionRequest` hook (`tg-bridge hook permission`) that talks to the bridge
  over a per-session Unix socket.
- The folder-trust dialog is accepted automatically.
- Stop sends Esc to the TUI.

tui limits:

- `/mode` and `/model` apply from the next message. Typing `/model` inside the TUI would save your default model,
  so the bridge passes the choice as a flag instead.
- No cost in the footer, only token counts.
- `AskUserQuestion` is disabled, so the model asks its questions in plain text.
- Tool calls made by subagents are not in the live feed.
- Auto titles use `TG_TITLE_LAUNCHER`, which must stay a print launcher (or be empty).
- A service may cap how many sessions run at once.

## Sessions in the terminal: `tg-bridge view`

`tg-bridge view` attaches to a tmux session named `tg-bridge`. It has one window per Telegram topic. A window is
named with a shortened, tmux-safe form of the topic name (dots and spaces become dashes, and long names are cut),
so it may not match the topic name exactly. New topics get a window automatically. A window of a deleted topic
closes, and a renamed topic's window gets the new name.

Each tab is the topic's real Claude Code session. Enter opens it, using the same launcher Telegram uses for that
topic (`claude-work`, `claude-alt`, `claude` or another one you set), so the session you get is the one Telegram talks to:

- Enter in a tab starts Claude Code in that topic's folder and resumes its session, so you continue the same
  conversation in the terminal.
- Exit Claude Code to give the session back to Telegram. The next Telegram message resumes it from where you left it.
- While the session is open in the terminal, Telegram messages for that topic wait, and the bot posts a notice that
  the session is open in the terminal. They run when you close Claude Code.
- While Telegram runs a turn for the topic, or keeps the session for background work, the tab shows a busy screen
  with the live output instead of opening Claude Code. Enter does nothing there; press Enter again once the bot is
  idle.

The ready screen lists the Claude launchers under the topic's details, numbered, with the current one marked. Press a
number (1 to 9) to change the topic's launcher: it is the same change as Telegram's `/launcher`, so Telegram uses the new
launcher too, and it is refused while the topic is busy.

A right-click on a tab offers Delete topic, which deletes the topic here and in Telegram after a y/n confirmation; the conversation file stays on disk.

Switching tabs: click a tab with the mouse, or use the tmux keys (prefix `Ctrl-b`):

- `Ctrl-b n` / `Ctrl-b p`: next / previous topic tab
- `Ctrl-b w`: pick a topic from the list
- `Ctrl-b d`: leave the view; it keeps running in the background

A window follows the size of the terminal you attach with, so it fills that terminal. The busy screen is drawn at
`tui_size` (see the config reference): on a bigger terminal the rest of the window stays empty, and on a smaller one the
lines wrap badly and part of the window is out of view (tmux pans to follow the cursor). Set `tui_size` to the size you
usually use.

The view files live in `<state>/view` (mode 0700, `<state>` is `~/.local/state/tg-bridge`):

- `<thread>.pty`: the raw terminal mirror of a `tui` topic
- `<thread>.feed`: plain lines of a `print` topic
- `<thread>.json`: metadata: topic name, transport, state (`running`, `idle` or `lingering`), updated epoch, launcher
- `<thread>.term`: the claim of an open Claude Code session (JSON with the pid of the pane that holds it)
- `<thread>.lock`: a lock that keeps the claim and the start of a Telegram turn from racing

## Config reference

| key | default | meaning |
|---|---|---|
| `allow_user_ids` | `[]` | Telegram user ids allowed to use the bot (required) |
| `topics` | `{}` | per-topic settings: `launcher`, `mode`, `model`, `cwd`, `allowed_tools`, `note` |
| `default_topic` | — | same fields, used for topics with no entry |
| `launchers` / `agent_launchers` | `claude` | see above; a Claude entry may set `transport` (`print` or `tui`) |
| `default_mode` | `bypassPermissions` | mode for topics without one |
| `tui_size` | `200x50` | terminal size for `tui` topics, as `COLSxROWS` (each side at most 1000); see `tg-bridge view` |
| `models` | `opus, sonnet, haiku, fable, default` | the `/model` buttons |
| `max_turn_seconds` / `hard_cap_seconds` | `1800` / `21600` | active-time cap and hard ceiling |
| `poll_seconds` | `50` | Telegram long-poll timeout |
| `progress_flush_seconds` / `idle_ping_seconds` | `3` / `60` | feed batching and idle ping |
| `answer_doc_threshold` | `9000` | answer length that becomes a `.md` document |
| `linger_seconds` | `1800` | seconds a session stays open after your turn while background work runs; `0` turns it off |
| `linger_max_sessions` | `3` | how many sessions may be kept open for background work at once; `0` keeps none |
| `topic_probe_seconds` | `600` | how often idle topics are checked for a topic you deleted in Telegram (it is then removed here too); `0` turns the check off |
| `max_output_chunks` | `4` | max Telegram messages per answer |
| `cwd_shortcuts` / `search_roots` | `~/workspace`, home | `/cwd` buttons and search roots |
| `egress_proxy` | — | HTTP proxy for all Telegram calls |

### Background work

A Claude session stays open after your turn while a Monitor, a background command, a background agent or a scheduled wakeup is still running, so its result can reach the topic. Such a report starts with 🔔 background update. `/stop` closes a kept session. The `linger_seconds` countdown starts at your last message, not at the background work.
It works on both Claude transports (`print` and `tui`); a Codex session is never kept open.

Secrets never live in `config.json`: the bot token is `~/.local/state/tg-bridge/token`, the Soniox key is
`soniox-key` next to it, both mode 600.

## Bridge commands

`/status` `/stop` `/mode` `/model` `/launcher` `/new` (`/clear`) `/resume` `/cwd` `/queue` `/cost` `/rules`
`/forget` `/help` — answered by the bridge, never sent to the agent. Any other `/command` is passed through
as text.

## Development

```sh
python3 -I -m unittest discover -s tests -t .
python3 -I tests/check_layers.py      # import-direction rules
python3 -I tests/check_globals.py     # no mutable module state
```

The package is layered: `telegram/` (client, keyboards, messenger), `store/` (SQLite repositories),
`agents/` (backend interface, `claude/` and `codex/` adapters), `core/` (turn runner, permissions, handoff),
`feed/` (the live tool feed), `bot/` (commands and callbacks), `daemon/` (poll loop) — wired by constructor
injection in `app.py`.

## License

[MIT](LICENSE)
