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
- A Codex entry turns its name into a Codex launcher. `/launcher codex` switches a topic to it; a switch
  between Claude and Codex starts a fresh session with a summary of the earlier conversation. A Codex
  `config_dir` is the Codex home (default `$CODEX_HOME`, else `~/.codex`).
- Auto-generated topic titles use `TG_TITLE_LAUNCHER` (env var, default `claude`); set it to a cheap seat or
  to an empty string to turn titles off.

## Config reference

| key | default | meaning |
|---|---|---|
| `allow_user_ids` | `[]` | Telegram user ids allowed to use the bot (required) |
| `topics` | `{}` | per-topic settings: `launcher`, `mode`, `model`, `cwd`, `allowed_tools`, `note` |
| `default_topic` | — | same fields, used for topics with no entry |
| `launchers` / `agent_launchers` | `claude` | see above |
| `default_mode` | `bypassPermissions` | mode for topics without one |
| `models` | `opus, sonnet, haiku, fable, default` | the `/model` buttons |
| `max_turn_seconds` / `hard_cap_seconds` | `1800` / `21600` | active-time cap and hard ceiling |
| `poll_seconds` | `50` | Telegram long-poll timeout |
| `progress_flush_seconds` / `idle_ping_seconds` | `3` / `60` | feed batching and idle ping |
| `answer_doc_threshold` | `9000` | answer length that becomes a `.md` document |
| `max_output_chunks` | `4` | max Telegram messages per answer |
| `cwd_shortcuts` / `search_roots` | `~/workspace`, home | `/cwd` buttons and search roots |
| `egress_proxy` | — | HTTP proxy for all Telegram calls |

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
