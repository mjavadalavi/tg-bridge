"""tg-bridge — talk to your Claude sessions from a Telegram forum supergroup.

One private supergroup ("My AIs"), one topic per session. Each topic is bound (by its NAME) to a launcher in
config.json. The first message in a topic starts a headless session; every later turn resumes the same session
by its UUID. The UUID is stored before the process starts, so a crash or restart keeps the context. Interactive
tmux panes are never touched. Never uses --continue.

Each turn runs one `claude -p` process over the stream-json stdio protocol (TopicSession). While it runs:
  live feed   Claude's intermediate text goes out as its own messages (Turn). The tool calls between two texts share
              one tool-log message, edited as the turn goes on: a header with the count, the elapsed time, idle time
              and API retries, the calls in an expandable quote, and a Stop button while it runs. A failed call is
              marked ❌. When the turn ends the log's header holds the summary (tools, duration, tokens, cost).
  answer      Telegram HTML in up to three messages, or an answer-<ts>.md document when longer.
  permissions per-topic mode: bypassPermissions (default), acceptEdits, default or plan. Outside bypass, each
              permission request becomes a card: Allow once / Always (saved as a rule) / Deny / Bypass this turn.
              AskUserQuestion always becomes option buttons.
  queue       messages sent during a turn queue per topic (Now / Next / Drop buttons). The inbox row is stored
              before any work starts, so a restart replays what was pending.

Bridge slash commands are answered here and never reach Claude: /status /stop /mode /model /launcher /new /clear /resume /cwd
/queue /cost /rules /forget /help. Any other /command is passed to Claude as text. /clear is an alias of /new.
Button pickers: /model, /mode, /launcher, /resume, /rules and /cwd show buttons when sent without an argument; /status shows
shortcut buttons (Stop, New, Mode, Model, Cost). A tap applies the choice and edits the same message, with ✓ on the
current value. Config keys: 'models' (the /model buttons, default opus, sonnet, haiku, fable, default),
'cwd_shortcuts' (directories the /cwd buttons offer first) and 'search_roots' (where folder-name searches look,
default ~/workspace and the home directory).
Launchers: a topic runs the command its launcher names, split like a shell line, so 'myclaude run' works as a launcher.
/launcher picks one per topic (buttons, or /launcher <name>). Each launcher keeps its own transcripts in its config folder,
so a change carries the session over: launchers sharing a folder keep it as it is, others get a copy of the transcript
(thinking blocks stripped) in their folder. The launchers come from config key 'launchers' (name -> command, or
{"cmd": ..., "config_dir": ...}; the default is claude). A topic's configured 'launcher' is its
default until a pick is made.
Codex: a launcher under the config key 'agent_launchers' (e.g. "codex": {"backend": "codex", "command": "codex",
"models": [...], "config_dir": "~/.codex"}) runs the Codex CLI instead of Claude. Each turn drives `codex app-server`
over JSON-RPC; its live feed, approval cards, modes and Stop button work as they do for Claude. Its footer shows the tokens
and 'cost —', because Codex reports no cost; /cost prints the Codex tokens apart. /launcher codex switches a topic to it. A
switch between Claude and Codex starts a fresh session with a summary of the earlier conversation, since session ids do not
carry across. /model lists the launcher's 'models' and applies the choice from the next message. A Codex config_dir is the
Codex home (default $CODEX_HOME, else ~/.codex).
Folders: '/cwd <folder name>' searches the search roots (5 levels deep, 5 seconds at most) and shows the matches as
buttons; a full path is applied at once. A new forum topic whose name matches folders gets a card first ("start this
topic in one of these folders?", with a keep button for its default folder). Changing the cwd starts a fresh session,
because claude keeps transcripts per project directory; /resume still reaches the old one, and it
returns the topic to the folder that session ran in.

Safety: only allow-listed Telegram user ids are served (listed by hand in config.json, never learned). The default
mode is bypassPermissions, the owner's terminal habit, so the standing rules in RULES are the guard; /mode tightens
it per topic. Token + config live under ~/.local/state (mode 600), never in the repo. Set 'egress_proxy' to route all
Telegram calls through a local HTTP proxy.

Subcommands:
  daemon        long-poll Telegram and serve turns (the service; run under tmux)
  send <topic> <text>   push one message to a known topic (for scripts)
  sendfile <topic> <path> [caption]   push a photo (sendPhoto) or any file (sendDocument)
  type <pane> <text>   type text into a live tmux pane
  spawn <launcher> <name> <text>   start a live seat in a new tmux window
"""

__version__ = "3.0.0"
