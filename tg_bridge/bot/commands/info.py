"""Info commands: /status, /cost, /queue, /rules, /forget and /help (C19 handle_command, the read-only part).

The reply texts and button callbacks are the live file's. Every command here reads state and replies; none of them
changes a running turn. /status shows the backend, the resume hint of the backend (empty for a backend that has none)
and the tool count of the running turn. /cost splits the totals by backend when a topic ran on more than one.
"""

import json

from tg_bridge.agents.registry import BackendRegistry, UnknownBackend
from tg_bridge.bot.commands.base import Command, CommandContext, CommandRegistry
from tg_bridge.config import Config
from tg_bridge.bot.pickers import rules_picker, status_rows
from tg_bridge.core.live import RunningTurn
from tg_bridge.core.topic_backend import topic_backend
from tg_bridge.core.turn_listeners import slash_key
from tg_bridge.feed.format import fmt_k, short
from tg_bridge.store.sessions import SessionRecord
from tg_bridge.store.turns import CostTotals
from tg_bridge.telegram.keyboards import CallbackData

DEFAULT_BACKEND = "claude"
QUEUE_TEXT_CHARS = 80
MEDIA_PLACEHOLDER = "(voice or photo, still processing)"  # a queued voice/photo whose text is not in yet
SQLITE_MAX_INT = 2**63 - 1  # a /forget number above this cannot be a rule id (and would overflow the query)


def register_info_commands(registry: CommandRegistry) -> None:
    """Adds the info commands to the registry. The App sets the menu order (LIVE_MENU_ORDER), so /help and the
    setMyCommands list do not depend on the order of these calls."""
    registry.register(Command("status", "show mode, model, session, cwd and queue", _status))
    registry.register(Command("cost", "token and cost totals for this topic", _cost))
    registry.register(Command("queue", "show queued messages", _queue))
    registry.register(Command("rules", "list the always-allow rules, each with a 🗑 button", _rules))
    registry.register(Command("forget", "remove an always-allow rule by number, or all", _forget))
    registry.register(Command("help", "list the bridge commands", _help))


def launcher_state(config: Config, row: SessionRecord | None, topic: str) -> tuple[str | None, str]:
    """(launcher name, command) the topic runs: the stored pick while it is still in config, else the topic's configured
    launcher, else the command stored with its session. The name is None when no visible launcher has that command."""
    choices = config.launcher_choices()
    pick = row.launcher_pick if row else ""
    if pick and pick in choices:
        return pick, choices[pick]
    cmd = str(config.topic(topic).get("launcher") or "") or (row.launcher if row else "")
    return next((name for name, value in choices.items() if value == cmd), None), cmd


def _status(ctx: CommandContext, args: str) -> None:
    deps = ctx.deps
    row = deps.sessions.get(ctx.thread_id)
    running = deps.live.running(ctx.thread_id)
    backend = topic_backend(deps.config, deps.backends, row, ctx.topic).name if row else DEFAULT_BACKEND
    mode = (row.mode if row else "") or deps.config.default_mode()
    model = (row.model if row else "") or "default"
    cwd = row.cwd if row else ""
    uuid = row.uuid if row else ""
    cur, cmd = launcher_state(deps.config, row, ctx.topic)
    launcher_line = f"launcher: {cur} ({cmd})" if cur else f"launcher: {cmd or 'none'}"
    lines = [f"topic: {ctx.topic}", f"backend: {backend}", f"mode: {mode}", f"model: {model}", launcher_line,
             f"cwd: {cwd}", f"session: {uuid or 'none yet (the next message starts one)'}"]
    hint = _resume_hint(deps.backends, backend, uuid, cwd) if uuid else ""
    if hint:
        lines.append(f"in a terminal: {hint}")
    lines.append(_turn_line(running))
    lines += [f"queued: {len(deps.queues.snapshot(ctx.thread_id))}",
              f"always-allow rules: {len(deps.rules.for_thread(ctx.thread_id))}"]
    ctx.say_buttons("\n".join(lines), status_rows(ctx.thread_id, running is not None))


def _turn_line(running: RunningTurn | None) -> str:
    if running is None:
        return "turn: idle"
    return f"turn: running, {running.feed.tools} tool calls so far"


def _resume_hint(backends: BackendRegistry, backend: str, uuid: str, cwd: str) -> str:
    """The backend's command that resumes the session in a terminal, or '' (also when the backend is not registered)."""
    try:
        return backends.get(backend).resume_hint(uuid, cwd)
    except UnknownBackend:
        return ""


def _cost(ctx: CommandContext, args: str) -> None:
    total = ctx.deps.turns.totals(ctx.thread_id)
    by_backend = ctx.deps.turns.summary_by_backend(ctx.thread_id)
    lines = [_cost_line(total)]
    if len(by_backend) > 1:
        lines += [f"{name}: {_cost_line(totals)}" for name, totals in by_backend.items()]
    text = "\n".join(lines)
    if total.unknown:
        text += "\n* some turns ran on a model without a known price"
    ctx.say(text)


def _cost_line(totals: CostTotals) -> str:
    cost = "unknown" if totals.cost is None else f"${totals.cost:.2f}" + ("*" if totals.unknown else "")
    return (f"{totals.turns} turns · ↑{fmt_k(totals.tokens_in)} ↓{fmt_k(totals.tokens_out)} tokens · cost {cost}")


def _queue(ctx: CommandContext, args: str) -> None:
    items = ctx.deps.queues.snapshot(ctx.thread_id)
    if not items:
        ctx.say("the queue is empty.")
        return
    for index, item in enumerate(items, 1):
        body = MEDIA_PLACEHOLDER if item.text is None else item.text
        ctx.say_buttons(f"{index}. {short(body, QUEUE_TEXT_CHARS)}", [[
            ("⏭ Now", CallbackData("now", (ctx.thread_id, item.nid))),
            ("⏩ Next", CallbackData("jump", (ctx.thread_id, item.nid))),
            ("❌ Drop", CallbackData("drop", (ctx.thread_id, item.nid))),
        ]])


def _rules(ctx: CommandContext, args: str) -> None:
    picker = rules_picker(ctx.thread_id, ctx.deps.rules.for_thread(ctx.thread_id))
    if picker.rows:
        ctx.say_buttons(picker.text, picker.rows)
    else:
        ctx.say(picker.text)


def _forget(ctx: CommandContext, args: str) -> None:
    thread = ctx.thread_id
    if args.lower() == "all":
        removed = ctx.deps.rules.delete_all(thread)
    elif args.isdecimal():
        rule_id = int(args)
        removed = 1 if rule_id <= SQLITE_MAX_INT and ctx.deps.rules.delete(thread, rule_id) else 0
    else:
        ctx.say("usage: /forget <number> or /forget all (numbers come from /rules)")
        return
    ctx.say(f"🗑 removed {removed} rule(s). They stop applying from the next turn." if removed
            else "no such rule — see /rules")


def _help(ctx: CommandContext, args: str) -> None:
    lines = ["Bridge commands:"] + [f"/{word} — {text}" for word, text in ctx.registry.bridge_commands()]
    claude_commands = _stored_slash_commands(ctx)
    if claude_commands:
        lines += ["", "Claude commands (sent to Claude as text):"] + [f"/{name}" for name in claude_commands]
    ctx.say("\n".join(lines))


def _stored_slash_commands(ctx: CommandContext) -> list[str]:
    """The Claude slash commands the topic's last session reported (kv slash:<thread>), or [] when none are usable."""
    stored = ctx.deps.kv.get(slash_key(ctx.thread_id))
    if not stored:
        return []
    try:
        names = json.loads(stored)
    except ValueError:
        return []
    return [str(name) for name in names] if isinstance(names, list) else []
