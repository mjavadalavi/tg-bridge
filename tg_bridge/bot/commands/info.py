"""Info commands: /status, /cost, /queue, /rules, /forget and /help (C19 handle_command, the read-only part).

The reply texts and button callbacks are the live file's. Every command here reads state and replies; none of them
changes a running turn. /status shows the backend, the resume hint of the backend (empty for a backend that has none)
and the tool count of the running turn. /cost splits the totals by backend when a topic ran on more than one.
"""

import json
import math

from tg_bridge.agents.registry import BackendRegistry, UnknownBackend
from tg_bridge.bot.commands.base import Command, CommandContext, CommandRegistry
from tg_bridge.config import Config
from tg_bridge.bot.pickers import rules_picker, status_rows
from tg_bridge.core import terminal_claim
from tg_bridge.core.linger import LingerStatus
from tg_bridge.core.live import RunningTurn
from tg_bridge.core.topic_queues import QueuedItem
from tg_bridge.core.topic_backend import topic_backend
from tg_bridge.core.turn_listeners import slash_key
from tg_bridge.feed.format import fmt_k, short
from tg_bridge.store.sessions import SessionRecord
from tg_bridge.store.turns import CostTotals
from tg_bridge.telegram.keyboards import CallbackData

DEFAULT_BACKEND = "claude"
QUEUE_TEXT_CHARS = 80  # a queued message as one /queue line is cut to this many characters
QUEUE_HEADER = "📥 {count} queued:"
QUEUE_LIST_CHARS = 3500  # the /queue text stops growing here; Telegram allows 4096 characters per message
QUEUE_MORE_TEXT = "… and {more} more"  # the last line of /queue when the list is cut
QUEUE_BUTTON_ITEMS = 10  # the first queued messages that get a ⏭ / ⏩ / ❌ button row each
MEDIA_PLACEHOLDER = "(voice or photo, still processing)"  # a queued voice/photo whose text is not in yet
SQLITE_MAX_INT = 2**63 - 1  # a /forget number above this cannot be a rule id (and would overflow the query)
SECONDS_PER_MINUTE = 60
SELF_TURN_STATUS_TEXT = "turn: background update running, {tools} tool calls so far"
LINGER_STATUS_TEXT = ("turn: idle — session kept open for background work ({pending} running, closes in {minutes} min; "
                      "/stop closes it)")
LINGER_QUIET_STATUS_TEXT = ("turn: idle — background work finished; the session closes in a few seconds (/stop closes "
                            "it now)")
TERMINAL_STATUS_TEXT = "turn: open in the terminal"
ATTACHED_STATUS_TEXT = "turn: in the terminal session (tg-bridge view)"


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
    linger = deps.linger.status(ctx.thread_id) if deps.linger is not None else None
    lines.append(_turn_line(running, linger, _terminal_holds(deps.view_dir, ctx.thread_id)))
    lines += [f"queued: {len(deps.queues.snapshot(ctx.thread_id))}",
              f"always-allow rules: {len(deps.rules.for_thread(ctx.thread_id))}"]
    ctx.say_buttons("\n".join(lines), status_rows(ctx.thread_id, running is not None or linger is not None))


def _terminal_holds(view_dir: str, thread_id: int) -> bool:
    """True while a live terminal claim (tg-bridge view) holds the topic. Views off ('') never reads the view folder.
    A view folder that cannot be read counts as no claim, so /status still answers."""
    if view_dir == "":
        return False
    try:
        return terminal_claim.live_claim(view_dir, thread_id) is not None
    except OSError:
        return False


def _turn_line(running: RunningTurn | None, linger: LingerStatus | None, terminal: bool = False) -> str:
    """The turn line of /status: the terminal when it holds the topic's session, else a running turn (a background
    update of a kept session is named as one), else the kept session with its background work and minutes left, else
    idle."""
    if terminal:  # a running turn while a claim holds the topic is a turn in that terminal session
        return ATTACHED_STATUS_TEXT if running is not None else TERMINAL_STATUS_TEXT
    if running is not None:
        if linger is not None and linger.self_turn:
            return SELF_TURN_STATUS_TEXT.format(tools=running.feed.tools)
        return f"turn: running, {running.feed.tools} tool calls so far"
    if linger is None:
        return "turn: idle"
    if linger.pending > 0:
        minutes = math.ceil(linger.seconds_left / SECONDS_PER_MINUTE)
        return LINGER_STATUS_TEXT.format(pending=linger.pending, minutes=minutes)
    return LINGER_QUIET_STATUS_TEXT


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
    """One message: a header with the count, a numbered line per queued message, and a button row for each of the
    first QUEUE_BUTTON_ITEMS messages (⏭ Now, ⏩ Next, ❌ Drop, labelled with the number)."""
    items = ctx.deps.queues.snapshot(ctx.thread_id)
    if not items:
        ctx.say("the queue is empty.")
        return
    rows = []
    for index, item in enumerate(items[:QUEUE_BUTTON_ITEMS], 1):
        rows.append([
            (f"⏭ {index}", CallbackData("now", (ctx.thread_id, item.nid))),
            (f"⏩ {index}", CallbackData("jump", (ctx.thread_id, item.nid))),
            (f"❌ {index}", CallbackData("drop", (ctx.thread_id, item.nid))),
        ])
    ctx.say_buttons(_queue_text(items), rows)


def _queue_text(items: tuple[QueuedItem, ...]) -> str:
    """The /queue message. When the whole list would pass QUEUE_LIST_CHARS, the lines stop early (room is kept for the
    last line, which tells how many are not listed)."""
    lines = [QUEUE_HEADER.format(count=len(items))]
    for index, item in enumerate(items, 1):
        body = MEDIA_PLACEHOLDER if item.text is None else item.text
        lines.append(f"{index}. {short(body, QUEUE_TEXT_CHARS)}")
    text = "\n".join(lines)
    if len(text) <= QUEUE_LIST_CHARS:
        return text
    room = len("\n" + QUEUE_MORE_TEXT.format(more=len(items)))  # the longest last line this list can need
    kept = [lines[0]]
    for line in lines[1:]:
        if len("\n".join(kept + [line])) + room > QUEUE_LIST_CHARS:
            break
        kept.append(line)
    more = len(items) - (len(kept) - 1)
    return "\n".join(kept + [QUEUE_MORE_TEXT.format(more=more)])


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
