"""CodexBackend: the Codex adapter of AgentBackend (capabilities, modes, models, sessions).

Codex differs from Claude in what it can do live (a mode or model change applies from the next message), in having no
cost in USD, no AskUserQuestion-style cards, no bridge-chosen session id, and native image input. Those differences are
the capability flags below. Sessions are built by CodexSession; this class never parses a protocol frame.

The rule matcher is CodexRuleMapper itself: it implements both matches() (through the shared RuleGrammar matching) and
rule_for() (the Always text), so saved rules use the same grammar as Claude's. One-shot titles and carry-over of
history stay with Claude (the base-class defaults), as the design says.
"""

from tg_bridge.agents.base import AgentBackend, AgentSession, Capabilities, EventSink, ModeChoice, SessionSpec
from tg_bridge.agents.codex.handoff import find_rollout, read_entries
from tg_bridge.agents.codex.rules import CodexRuleMapper
from tg_bridge.agents.codex.session import CodexSession
from tg_bridge.agents.events import HandoffEntry
from tg_bridge.clock import Clock
from tg_bridge.config import BACKEND_CODEX, Config, LauncherSpec
from tg_bridge.logging_ import Logger

CODEX_CAPABILITIES = Capabilities(
    live_mode=False,           # a mode change applies from the next message, not inside the running turn
    live_model=False,          # likewise for the model
    questions=False,           # no AskUserQuestion-like cards (answered with -32601)
    cost_usd=False,            # tokens only; the footer shows '—' for the cost
    images=True,               # Telegram photos go to turn/start as localImage input
    preassigned_id=False,      # the thread id is learned from thread/start
)
# Labels match the Claude picker, in the same order. Codex applies every mode from the next message.
CODEX_MODE_NOTE = "applies from your next message"
CODEX_MODES = (
    ModeChoice(key="bypassPermissions", label="bypass", note=CODEX_MODE_NOTE),
    ModeChoice(key="acceptEdits", label="acceptEdits", note=CODEX_MODE_NOTE),
    ModeChoice(key="default", label="default", note=CODEX_MODE_NOTE),
    ModeChoice(key="plan", label="plan", note=CODEX_MODE_NOTE),
)


class CodexBackend(AgentBackend):
    """Stateless apart from its logger and clock. One instance serves every Codex topic."""

    name = BACKEND_CODEX
    capabilities = CODEX_CAPABILITIES

    def __init__(self, *, logger: Logger, clock: Clock) -> None:
        self._logger = logger
        self._clock = clock
        self.rules = CodexRuleMapper()

    def modes(self) -> tuple[ModeChoice, ...]:
        return CODEX_MODES

    def models(self, cfg: Config) -> tuple[str, ...]:
        """No built-in Codex model list, so this returns (). The /model picker uses the 'models' that the topic's
        agent launcher lists (Config.launcher_models, through core.topic_backend.topic_models), so this fallback is
        only reached for a launcher with no list.
        """
        return ()

    def open_session(self, spec: SessionSpec, sink: EventSink) -> AgentSession:
        return CodexSession(spec, sink, logger=self._logger, clock=self._clock)

    def handoff_entries(self, session_id: str, cwd: str, spec: LauncherSpec) -> tuple[HandoffEntry, ...]:
        """The conversation of a finished Codex thread, read from its rollout under the launcher's Codex home.

        Rollouts are keyed by thread id alone, so cwd is not needed here. Empty when there is no rollout.
        """
        return read_entries(find_rollout(spec.config_dir, session_id))
