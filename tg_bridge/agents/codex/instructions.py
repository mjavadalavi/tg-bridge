"""Topic rules as Codex turn input (DESIGN section 4, 'RULES delivery').

developerInstructions is not used. On thread/resume it is accepted but never reaches the model or the rollout (live probe
run4, A.8 FAILED), and thread/start delivery is not verified either. The rules therefore travel as a leading text item of
every turn/start, which the probe showed the model reads. The item is wrapped in tags so that the handoff reader can tell
it apart from what the user typed: the rollout stores it as an input_text block of the user message.
"""

RULES_OPEN = "<tg-bridge-rules>"
RULES_CLOSE = "</tg-bridge-rules>"


def wrap_rules(rules: str) -> str:
    return f"{RULES_OPEN}\n{rules}\n{RULES_CLOSE}"


def is_rules_block(text: str) -> bool:
    """True for a text block written by wrap_rules (a rollout block, not something the user typed)."""
    stripped = text.strip()
    return stripped.startswith(RULES_OPEN) and stripped.endswith(RULES_CLOSE)
