from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# Prompts directory at: corindagpt/prompts
PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


def _prompt_path_for_phase(phase: int) -> Path:
    try:
        phase_int = int(phase)
    except Exception:
        phase_int = 1
    return PROMPTS_DIR / f"phase_{phase_int}_prompt.txt"


def _default_template() -> PromptTemplate:
    # Generic fallback prompt
    text = (
        "You are a concise, helpful assistant.\n"
        "User transcript: {transcript}\n"
        "Respond helpfully and briefly."
    )
    return PromptTemplate.from_template(text)


def load_prompt_for_phase(phase: int) -> PromptTemplate:
    """Load a PromptTemplate for the given phase; fallback to a default template if missing."""
    path = _prompt_path_for_phase(phase)
    try:
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError("Empty prompt file")
        return PromptTemplate.from_template(content)
    except Exception as exc:
        logger.warning("Prompt for phase %s not found or invalid (%s); using default.", phase, exc)
        return _default_template()


def render_prompt(template: PromptTemplate, context: Dict[str, object]) -> str:
    """Render a prompt safely by providing empty defaults for missing variables."""
    try:
        variables = {name: str(context.get(name, "")) for name in (getattr(template, "input_variables", []) or [])}
        return template.format(**variables)
    except Exception as exc:
        logger.error("Prompt render failed: %s; using minimal fallback.", exc)
        # Last-resort fallback using transcript only
        transcript = str(context.get("transcript", ""))
        return f"User transcript: {transcript}\nRespond briefly."
