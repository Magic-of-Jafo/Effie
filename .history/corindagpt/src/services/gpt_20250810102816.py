from __future__ import annotations

import logging
import json
from typing import Any, Dict, Optional, List, Tuple

import httpx

try:
    from ..utils.initialization import load_config  # type: ignore[relative-beyond-top-level]
except Exception:
    from utils.initialization import load_config

logger = logging.getLogger(__name__)


def _build_tools_config(cfg: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    transitions_cfg: Dict[str, Any] = (cfg.get("transitions") or {})
    llm_ctrl: Dict[str, Any] = transitions_cfg.get("llm_phase_control") or {}
    if not bool(llm_ctrl.get("enabled", False)):
        return None
    tool = {
        "type": "function",
        "function": {
            "name": "set_phase",
            "description": "Advance to next phase or set a specific phase number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["advance", "set"],
                        "description": "Advance to next phase or set a specific phase.",
                    },
                    "phase": {"type": "integer", "description": "Target phase when action is 'set'"},
                },
                "required": ["action"],
            },
        },
    }
    return [tool]


def _build_system_message(cfg: Dict[str, Any]) -> str:
    base = "You are a concise assistant."
    transitions_cfg: Dict[str, Any] = (cfg.get("transitions") or {})
    llm_ctrl: Dict[str, Any] = transitions_cfg.get("llm_phase_control") or {}
    if not bool(llm_ctrl.get("enabled", False)):
        return base
    phrases: List[str] = list(llm_ctrl.get("keyphrases") or [])
    phrases_text = ", ".join(phrases) if phrases else "next phase, advance phase, switch to phase {n}"
    instr = (
        "\nWhen the user expresses intent to change show phases (e.g., keyphrases: "
        f"{phrases_text}), call the function set_phase with action 'advance' or 'set' and 'phase' when setting."
    )
    return base + instr


async def generate_response(prompt_text: str, *, http_client: Optional[httpx.AsyncClient] = None, config: Optional[Dict[str, Any]] = None) -> str:
    """Generate a short text response from the configured LLM provider.

    Uses OpenAI Chat Completions via httpx.AsyncClient.
    """
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("prompt_text must be a non-empty string")

    cfg = config or load_config()
    api_key: Optional[str] = cfg.get("openai_api_key")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured (env or config.yaml)")

    model: str = (
        (cfg.get("model_names") or {}).get("text")
        if isinstance(cfg.get("model_names"), dict)
        else None
    ) or "gpt-4o-mini"

    own_client = http_client is None
    client = http_client or httpx.AsyncClient(base_url="https://api.openai.com/v1", timeout=30.0)

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        system_message = _build_system_message(cfg)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_text.strip()},
        ]
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 64,
            "temperature": 0.7,
        }
        tools = _build_tools_config(cfg)
        if tools:
            payload["tools"] = tools
        resp = await client.post("/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content: str = message.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        logger.info("GPT: response received (%d chars)", len(content))
        return content.strip()
    finally:
        if own_client:
            await client.aclose()


async def chat_with_tools(prompt_text: str, *, http_client: Optional[httpx.AsyncClient] = None, config: Optional[Dict[str, Any]] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """Chat call that returns both content and tool_calls (if any)."""
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("prompt_text must be a non-empty string")

    cfg = config or load_config()
    api_key: Optional[str] = cfg.get("openai_api_key")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured (env or config.yaml)")

    model: str = (
        (cfg.get("model_names") or {}).get("text")
        if isinstance(cfg.get("model_names"), dict)
        else None
    ) or "gpt-4o-mini"

    own_client = http_client is None
    client = http_client or httpx.AsyncClient(base_url="https://api.openai.com/v1", timeout=30.0)

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        system_message = _build_system_message(cfg)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_text.strip()},
        ]
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 64,
            "temperature": 0.7,
        }
        tools = _build_tools_config(cfg)
        if tools:
            payload["tools"] = tools
        resp = await client.post("/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message: Dict[str, Any] = choice.get("message") or {}
        content: str = message.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        tool_calls = message.get("tool_calls") or []
        logger.info("GPT: response received (%d chars, %d tool calls)", len(content), len(tool_calls) if isinstance(tool_calls, list) else 0)
        # Normalize tool_calls to list[dict]
        if not isinstance(tool_calls, list):
            tool_calls = []
        return content.strip(), tool_calls
    finally:
        if own_client:
            await client.aclose()
