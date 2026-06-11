from __future__ import annotations

import logging
import json
from typing import Any, AsyncIterator, Dict, Optional, List, Tuple

import httpx

try:
    from ..utils.initialization import load_config  # type: ignore[relative-beyond-top-level]
except Exception:
    from utils.initialization import load_config

logger = logging.getLogger(__name__)

_shared_client: Optional[httpx.AsyncClient] = None


def get_shared_client() -> httpx.AsyncClient:
    """Process-wide AsyncClient so TLS connections are reused across requests."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(base_url="https://api.openai.com/v1", timeout=30.0)
    return _shared_client


async def warmup(config: Optional[Dict[str, Any]] = None) -> None:
    """Open the TLS connection to OpenAI ahead of the first real request."""
    cfg = config or load_config()
    api_key = cfg.get("openai_api_key")
    if not api_key:
        return
    try:
        await get_shared_client().get(
            "/models/gpt-4o-mini", headers={"Authorization": f"Bearer {api_key}"}
        )
        logger.info("OpenAI connection warmed up")
    except Exception as exc:
        logger.warning("OpenAI warmup failed (continuing): %s", exc)


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
    base = "Follow the character and brevity instructions in the user message exactly."
    transitions_cfg: Dict[str, Any] = (cfg.get("transitions") or {})
    llm_ctrl: Dict[str, Any] = transitions_cfg.get("llm_phase_control") or {}
    if not bool(llm_ctrl.get("enabled", False)):
        return base
    phrases: List[str] = list(llm_ctrl.get("keyphrases") or [])
    phrases_text = ", ".join(phrases) if phrases else "next phase, advance phase, switch to phase {n}"
    instr = (
        "\nCall the function set_phase ONLY when the user explicitly and unmistakably asks to "
        f"change show phases with wording like: {phrases_text}. Questions, requests, and "
        "performance dialogue are NEVER phase changes. When in any doubt, do not call the "
        "function. Every reply must contain spoken text; never respond with a function call alone."
    )
    return base + instr


def _completion_params(model: str) -> Dict[str, Any]:
    """Model-appropriate sampling/limit parameters.

    gpt-5-family reasoning models reject max_tokens and non-default
    temperature; they take max_completion_tokens (which also counts
    reasoning tokens) and reasoning_effort instead.
    """
    if model.startswith("gpt-5") and "chat" not in model:
        return {"max_completion_tokens": 2000, "reasoning_effort": "none"}
    return {"max_tokens": 64, "temperature": 0.7}


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

    client = http_client or get_shared_client()

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
        **_completion_params(model),
    }
    tools = _build_tools_config(cfg)
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        # gpt-5.4 rejects reasoning_effort + function tools on chat/completions;
        # the default effort is light enough for our short prompts
        payload.pop("reasoning_effort", None)
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


async def stream_chat(
    prompt_text: str,
    *,
    sink: Optional[Dict[str, Any]] = None,
    http_client: Optional[httpx.AsyncClient] = None,
    config: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[str]:
    """Stream text deltas from the LLM as they are generated.

    Tool calls cannot be acted on mid-stream, so they are accumulated and,
    together with the full response text, written into `sink` (if provided)
    under the keys 'content' and 'tool_calls' once the stream completes.
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

    client = http_client or get_shared_client()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _build_system_message(cfg)},
        {"role": "user", "content": prompt_text.strip()},
    ]
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        **_completion_params(model),
    }
    tools = _build_tools_config(cfg)
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        # gpt-5.4 rejects reasoning_effort + function tools on chat/completions;
        # the default effort is light enough for our short prompts
        payload.pop("reasoning_effort", None)

    parts: List[str] = []
    # tool_call deltas arrive fragmented; merge them by index
    tool_calls_acc: Dict[int, Dict[str, Any]] = {}
    async with client.stream("POST", "/chat/completions", headers=headers, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                delta = (json.loads(data_str).get("choices") or [{}])[0].get("delta") or {}
            except Exception:
                continue
            text = delta.get("content")
            if text:
                parts.append(text)
                yield text
            for tc in delta.get("tool_calls") or []:
                idx = int(tc.get("index", 0))
                acc = tool_calls_acc.setdefault(
                    idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
                if tc.get("id"):
                    acc["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    acc["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    acc["function"]["arguments"] += fn["arguments"]

    content = "".join(parts).strip()
    tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
    logger.info(
        "GPT: stream complete (%d chars, %d tool calls)", len(content), len(tool_calls)
    )
    if sink is not None:
        sink["content"] = content
        sink["tool_calls"] = tool_calls


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

    client = http_client or get_shared_client()

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
        **_completion_params(model),
    }
    tools = _build_tools_config(cfg)
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        # gpt-5.4 rejects reasoning_effort + function tools on chat/completions;
        # the default effort is light enough for our short prompts
        payload.pop("reasoning_effort", None)
    resp = await client.post("/chat/completions", headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    message: Dict[str, Any] = choice.get("message") or {}
    content: str = message.get("content") or ""
    if not isinstance(content, str):
        content = str(content)
    tool_calls = message.get("tool_calls") or []
    logger.info(
        "GPT: response received (%d chars, %d tool calls)",
        len(content),
        len(tool_calls) if isinstance(tool_calls, list) else 0,
    )
    # Normalize tool_calls to list[dict]
    if not isinstance(tool_calls, list):
        tool_calls = []
    return content.strip(), tool_calls
