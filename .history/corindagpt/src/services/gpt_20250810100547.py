from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

try:
    from ..utils.initialization import load_config  # type: ignore[relative-beyond-top-level]
except Exception:
    from utils.initialization import load_config

logger = logging.getLogger(__name__)


async def generate_response(prompt_text: str, *, http_client: Optional[httpx.AsyncClient] = None, config: Optional[Dict[str, Any]] = None) -> str:
    """Generate a short text response from the configured LLM provider.

    Uses OpenAI Chat Completions via httpx.AsyncClient.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

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
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": prompt.strip()},
            ],
            "max_tokens": 64,
            "temperature": 0.7,
        }
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
