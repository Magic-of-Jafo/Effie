"""Compare candidate LLM models on latency and in-character quality."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services import decoder  # noqa: E402
from src.services.prompt_loader import load_prompt_for_phase, render_prompt  # noqa: E402
from src.utils.initialization import load_config  # noqa: E402

MODELS = ["gpt-4o-mini", "gpt-5.4-nano"]
SPOKEN = "Cool. Could you tell me what playing card I am holding?"
RUNS = 3


async def bench(client: httpx.AsyncClient, api_key: str, model: str, prompt: str) -> None:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    times, answers = [], []
    for _ in range(RUNS):
        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if model.startswith("gpt-5") and "chat" not in model:
            payload["max_completion_tokens"] = 2000  # reasoning tokens count too
            payload["reasoning_effort"] = "none"
        else:
            payload["max_tokens"] = 64
            payload["temperature"] = 0.7
        t0 = time.monotonic()
        r = await client.post("/chat/completions", headers=headers, json=payload)
        ms = int((time.monotonic() - t0) * 1000)
        if r.status_code != 200:
            print(f"{model}: HTTP {r.status_code}: {r.text[:120]}")
            return
        content = (r.json()["choices"][0]["message"].get("content") or "").strip()
        times.append(ms)
        answers.append(content)
    med = sorted(times)[len(times) // 2]
    print(f"{model}: median {med} ms  {times}")
    for a in answers:
        print(f"   -> {a}")


async def main() -> None:
    cfg = load_config()
    decoded = decoder.decode(SPOKEN)
    prompt = render_prompt(load_prompt_for_phase(1), {"transcript": decoded})
    async with httpx.AsyncClient(base_url="https://api.openai.com/v1", timeout=60.0) as client:
        for model in MODELS:
            await bench(client, cfg["openai_api_key"], model, prompt)
            print()


if __name__ == "__main__":
    asyncio.run(main())
