"""Measure the warmed, overlapped LLM->TTS pipeline latency end to end."""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services import gpt, tts  # noqa: E402

PROMPT = "Greet the audience theatrically in exactly three short sentences."


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    await asyncio.gather(gpt.warmup(), tts.warmup())
    for run in range(2):
        sink: dict = {}
        t0 = time.monotonic()
        await tts.stream_sentences_and_play(
            gpt.stream_chat(PROMPT, sink=sink), started_at_monotonic=t0
        )
        text = (sink.get("content") or "")[:60]
        print(f"run {run + 1} text: {text}...")


if __name__ == "__main__":
    asyncio.run(main())
