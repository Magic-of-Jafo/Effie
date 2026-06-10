"""Quick in-character test: send sample questions through the real prompt pipeline."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services import gpt, tts  # noqa: E402
from src.services.prompt_loader import load_prompt_for_phase, render_prompt  # noqa: E402

QUESTIONS = [
    "Effie, are you awake?",
    "Tell everyone what you did last night.",
    "Effie, where did you come from?",
]


async def main() -> None:
    await asyncio.gather(gpt.warmup(), tts.warmup())
    phase = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for q in QUESTIONS:
        rendered = render_prompt(load_prompt_for_phase(phase), {"transcript": q})
        sink: dict = {}
        await tts.stream_sentences_and_play(gpt.stream_chat(rendered, sink=sink))
        print(f"Q: {q}")
        print(f"Effie: {sink.get('content')}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
