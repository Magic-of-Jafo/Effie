"""End-to-end decoder test: coded sentence -> decode -> prompt -> LLM -> TTS."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services import decoder, gpt, tts  # noqa: E402
from src.services.prompt_loader import load_prompt_for_phase, render_prompt  # noqa: E402

CODED = [
    # COOL COULD -> King of Hearts
    "Effie, cool, could you tell everyone what card this gentleman is holding?",
    # ALL RIGHT NOW CAN -> 51
    "All right, now can you guess the number she wrote down?",
]


async def main() -> None:
    await asyncio.gather(gpt.warmup(), tts.warmup())
    for spoken in CODED:
        decoded = decoder.decode(spoken)
        print("--- decoded prompt input ---")
        print(decoded)
        rendered = render_prompt(load_prompt_for_phase(2), {"transcript": decoded})
        sink: dict = {}
        await tts.stream_sentences_and_play(gpt.stream_chat(rendered, sink=sink))
        print("Effie:", sink.get("content"))
        print()


if __name__ == "__main__":
    asyncio.run(main())
