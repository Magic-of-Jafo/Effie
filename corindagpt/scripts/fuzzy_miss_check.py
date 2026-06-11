"""Verify the graceful-miss response when no code decodes."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services import decoder, gpt  # noqa: E402
from src.services.prompt_loader import load_prompt_for_phase, render_prompt  # noqa: E402

GARBLED = [
    "Tel me what card I am holding?",  # flubbed code word
    "Uh, what's the thing she wrote down?",  # no code at all
    "All right, um, what was it again?",  # prefix without completion
]


async def main() -> None:
    await gpt.warmup()
    for spoken in GARBLED:
        decoded = decoder.decode(spoken)
        rendered = render_prompt(load_prompt_for_phase(1), {"transcript": decoded})
        resp = await gpt.generate_response(rendered)
        print(f"{spoken!r} -> {resp}")


if __name__ == "__main__":
    asyncio.run(main())
