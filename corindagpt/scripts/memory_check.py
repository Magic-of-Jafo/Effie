"""Live check: Effie should remember earlier turns and her own played phrases."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services import decoder, gpt  # noqa: E402
from src.services.conversational_memory import ConversationalMemory  # noqa: E402
from src.services.prompt_loader import load_prompt_for_phase, render_prompt  # noqa: E402


async def ask(memory: ConversationalMemory, spoken: str) -> str:
    decoded = decoder.decode(spoken)
    rendered = render_prompt(load_prompt_for_phase(1), {"transcript": decoded})
    resp = await gpt.generate_response(rendered, history=memory.messages())
    memory.add_exchange(spoken, resp)
    return resp


async def main() -> None:
    await gpt.warmup()
    memory = ConversationalMemory(max_turns=10)

    r1 = await ask(memory, "Cool, could you tell me what playing card I am holding?")
    print("Q1 (coded card):", r1)

    # A pull-string phrase plays; she said it, so she remembers it
    memory.add_assistant("I hear whispers when you're not here.")

    r2 = await ask(memory, "Effie, remind everyone - what card did you see a moment ago?")
    print("Q2 (recall card):", r2)

    r3 = await ask(memory, "And what did you say about whispers?")
    print("Q3 (recall phrase):", r3)


if __name__ == "__main__":
    asyncio.run(main())
