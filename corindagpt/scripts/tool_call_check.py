"""Verify no spurious set_phase tool calls on normal performance dialogue."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services import decoder, gpt  # noqa: E402


async def main() -> None:
    await gpt.warmup()
    decoded = decoder.decode("Alright then, tell me what my favorite bird is?")
    print(decoded)
    print()
    for i in range(5):
        sink: dict = {}
        async for _ in gpt.stream_chat(decoded, sink=sink):
            pass
        tc = sink.get("tool_calls") or []
        print(f"run {i + 1}: tool_calls={len(tc)} text={sink.get('content')!r}")
    sink = {}
    async for _ in gpt.stream_chat("Effie, advance phase now please.", sink=sink):
        pass
    calls = [(c["function"]["name"], c["function"]["arguments"]) for c in (sink.get("tool_calls") or [])]
    print("explicit phase change ->", calls)


if __name__ == "__main__":
    asyncio.run(main())
