from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure src/ is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services import tts  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description="TTS smoke test using ElevenLabs")
    parser.add_argument("text", nargs="?", default="Hello from Effie. This is a TTS smoke test.")
    args = parser.parse_args()

    audio = await tts.synthesize(args.text)
    await tts.play(audio)


if __name__ == "__main__":
    asyncio.run(main())
