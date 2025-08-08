from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from time import monotonic
import traceback

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

    print("[tts_smoke] start", flush=True)
    try:
        t0 = monotonic()
        audio = await tts.synthesize(args.text)
        t1 = monotonic()
        await tts.play(audio)
        t2 = monotonic()

        synth_ms = int((t1 - t0) * 1000)
        play_ms = int((t2 - t1) * 1000)
        total_ms = int((t2 - t0) * 1000)
        print(f"latency_ms synth={synth_ms} play={play_ms} total={total_ms}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        print("[tts_smoke] done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
