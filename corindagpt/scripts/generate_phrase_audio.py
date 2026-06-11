"""Generate TTS audio files for the pre-programmed phrases CSV.

Reads knowledge-base/pre_programmed_phrases.csv (one phrase per line) and
writes numbered WAV files to assets/phrases/ using the configured ElevenLabs
voice. Existing files are kept unless --force is passed, so this only costs
API credits when the CSV or voice changes.

Run from corindagpt/:  ../.venv/Scripts/python -m scripts.generate_phrase_audio [--force]
"""
from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services import tts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "knowledge-base" / "pre_programmed_phrases.csv"
OUT_DIR = ROOT / "assets" / "phrases"


async def main() -> None:
    force = "--force" in sys.argv
    with CSV_PATH.open("r", encoding="utf-8-sig") as fh:
        phrases = [row[0].strip() for row in csv.reader(fh) if row and row[0].strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(phrases)} phrases from {CSV_PATH.name} -> {OUT_DIR}")
    for i, phrase in enumerate(phrases, start=1):
        out = OUT_DIR / f"phrase_{i:02d}.wav"
        if out.exists() and not force:
            print(f"  {out.name} exists, skipping")
            continue
        audio = await tts.synthesize(phrase)
        out.write_bytes(audio)
        print(f"  {out.name}: {phrase!r} ({len(audio)} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
