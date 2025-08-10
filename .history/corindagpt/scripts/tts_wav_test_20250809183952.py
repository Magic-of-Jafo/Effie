from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

# Ensure corindagpt/src is on sys.path when running from repo root
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.initialization import load_config  # type: ignore
from services import tts as tts_service  # type: ignore


async def main(play: bool = True) -> None:
    cfg: Dict[str, Any] = load_config()

    api_key: Optional[str] = cfg.get("elevenlabs_api_key")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY not configured (env or config.yaml)")
        sys.exit(1)

    tts_cfg: Dict[str, Any] = cfg.get("tts", {})
    ev_cfg: Dict[str, Any] = tts_cfg.get("elevenlabs", {})
    voice_id: str = ev_cfg.get("voice_id") or "EXAVITQu4vr4xnSDxMaL"
    model_id: str = ev_cfg.get("model_id") or "eleven_multilingual_v2"

    async with httpx.AsyncClient(base_url="https://api.elevenlabs.io/v1", timeout=60.0) as client:
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/wav",
        }
        payload = {
            "text": "The test is successful",
            "model_id": model_id,
            # Ask explicitly for WAV payload
            "output_format": "wav",
            "voice_settings": ev_cfg.get("voice_settings") or {"stability": 0.3, "similarity_boost": 0.7},
        }
        resp = await client.post(f"/text-to-speech/{voice_id}", headers=headers, json=payload)
        print(f"HTTP {resp.status_code} Content-Type={resp.headers.get('Content-Type')}")
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            print("Body:", exc.response.text)
            sys.exit(2)
        audio = resp.content
        if not audio or not (len(audio) >= 4 and audio[:4] == b"RIFF"):
            print("Received non-WAV or empty payload (head=", audio[:16], ")")
            sys.exit(3)

        out_path = Path.cwd() / "wav_test_output.wav"
        out_path.write_bytes(audio)
        print(f"Saved WAV: {out_path} ({len(audio)} bytes)")

        if play:
            print("Playing...")
            await tts_service.play(audio)
            print("Done.")


if __name__ == "__main__":
    # Usage: python corindagpt/scripts/tts_wav_test.py [--no-play]
    do_play = True
    if len(sys.argv) > 1 and sys.argv[1] == "--no-play":
        do_play = False
    asyncio.run(main(play=do_play))
