from __future__ import annotations

import os
import sys
from pathlib import Path
import json

import httpx

# Ensure src/ is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.initialization import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    env_key = os.getenv("ELEVENLABS_API_KEY")
    cfg_key = cfg.get("elevenlabs_api_key")
    print(f"ENV ELEVENLABS_API_KEY present: {bool(env_key)}")
    print(f"CFG ELEVENLABS_API_KEY present: {bool(cfg_key)}")
    key = env_key or cfg_key
    if not key:
        print("No ElevenLabs API key found in env or config.")
        return

    voice_id = (cfg.get("tts") or {}).get("elevenlabs", {}).get("voice_id")
    print(f"Configured voice_id: {voice_id!r}")

    headers = {"xi-api-key": key}
    try:
        r = httpx.get("https://api.elevenlabs.io/v1/voices", headers=headers, timeout=20.0)
        r.raise_for_status()
        data = r.json()
        voices = data.get("voices", [])
        print(f"Voices visible to account: {len(voices)}")
        found = any(v.get("voice_id") == voice_id for v in voices)
        print(f"Configured voice_id accessible: {found}")
        if not found:
            sample = [v.get("voice_id") for v in voices[:5]]
            print("First few voice_ids:", sample)
    except Exception as exc:
        print("Failed to list voices:", repr(exc))


if __name__ == "__main__":
    main()
