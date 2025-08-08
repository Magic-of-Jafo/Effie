from typing import Any, Dict

import httpx
import pytest

from services import tts


def test_synthesize_success(httpx_mock) -> None:
    # Return 160 bytes of PCM16LE (dummy)
    httpx_mock.add_response(
        method="POST",
        url="https://api.elevenlabs.io/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL",
        content=b"\x00\x00" * 80,
        status_code=200,
        headers={"Content-Type": "application/octet-stream"},
    )

    cfg: Dict[str, Any] = {
        "elevenlabs_api_key": "fake-key",
        "tts": {
            "provider": "elevenlabs",
            "elevenlabs": {
                "voice_id": "EXAVITQu4vr4xnSDxMaL",
                "model_id": "eleven_monolingual_v1",
            },
        },
    }

    async def runner() -> None:
        async with httpx.AsyncClient() as client:
            wav_bytes = await tts.synthesize("Hello world", http_client=client, config=cfg)
        assert isinstance(wav_bytes, (bytes, bytearray))
        assert len(wav_bytes) > 44  # WAV header + data

    import asyncio

    asyncio.run(runner())


def test_play_handles_missing_simpleaudio(monkeypatch) -> None:
    # Force import failure only for 'simpleaudio'
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "simpleaudio":
            raise ImportError("no simpleaudio")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    # Should not raise
    import asyncio

    asyncio.run(tts.play(b"RIFF....fake"))
