from typing import Any, Dict

import httpx
import pytest

from services import tts


def test_synthesize_success() -> None:
    # Return 160 bytes of PCM16LE (dummy)
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.endswith("/text-to-speech/EXAVITQu4vr4xnSDxMaL/stream")
        q = request.url.query
        if isinstance(q, (bytes, bytearray)):
            q = q.decode()
        assert "output_format=pcm_44100" in (q or "")
        return httpx.Response(200, content=b"RIFF" + (b"\x00\x00" * 80), headers={"Content-Type": "audio/wav"})

    transport = httpx.MockTransport(handler)

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
        async with httpx.AsyncClient(transport=transport, base_url="https://api.elevenlabs.io/v1") as client:
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
