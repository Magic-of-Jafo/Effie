from __future__ import annotations

import asyncio
import io
import logging
import wave
from typing import Any, Dict, Optional

import httpx

from utils.initialization import load_config

logger = logging.getLogger(__name__)


def _looks_like_wav(b: bytes) -> bool:
    # Minimal check: "RIFF"...."WAVE"
    return len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE"


async def synthesize(
    text: str,
    *,
    http_client: Optional[httpx.AsyncClient] = None,
    config: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Synthesize speech and return WAV bytes.

    Strategy:
      1) Non-streaming endpoint requesting WAV (most reliable).
      2) If that fails, try streaming endpoint requesting WAV as well.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    cfg = config or load_config()
    api_key: Optional[str] = cfg.get("elevenlabs_api_key")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not configured (env or config.yaml)")

    tts_cfg: Dict[str, Any] = cfg.get("tts", {})
    provider = (tts_cfg.get("provider") or "elevenlabs").lower()
    if provider != "elevenlabs":
        raise NotImplementedError(f"TTS provider '{provider}' not supported yet")

    ev_cfg: Dict[str, Any] = tts_cfg.get("elevenlabs", {})
    # Use a commonly available default voice/model unless provided
    voice_id: str = ev_cfg.get("voice_id") or "EXAVITQu4vr4xnSDxMaL"  # demo voice
    model_id: str = ev_cfg.get("model_id") or "eleven_multilingual_v2"  # current default

    own_client = http_client is None
    client = http_client or httpx.AsyncClient(
        base_url="https://api.elevenlabs.io/v1",
        timeout=httpx.Timeout(60.0),
    )

    headers_common = {
        "xi-api-key": api_key,
    }

    # ---------- Primary: Non-streaming, ask for WAV ----------
    payload_wav: Dict[str, Any] = {
        "text": text.strip(),
        "model_id": model_id,
        # Request explicit WAV to avoid raw PCM/MP3 surprises
        # ElevenLabs supports: "wav", "mp3_44100_128", "ulaw_8000", "pcm_16000", etc.
        "output_format": "wav",
        "voice_settings": ev_cfg.get("voice_settings")
        or {"stability": 0.3, "similarity_boost": 0.7},
    }

    try:
        resp = await client.post(
            f"/text-to-speech/{voice_id}",
            headers={**headers_common, "Accept": "audio/wav", "Content-Type": "application/json"},
            json=payload_wav,
        )
        resp.raise_for_status()
        data = resp.content
        if not data:
            raise RuntimeError("Empty audio payload returned from TTS provider (WAV).")
        if not _looks_like_wav(data):
            # Some CDNs may drop/alter headers; validate bytes too
            ct = resp.headers.get("Content-Type", "")
            raise RuntimeError(
                f"Unexpected non-WAV response for WAV request (Content-Type={ct!r}, first 12 bytes={data[:12]!r})."
            )
        logger.info("TTS: synthesized %d bytes (non-streaming WAV).", len(data))
        return data
    except Exception as e:
        logger.warning("Primary WAV request failed (%s). Trying streaming WAV fallback.", e)

    # ---------- Fallback: Streaming endpoint, request WAV ----------
    try:
        # For streaming, explicitly ask for WAV as well.
        # Note: Some accounts may require specific model/sample_rate combos; WAV avoids ambiguity.
        resp_stream = await client.post(
            f"/text-to-speech/{voice_id}/stream?output_format=wav",
            headers={**headers_common, "Accept": "audio/wav", "Content-Type": "application/json"},
            json={
                "text": text.strip(),
                "model_id": model_id,
                "voice_settings": ev_cfg.get("voice_settings")
                or {"stability": 0.3, "similarity_boost": 0.7},
            },
        )
        resp_stream.raise_for_status()
        data = resp_stream.content
        if not data:
            raise RuntimeError("Empty audio payload returned from TTS provider (streaming WAV).")
        if not _looks_like_wav(data):
            ct = resp_stream.headers.get("Content-Type", "")
            raise RuntimeError(
                f"Unexpected non-WAV response for streaming WAV request (Content-Type={ct!r}, head={data[:12]!r})."
            )
        logger.info("TTS: synthesized %d bytes (streaming WAV).", len(data))
        return data
    finally:
        if own_client:
            await client.aclose()


async def play(audio_bytes: bytes) -> None:
    """
    Play WAV bytes via the system default audio device without blocking the event loop.

    Uses simpleaudio when available; falls back to winsound on Windows.
    Expects valid WAV bytes (we synthesize WAV above).
    """
    if not audio_bytes:
        return

    # Sanity: ensure we have WAV, not raw PCM/compressed content
    if not _looks_like_wav(audio_bytes):
        raise ValueError("play() expected WAV bytes, but payload is not a WAV file.")

    # Attempt simpleaudio first
    try:
        import simpleaudio  # type: ignore

        def _play_blocking_sa(data: bytes) -> None:
            with wave.open(io.BytesIO(data), "rb") as wf:
                wave_obj = simpleaudio.WaveObject.from_wave_read(wf)
                play_obj = wave_obj.play()
                play_obj.wait_done()

        await asyncio.to_thread(_play_blocking_sa, audio_bytes)
        return
    except Exception as exc:
        logger.warning("simpleaudio unavailable or failed; falling back to winsound (%s).", exc)

    # Fallback: winsound on Windows
    try:
        import winsound  # type: ignore
        import tempfile
        import os

        def _play_blocking_ws(data: bytes) -> None:
            tmp_path: Optional[str] = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                with open(tmp_path, "wb") as f:
                    f.write(data)
                winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
            finally:
                if tmp_path:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        await asyncio.to_thread(_play_blocking_ws, audio_bytes)
    except Exception as exc:
        logger.error("winsound playback failed: %s", exc)
