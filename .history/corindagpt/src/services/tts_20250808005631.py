from __future__ import annotations

import asyncio
import io
import logging
import wave
from typing import Any, Dict, Optional

import httpx

from utils.initialization import load_config

logger = logging.getLogger(__name__)


def _pcm16le_to_wav_bytes(pcm_bytes: bytes, *, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw PCM16LE bytes in a minimal WAV container and return full WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


async def synthesize(text: str, *, http_client: Optional[httpx.AsyncClient] = None, config: Optional[Dict[str, Any]] = None) -> bytes:
    """Synthesize speech for provided text and return WAV bytes.

    Uses ElevenLabs non-streaming API with output_format=pcm_16000 and wraps to WAV.
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
    voice_id: str = ev_cfg.get("voice_id") or "EXAVITQu4vr4xnSDxMaL"  # default demo voice
    model_id: str = ev_cfg.get("model_id") or "eleven_monolingual_v1"

    own_client = http_client is None
    client = http_client or httpx.AsyncClient(base_url="https://api.elevenlabs.io/v1", timeout=60.0)
    try:
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/wav",
        }
        payload: Dict[str, Any] = {
            "text": text.strip(),
            "model_id": model_id,
            "voice_settings": ev_cfg.get("voice_settings") or {"stability": 0.3, "similarity_boost": 0.7},
            # Use streaming endpoint recommended; let output_format be query not body
        }
        # Prefer streaming endpoint and request WAV format
        url = f"/text-to-speech/{voice_id}/stream?output_format=wav"
        resp = await client.post(url, headers=headers, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:  # surface provider error body for debugging
            body = None
            try:
                body = resp.text
            except Exception:
                body = "<unreadable>"
            logger.error("TTS provider error %s: %s", resp.status_code, body)
            raise
        audio_bytes = resp.content  # should already be WAV bytes
        if not audio_bytes:
            raise RuntimeError("Empty audio payload returned from TTS provider")
        # Validate WAV header minimally
        if not audio_bytes.startswith(b"RIFF"):
            # fallback: wrap assuming PCM16LE
            audio_bytes = _pcm16le_to_wav_bytes(audio_bytes, sample_rate=16000, channels=1)
        logger.info("TTS: synthesized %d bytes (streaming)", len(audio_bytes))
        return audio_bytes
    finally:
        if own_client:
            await client.aclose()


async def play(audio_bytes: bytes) -> None:
    """Play WAV bytes via the system default audio device without blocking the event loop.

    Falls back to Windows winsound if simpleaudio is unavailable.
    """
    if not audio_bytes:
        return

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
    except Exception as exc:  # pragma: no cover
        logger.error("simpleaudio not available for playback: %s", exc)

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
            except Exception as ws_exc:
                logger.error("winsound playback failed: %s", ws_exc)
            finally:
                if tmp_path:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        await asyncio.to_thread(_play_blocking_ws, audio_bytes)
    except Exception as exc:  # pragma: no cover
        logger.error("winsound module not available for playback: %s", exc)
