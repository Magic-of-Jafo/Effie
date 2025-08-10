from __future__ import annotations

import asyncio
import io
import logging
import shutil
import subprocess
import tempfile
import wave
from typing import Any, Dict, Optional

import httpx

from utils.initialization import load_config

logger = logging.getLogger(__name__)


# ---------------------------
# Byte-format helpers
# ---------------------------

def _looks_like_wav(b: bytes) -> bool:
    # "RIFF....WAVE"
    return len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE"


def _pcm16le_to_wav_bytes(pcm_bytes: bytes, *, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw PCM16LE bytes in a minimal WAV container and return WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _mp3_to_wav_bytes(data: bytes) -> bytes:
    """
    Convert MP3 bytes to WAV bytes.
    Prefers pydub (with ffmpeg/ffprobe available). Falls back to ffmpeg CLI if needed.
    """
    # Try pydub first
    try:
        from pydub import AudioSegment  # type: ignore
        # AudioSegment.from_file requires ffmpeg in PATH or pydub configured
        mp3_buf = io.BytesIO(data)
        seg = AudioSegment.from_file(mp3_buf, format="mp3")
        out = io.BytesIO()
        seg.export(out, format="wav")
        wav_bytes = out.getvalue()
        if not _looks_like_wav(wav_bytes):
            raise RuntimeError("pydub produced non-WAV data.")
        return wav_bytes
    except Exception as e:
        logger.info("pydub path not available or failed (%s); trying ffmpeg CLI fallback.", e)

    # Fallback to ffmpeg CLI if available
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "MP3 received but cannot convert to WAV: pydub/ffmpeg not available. "
            "Install pydub (`pip install pydub`) and have ffmpeg in PATH."
        )

    with tempfile.TemporaryDirectory() as td:
        in_path = f"{td}/in.mp3"
        out_path = f"{td}/out.wav"
        with open(in_path, "wb") as f:
            f.write(data)

        # -y overwrite, -i input, default PCM s16le
        cmd = [ffmpeg, "-y", "-i", in_path, out_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to convert MP3 to WAV: {proc.stderr.decode(errors='ignore')}")
        with open(out_path, "rb") as f:
            wav_bytes = f.read()
        if not _looks_like_wav(wav_bytes):
            raise RuntimeError("ffmpeg produced non-WAV data.")
        return wav_bytes


# ---------------------------
# TTS
# ---------------------------

async def synthesize(
    text: str,
    *,
    http_client: Optional[httpx.AsyncClient] = None,
    config: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Synthesize speech and return WAV bytes.

    Reliable strategy across plan tiers:
      - Request MP3 from ElevenLabs (non-streaming).
      - Detect actual payload and transcode to WAV if needed.
      - Correctly wrap PCM if the API returns raw PCM.
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
    voice_id: str = ev_cfg.get("voice_id") or "EXAVITQu4vr4xnSDxMaL"  # demo voice
    model_id: str = ev_cfg.get("model_id") or "eleven_multilingual_v2"

    own_client = http_client is None
    client = http_client or httpx.AsyncClient(
        base_url="https://api.elevenlabs.io/v1",
        timeout=httpx.Timeout(60.0),
    )

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        # Ask for MP3 explicitly; we will normalize to WAV locally
        "Accept": "audio/mpeg, application/octet-stream, audio/wav, */*",
    }

    # Non-streaming request; streaming WAV is 403 on some accounts
    payload: Dict[str, Any] = {
        "text": text.strip(),
        "model_id": model_id,
        # Request a common MP3 profile for reliability
        "output_format": "mp3_44100_128",
        "voice_settings": ev_cfg.get("voice_settings") or {"stability": 0.3, "similarity_boost": 0.7},
    }

    try:
        resp = await client.post(f"/text-to-speech/{voice_id}/stream", headers=headers, json=payload, params={"output_format": "pcm_44100"})
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            body = None
            try:
                body = resp.text
            except Exception:
                body = "<unreadable>"
            logger.error("TTS provider error %s: %s", resp.status_code, body)
            raise
        audio_bytes = resp.content
        if not audio_bytes:
            raise RuntimeError("Empty audio payload returned from TTS provider")

        content_type = (resp.headers.get("Content-Type") or "").lower()

        # Case 1: Already WAV
        if _looks_like_wav(audio_bytes) or "audio/wav" in content_type:
            logger.info("TTS: synthesized %d bytes (WAV).", len(audio_bytes))
            return audio_bytes

        # Case 2: MP3
        if ("audio/mpeg" in content_type) or audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
            logger.info("TTS: received MP3; converting to WAV.")
            wav_bytes = _mp3_to_wav_bytes(audio_bytes)
            logger.info("TTS: MP3->WAV conversion complete (%d bytes).", len(wav_bytes))
            return wav_bytes

        # Case 3: Raw PCM16LE (some configs return this with octet-stream)
        # Heuristic: not RIFF/WAVE, length is even, and not empty
        if "octet-stream" in content_type:
            logger.info("TTS: octet-stream payload; assuming PCM16LE@16k mono and wrapping to WAV.")
            wav_bytes = _pcm16le_to_wav_bytes(audio_bytes, sample_rate=16000, channels=1)
            return wav_bytes

        # Unknown format: last resort try MP3 decode; if fails, raise
        logger.warning("TTS: Unknown content-type '%s'; attempting MP3 decode fallback.", content_type or "<none>")
        try:
            wav_bytes = _mp3_to_wav_bytes(audio_bytes)
            return wav_bytes
        except Exception as e:
            raise RuntimeError(
                f"Unknown audio payload; could not decode. Content-Type={content_type!r}, head={audio_bytes[:16]!r}"
            ) from e

    finally:
        if own_client:
            await client.aclose()


# ---------------------------
# Playback
# ---------------------------

async def play(audio_bytes: bytes) -> None:
    """
    Play WAV bytes via the system default audio device without blocking the event loop.

    Uses simpleaudio when available; falls back to winsound on Windows.
    Expects valid WAV bytes (synthesize() guarantees WAV).
    """
    if not audio_bytes:
        return

    if not _looks_like_wav(audio_bytes):
        raise ValueError("play() expected WAV bytes, but payload is not a WAV file.")

    # Prefer simpleaudio
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
        logger.info("simpleaudio unavailable or failed; falling back to winsound (%s).", exc)

    # Fallback: winsound on Windows
    try:
        import winsound  # type: ignore

        def _play_blocking_ws(data: bytes) -> None:
            # PlaySound with memory flag can be unreliable; use temp file for robustness
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
            finally:
                try:
                    import os
                    os.remove(tmp_path)
                except Exception:
                    pass

        await asyncio.to_thread(_play_blocking_ws, audio_bytes)
    except Exception as exc:
        logger.error("winsound playback failed: %s", exc)
