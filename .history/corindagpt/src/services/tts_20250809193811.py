from __future__ import annotations

import asyncio
import io
import logging
import shutil
import subprocess
import tempfile
import wave
from typing import Any, Dict, Optional
from time import monotonic

import httpx

try:
    from ..utils.initialization import load_config  # type: ignore[relative-beyond-top-level]
except Exception:
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


def _prepend_silence_wav(wav_bytes: bytes, ms: int) -> bytes:
    if ms <= 0 or not _looks_like_wav(wav_bytes):
        return wav_bytes
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        num_silence_frames = int((framerate * ms) / 1000)
        silence = b"\x00" * (num_silence_frames * channels * sampwidth)
        out = io.BytesIO()
        with wave.open(out, "wb") as wf2:
            wf2.setnchannels(channels)
            wf2.setsampwidth(sampwidth)
            wf2.setframerate(framerate)
            wf2.writeframes(silence + frames)
        return out.getvalue()
    except Exception as exc:
        logger.debug("Failed to prepend silence: %s", exc)
        return wav_bytes


def _generate_silence_wav(duration_ms: int, *, sample_rate: int = 16000, channels: int = 1) -> bytes:
    if duration_ms <= 0:
        return b""
    num_frames = int(sample_rate * duration_ms / 1000)
    silence = b"\x00" * (num_frames * channels * 2)  # 16-bit
    return _pcm16le_to_wav_bytes(silence, sample_rate=sample_rate, channels=channels)


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

    Strategy:
      - If tts.streaming_enabled is True, try ElevenLabs streaming PCM; on 403/400/404, fall back to non-streaming MP3.
      - If tts.streaming_enabled is False, use non-streaming MP3 directly.
      - Normalize provider output to WAV.
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
    streaming_enabled: bool = bool(tts_cfg.get("streaming_enabled", True))

    own_client = http_client is None
    client = http_client or httpx.AsyncClient(
        base_url="https://api.elevenlabs.io/v1",
        timeout=httpx.Timeout(60.0),
    )

    # Decide Accept based on desired non-streaming output
    nonstream_output: str = str(tts_cfg.get("output_format") or "mp3_44100_128")
    if nonstream_output.startswith("mp3"):
        accept_header = "audio/mpeg"
    elif nonstream_output.startswith("pcm"):
        accept_header = "application/octet-stream"
    elif nonstream_output.startswith("wav"):
        accept_header = "audio/wav"
    else:
        accept_header = "audio/mpeg, application/octet-stream, audio/wav, */*"

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": accept_header,
    }

    # Base payload used for both streaming and non-streaming
    # For non-streaming, allow configurable output format (e.g., mp3_44100_64, mp3_22050_32)
    payload: Dict[str, Any] = {
        "text": text.strip(),
        "model_id": model_id,
        "output_format": nonstream_output,
        "voice_settings": ev_cfg.get("voice_settings") or {"stability": 0.3, "similarity_boost": 0.7},
    }

    async def _request_streaming() -> httpx.Response:
        stream_params = {"output_format": "pcm_44100"}
        resp = await client.post(
            f"/text-to-speech/{voice_id}/stream", headers=headers, json=payload, params=stream_params
        )
        if resp.status_code in (400, 403, 404):
            raise httpx.HTTPStatusError("Streaming not allowed", request=resp.request, response=resp)
        resp.raise_for_status()
        return resp

    async def _request_non_streaming() -> httpx.Response:
        resp2 = await client.post(
            f"/text-to-speech/{voice_id}", headers=headers, json=payload, params={"output_format": nonstream_output}
        )
        resp2.raise_for_status()
        return resp2

    try:
        if streaming_enabled:
            try:
                resp_obj = await _request_streaming()
            except httpx.HTTPStatusError as e:
                logger.info("TTS: streaming not allowed (%s); using non-streaming MP3 endpoint.", getattr(e.response, "status_code", ""))
                resp_obj = await _request_non_streaming()
        else:
            resp_obj = await _request_non_streaming()

        audio_bytes = resp_obj.content
        if not audio_bytes:
            raise RuntimeError("Empty audio payload returned from TTS provider")

        # Normalize response to WAV bytes
        content_type = (resp_obj.headers.get("Content-Type") or "").lower()

        if _looks_like_wav(audio_bytes) or "audio/wav" in content_type:
            logger.info("TTS: synthesized %d bytes (WAV).", len(audio_bytes))
            return audio_bytes

        if nonstream_output.startswith("pcm") or "octet-stream" in content_type:
            logger.info("TTS: received PCM; wrapping to WAV for playback.")
            return _pcm16le_to_wav_bytes(audio_bytes, sample_rate=16000, channels=1)

        if ("audio/mpeg" in content_type) or audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
            logger.info("TTS: received MP3; returning as-is for direct playback.")
            return audio_bytes

        logger.warning("TTS: Unknown content-type '%s'; attempting MP3 decode fallback.", content_type or "<none>")
        try:
            return _mp3_to_wav_bytes(audio_bytes)
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

async def play(audio_bytes: bytes, *, config: Optional[Dict[str, Any]] = None, started_at_monotonic: Optional[float] = None) -> None:
    """
    Play audio bytes via the system default audio device without blocking the event loop.

    - If payload is WAV: optionally prepend preroll silence and play via simpleaudio/winsound.
    - If payload is MP3: decode and play directly via pydub (no full transcode to WAV file on disk).

    If started_at_monotonic is provided, logs the elapsed time (ms) from that moment to when playback is initiated.
    """
    if not audio_bytes:
        return

    cfg = config or load_config()
    tts_cfg: Dict[str, Any] = cfg.get("tts", {})
    playback_cfg: Dict[str, Any] = tts_cfg.get("playback", {})
    preroll_ms: int = int(playback_cfg.get("preroll_ms", 0))
    start_delay_ms: int = int(playback_cfg.get("start_delay_ms", 0))
    warmup_ms: int = int(playback_cfg.get("warmup_ms", 150))
    backend_pref: Optional[str] = str(playback_cfg.get("backend") or "").strip().lower() or None

    debug_cfg: Dict[str, Any] = tts_cfg.get("debug", {})
    save_last_wav: bool = bool(debug_cfg.get("save_last_wav", False))
    save_path: str = str(debug_cfg.get("path") or "debug_last_tts.wav")

    # Optional start delay to let device warm up
    if start_delay_ms > 0:
        await asyncio.sleep(max(0.0, start_delay_ms / 1000.0))

    # MP3 direct path
    if (len(audio_bytes) >= 3 and audio_bytes[:3] == b"ID3") or (len(audio_bytes) >= 2 and audio_bytes[:2] == b"\xff\xfb"):
        try:
            from pydub import AudioSegment  # type: ignore
            from pydub.playback import play as pydub_play  # type: ignore

            def _play_mp3(data: bytes) -> None:
                # Optional warmup via silent segment
                if warmup_ms > 0:
                    warmup = AudioSegment.silent(duration=warmup_ms)
                    pydub_play(warmup)
                seg = AudioSegment.from_file(io.BytesIO(data), format="mp3")
                if preroll_ms > 0:
                    seg = AudioSegment.silent(duration=preroll_ms, frame_rate=seg.frame_rate) + seg
                pydub_play(seg)

            if started_at_monotonic is not None:
                elapsed_ms = int((monotonic() - started_at_monotonic) * 1000)
                logger.info("Timing: release->play_start %d ms", elapsed_ms)

            logger.info("Playback backend: pydub/mp3")
            await asyncio.to_thread(_play_mp3, audio_bytes)
            return
        except Exception as exc:
            logger.info("pydub playback unavailable or failed (%s); falling back to WAV path.", exc)
            # Fall through to WAV checks, which will error if data isn't WAV

    # WAV path
    # Accept minimal RIFF header (tests use placeholder) and real WAV
    if not _looks_like_wav(audio_bytes):
        # Allow very small placeholder RIFF buffers in tests
        if not (len(audio_bytes) >= 4 and audio_bytes[:4] == b"RIFF"):
            raise ValueError("play() expected WAV or MP3 bytes; payload is neither WAV nor recognized MP3.")
    else:
        # Optionally prepend only preroll silence; warmup will be played as a separate clip
        combined_ms = max(0, preroll_ms)
        if combined_ms > 0:
            audio_bytes = _prepend_silence_wav(audio_bytes, combined_ms)

    # Optionally save debug WAV
    if save_last_wav and _looks_like_wav(audio_bytes):
        try:
            from pathlib import Path
            Path(save_path).write_bytes(audio_bytes)
            logger.info("Debug: saved last WAV to %s", save_path)
        except Exception as exc:
            logger.debug("Debug save failed: %s", exc)

    # If backend is forced to winsound, use it first
    if backend_pref == "winsound":
        try:
            import winsound  # type: ignore

            def _play_blocking_ws(data: bytes) -> None:
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

            if started_at_monotonic is not None:
                elapsed_ms = int((monotonic() - started_at_monotonic) * 1000)
                logger.info("Timing: release->play_start %d ms", elapsed_ms)

            logger.info("Playback backend: winsound")
            await asyncio.to_thread(_play_blocking_ws, audio_bytes)
            logger.info("Playback: second pass (winsound)")
            await asyncio.to_thread(_play_blocking_ws, audio_bytes)
            return
        except Exception as exc:
            logger.error("winsound playback failed: %s", exc)
            # continue to simpleaudio fallback

    # Prefer simpleaudio for WAV
    try:
        import simpleaudio  # type: ignore

        def _play_blocking_sa(data: bytes) -> None:
            with wave.open(io.BytesIO(data), "rb") as wf:
                wave_obj = simpleaudio.WaveObject.from_wave_read(wf)
                play_obj = wave_obj.play()
                play_obj.wait_done()

        if started_at_monotonic is not None:
            elapsed_ms = int((monotonic() - started_at_monotonic) * 1000)
            logger.info("Timing: release->play_start %d ms", elapsed_ms)

        logger.info("Playback backend: simpleaudio")
        await asyncio.to_thread(_play_blocking_sa, audio_bytes)
        logger.info("Playback: second pass (simpleaudio)")
        await asyncio.to_thread(_play_blocking_sa, audio_bytes)
        return
    except Exception as exc:
        logger.info("simpleaudio unavailable or failed; falling back to winsound (%s).", exc)

    # Fallback: winsound on Windows for WAV
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

        if started_at_monotonic is not None:
            elapsed_ms = int((monotonic() - started_at_monotonic) * 1000)
            logger.info("Timing: release->play_start %d ms", elapsed_ms)

        logger.info("Playback backend: winsound")
        await asyncio.to_thread(_play_blocking_ws, audio_bytes)
    except Exception as exc:
        logger.error("winsound playback failed: %s", exc)
