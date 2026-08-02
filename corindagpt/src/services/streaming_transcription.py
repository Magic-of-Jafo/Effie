"""Continuous streaming transcription (voice-control PRD story 1).

Keeps a microphone stream open and feeds it to OpenAI's realtime
transcription WebSocket (gpt-live-transcribe), maintaining a rolling,
timestamped transcript. The record gestures then slice this transcript by
press/release time instead of recording-and-uploading, which removes the
STT leg from release latency.

Toggled live from the dashboard (listening_mode: streaming | push_hold).
This module never replaces the push-hold recorder - it runs alongside it,
and the caller falls back to the recorder whenever the stream is not
connected.

Protocol notes (verified against the live API 2026-08-02):
- GA shape only: no OpenAI-Beta header; session.update with
  session.type "transcription"; audio/pcm at 24 kHz.
- gpt-live-transcribe rejects turn_detection (it is continuous); the
  server still emits item boundaries via its own segmentation.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover
    websockets = None  # type: ignore

try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

logger = logging.getLogger(__name__)

WS_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
SAMPLE_RATE = 24000
CHUNK_FRAMES = 2400  # 100 ms


class StreamingTranscriptionService:
    """Rolling-transcript engine with reconnect; safe to start/stop at runtime."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._apply_cfg()
        self._api_key: Optional[str] = config.get("openai_api_key")
        self.status: str = "off"  # off | connecting | live | reconnecting
        self.last_error: str = ""
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._stream = None
        self._audio_q: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # (arrival time.monotonic(), delta text); ~5 min of speech at speaking pace
        self._entries: Deque[Tuple[float, str]] = deque(maxlen=4000)
        self._last_delta_at: float = 0.0
        self._last_completed_at: float = 0.0
        # Mic diagnostics for the dashboard Live view
        self.level: float = 0.0  # decaying RMS, 0..1
        self.frames: int = 0
        self.device_name: str = ""
        self._last_loud_at: float = 0.0  # last moment mic RMS indicated speech

    def _apply_cfg(self) -> None:
        streaming_cfg = (self._config.get("transcription") or {}).get("streaming") or {}
        self._model = str(streaming_cfg.get("model") or "gpt-live-transcribe")
        # gpt-live-transcribe defers sentence-final punctuation until later
        # context (often the next utterance) and sometimes omits it. A gap in
        # delta arrivals is a speech pause; treat it as a sentence boundary
        # so decoding never depends on the model's punctuation.
        self._pause_split_s = float(streaming_cfg.get("pause_split_ms", 700)) / 1000.0
        # The 4o-transcribe family supports server VAD instead: the API closes
        # each utterance at a silence and returns it finalized AND punctuated.
        self._vad_silence_ms = int(streaming_cfg.get("vad_silence_ms", 500))
        # Window close: mic RMS above speech_rms marks "still speaking"; the
        # window is settled once transcription has caught up to the last
        # speech and the delta stream has been quiet for window_settle_ms.
        self._speech_rms = float(streaming_cfg.get("speech_rms", 0.004))
        self._settle_s = float(streaming_cfg.get("window_settle_ms", 400)) / 1000.0

    def _refresh_config(self) -> None:
        """Re-read config so dashboard edits apply on the next start()."""
        try:
            from ..utils.initialization import load_config

            self._config = load_config()
        except Exception as exc:
            logger.debug("Streaming transcription: config refresh failed (%s)", exc)
        self._apply_cfg()

    def _session_input_config(self) -> Dict[str, Any]:
        inp: Dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
            "transcription": {"model": self._model},
        }
        if self._model != "gpt-live-transcribe":
            inp["turn_detection"] = {
                "type": "server_vad",
                "silence_duration_ms": self._vad_silence_ms,
            }
        return inp

    # ------------------------------------------------------------ control

    def is_active(self) -> bool:
        return self._running and self.status == "live"

    async def start(self) -> None:
        if self._running:
            return
        self._refresh_config()
        if websockets is None or sd is None:
            logger.warning("Streaming transcription unavailable (missing websockets/sounddevice)")
            return
        if not self._api_key:
            logger.warning("Streaming transcription: no OpenAI API key")
            return
        self._loop = asyncio.get_running_loop()
        self._audio_q = asyncio.Queue(maxsize=100)
        self._running = True
        self.status = "connecting"
        try:
            self._stream = self._open_input_stream()
        except Exception as exc:
            logger.error("Streaming transcription: microphone open failed: %s", exc)
            self._running = False
            self.status = "off"
            self.last_error = f"mic: {exc}"
            return
        self._task = asyncio.create_task(self._run())
        logger.info("Streaming transcription: started (model=%s)", self._model)

    async def stop(self) -> None:
        self._running = False
        self.status = "off"
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        logger.info("Streaming transcription: stopped")

    def _resolve_device(self):
        """Optional transcription.streaming.input_device: index or name substring."""
        want = ((self._config.get("transcription") or {}).get("streaming") or {}).get("input_device")
        if want in (None, "", "default"):
            return None
        try:
            return int(want)
        except (TypeError, ValueError):
            pass
        want_l = str(want).lower()
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and want_l in dev["name"].lower():
                return i
        logger.warning("Streaming transcription: input_device %r not found; using default", want)
        return None

    def _open_input_stream(self):
        def callback(indata, frames, time_info, status) -> None:
            if status:
                logger.debug("Streaming mic status: %s", status)
            data = bytes(indata)
            self.frames += 1
            if np is not None and data:
                pcm = np.frombuffer(data, dtype=np.int16)
                rms = float(np.sqrt(np.mean((pcm / 32768.0) ** 2)))
                # Peak-hold with decay so brief words stay visible on the meter
                self.level = max(self.level * 0.7, rms)
                if rms > self._speech_rms:
                    self._last_loud_at = time.monotonic()
            if self._loop is not None and self._audio_q is not None:
                def _put() -> None:
                    try:
                        self._audio_q.put_nowait(data)
                    except asyncio.QueueFull:
                        pass  # drop oldest behavior: full queue means ws is behind

                self._loop.call_soon_threadsafe(_put)

        device = self._resolve_device()
        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_FRAMES,
            device=device,
            callback=callback,
        )
        stream.start()
        try:
            self.device_name = str(sd.query_devices(stream.device)["name"])
        except Exception:
            self.device_name = str(device if device is not None else "default")
        logger.info("Streaming transcription: capturing from '%s'", self.device_name)
        return stream

    # ------------------------------------------------------------- engine

    async def _run(self) -> None:
        backoff = 1.0
        while self._running:
            try:
                await self._session()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc)
                if not self._running:
                    break
                self.status = "reconnecting"
                logger.warning(
                    "Streaming transcription: connection lost (%s); retrying in %.0fs", exc, backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)

    async def _session(self) -> None:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with websockets.connect(WS_URL, additional_headers=headers, max_size=None) as ws:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {"input": self._session_input_config()},
                },
            }))
            self.status = "live"
            logger.info("Streaming transcription: session live")
            sender = asyncio.create_task(self._send_audio(ws))
            try:
                async for raw in ws:
                    self._handle_event(json.loads(raw))
            finally:
                sender.cancel()
            raise ConnectionError("websocket closed")

    async def _send_audio(self, ws) -> None:
        assert self._audio_q is not None
        while True:
            data = await self._audio_q.get()
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(data).decode(),
            }))

    def _handle_event(self, evt: Dict[str, Any]) -> None:
        etype = evt.get("type", "")
        now = time.monotonic()
        if etype == "conversation.item.input_audio_transcription.delta":
            delta = evt.get("delta") or ""
            if delta:
                self._entries.append((now, delta))
                self._last_delta_at = now
                logger.debug("Streaming delta: %r", delta)
        elif etype == "conversation.item.input_audio_transcription.completed":
            self._last_completed_at = now
            logger.info("Streaming segment: %s", (evt.get("transcript") or "").strip())
        elif etype == "error":
            self.last_error = json.dumps(evt.get("error") or {})[:300]
            logger.warning("Streaming transcription error event: %s", self.last_error)

    # -------------------------------------------------------------- reads

    def _assemble(self, entries: List[Tuple[float, str]]) -> str:
        """Join deltas, inserting sentence boundaries at speech pauses.

        - A delta-arrival gap > pause_split_s ends the sentence; a period is
          inserted if the model has not already punctuated.
        - The model often emits the previous sentence's period as the FIRST
          delta after the pause; that late punctuation is re-attached to the
          sentence it belongs to instead of orphaning it after the boundary.
        """
        parts: List[str] = []
        prev_t: Optional[float] = None
        for t, d in entries:
            if prev_t is not None and (t - prev_t) > self._pause_split_s:
                stripped = d.lstrip()
                while stripped and stripped[0] in ".!?,;:":
                    parts.append(stripped[0])
                    stripped = stripped[1:].lstrip()
                d = (" " + stripped) if stripped else ""
                joined_tail = "".join(parts).rstrip()
                if joined_tail and joined_tail[-1] not in ".!?":
                    parts.append(".")
            parts.append(d)
            prev_t = t
        text = "".join(parts).strip()
        if text and text[-1] not in ".!?":
            text += "."
        return text

    async def get_window(self, t_start: float, t_end: float, *, tail_s: float = 1.5) -> str:
        """Text whose deltas arrived between t_start and t_end (+ settle tail).

        Transcription lags speech, so the window closes on evidence, not a
        fixed wait: once deltas have arrived covering the last moment the mic
        heard speech AND the delta stream has been quiet for settle_s, the
        transcript is complete. If speech ended well before release (the
        natural finish-pause-release rhythm), that is true immediately and
        the handoff is near-instant; releasing mid-word waits only for the
        in-flight words. tail_s is the hard cap either way.
        """
        t0 = time.monotonic()
        # Last speech the mic heard before release. >= not >: same-instant
        # press-and-speak must count as speech (Windows clock ticks ~16 ms)
        target = self._last_loud_at
        if target >= t_start:
            deadline = t_end + tail_s
            while True:
                now = time.monotonic()
                if now >= deadline:
                    logger.info("Streaming window: settle cap hit (%d ms)", int((now - t0) * 1000))
                    break
                if self._last_delta_at >= target and (now - self._last_delta_at) >= self._settle_s:
                    logger.info("Streaming window: settled in %d ms", int((now - t0) * 1000))
                    break
                await asyncio.sleep(0.05)
        cutoff = time.monotonic()
        return self._assemble([(t, d) for t, d in list(self._entries) if t_start <= t <= cutoff])

    def recent(self, seconds: float = 45.0) -> List[Dict[str, Any]]:
        """Recent deltas for the dashboard Live view (thread-safe snapshot)."""
        now = time.monotonic()
        out = []
        for t, d in list(self._entries):
            if now - t <= seconds:
                out.append({"ago": round(now - t, 1), "text": d})
        return out

    def recent_text(self, seconds: float = 45.0) -> str:
        """Recent speech, pause-segmented - what the decoder would see."""
        now = time.monotonic()
        return self._assemble([(t, d) for t, d in list(self._entries) if now - t <= seconds])
