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

logger = logging.getLogger(__name__)

WS_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
SAMPLE_RATE = 24000
CHUNK_FRAMES = 2400  # 100 ms


class StreamingTranscriptionService:
    """Rolling-transcript engine with reconnect; safe to start/stop at runtime."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._model = str(
            ((config.get("transcription") or {}).get("streaming") or {}).get("model")
            or "gpt-live-transcribe"
        )
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

    # ------------------------------------------------------------ control

    def is_active(self) -> bool:
        return self._running and self.status == "live"

    async def start(self) -> None:
        if self._running:
            return
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

    def _open_input_stream(self):
        def callback(indata, frames, time_info, status) -> None:
            if status:
                logger.debug("Streaming mic status: %s", status)
            data = bytes(indata)
            if self._loop is not None and self._audio_q is not None:
                def _put() -> None:
                    try:
                        self._audio_q.put_nowait(data)
                    except asyncio.QueueFull:
                        pass  # drop oldest behavior: full queue means ws is behind

                self._loop.call_soon_threadsafe(_put)

        return sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_FRAMES,
            callback=callback,
        )

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
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                            "transcription": {"model": self._model},
                        },
                    },
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

    async def get_window(self, t_start: float, t_end: float, *, tail_s: float = 1.2) -> str:
        """Text whose deltas arrived between t_start and t_end (+ settle tail).

        Deltas lag speech by ~0.4-0.8 s, so after release we wait for the
        trailing words: up to tail_s, ending early once deltas go quiet.
        """
        deadline = t_end + tail_s
        while time.monotonic() < deadline:
            quiet_for = time.monotonic() - self._last_delta_at
            if self._last_delta_at > t_end and quiet_for > 0.35:
                break
            await asyncio.sleep(0.05)
        cutoff = time.monotonic()
        text = "".join(d for t, d in list(self._entries) if t_start <= t <= cutoff).strip()
        return text

    def recent(self, seconds: float = 45.0) -> List[Dict[str, Any]]:
        """Recent deltas for the dashboard Live view (thread-safe snapshot)."""
        now = time.monotonic()
        out = []
        for t, d in list(self._entries):
            if now - t <= seconds:
                out.append({"ago": round(now - t, 1), "text": d})
        return out
