from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional, Tuple, Dict, Any, List, Callable

import speech_recognition as sr
import httpx
import io
import wave
from time import monotonic

from utils.initialization import load_config

from io import BytesIO




# Optional, for robust press/hold capture
try:  # pragma: no cover - optional dependency
    import sounddevice as sd  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    sd = None  # type: ignore
    np = None  # type: ignore

# Optional ElevenLabs SDK
try:  # pragma: no cover - optional dependency
    from elevenlabs.client import ElevenLabs  # type: ignore
except Exception:  # pragma: no cover
    ElevenLabs = None  # type: ignore

logger = logging.getLogger(__name__)

# Type alias for benchmark results
BenchmarkResult = Dict[str, Any]


class VoiceRecorder:
    """Basic press-and-hold audio capture built on SpeechRecognition.

    Captures mono audio from the system default microphone between start/stop calls
    and returns an in-memory buffer (WAV bytes) suitable for low-latency transfer.
    """

    def __init__(self) -> None:
        self._recognizer = sr.Recognizer()
        # Force consistent capture rate/chunk for predictable WAV duration
        self._microphone = sr.Microphone(sample_rate=16000, chunk_size=1024)
        self._stop_event = threading.Event()
        self._record_task: Optional[asyncio.Task[Tuple[bytes, int, int]]] = None
        self._lock = threading.Lock()
        self._calibrated = False
        self._t_start: Optional[float] = None
        self._min_hold_sec: float = 0.35
        self._flush_after_stop_chunks: int = 3

    async def calibrate_ambient_noise(self, duration: float = 0.2) -> None:
        if self._calibrated:
            return
        await asyncio.to_thread(self._calibrate_blocking, duration)
        self._calibrated = True
        logger.debug("VoiceRecorder: ambient noise calibrated (%.2fs)", duration)

    def _calibrate_blocking(self, duration: float) -> None:
        try:
            with self._microphone as source:
                try:
                    self._recognizer.adjust_for_ambient_noise(source, duration=duration)
                except Exception:
                    pass
        except Exception as exc:
            logger.exception("VoiceRecorder: error during calibration: %s", exc)

    async def start_recording(self) -> None:
        if self._record_task is not None and not self._record_task.done():
            return
        self._stop_event.clear()
        # Run the blocking recording function in a worker thread
        self._record_task = asyncio.create_task(asyncio.to_thread(self._record_blocking))
        self._t_start = monotonic()
        logger.info("VoiceRecorder: recording started")

    async def stop_recording(self) -> bytes:
        if self._record_task is None:
            logger.debug("VoiceRecorder: stop called with no active recording")
            return b""
        # Enforce minimum hold duration to avoid sub-100ms clips
        if self._t_start is not None:
            held = monotonic() - self._t_start
            if held < self._min_hold_sec:
                await asyncio.sleep(self._min_hold_sec - held)
        self._stop_event.set()
        frames, sample_rate, sample_width = await self._record_task
        self._record_task = None
        self._t_start = None
        # Convert frames to WAV bytes via AudioData for consistency (force 16kHz/16-bit)
        audio = sr.AudioData(frames, sample_rate, sample_width)
        wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
        # Optional: log computed duration for diagnostics
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                duration = wf.getnframes() / float(wf.getframerate() or 1)
        except Exception:
            duration = -1.0
        logger.info(
            "VoiceRecorder: recording stopped (bytes=%d, duration=%.3fs)",
            len(wav_bytes),
            duration,
        )
        return wav_bytes

    # Internal blocking routine executed in a worker thread
    def _record_blocking(self) -> Tuple[bytes, int, int]:
        frames = bytearray()
        # Use fixed output parameters for consistency
        sample_rate = 16000
        sample_width = 2
        try:
            with self._microphone as source:
                # Read until stop; include one trailing chunk to flush buffered audio
                while True:
                    try:
                        data = source.stream.read(source.CHUNK, exception_on_overflow=False)
                    except Exception:
                        # Attempt to continue on intermittent read errors/overflows
                        continue
                    with self._lock:
                        frames.extend(data)
                    if self._stop_event.is_set():
                        # Read a few trailing chunks to capture buffered audio
                        for _ in range(self._flush_after_stop_chunks):
                            try:
                                data2 = source.stream.read(source.CHUNK, exception_on_overflow=False)
                                with self._lock:
                                    frames.extend(data2)
                            except Exception:
                                break
                        break
        except Exception as exc:
            logger.exception("VoiceRecorder: error during capture: %s", exc)
        return bytes(frames), sample_rate, sample_width


class SDPressHoldRecorder:
    """SoundDevice-based press/hold recorder producing WAV bytes in memory.

    Preferred on Windows for robust start/stop and latency characteristics.
    """

    def __init__(self, samplerate: int = 16000, channels: int = 1, input_gain: float = 1.25) -> None:
        if sd is None or np is None:
            raise RuntimeError("sounddevice/numpy not available")
        self.samplerate = samplerate
        self.channels = channels
        self._q: "queue.Queue[np.ndarray]" = self._make_queue()
        self._stream: Optional["sd.InputStream"] = None
        self._lock = threading.Lock()
        self._recording = False
        self._t_start: Optional[float] = None
        self._min_hold_sec: float = 0.35
        self._flush_after_stop_sec: float = 0.15
        self._dtype = "float32"
        self._input_gain = float(max(0.5, min(input_gain, 3.0)))

    @staticmethod
    def is_available() -> bool:
        return sd is not None and np is not None

    @staticmethod
    def _make_queue():  # separate to avoid importing typing queue generics
        import queue

        return queue.Queue()

    def _callback(self, indata, frames, time, status):  # type: ignore[no-untyped-def]
        if status:
            pass
        # Apply simple gain to improve SNR before quantization
        data = (indata.astype(self._dtype) * self._input_gain).clip(-1.0, 1.0)
        # Copy as PortAudio reuses buffers
        self._q.put(data.copy())

    def _start_blocking(self) -> None:
        with self._lock:
            if self._recording:
                return
            # clear queue
            try:
                while True:
                    self._q.get_nowait()
            except Exception:
                pass
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype=self._dtype,
                blocksize=0,
                callback=self._callback,
            )
            self._stream.start()
            self._recording = True
            self._t_start = monotonic()
        logger.info("SDPressHoldRecorder: recording started")

    async def start_recording(self) -> None:
        await asyncio.to_thread(self._start_blocking)

    def _stop_blocking(self) -> bytes:
        # Enforce minimum hold
        if self._t_start is not None:
            held = monotonic() - self._t_start
            if held < self._min_hold_sec:
                import time as _t

                _t.sleep(self._min_hold_sec - held)
        # Allow callback to flush late frames
        import time as _t

        _t.sleep(self._flush_after_stop_sec)

        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                finally:
                    try:
                        self._stream.close()
                    finally:
                        self._stream = None
            self._recording = False

        # Drain queue
        chunks = []
        try:
            while True:
                chunks.append(self._q.get_nowait())
        except Exception:
            pass

        if not chunks:
            logger.info("SDPressHoldRecorder: no audio captured")
            return b""

        audio = np.concatenate(chunks, axis=0)
        # Convert to int16 PCM
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16).tobytes()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes(pcm16)
        wav_bytes = buf.getvalue()

        # Duration log
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                duration = wf.getnframes() / float(wf.getframerate() or 1)
        except Exception:
            duration = -1.0
        logger.info(
            "SDPressHoldRecorder: recording stopped (bytes=%d, duration=%.3fs)",
            len(wav_bytes),
            duration,
        )
        return wav_bytes

    async def stop_recording(self) -> bytes:
        return await asyncio.to_thread(self._stop_blocking)


class TranscriptionService:
    """Asynchronous transcription service with provider configurability.

    Currently supports non-streaming transcription via OpenAI's Audio Transcriptions API.
    Streaming mode is accepted as a parameter but will fall back to non-streaming for OpenAI
    until a streaming provider/implementation is available.
    """

    def __init__(
        self,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._client = http_client
        self._config = config or load_config()
        transcription_cfg: Dict[str, Any] = self._config.get("transcription", {})
        self._provider: str = transcription_cfg.get("provider", "openai").lower()
        # Prefer explicit transcription.model; fallback to legacy model_names.transcription
        model_from_cfg = transcription_cfg.get("model")
        legacy_model = (
            (self._config.get("model_names") or {}).get("transcription")
            if isinstance(self._config.get("model_names"), dict)
            else None
        )
        self._model: str = model_from_cfg or legacy_model or "whisper-1"
        self._models_list: List[str] = list(transcription_cfg.get("models", [])) or [self._model]
        self._streaming_enabled_default: bool = bool(transcription_cfg.get("streaming_enabled", False))
        self._language: Optional[str] = transcription_cfg.get("language")
        self._prompt: Optional[str] = transcription_cfg.get("prompt")
        self._eleven_cfg: Dict[str, Any] = transcription_cfg.get("elevenlabs", {})
        self._eleven_client: Optional[ElevenLabs] = None
        if ElevenLabs is not None and self._config.get("elevenlabs_api_key"):
            try:
                self._eleven_client = ElevenLabs(api_key=self._config.get("elevenlabs_api_key"))
            except Exception:
                self._eleven_client = None

    def is_streaming_enabled(self) -> bool:
        return self._streaming_enabled_default

    async def transcribe(self, buffer: bytes, *, streaming: Optional[bool] = None) -> str:
        if not buffer:
            logger.info("TranscriptionService: empty audio buffer, returning empty transcript")
            return ""

        # Skip provider call if WAV duration is below provider minimum (0.1s)
        try:
            with wave.open(io.BytesIO(buffer), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate() or 1
                duration_sec = frames / float(rate)
        except Exception:
            duration_sec = None
        if duration_sec is not None and duration_sec < 0.08:
            logger.info("TranscriptionService: audio too short (%.3fs); skipping transcription", duration_sec)
            return ""

        use_streaming = self._streaming_enabled_default if streaming is None else bool(streaming)

        if self._provider == "openai":
            if use_streaming:
                logger.warning(
                    "TranscriptionService: streaming requested for provider 'openai' but not implemented; falling back to non-streaming"
                )
            # Try models in order
            last_err: Optional[Exception] = None
            for model in self._models_list:
                try:
                    return await self._transcribe_openai_non_streaming(buffer, model=model)
                except Exception as exc:  # capture and try next
                    last_err = exc
                    logger.warning("OpenAI transcription failed on model %s: %s", model, exc)
                    continue
            assert last_err is not None
            raise last_err

        if self._provider == "elevenlabs":
            return await self._transcribe_elevenlabs(buffer)

        raise ValueError(f"Unsupported transcription provider: {self._provider}")

    async def _transcribe_openai_non_streaming(self, buffer: bytes, *, model: Optional[str] = None) -> str:
        api_key = self._config.get("openai_api_key")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not configured. Set in config or environment.")

        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}

        files = {
            "file": ("audio.wav", buffer, "audio/wav"),
        }
        data: Dict[str, Any] = {"model": model or self._model}
        if self._language:
            data["language"] = self._language
        if self._prompt:
            data["prompt"] = self._prompt

        # Use provided client or create a short-lived one
        if self._client is None:
            async with httpx.AsyncClient(timeout=60) as client:
                return await self._post_and_parse_text(client, url, headers, data, files)
        return await self._post_and_parse_text(self._client, url, headers, data, files)

    async def _transcribe_elevenlabs(self, buffer: bytes) -> str:
        eleven_cfg = self._eleven_cfg or {}
        lang = eleven_cfg.get("language_code") or ("eng" if (self._language or "").lower().startswith("en") else None)
        diarize = bool(eleven_cfg.get("diarize", True))
        tag_events = bool(eleven_cfg.get("tag_audio_events", True))
        model_id = eleven_cfg.get("model") or "scribe_v1"

        if self._eleven_client is not None and hasattr(self._eleven_client, "speech_to_text"):
            try:
                transcription = await asyncio.to_thread(
                    self._eleven_client.speech_to_text.convert,
                    file=BytesIO(buffer),             # <-- keyword arg
                    model_id=model_id,
                    tag_audio_events=tag_events,
                    language_code=lang,
                    diarize=diarize,
                )
            except Exception as exc:
                # Fallback if anything SDK-related goes sideways
                if "has no attribute 'speech_to_text'" in str(exc):
                    return await self._transcribe_elevenlabs_non_streaming_http(buffer)
                raise RuntimeError(f"ElevenLabs SDK error: {exc}") from exc

            # Normalize to text
            if isinstance(transcription, dict):
                text = transcription.get("text") or transcription.get("transcript")
                if isinstance(text, str):
                    return text
            text = getattr(transcription, "text", None)
            return text if isinstance(text, str) else str(transcription)

        # No SDK / too old → HTTP
        return await self._transcribe_elevenlabs_non_streaming_http(buffer)


    async def _transcribe_elevenlabs_non_streaming_http(self, buffer: bytes) -> str:
        api_key = self._config.get("elevenlabs_api_key")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not configured. Set in config or environment.")

        endpoint = self._eleven_cfg.get("endpoint") or "https://api.elevenlabs.io/v1/speech-to-text"
        model = self._eleven_cfg.get("model") or "scribe_v1"
        diarize = bool(self._eleven_cfg.get("diarize", True))
        tag_events = bool(self._eleven_cfg.get("tag_audio_events", True))
        lang_cfg = self._eleven_cfg.get("language_code")
        lang = lang_cfg
        if not lang and self._language:
            lang = "eng" if self._language.lower().startswith("en") else None

        headers = {"xi-api-key": api_key}
        files = {"file": ("audio.wav", buffer, "audio/wav")}
        data: Dict[str, Any] = {"model_id": model, "diarize": diarize, "tag_audio_events": tag_events}
        if lang:
            data["language_code"] = lang

        if self._client is None:
            async with httpx.AsyncClient(timeout=60) as client:
                return await self._post_and_parse_text_elevenlabs(client, endpoint, headers, data, files)
        return await self._post_and_parse_text_elevenlabs(self._client, endpoint, headers, data, files)

    async def _post_and_parse_text(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        data: Dict[str, Any],
        files: Dict[str, Any],
    ) -> str:
        try:
            resp = await client.post(url, headers=headers, data=data, files=files)
            resp.raise_for_status()
            payload = resp.json()
            text = payload.get("text")
            if isinstance(text, str):
                logger.info("TranscriptionService: received transcript (%d chars)", len(text))
                return text
            # Some providers may return alternatives; be defensive
            if "results" in payload:
                # Attempt basic extraction
                results = payload.get("results")
                if isinstance(results, list) and results:
                    alt = results[0]
                    if isinstance(alt, dict) and isinstance(alt.get("text"), str):
                        return alt["text"]
            raise RuntimeError("Unexpected transcription response structure")
        except httpx.RequestError as exc:
            logger.error("TranscriptionService: network error: %s", exc)
            raise RuntimeError(f"Network error during transcription: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.text
            except Exception:
                detail = str(exc)
            logger.error("TranscriptionService: HTTP error: %s", detail)
            raise RuntimeError(f"HTTP error during transcription: {detail}") from exc

    async def _post_and_parse_text_elevenlabs(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        data: Dict[str, Any],
        files: Dict[str, Any],
    ) -> str:
        try:
            resp = await client.post(url, headers=headers, data=data, files=files)
            resp.raise_for_status()
            payload = resp.json()
            # ElevenLabs may return { text: "..." } or nested structure
            text = payload.get("text")
            if isinstance(text, str):
                return text
            # Fallback generic extraction attempts
            if isinstance(payload, dict):
                for key in ("transcript", "transcription", "result"):
                    val = payload.get(key)
                    if isinstance(val, str):
                        return val
            raise RuntimeError("Unexpected ElevenLabs transcription response structure")
        except httpx.RequestError as exc:
            logger.error("TranscriptionService (11L): network error: %s", exc)
            raise RuntimeError(f"Network error during transcription (11L): {exc}") from exc
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.text
            except Exception:
                detail = str(exc)
            logger.error("TranscriptionService (11L): HTTP error: %s", detail)
            raise RuntimeError(f"HTTP error during transcription (11L): {detail}") from exc

    async def benchmark_three(self, buffer: bytes) -> List[BenchmarkResult]:
        """Run the same audio through three providers/models in parallel and return results.

        - openai: whisper-1
        - openai: gpt-4o-transcribe
        - elevenlabs: default endpoint
        """
        results: List[BenchmarkResult] = []

        async def timed(label: str, func: Callable[[], asyncio.Future]) -> BenchmarkResult:
            start = monotonic()
            err: Optional[str] = None
            text: str = ""
            try:
                text = await func()
            except Exception as exc:
                err = str(exc)
            ms = int((monotonic() - start) * 1000)
            return {"label": label, "ms": ms, "text": text, "error": err}

        # Build callables
        async def call_openai(model: str) -> str:
            return await self._transcribe_openai_non_streaming(buffer, model=model)

        async def call_eleven() -> str:
            return await self._transcribe_elevenlabs(buffer)

        tasks = [
            timed("openai:whisper-1", lambda: call_openai("whisper-1")),
            timed("openai:gpt-4o-transcribe", lambda: call_openai("gpt-4o-transcribe")),
            timed("elevenlabs", call_eleven),
        ]

        results = await asyncio.gather(*tasks)
        return list(results)
