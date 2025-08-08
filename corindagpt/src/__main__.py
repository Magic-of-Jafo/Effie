from __future__ import annotations

import asyncio
import logging
import os
from logging import StreamHandler

from components.input_handler import InputHandler
from services.voice_to_text import VoiceRecorder, TranscriptionService
try:  # optional import of SDPressHoldRecorder symbol
    from services.voice_to_text import SDPressHoldRecorder  # type: ignore
except Exception:  # pragma: no cover
    SDPressHoldRecorder = None  # type: ignore


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        handlers=[StreamHandler()],
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("corindagpt")

    # Benchmark toggle via env
    benchmark = os.getenv("BENCHMARK_TRANSCRIPTION", "0") == "1"

    # Prefer sounddevice-based recorder on Windows if available
    if SDPressHoldRecorder is not None:
        try:
            recorder = SDPressHoldRecorder()
            logger.info("Using SDPressHoldRecorder for capture")
        except Exception as exc:
            logger.warning("Falling back to SpeechRecognition recorder: %s", exc)
            recorder = VoiceRecorder()
    else:
        recorder = VoiceRecorder()
    transcriber = TranscriptionService()

    # Calibrate ambient noise once to avoid consuming initial press audio
    try:
        # Only VoiceRecorder supports calibration
        if isinstance(recorder, VoiceRecorder):
            await recorder.calibrate_ambient_noise(duration=0.2)
    except Exception as exc:
        logger.warning("Recorder calibration failed (continuing): %s", exc)

    async def on_press() -> None:
        await recorder.start_recording()

    async def on_release() -> None:
        data = await recorder.stop_recording()
        if data:
            logger.info("Audio captured")
            try:
                if benchmark:
                    results = await transcriber.benchmark_three(data)
                    for r in results:
                        label = r.get("label")
                        ms = r.get("ms")
                        text = r.get("text")
                        err = r.get("error")
                        if err:
                            logger.info("%s -> %d ms ERROR: %s", label, ms, err)
                        else:
                            logger.info("%s -> %d ms TEXT: %s", label, ms, text)
                else:
                    transcript = await transcriber.transcribe(data)
                    logger.info("Transcript: %s", transcript)
            except Exception as exc:
                logger.error("Transcription failed: %s", exc)
        else:
            logger.info("Audio captured (empty)")

    loop = asyncio.get_running_loop()
    input_handler = InputHandler(
        loop=loop,
        on_press_active=on_press,
        on_release_active=on_release,
        hotkey_name="f12",
    )

    # Start keyboard listener in background thread
    listener = input_handler.start_keyboard_listener()
    if listener is None:
        logger.error("Keyboard listener could not be started. Ensure 'pynput' is installed and permissions are granted.")
    else:
        logger.info("Ready. Hold F12 to record; release to %s.", "benchmark" if benchmark else "transcribe")

    try:
        # Keep the event loop alive indefinitely
        await asyncio.Event().wait()
    finally:
        input_handler.stop_keyboard_listener()


if __name__ == "__main__":
    asyncio.run(main())
