from __future__ import annotations

import asyncio
import logging
import os
from logging import StreamHandler

from .components.input_handler import InputHandler
from .components.state_machine import StateMachine, State
from .services.voice_to_text import VoiceRecorder, TranscriptionService
try:  # optional import of SDPressHoldRecorder symbol
    from .services.voice_to_text import SDPressHoldRecorder  # type: ignore
except Exception:  # pragma: no cover
    SDPressHoldRecorder = None  # type: ignore

# Config loader
from .utils.initialization import load_config

# Prompt loader
from .services.prompt_loader import load_prompt_for_phase, render_prompt


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        handlers=[StreamHandler()],
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("corindagpt")
    # Suppress all logging output per request
    logging.disable(logging.CRITICAL)

    # Load configuration
    cfg = load_config()

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

    # Load and validate performance plan
    plan = cfg.get("performance_plan")
    if not isinstance(plan, list) or not plan:
        logger = logging.getLogger("corindagpt")
        logger.warning("Config: performance_plan missing/invalid; defaulting to [1]")
        plan = [1]

    # State machine with phase management
    fsm = StateMachine(performance_plan=plan)

    # Log initial phase
    logger.info("Current phase on startup: %s", fsm.get_current_phase())

    # Calibrate ambient noise once to avoid consuming initial press audio
    try:
        # Only VoiceRecorder supports calibration
        if isinstance(recorder, VoiceRecorder):
            await recorder.calibrate_ambient_noise(duration=0.2)
    except Exception as exc:
        logger.warning("Recorder calibration failed (continuing): %s", exc)

    async def on_press() -> None:
        # Transition IDLE -> LISTENING; ignore press if not allowed
        if fsm.transition(State.LISTENING):
            await recorder.start_recording()
        else:
            logger.debug("Ignoring press: transition to LISTENING not allowed from %s", fsm.state)

    async def on_release() -> None:
        # Only handle release if we are in LISTENING state
        if fsm.state != State.LISTENING:
            logger.debug("Ignoring release: not in LISTENING (current=%s)", fsm.state)
            return

        t_release = asyncio.get_running_loop().time()
        # For higher precision cross-threads, also compute monotonic() in Python space at this point
        from time import monotonic as _mono
        t_release_mono = _mono()

        data = await recorder.stop_recording()
        if data:
            logger.info("Audio captured")
            # LISTENING -> PROCESSING
            if not fsm.transition(State.PROCESSING):
                logger.warning("Unexpected state; cannot enter PROCESSING from %s", fsm.state)
                return
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
                    # Load a phase-specific prompt and render with context
                    phase = fsm.get_current_phase()
                    template = load_prompt_for_phase(phase)
                    rendered = render_prompt(template, {"transcript": transcript})

                    # Call LLM with rendered prompt
                    from .services.gpt import generate_response
                    from .services import tts as tts_service

                    try:
                        llm_text = await generate_response(rendered)
                        logger.info("LLM Response: %s", llm_text)
                    except Exception as llm_exc:
                        logger.error("LLM generate_response failed: %s", llm_exc)
                        # PROCESSING -> IDLE on failure as well
                        fsm.transition(State.IDLE)
                        return

                    try:
                        # If streaming is enabled, stream directly; otherwise synth then play
                        if tts_service.is_streaming_enabled():
                            await tts_service.stream_and_play(llm_text, started_at_monotonic=t_release_mono)
                            audio_bytes = b""
                        else:
                            audio_bytes = await tts_service.synthesize(llm_text)
                    except Exception as tts_exc:
                        logger.error("TTS synthesize/stream failed: %s", tts_exc)
                        fsm.transition(State.IDLE)
                        return

                    try:
                        if audio_bytes:
                            await tts_service.play(audio_bytes, started_at_monotonic=t_release_mono)
                    except Exception as play_exc:
                        logger.error("Audio playback failed: %s", play_exc)
                    finally:
                        # PROCESSING -> IDLE after playback completes (or errors)
                        fsm.transition(State.IDLE)
                        # Demo behavior: optionally advance phase after each full cycle
                        new_phase = fsm.advance_phase_if_requested(advance=True)
                        if new_phase is not None:
                            logger.info("Current phase changed -> %s", new_phase)
            except Exception as exc:
                logger.error("Transcription failed: %s", exc)
                # Return to IDLE on error
                fsm.transition(State.IDLE)
        else:
            logger.info("Audio captured (empty)")
            # Allow LISTENING -> IDLE when no data captured
            fsm.transition(State.IDLE)

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
