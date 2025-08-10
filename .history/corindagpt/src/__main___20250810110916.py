from __future__ import annotations

import asyncio
import logging
import os
from logging import StreamHandler

from .components.input_handler import InputHandler
from .components.state_machine import StateMachine, State
from .components.phase_manager import PhaseManager
from .services.voice_to_text import VoiceRecorder, TranscriptionService
try:  # optional import of SDPressHoldRecorder symbol
    from .services.voice_to_text import SDPressHoldRecorder  # type: ignore
except Exception:  # pragma: no cover
    SDPressHoldRecorder = None  # type: ignore

# Config loader
from .utils.initialization import load_config

# Prompt loader
from .services.prompt_loader import load_prompt_for_phase, render_prompt

# Abstract input handler
try:
    from .components.abstract_input_handler import KeyboardInputHandler, InputEvent, InputPattern  # type: ignore
except Exception:  # pragma: no cover
    KeyboardInputHandler = None  # type: ignore
    InputEvent = None  # type: ignore
    InputPattern = None  # type: ignore


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

    # PhaseManager for 2.4/2.5
    phase_manager = PhaseManager(performance_plan=plan)

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
                    phase = phase_manager.current_phase
                    template = load_prompt_for_phase(phase)
                    rendered = render_prompt(template, {"transcript": transcript})

                    # Call LLM with rendered prompt
                    from .services import gpt as gpt_service
                    from .services import tts as tts_service

                    llm_ctrl = ((cfg.get("transitions") or {}).get("llm_phase_control") or {})
                    use_tools = bool(llm_ctrl.get("enabled", False))

                    try:
                        if use_tools:
                            content, tool_calls = await gpt_service.chat_with_tools(rendered)
                        else:
                            content = await gpt_service.generate_response(rendered)
                            tool_calls = []
                        logger.info("LLM Response: %s", content)
                    except Exception as llm_exc:
                        logger.error("LLM generate_response failed: %s", llm_exc)
                        # PROCESSING -> IDLE on failure as well
                        fsm.transition(State.IDLE)
                        return

                    # Handle tool calls
                    try:
                        for call in tool_calls:
                            fn = ((call or {}).get("function") or {})
                            name = fn.get("name")
                            args_str = fn.get("arguments")
                            if name != "set_phase":
                                continue
                            # Parse arguments (JSON string per OpenAI)
                            try:
                                import json as _json
                                args = _json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                            except Exception:
                                args = {}
                            action = (args.get("action") or "").lower()
                            phase_arg = args.get("phase")
                            if action == "advance":
                                new_phase = phase_manager.advance()
                                try:
                                    fsm.set_current_phase(new_phase)
                                except Exception:
                                    pass
                                logger.info("Phase transitioned to %s (via LLM)", new_phase)
                            elif action == "set" and phase_arg is not None:
                                new_phase = phase_manager.set(phase_arg)
                                try:
                                    fsm.set_current_phase(new_phase)
                                except Exception:
                                    pass
                                logger.info("Phase set to %s (via LLM)", new_phase)
                    except Exception as tool_exc:
                        logger.warning("Tool-call handling error: %s", tool_exc)

                    try:
                        # If streaming is enabled, stream directly; otherwise synth then play
                        if tts_service.is_streaming_enabled():
                            await tts_service.stream_and_play(content, started_at_monotonic=t_release_mono)
                            audio_bytes = b""
                        else:
                            audio_bytes = await tts_service.synthesize(content)
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
            except Exception as exc:
                logger.error("Transcription failed: %s", exc)
                # Return to IDLE on error
                fsm.transition(State.IDLE)
        else:
            logger.info("Audio captured (empty)")
            # Allow LISTENING -> IDLE when no data captured
            fsm.transition(State.IDLE)

    # Transition trigger from long-press hotkey per config
    async def on_transition_trigger() -> None:
        new_phase = phase_manager.advance()
        # Sync FSM display/logging phase
        try:
            fsm.set_current_phase(new_phase)
        except Exception:
            pass
        logger.info("Phase transitioned to %s", new_phase)

    # Extract transition hotkey settings
    trans_cfg = (cfg.get("transitions") or {}).get("phase_transition", {})
    trans_hotkey = str(trans_cfg.get("hotkey") or "f11")
    trans_ms = int(trans_cfg.get("long_press_ms", 3000))

    loop = asyncio.get_running_loop()

    # Abstract input (log-only integration)
    abs_handler = None
    ip_cfg = (cfg.get("input_patterns") or {})
    enabled_sources = [str(s).lower() for s in (ip_cfg.get("enabled_sources") or [])]
    if KeyboardInputHandler is not None and ("keyboard" in enabled_sources):
        async def on_abs_event(evt: InputEvent) -> None:
            try:
                held_ms = int((evt.meta or {}).get("held_ms", 0)) if isinstance(evt.meta, dict) else 0
            except Exception:
                held_ms = 0
            logging.getLogger("input_patterns").info(
                "Detected pattern: %s (held=%d ms)", getattr(evt, "pattern", "<unknown>"), held_ms
            )
        abs_handler = KeyboardInputHandler(
            loop=loop,
            on_event=on_abs_event,  # type: ignore[arg-type]
            hotkey_name=str(ip_cfg.get("hotkey") or "f12"),
            brief_max_ms=int(ip_cfg.get("brief_max_ms", 250)),
            sustained_min_ms=int(ip_cfg.get("sustained_min_ms", 600)),
            compound_double_press_window_ms=int(ip_cfg.get("compound_double_press_window_ms", 350)),
        )
        try:
            await abs_handler.start()
        except Exception as exc:
            logging.getLogger("input_patterns").warning("Abstract input handler failed to start: %s", exc)

    input_handler = InputHandler(
        loop=loop,
        on_press_active=on_press,
        on_release_active=on_release,
        hotkey_name="f12",
        transition_hotkey_name=trans_hotkey,
        transition_long_press_ms=trans_ms,
        on_transition_trigger=on_transition_trigger,
    )

    # Start keyboard listener in background thread
    listener = input_handler.start_keyboard_listener()
    if listener is None:
        logger.error("Keyboard listener could not be started. Ensure 'pynput' is installed and permissions are granted.")
    else:
        logger.info(
            "Ready. Hold F12 to record; release to %s. Long-press %s for %d ms to advance phase.",
            "benchmark" if benchmark else "transcribe",
            trans_hotkey,
            trans_ms,
        )

    try:
        # Keep the event loop alive indefinitely
        await asyncio.Event().wait()
    finally:
        input_handler.stop_keyboard_listener()
        if abs_handler is not None:
            try:
                await abs_handler.stop()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
