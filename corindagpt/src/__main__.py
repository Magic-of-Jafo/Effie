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
    # Suppress all logging output (stealth) unless CORINDA_DEBUG=1
    if os.getenv("CORINDA_DEBUG", "0") != "1":
        logging.disable(logging.CRITICAL)

    # Load configuration
    cfg = load_config()

    # Pre-open TLS connections to both APIs so the first interaction is fast
    from .services import gpt as _gpt_warm, tts as _tts_warm
    warmup_tasks = [
        asyncio.create_task(_gpt_warm.warmup(cfg)),
        asyncio.create_task(_tts_warm.warmup(cfg)),
    ]

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

    # Priority audio queue for BRIEF inputs (FR8/FR9)
    from .services.audio_queue import AudioItem, build_default_queue, response_playback_mode
    from .services import decoder as decoder_service
    from .services.conversational_memory import build_memory

    audio_queue = build_default_queue(cfg)
    memory = build_memory(cfg)
    logger.info("LLM response playback mode: %s", response_playback_mode(cfg))

    # Live show settings - the dashboard mutates these while running; the
    # handler reads them per question so changes apply without a restart
    live = {
        "performance_plan": list(cfg.get("performance_plan") or [1]),
        "response_mode": response_playback_mode(cfg),
        "keepalive_interval_s": float((cfg.get("network") or {}).get("keepalive_interval_s", 60)),
        "listening_mode": str(cfg.get("listening_mode") or "push_hold"),
    }

    # Streaming ears (voice-control story 1): rolling transcript over a
    # continuous WebSocket. Runs only while listening_mode == "streaming";
    # push-hold recording stays fully intact as baseline and fallback.
    from .services.streaming_transcription import StreamingTranscriptionService

    streaming_svc = StreamingTranscriptionService(cfg)
    if live["listening_mode"] == "streaming":
        await streaming_svc.start()

    # Interaction mode for the press in progress, set by pattern events:
    # None (brief/dead-zone), "decode" (sustained), or "bypass" (compound)
    interaction = {"mode": None, "discard": None}

    async def on_press() -> None:
        # Transition IDLE -> LISTENING; ignore press if not allowed
        if fsm.transition(State.LISTENING):
            interaction["mode"] = None
            interaction["stream_press"] = None
            # Streaming mode: the transcript already exists - just mark the
            # window start. Falls through to the recorder when the stream
            # is down so a dead connection can never lose a question.
            if live["listening_mode"] == "streaming" and streaming_svc.is_active():
                from time import monotonic as _mono
                interaction["stream_press"] = _mono()
                logger.debug("Streaming window opened")
                return
            # A discard from a just-released tap may still hold the recorder
            # lock; wait for it (milliseconds) or start_recording is a no-op
            pending = interaction.get("discard")
            if pending is not None and not pending.done():
                await pending
            await recorder.start_recording()
        else:
            logger.debug("Ignoring press: transition to LISTENING not allowed from %s", fsm.state)

    async def play_queue_next() -> None:
        try:
            played = await audio_queue.play_next(config=cfg)
            if played is None:
                logger.info("BRIEF input: audio queue empty")
            elif played.text:
                # The audience heard Effie say this; she must remember it
                memory.add_assistant(played.text)
        except Exception as exc:
            logger.error("Audio queue playback failed: %s", exc)

    async def on_release() -> None:
        # Only handle release if we are in LISTENING state
        if fsm.state != State.LISTENING:
            logger.debug("Ignoring release: not in LISTENING (current=%s)", fsm.state)
            return

        mode = interaction["mode"]
        if mode is None:
            # Brief tap or dead-zone release: discard the recording and free
            # the state machine immediately so a follow-up press (second tap
            # of a COMPOUND) is not ignored. BRIEF queue playback is
            # triggered by its own pattern event.
            fsm.transition(State.IDLE)

            async def _discard() -> None:
                try:
                    abort = getattr(recorder, "abort_recording", None)
                    if abort is not None:
                        await abort()
                    else:
                        await recorder.stop_recording()
                except Exception as exc:
                    logger.debug("Recording discard failed: %s", exc)

            interaction["discard"] = asyncio.create_task(_discard())
            logger.debug("Release without sustained/compound pattern; recording discarded")
            return

        t_release = asyncio.get_running_loop().time()
        # For higher precision cross-threads, also compute monotonic() in Python space at this point
        from time import monotonic as _mono
        t_release_mono = _mono()

        stream_press = interaction.get("stream_press")
        use_stream = stream_press is not None and streaming_svc.is_active()
        data = b"" if use_stream else await recorder.stop_recording()
        if use_stream or data:
            logger.info("Audio captured" + (" (streaming window)" if use_stream else ""))
            # LISTENING -> PROCESSING
            if not fsm.transition(State.PROCESSING):
                logger.warning("Unexpected state; cannot enter PROCESSING from %s", fsm.state)
                return
            try:
                if benchmark and not use_stream:
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
                    if use_stream:
                        transcript = await streaming_svc.get_window(stream_press, t_release_mono)
                    else:
                        transcript = await transcriber.transcribe(data)
                    logger.info(
                        "Transcript (release->transcript %d ms): %s",
                        int((_mono() - t_release_mono) * 1000),
                        transcript,
                    )
                    raw_transcript = transcript
                    # SUSTAINED runs the decoder; COMPOUND bypasses it (FR4/FR6)
                    if mode == "decode":
                        transcript = decoder_service.decode(transcript, config=cfg)
                    # Load a phase-specific prompt and render with context.
                    # Single-phase shows (settings.yaml phases: 1) run on the
                    # persona alone - no per-phase flavor.
                    phase = phase_manager.current_phase
                    multi_phase = len(live["performance_plan"]) > 1
                    template = load_prompt_for_phase(phase, use_phase_files=multi_phase)
                    rendered = render_prompt(template, {"transcript": transcript})

                    # Call LLM with rendered prompt
                    from .services import gpt as gpt_service
                    from .services import tts as tts_service

                    llm_ctrl = ((cfg.get("transitions") or {}).get("llm_phase_control") or {})
                    use_tools = bool(llm_ctrl.get("enabled", False))
                    sentence_streaming = tts_service.is_streaming_enabled() and bool(
                        (cfg.get("tts") or {}).get("sentence_streaming", True)
                    )

                    # Attaching tools forces reasoning_effort off the request
                    # (gpt-5.4 rejects the combination), which makes the model
                    # reason on every answer. Only pay that cost when the
                    # transcript could actually be a phase command.
                    # (single-phase shows have nothing to advance to)
                    wants_tools = use_tools and multi_phase and ("phase" in (raw_transcript or "").lower())

                    def handle_tool_calls(tool_calls: list) -> None:
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

                    if live["response_mode"] == "queued":
                        # Pull-string mode (FR8): the response is synthesized
                        # and parked at the queue front; the next BRIEF input
                        # (string pull) speaks it. Nothing plays now.
                        try:
                            if wants_tools:
                                content, tool_calls = await gpt_service.chat_with_tools(rendered, history=memory.messages())
                            else:
                                content = await gpt_service.generate_response(rendered, history=memory.messages())
                                tool_calls = []
                            logger.info("LLM Response (queued): %s", content)
                            memory.add_exchange(raw_transcript, content)
                        except Exception as llm_exc:
                            logger.error("LLM request failed: %s", llm_exc)
                            fsm.transition(State.IDLE)
                            return
                        handle_tool_calls(tool_calls)
                        try:
                            if content and content.strip():
                                audio_bytes = await tts_service.synthesize(content)
                                audio_queue.push_priority(
                                    AudioItem(label=f"LLM: {content[:40]}", data=audio_bytes)
                                )
                        except Exception as tts_exc:
                            logger.error("TTS synthesis for queue failed: %s", tts_exc)
                        finally:
                            fsm.transition(State.IDLE)
                        return

                    if sentence_streaming:
                        # Overlapped pipeline: TTS begins on the first complete
                        # sentence while the LLM is still generating (NFR5).
                        sink: dict = {}
                        try:
                            await tts_service.stream_sentences_and_play(
                                gpt_service.stream_chat(rendered, sink=sink, history=memory.messages()),
                                started_at_monotonic=t_release_mono,
                            )
                            logger.info("LLM Response: %s", sink.get("content", ""))
                            memory.add_exchange(raw_transcript, sink.get("content", ""))
                        except Exception as pipe_exc:
                            logger.error("Streamed LLM->TTS pipeline failed: %s", pipe_exc)
                            fsm.transition(State.IDLE)
                            return
                        finally:
                            handle_tool_calls(sink.get("tool_calls") or [])
                            fsm.transition(State.IDLE)
                        return

                    try:
                        if wants_tools:
                            content, tool_calls = await gpt_service.chat_with_tools(rendered, history=memory.messages())
                        else:
                            content = await gpt_service.generate_response(rendered, history=memory.messages())
                            tool_calls = []
                        logger.info("LLM Response: %s", content)
                        memory.add_exchange(raw_transcript, content)
                    except Exception as llm_exc:
                        logger.error("LLM generate_response failed: %s", llm_exc)
                        # PROCESSING -> IDLE on failure as well
                        fsm.transition(State.IDLE)
                        return

                    handle_tool_calls(tool_calls)

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

    # Abstract input handler owns the record hotkey: raw press/release drive
    # recording, pattern events select the workflow (Story 3.2)
    abs_handler = None
    ip_cfg = (cfg.get("input_patterns") or {})
    enabled_sources = [str(s).lower() for s in (ip_cfg.get("enabled_sources") or [])]
    play_handler = None
    # Separate play trigger (the pull string); when set, BRIEF on the record
    # hotkey is ignored and only this key plays the queue front.
    play_hotkey = str(ip_cfg.get("play_hotkey") or "").strip() or None
    if KeyboardInputHandler is not None and ("keyboard" in enabled_sources):
        async def on_abs_event(evt: InputEvent) -> None:
            try:
                held_ms = int((evt.meta or {}).get("held_ms", 0)) if isinstance(evt.meta, dict) else 0
            except Exception:
                held_ms = 0
            logging.getLogger("input_patterns").info(
                "Detected pattern: %s (held=%d ms)", getattr(evt, "pattern", "<unknown>"), held_ms
            )
            if evt.pattern == InputPattern.SUSTAINED:
                interaction["mode"] = "decode"
            elif evt.pattern == InputPattern.COMPOUND:
                interaction["mode"] = "bypass"
            elif evt.pattern == InputPattern.BRIEF and play_hotkey is None:
                asyncio.create_task(play_queue_next())

        abs_handler = KeyboardInputHandler(
            loop=loop,
            on_event=on_abs_event,  # type: ignore[arg-type]
            hotkey_name=str(ip_cfg.get("hotkey") or "f12"),
            brief_max_ms=int(ip_cfg.get("brief_max_ms", 250)),
            sustained_min_ms=int(ip_cfg.get("sustained_min_ms", 600)),
            compound_double_press_window_ms=int(ip_cfg.get("compound_double_press_window_ms", 350)),
            on_press_raw=on_press,
            on_release_raw=on_release,
        )
        try:
            await abs_handler.start()
        except Exception as exc:
            logging.getLogger("input_patterns").warning("Abstract input handler failed to start: %s", exc)
            abs_handler = None

    if abs_handler is not None and play_hotkey is not None:
        async def on_play_event(evt: InputEvent) -> None:
            if evt.pattern == InputPattern.BRIEF:
                logging.getLogger("input_patterns").info("Play trigger: tap on '%s'", play_hotkey)
                asyncio.create_task(play_queue_next())

        play_handler = KeyboardInputHandler(
            loop=loop,
            on_event=on_play_event,  # type: ignore[arg-type]
            hotkey_name=play_hotkey,
            brief_max_ms=int(ip_cfg.get("brief_max_ms", 250)),
            sustained_min_ms=int(ip_cfg.get("sustained_min_ms", 600)),
            # No double-tap semantics on the string: window 0 emits BRIEF immediately
            compound_double_press_window_ms=0,
        )
        try:
            await play_handler.start()
        except Exception as exc:
            logging.getLogger("input_patterns").warning("Play trigger handler failed to start: %s", exc)
            play_handler = None

    if abs_handler is not None:
        # Legacy handler keeps only the phase-transition hotkey
        input_handler = InputHandler(
            loop=loop,
            hotkey_name=None,
            transition_hotkey_name=trans_hotkey,
            transition_long_press_ms=trans_ms,
            on_transition_trigger=on_transition_trigger,
        )
    else:
        # Fallback: no abstract handler available; legacy handler drives
        # recording directly (sustained-style only, decoder applied)
        logger.warning("Abstract input handler unavailable; falling back to legacy press/hold input")

        async def legacy_press() -> None:
            await on_press()
            interaction["mode"] = "decode"

        input_handler = InputHandler(
            loop=loop,
            on_press_active=legacy_press,
            on_release_active=on_release,
            hotkey_name=str(ip_cfg.get("hotkey") or "f12"),
            transition_hotkey_name=trans_hotkey,
            transition_long_press_ms=trans_ms,
            on_transition_trigger=on_transition_trigger,
        )

    # Start keyboard listener in background thread
    listener = input_handler.start_keyboard_listener()
    if listener is None:
        logger.error("Keyboard listener could not be started. Ensure 'pynput' is installed and permissions are granted.")
    else:
        record_key = str(ip_cfg.get("hotkey") or "f12").upper()
        logger.info(
            "Ready. Tap %s: play next queued clip | hold %s + speak: decoded pipeline | "
            "double-tap + hold %s + speak: bypass decoder | long-press %s %d ms: advance phase.%s",
            (play_hotkey.upper() if play_hotkey else record_key),
            record_key,
            record_key,
            trans_hotkey,
            trans_ms,
            " (BENCHMARK MODE)" if benchmark else "",
        )

    # Keep API TLS connections open between interactions: questions are minutes
    # apart on stage, far past any keepalive expiry, so without a heartbeat
    # every leg of every question pays a fresh handshake.
    from .services import gpt as _gpt_ka, tts as _tts_ka

    async def _keepalive_loop() -> None:
        oa_key = cfg.get("openai_api_key")
        ev_key = cfg.get("elevenlabs_api_key")
        while True:
            # Interval is live-tunable from the dashboard; 0 pauses pinging
            interval = float(live["keepalive_interval_s"])
            if interval <= 0:
                await asyncio.sleep(30)
                continue
            await asyncio.sleep(interval)
            try:
                if oa_key:
                    await _gpt_ka.get_shared_client().get(
                        "/models/gpt-4o-mini", headers={"Authorization": f"Bearer {oa_key}"}
                    )
                if ev_key:
                    await _tts_ka.get_shared_async_client().get(
                        "/models", headers={"xi-api-key": ev_key}
                    )
                    await asyncio.to_thread(_tts_ka.get_elevenlabs_client(ev_key).models.list)
            except Exception as exc:
                logger.debug("Keepalive ping failed (continuing): %s", exc)

    keepalive_task = asyncio.create_task(_keepalive_loop())

    # Localhost settings dashboard: edits config/settings.yaml and applies
    # supported values to the running show through `live`
    from .services.dashboard import start_dashboard
    from .utils.initialization import PROJECT_ROOT

    def _apply_dashboard_settings(vals: dict) -> None:
        # Called from the dashboard's server thread; mutate app state on the loop
        def _apply() -> None:
            if "phases" in vals:
                new_plan = list(range(1, int(vals["phases"]) + 1))
                live["performance_plan"] = new_plan
                phase_manager.set_plan(new_plan)
            if "response_playback" in vals:
                live["response_mode"] = str(vals["response_playback"])
            if "keepalive_interval_s" in vals:
                live["keepalive_interval_s"] = float(vals["keepalive_interval_s"])
            if "listening_mode" in vals:
                mode = str(vals["listening_mode"])
                if mode != live["listening_mode"]:
                    live["listening_mode"] = mode
                    logger.info("Listening mode switched to %s", mode)
                    if mode == "streaming":
                        asyncio.create_task(streaming_svc.start())
                    else:
                        asyncio.create_task(streaming_svc.stop())

        loop.call_soon_threadsafe(_apply)

    def _live_state() -> dict:
        # Read by the dashboard's Live view (server thread; snapshot reads only)
        return {
            "mode": live["listening_mode"],
            "status": streaming_svc.status,
            "last_error": streaming_svc.last_error,
            "entries": streaming_svc.recent(45.0),
        }

    dashboard_server = start_dashboard(
        cfg, PROJECT_ROOT / "config" / "settings.yaml", _apply_dashboard_settings,
        live_state_cb=_live_state,
    )

    try:
        # Keep the event loop alive indefinitely
        await asyncio.Event().wait()
    finally:
        try:
            await streaming_svc.stop()
        except Exception:
            pass
        if dashboard_server is not None:
            dashboard_server.shutdown()
        if keepalive_task is not None:
            keepalive_task.cancel()
        input_handler.stop_keyboard_listener()
        if abs_handler is not None:
            try:
                await abs_handler.stop()
            except Exception:
                pass
        if play_handler is not None:
            try:
                await play_handler.stop()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
