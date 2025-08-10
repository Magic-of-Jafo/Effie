from __future__ import annotations

import asyncio
import logging
from logging import StreamHandler
from pathlib import Path

from ..utils.initialization import load_config
from ..components.phase_manager import PhaseManager
from ..components.input_handler import InputHandler
from ..components.state_machine import StateMachine, State
from ..services.voice_to_text import VoiceRecorder, TranscriptionService
from ..services.prompt_loader import load_prompt_for_phase, render_prompt
from ..services import gpt as gpt_service
from ..services import tts as tts_service


async def run() -> None:
    logging.basicConfig(level=logging.INFO, handlers=[StreamHandler()], format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("phase_prompt_test")

    cfg = load_config()

    # Plan and managers
    plan = cfg.get("performance_plan") or [1]
    if not isinstance(plan, list) or not plan:
        plan = [1]
    phase_manager = PhaseManager(performance_plan=plan)
    fsm = StateMachine(performance_plan=plan)

    # Print current phase and prompt contents
    current_phase = phase_manager.current_phase
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / f"phase_{current_phase}_prompt.txt"
    try:
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    except Exception:
        prompt_text = "<default prompt (fallback)>"
    print(f"Current phase: {current_phase}")
    print("Prompt template:\n" + prompt_text + "\n---")

    # Setup recorder/transcriber
    recorder = VoiceRecorder()
    transcriber = TranscriptionService()

    loop = asyncio.get_running_loop()

    async def on_press() -> None:
        if fsm.transition(State.LISTENING):
            await recorder.start_recording()

    async def on_release() -> None:
        if fsm.state != State.LISTENING:
            return
        data = await recorder.stop_recording()
        if not data:
            fsm.transition(State.IDLE)
            return
        if not fsm.transition(State.PROCESSING):
            return
        try:
            transcript = await transcriber.transcribe(data)
            phase = phase_manager.current_phase
            template = load_prompt_for_phase(phase)
            rendered = render_prompt(template, {"transcript": transcript})

            llm_ctrl = ((cfg.get("transitions") or {}).get("llm_phase_control") or {})
            use_tools = bool(llm_ctrl.get("enabled", False))

            if use_tools:
                content, tool_calls = await gpt_service.chat_with_tools(rendered)
            else:
                content = await gpt_service.generate_response(rendered)
                tool_calls = []

            # Handle tool calls
            for call in tool_calls:
                fn = ((call or {}).get("function") or {})
                name = fn.get("name")
                args_str = fn.get("arguments")
                if name != "set_phase":
                    continue
                try:
                    import json as _json
                    args = _json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                except Exception:
                    args = {}
                action = (args.get("action") or "").lower()
                phase_arg = args.get("phase")
                if action == "advance":
                    new_phase = phase_manager.advance()
                    print(f"Phase transitioned to {new_phase}")
                elif action == "set" and phase_arg is not None:
                    new_phase = phase_manager.set(phase_arg)
                    print(f"Phase set to {new_phase}")

            # Stream TTS via ElevenLabs if enabled
            if tts_service.is_streaming_enabled(cfg):
                await tts_service.stream_and_play(content)
            else:
                audio = await tts_service.synthesize(content, config=cfg)
                await tts_service.play(audio)
        finally:
            fsm.transition(State.IDLE)

    input_handler = InputHandler(
        loop=loop,
        on_press_active=on_press,
        on_release_active=on_release,
        hotkey_name="f12",
    )
    input_handler.start_keyboard_listener()

    print("Press and hold F12 to speak. Release to process. Ctrl+C to exit.")
    try:
        await asyncio.Event().wait()
    finally:
        input_handler.stop_keyboard_listener()


if __name__ == "__main__":
    asyncio.run(run())
