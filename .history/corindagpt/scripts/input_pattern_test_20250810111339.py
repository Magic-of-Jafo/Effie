from __future__ import annotations

import asyncio
import logging
from logging import StreamHandler

from corindagpt.src.utils.initialization import load_config

try:
    from corindagpt.src.components.abstract_input_handler import (
        KeyboardInputHandler,
        InputEvent,
        InputPattern,
    )
except Exception as exc:  # pragma: no cover
    KeyboardInputHandler = None  # type: ignore
    InputEvent = None  # type: ignore
    InputPattern = None  # type: ignore


async def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        handlers=[StreamHandler()],
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("input_pattern_test")

    cfg = load_config()
    ip_cfg = (cfg.get("input_patterns") or {})

    if KeyboardInputHandler is None:
        print("Abstract input handler unavailable (pynput not installed?).")
        return

    hotkey = str(ip_cfg.get("hotkey") or "f12")
    brief_max_ms = int(ip_cfg.get("brief_max_ms", 250))
    sustained_min_ms = int(ip_cfg.get("sustained_min_ms", 600))
    compound_win_ms = int(ip_cfg.get("compound_double_press_window_ms", 350))

    print("Input Pattern Test")
    print(f"Hotkey: {hotkey}")
    print(f"Thresholds: brief<= {brief_max_ms} ms, sustained>= {sustained_min_ms} ms, compound window= {compound_win_ms} ms")
    print("Actions:")
    print(" - BRIEF: quick tap")
    print(" - SUSTAINED: press and hold")
    print(" - COMPOUND: quick tap, then within window press and hold")
    print("Press Ctrl+C to exit.\n")

    loop = asyncio.get_running_loop()

    async def on_event(evt: InputEvent) -> None:
        try:
            held_ms = int((evt.meta or {}).get("held_ms", 0)) if isinstance(evt.meta, dict) else 0
        except Exception:
            held_ms = 0
        print(f"Detected: {getattr(evt, 'pattern', '<unknown>')} (held={held_ms} ms)")

    handler = KeyboardInputHandler(
        loop=loop,
        on_event=on_event,  # type: ignore[arg-type]
        hotkey_name=hotkey,
        brief_max_ms=brief_max_ms,
        sustained_min_ms=sustained_min_ms,
        compound_double_press_window_ms=compound_win_ms,
    )

    await handler.start()
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        await handler.stop()


if __name__ == "__main__":
    asyncio.run(run())
