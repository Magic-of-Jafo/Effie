from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Dict, Optional

try:
    from pynput import keyboard  # type: ignore
except Exception:  # pragma: no cover
    keyboard = None  # type: ignore

logger = logging.getLogger(__name__)


class InputPattern(str, Enum):
    BRIEF = "BRIEF"
    SUSTAINED = "SUSTAINED"
    COMPOUND = "COMPOUND"  # double-tap then hold


@dataclass
class InputEvent:
    pattern: InputPattern
    timestamp: float
    meta: Dict[str, object]


EventHandler = Callable[[InputEvent], Awaitable[None]]


class AbstractInputHandler:
    """Abstract input handler interface. Concrete sources must implement start/stop."""

    def __init__(self, *, on_event: EventHandler) -> None:
        self._on_event = on_event

    async def start(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def stop(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class KeyboardInputHandler(AbstractInputHandler):
    """Keyboard-based input handler producing BRIEF/SUSTAINED/COMPOUND events.

    Detection rules (configurable thresholds):
      - BRIEF: press duration <= brief_max_ms
      - SUSTAINED: hold duration >= sustained_min_ms (emitted when threshold crossed or at release)
      - COMPOUND: a BRIEF press followed by another press within compound_window, held for >= sustained_min_ms
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        on_event: EventHandler,
        hotkey_name: str = "f12",
        brief_max_ms: int = 250,
        sustained_min_ms: int = 600,
        compound_double_press_window_ms: int = 350,
    ) -> None:
        super().__init__(on_event=on_event)
        self.loop = loop
        self.hotkey_name = (hotkey_name or "f12").lower()
        self.brief_max_ms = int(max(0, brief_max_ms))
        self.sustained_min_ms = int(max(0, sustained_min_ms))
        self.compound_window_ms = int(max(0, compound_double_press_window_ms))
        self._listener: Optional["keyboard.Listener"] = None
        self._pressed: bool = False
        self._t_press: Optional[float] = None
        self._sustained_timer: Optional[asyncio.TimerHandle] = None
        self._last_brief_at: Optional[float] = None
        self._double_tap_pending: bool = False
        self._event_emitted: bool = False

    def _is_hotkey(self, key: object) -> bool:
        return self.hotkey_name in str(key).lower()

    async def start(self) -> None:
        if keyboard is None:
            logger.warning("KeyboardInputHandler: pynput not available; not started")
            return
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release, suppress=False)
        self._listener.start()
        logger.info(
            "KeyboardInputHandler started for hotkey '%s' (brief<=%dms, sustained>=%dms, compound window=%dms)",
            self.hotkey_name,
            self.brief_max_ms,
            self.sustained_min_ms,
            self.compound_window_ms,
        )

    async def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            finally:
                self._listener = None
        if self._sustained_timer is not None:
            try:
                self._sustained_timer.cancel()
            except Exception:
                pass
            finally:
                self._sustained_timer = None

    def _schedule_sustained(self) -> None:
        if self._sustained_timer is not None:
            try:
                self._sustained_timer.cancel()
            except Exception:
                pass
        delay = max(0.0, self.sustained_min_ms / 1000.0)
        def _fire():
            if self._pressed and not self._event_emitted and self._t_press is not None:
                now = time.monotonic()
                held_ms = int((now - self._t_press) * 1000)
                pattern = InputPattern.SUSTAINED
                if self._double_tap_pending:
                    pattern = InputPattern.COMPOUND
                self._event_emitted = True
                self.loop.create_task(self._on_event(InputEvent(pattern=pattern, timestamp=now, meta={"held_ms": held_ms})))
        try:
            self._sustained_timer = self.loop.call_later(delay, _fire)
        except Exception as exc:
            logger.debug("KeyboardInputHandler: failed scheduling sustained timer: %s", exc)

    def _on_press(self, key: object) -> None:
        if not self._is_hotkey(key):
            return
        if self._pressed:
            return
        self._pressed = True
        self._event_emitted = False
        now = time.monotonic()
        self._t_press = now
        # Detect double-tap window
        if self._last_brief_at is not None:
            if (now - self._last_brief_at) * 1000.0 <= self.compound_window_ms:
                self._double_tap_pending = True
            else:
                self._double_tap_pending = False
        else:
            self._double_tap_pending = False
        # Schedule sustained threshold
        self._schedule_sustained()

    def _on_release(self, key: object) -> None:
        if not self._is_hotkey(key):
            return
        if not self._pressed:
            return
        now = time.monotonic()
        self._pressed = False
        if self._sustained_timer is not None:
            try:
                self._sustained_timer.cancel()
            except Exception:
                pass
            finally:
                self._sustained_timer = None
        t_press = self._t_press or now
        duration_ms = int((now - t_press) * 1000)
        self._t_press = None
        # If nothing emitted yet, decide BRIEF or SUSTAINED by duration
        if not self._event_emitted:
            if duration_ms <= self.brief_max_ms:
                self._last_brief_at = now
                self.loop.create_task(self._on_event(InputEvent(pattern=InputPattern.BRIEF, timestamp=now, meta={"held_ms": duration_ms})))
            elif duration_ms >= self.sustained_min_ms:
                pattern = InputPattern.COMPOUND if self._double_tap_pending else InputPattern.SUSTAINED
                self.loop.create_task(self._on_event(InputEvent(pattern=pattern, timestamp=now, meta={"held_ms": duration_ms})))
            # else: between thresholds; emit nothing
        # Reset double-tap state if release ends the sequence
        self._double_tap_pending = False
