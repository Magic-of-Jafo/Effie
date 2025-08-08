from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

try:
    from pynput import keyboard
except Exception:  # pragma: no cover - optional import for environments without GUI hooks
    keyboard = None  # type: ignore

logger = logging.getLogger(__name__)


KeyHandler = Callable[[], Awaitable[None]]


class InputHandler:
    """Translates keyboard events into async callbacks.

    This class is designed to be testable by calling `handle_press` and `handle_release`
    directly, without requiring a real system keyboard hook.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_press_active: KeyHandler,
        on_release_active: KeyHandler,
        hotkey_name: str = "f12",
    ) -> None:
        self.loop = loop
        self.on_press_active = on_press_active
        self.on_release_active = on_release_active
        self.hotkey_name = hotkey_name.lower()
        self._is_pressed = False
        self._listener: Optional["keyboard.Listener"] = None

    def _is_hotkey(self, key: object) -> bool:
        name = str(key).lower()
        return self.hotkey_name in name

    def _submit(self, coro: Awaitable[None]) -> None:
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    # Public handlers that can be unit-tested directly
    def handle_press(self, key: object) -> None:
        if not self._is_hotkey(key):
            return
        if self._is_pressed:
            logger.debug("InputHandler: %s pressed but already active; ignoring", self.hotkey_name)
            return
        self._is_pressed = True
        logger.info("Hotkey '%s' pressed", self.hotkey_name)
        self._submit(self.on_press_active())

    def handle_release(self, key: object) -> None:
        if not self._is_hotkey(key):
            return
        if not self._is_pressed:
            logger.debug("InputHandler: %s released but not active; ignoring", self.hotkey_name)
            return
        self._is_pressed = False
        logger.info("Hotkey '%s' released", self.hotkey_name)
        self._submit(self.on_release_active())

    def start_keyboard_listener(self) -> Optional["keyboard.Listener"]:
        """Start a global keyboard listener in a background thread.

        Returns the listener instance or None if `pynput` is unavailable.
        """
        if keyboard is None:
            logger.warning("pynput is not available; keyboard listener not started")
            return None
        self._listener = keyboard.Listener(
            on_press=self.handle_press,
            on_release=self.handle_release,
            suppress=False,
        )
        self._listener.start()
        logger.info("Keyboard listener started for hotkey '%s'", self.hotkey_name)
        return self._listener

    def stop_keyboard_listener(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            finally:
                self._listener = None
