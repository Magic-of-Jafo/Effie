from __future__ import annotations

import asyncio

from pynput.keyboard import Key, KeyCode

from src.components.abstract_input_handler import KeyboardInputHandler


def _handler(hotkey: str) -> KeyboardInputHandler:
    loop = asyncio.new_event_loop()
    try:
        async def noop(evt):
            return None

        return KeyboardInputHandler(loop, on_event=noop, hotkey_name=hotkey)
    finally:
        loop.close()


def test_numpad_minus_matches_by_vk_only():
    h = _handler("numpad_minus")
    assert h._is_hotkey(KeyCode.from_vk(109)) is True          # numpad minus
    assert h._is_hotkey(KeyCode.from_char("-")) is False       # typed hyphen
    assert h._is_hotkey(Key.space) is False


def test_character_hotkeys_still_match_by_string():
    h = _handler("`")
    assert h._is_hotkey(KeyCode.from_char("`")) is True
    assert h._is_hotkey(KeyCode.from_char("-")) is False
    h2 = _handler("space")
    assert h2._is_hotkey(Key.space) is True
