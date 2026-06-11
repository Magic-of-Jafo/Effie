from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 10


class ConversationalMemory:
    """Rolling history of the performance conversation (FR10).

    A turn is one user utterance and/or one Effie utterance. Pre-recorded
    phrases played from the audio queue count as Effie speaking - she must
    remember everything the audience heard her say.
    """

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        self.max_turns = max(1, int(max_turns))
        # Each entry: {"role": "user" | "assistant", "content": str}
        self._messages: Deque[Dict[str, str]] = deque(maxlen=self.max_turns * 2)

    def add_user(self, text: str) -> None:
        if text and text.strip():
            self._messages.append({"role": "user", "content": text.strip()})

    def add_assistant(self, text: str) -> None:
        if text and text.strip():
            self._messages.append({"role": "assistant", "content": text.strip()})

    def add_exchange(self, user_text: str, assistant_text: str) -> None:
        self.add_user(user_text)
        self.add_assistant(assistant_text)

    def messages(self) -> List[Dict[str, str]]:
        """History as chat messages, oldest first, ready for the API."""
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        logger.info("Conversational memory cleared")


def build_memory(cfg: Optional[Dict[str, Any]] = None) -> ConversationalMemory:
    mem_cfg: Dict[str, Any] = (cfg or {}).get("memory") or {}
    max_turns = int(mem_cfg.get("max_turns", DEFAULT_MAX_TURNS))
    memory = ConversationalMemory(max_turns=max_turns)
    logger.info("Conversational memory: remembering up to %d turns", max_turns)
    return memory
