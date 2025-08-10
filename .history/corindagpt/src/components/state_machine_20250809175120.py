from __future__ import annotations

import logging
from enum import Enum
from typing import Callable, Optional


logger = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"


TransitionHook = Callable[[State, State], None]


class StateMachine:
    """Simple finite state machine for the core application lifecycle.

    Allowed transitions:
      - IDLE -> LISTENING
      - LISTENING -> PROCESSING
      - PROCESSING -> IDLE
    """

    def __init__(self, *, on_transition: Optional[TransitionHook] = None) -> None:
        self._state: State = State.IDLE
        self._on_transition: Optional[TransitionHook] = on_transition
        # adjacency map for valid transitions
        self._allowed = {
            State.IDLE: {State.LISTENING},
            # Allow cancel/no-data path back to IDLE
            State.LISTENING: {State.PROCESSING, State.IDLE},
            State.PROCESSING: {State.IDLE},
        }
        logger.info("StateMachine initialized in state %s", self._state)

    @property
    def state(self) -> State:
        return self._state

    def can_transition(self, next_state: State) -> bool:
        allowed_next = self._allowed.get(self._state, set())
        return next_state in allowed_next

    def transition(self, next_state: State) -> bool:
        """Attempt to transition to next_state.

        Returns True on success; False if the transition is not allowed.
        Emits on_transition hook and logs on successful transitions.
        """
        if not isinstance(next_state, State):
            raise ValueError("next_state must be a State enum value")

        if not self.can_transition(next_state):
            logger.warning(
                "StateMachine: invalid transition %s -> %s (ignored)", self._state, next_state
            )
            return False

        prev = self._state
        self._state = next_state
        try:
            logger.info("State transition: %s -> %s", prev, next_state)
            if self._on_transition is not None:
                self._on_transition(prev, next_state)
        except Exception as exc:  # hook errors must not break state
            logger.error("on_transition hook error: %s", exc)
        return True

    def set_on_transition(self, hook: Optional[TransitionHook]) -> None:
        self._on_transition = hook
