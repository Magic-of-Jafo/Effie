from __future__ import annotations

import logging
from enum import Enum
from typing import Callable, Optional, List


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

    def __init__(self, *, on_transition: Optional[TransitionHook] = None, performance_plan: Optional[List[int]] = None) -> None:
        self._state: State = State.IDLE
        self._on_transition: Optional[TransitionHook] = on_transition
        # adjacency map for valid transitions
        self._allowed = {
            State.IDLE: {State.LISTENING},
            # Allow cancel/no-data path back to IDLE
            State.LISTENING: {State.PROCESSING, State.IDLE},
            State.PROCESSING: {State.IDLE},
        }
        # --- Performance phase management ---
        plan = list(performance_plan or [1])
        # Validate plan: non-empty list of ints; else fallback to [1]
        validated: List[int] = []
        for item in plan:
            try:
                validated.append(int(item))
            except Exception:
                continue
        if not validated:
            logger.warning("Performance plan invalid or empty; defaulting to [1]")
            validated = [1]
        self._performance_plan: List[int] = validated
        self._current_phase_index: int = 0
        self._current_phase: int = self._performance_plan[self._current_phase_index]

        logger.info("StateMachine initialized in state %s (phase=%s)", self._state, self._current_phase)

    # -----------------
    # State functions
    # -----------------
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
            logger.info("State transition: %s -> %s (phase=%s)", prev, next_state, self._current_phase)
            if self._on_transition is not None:
                self._on_transition(prev, next_state)
        except Exception as exc:  # hook errors must not break state
            logger.error("on_transition hook error: %s", exc)
        return True

    def set_on_transition(self, hook: Optional[TransitionHook]) -> None:
        self._on_transition = hook

    # -----------------
    # Phase helpers
    # -----------------
    def get_current_phase(self) -> int:
        return self._current_phase

    def get_next_phase(self) -> Optional[int]:
        next_index = self._current_phase_index + 1
        if 0 <= next_index < len(self._performance_plan):
            return self._performance_plan[next_index]
        return None

    def advance_phase_if_requested(self, *, advance: bool = True) -> Optional[int]:
        """Advance to the next configured phase if requested and available.

        Returns the new current phase on success, or None if at end/no advancement.
        """
        if not advance:
            return self._current_phase
        next_phase = self.get_next_phase()
        if next_phase is None:
            logger.info("Phase: at end of performance plan; no further phases.")
            return None
        self._current_phase_index += 1
        self._current_phase = self._performance_plan[self._current_phase_index]
        logger.info("Phase advanced -> %s", self._current_phase)
        return self._current_phase
