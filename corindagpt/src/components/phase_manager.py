from __future__ import annotations

import logging
from typing import List, Optional

from ..services.prompt_loader import load_prompt_for_phase

logger = logging.getLogger(__name__)


class PhaseManager:
    def __init__(self, performance_plan: List[int]):
        if not isinstance(performance_plan, list) or not performance_plan:
            logger.warning("PhaseManager: invalid performance_plan; defaulting to [1]")
            performance_plan = [1]
        # Normalize to ints, filter invalid entries
        normalized: List[int] = []
        for item in performance_plan:
            try:
                normalized.append(int(item))
            except Exception:
                continue
        if not normalized:
            normalized = [1]
        self._plan: List[int] = normalized
        self._index: int = 0
        self._current: int = self._plan[self._index]
        logger.info("PhaseManager initialized: current_phase=%s, plan=%s", self._current, self._plan)

    @property
    def current_phase(self) -> int:
        return self._current

    def get_next_phase(self) -> Optional[int]:
        nxt_idx = self._index + 1
        if 0 <= nxt_idx < len(self._plan):
            return self._plan[nxt_idx]
        return None

    def advance(self) -> int:
        nxt = self.get_next_phase()
        if nxt is None:
            logger.info("PhaseManager: already at end of plan (%s)", self._current)
            return self._current
        self._index += 1
        self._current = self._plan[self._index]
        logger.info("Phase transitioned to %s", self._current)
        # Eagerly load prompt for side-effects/logging correctness
        try:
            _ = self.load_prompt_for_current_phase()
        except Exception:
            pass
        return self._current

    def set(self, phase: int) -> int:
        try:
            desired = int(phase)
        except Exception:
            logger.warning("PhaseManager.set: invalid phase '%s'", phase)
            return self._current
        # Find first index of desired in plan; if not present, clamp to nearest existing?
        if desired in self._plan:
            self._index = self._plan.index(desired)
            self._current = self._plan[self._index]
            logger.info("Phase set to %s", self._current)
            try:
                _ = self.load_prompt_for_current_phase()
            except Exception:
                pass
        else:
            logger.warning("PhaseManager.set: phase %s not in plan %s", desired, self._plan)
        return self._current

    def load_prompt_for_current_phase(self):
        return load_prompt_for_phase(self._current)
