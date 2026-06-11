from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def decode(text: str, *, config: Optional[Dict[str, Any]] = None) -> str:
    """Translate the magician's coded phrases into secret data for the LLM.

    Pass-through until Story 3.3 ports Hadley's system from the original
    project. The seam exists now so the sustained-input workflow already
    routes through it and only this module changes later.
    """
    if not isinstance(text, str):
        return ""
    logger.debug("Decoder (pass-through): %r", text)
    return text
