from __future__ import annotations

import asyncio
import logging
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional

try:
    from . import tts as tts_service
except Exception:  # pragma: no cover
    import corindagpt.src.services.tts as tts_service  # type: ignore

try:
    from ..utils.initialization import load_config  # type: ignore[relative-beyond-top-level]
except Exception:
    from corindagpt.src.utils.initialization import load_config  # type: ignore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AudioItem:
    """One queued utterance: either raw audio bytes or a file on disk."""

    label: str
    data: Optional[bytes] = None
    path: Optional[Path] = None
    preloaded: bool = False  # preloaded clips can loop; LLM responses never do

    def load_bytes(self) -> Optional[bytes]:
        if self.data is not None:
            return self.data
        if self.path is not None:
            try:
                return self.path.read_bytes()
            except Exception as exc:
                logger.error("AudioQueue: failed to read %s: %s", self.path, exc)
        return None


class AudioQueue:
    """Priority audio queue (FR8/FR9).

    Pre-loaded phrases sit in order at the back; new LLM responses are
    pushed to the front so the next BRIEF input (the string pull) speaks
    them ahead of anything else.
    """

    def __init__(self, *, loop_preloaded: bool = True) -> None:
        self._items: Deque[AudioItem] = deque()
        self._loop_preloaded = loop_preloaded
        self._play_lock = asyncio.Lock()

    def __len__(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return not self._items

    def push_front(self, item: AudioItem) -> None:
        self._items.appendleft(item)
        logger.info("AudioQueue: prioritized %r at front (queue size %d)", item.label, len(self._items))

    def append(self, item: AudioItem) -> None:
        self._items.append(item)

    def preload_paths(self, paths: Iterable[Path], *, order: str = "sequential") -> int:
        items = [
            AudioItem(label=p.name, path=p, preloaded=True)
            for p in paths
            if p.exists() and p.is_file()
        ]
        if order.lower() == "random":
            random.shuffle(items)
        for item in items:
            self._items.append(item)
        logger.info("AudioQueue: preloaded %d clips (%s order)", len(items), order)
        return len(items)

    async def play_next(self, *, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Play and consume the front item; returns its label or None if empty.

        Looping preloaded clips are re-appended to the back after playing.
        """
        async with self._play_lock:
            if not self._items:
                logger.info("AudioQueue: empty")
                return None
            item = self._items.popleft()
            data = item.load_bytes()
            if data is None:
                logger.warning("AudioQueue: skipping unreadable item %r", item.label)
                return item.label
            logger.info("AudioQueue: playing %r (%d left)", item.label, len(self._items))
            await tts_service.play(data, config=config)
            if item.preloaded and self._loop_preloaded:
                self._items.append(item)
            return item.label


def _resolve_dir(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    # Tolerate configured paths that include the project dir prefix
    if p.parts and p.parts[0] == PROJECT_ROOT.name:
        p = Path(*p.parts[1:]) if len(p.parts) > 1 else Path(".")
    return (PROJECT_ROOT / p).resolve()


def default_sfx_dir(cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or load_config()
    assets_cfg: Dict[str, Any] = cfg.get("assets", {})
    return _resolve_dir(str(assets_cfg.get("sfx_dir") or "assets/sfx"))


def build_default_queue(cfg: Optional[Dict[str, Any]] = None) -> AudioQueue:
    """Build the queue from config: audio_queue.preload, else legacy sfx dir."""
    cfg = cfg or load_config()
    aq_cfg: Dict[str, Any] = cfg.get("audio_queue") or {}
    preload_cfg: Dict[str, Any] = aq_cfg.get("preload") or {}

    loop_preloaded = bool(preload_cfg.get("loop", True))
    queue = AudioQueue(loop_preloaded=loop_preloaded)

    directory = (
        _resolve_dir(str(preload_cfg["dir"])) if preload_cfg.get("dir") else default_sfx_dir(cfg)
    )
    directory.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = sorted(
        [*directory.glob("*.mp3"), *directory.glob("*.wav")], key=lambda p: p.name.lower()
    )
    queue.preload_paths(paths, order=str(preload_cfg.get("order") or "sequential"))
    return queue


def response_playback_mode(cfg: Optional[Dict[str, Any]] = None) -> str:
    """'immediate' streams LLM responses as they generate; 'queued' parks the
    synthesized response at the queue front for the next string pull."""
    cfg = cfg or load_config()
    aq_cfg: Dict[str, Any] = cfg.get("audio_queue") or {}
    mode = str(aq_cfg.get("response_playback") or "immediate").lower()
    return mode if mode in ("immediate", "queued") else "immediate"
