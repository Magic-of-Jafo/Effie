from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Iterable, Dict, Any

try:
    from . import tts as tts_service
except Exception:  # pragma: no cover
    import corindagpt.src.services.tts as tts_service  # type: ignore

try:
    from ..utils.initialization import load_config  # type: ignore[relative-beyond-top-level]
except Exception:
    from corindagpt.src.utils.initialization import load_config  # type: ignore

logger = logging.getLogger(__name__)


class AudioQueue:
    """Simple rotating queue of local audio files (MP3/WAV) for quick playback.

    Designed for mapping BRIEF inputs to preloaded clips.
    """

    def __init__(self, paths: Iterable[Path]) -> None:
        self._items: List[Path] = [p for p in paths if p.exists() and p.is_file()]
        self._index: int = 0
        logger.info("AudioQueue initialized with %d items", len(self._items))

    @classmethod
    def from_directory(cls, directory: Path, pattern: str = "*.mp3") -> "AudioQueue":
        items = sorted(directory.glob(pattern), key=lambda p: p.name.lower())
        return cls(items)

    def reload_from_directory(self, directory: Path, pattern: str = "*.mp3") -> None:
        self._items = sorted(directory.glob(pattern), key=lambda p: p.name.lower())
        self._index = 0
        logger.info("AudioQueue reloaded: %d items", len(self._items))

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def next_path(self) -> Optional[Path]:
        if self.is_empty():
            return None
        path = self._items[self._index]
        self._index = (self._index + 1) % len(self._items)
        return path

    async def play_next(self, *, config: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """Load bytes of the next file and play using tts.play.

        Returns the Path played or None if queue empty.
        """
        path = self.next_path()
        if path is None:
            logger.info("AudioQueue: no items to play")
            return None
        try:
            data = path.read_bytes()
        except Exception as exc:
            logger.error("AudioQueue: failed to read %s: %s", path, exc)
            return None
        logger.info("AudioQueue: playing %s", path.name)
        # tts.play can handle MP3 via pydub or WAV paths
        await tts_service.play(data, config=config)
        return path


def default_sfx_dir(cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or load_config()
    assets_cfg: Dict[str, Any] = cfg.get("assets", {})
    sfx_dir_str = assets_cfg.get("sfx_dir") or "corindagpt/assets/sfx"
    return (Path(__file__).resolve().parents[2] / Path(sfx_dir_str)).resolve()


def build_default_queue(cfg: Optional[Dict[str, Any]] = None) -> AudioQueue:
    directory = default_sfx_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    return AudioQueue.from_directory(directory, pattern="*.mp3")
