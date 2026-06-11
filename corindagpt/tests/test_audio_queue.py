from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.services.audio_queue import AudioItem, AudioQueue


def _wav_file(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake")
    return p


def test_preload_sequential_order(tmp_path):
    q = AudioQueue()
    paths = [_wav_file(tmp_path, f"clip_{i}.wav") for i in range(3)]
    assert q.preload_paths(paths) == 3
    assert len(q) == 3


def test_push_front_takes_priority(tmp_path):
    q = AudioQueue()
    q.preload_paths([_wav_file(tmp_path, "clip.wav")])
    q.push_front(AudioItem(label="llm-response", data=b"RIFFxxxxWAVE"))
    assert q._items[0].label == "llm-response"
    assert q._items[1].label == "clip.wav"


@pytest.mark.asyncio
async def test_play_next_consumes_priority_item_then_loops_preloaded(tmp_path):
    q = AudioQueue(loop_preloaded=True)
    q.preload_paths([_wav_file(tmp_path, "clip.wav")])
    q.push_front(AudioItem(label="llm-response", data=b"RIFFxxxxWAVE"))
    with patch("src.services.audio_queue.tts_service.play", new=AsyncMock()) as mock_play:
        assert (await q.play_next()).label == "llm-response"
        assert (await q.play_next()).label == "clip.wav"
        # LLM item is gone; looping preloaded clip returned to the back
        assert (await q.play_next()).label == "clip.wav"
        assert mock_play.await_count == 3
    assert len(q) == 1  # only the looping clip remains


@pytest.mark.asyncio
async def test_play_next_no_loop_consumes(tmp_path):
    q = AudioQueue(loop_preloaded=False)
    q.preload_paths([_wav_file(tmp_path, "clip.wav")])
    with patch("src.services.audio_queue.tts_service.play", new=AsyncMock()):
        assert (await q.play_next()).label == "clip.wav"
        assert await q.play_next() is None
    assert q.is_empty()


def test_preload_random_keeps_all_items(tmp_path):
    q = AudioQueue()
    paths = [_wav_file(tmp_path, f"clip_{i}.wav") for i in range(5)]
    q.preload_paths(paths, order="random")
    labels = {item.label for item in q._items}
    assert labels == {f"clip_{i}.wav" for i in range(5)}


def test_missing_files_are_skipped(tmp_path):
    q = AudioQueue()
    count = q.preload_paths([tmp_path / "ghost.wav", _wav_file(tmp_path, "real.wav")])
    assert count == 1
