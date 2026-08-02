from __future__ import annotations

import asyncio
import time

from src.services.streaming_transcription import StreamingTranscriptionService


def _svc() -> StreamingTranscriptionService:
    return StreamingTranscriptionService({"openai_api_key": "test-key"})


def test_window_slices_by_arrival_time():
    svc = _svc()
    now = time.monotonic()
    svc._entries.extend([
        (now - 10.0, " way"), (now - 10.0, " earlier"),   # before the window
        (now - 2.0, " Guess"), (now - 1.5, " the"), (now - 1.0, " card"),
    ])
    svc._last_delta_at = now - 1.0
    text = asyncio.run(svc.get_window(now - 3.0, now - 0.5, tail_s=0.0))
    assert text == "Guess the card"


def test_window_waits_for_trailing_deltas():
    svc = _svc()
    t_press = time.monotonic()

    async def run() -> str:
        async def late_delta() -> None:
            await asyncio.sleep(0.15)
            now = time.monotonic()
            svc._entries.append((now, " late word"))
            svc._last_delta_at = now

        task = asyncio.create_task(late_delta())
        text = await svc.get_window(t_press, time.monotonic(), tail_s=0.8)
        await task
        return text

    assert asyncio.run(run()) == "late word"


def test_recent_returns_only_fresh_entries():
    svc = _svc()
    now = time.monotonic()
    svc._entries.extend([(now - 100.0, " stale"), (now - 1.0, " fresh")])
    entries = svc.recent(45.0)
    assert [e["text"] for e in entries] == [" fresh"]


def test_inactive_until_started():
    svc = _svc()
    assert svc.is_active() is False
    assert svc.status == "off"
