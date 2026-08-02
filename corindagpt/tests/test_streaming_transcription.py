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
    assert text == "Guess the card."


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

    assert asyncio.run(run()) == "late word."


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


def test_pause_inserts_missing_sentence_boundary():
    # Model never punctuated; the 0.9s arrival gap is the sentence break
    svc = _svc()
    t = 100.0
    entries = [
        (t, "Guess"), (t + 0.3, " the"), (t + 0.6, " card"),
        (t + 1.5, " Give"), (t + 1.8, " us"), (t + 2.1, " the minute"),
    ]
    assert svc._assemble(entries) == "Guess the card. Give us the minute."


def test_late_period_reattaches_to_previous_sentence():
    # The model emits sentence 1's period as the first delta after the pause
    svc = _svc()
    t = 100.0
    entries = [
        (t, "Guess"), (t + 0.3, " the"), (t + 0.6, " card"),
        (t + 1.6, "."), (t + 1.61, " Give"), (t + 1.9, " us the minute"),
    ]
    assert svc._assemble(entries) == "Guess the card. Give us the minute."


def test_model_punctuation_not_doubled():
    # When the model DID punctuate before the pause, no extra period is added
    svc = _svc()
    t = 100.0
    entries = [
        (t, "Guess"), (t + 0.3, " the card"), (t + 0.5, "."),
        (t + 1.6, " Give"), (t + 1.9, " us the minute"),
    ]
    assert svc._assemble(entries) == "Guess the card. Give us the minute."


def test_trailing_period_added_at_window_end():
    svc = _svc()
    assert svc._assemble([(1.0, "Guess"), (1.3, " the card")]) == "Guess the card."
